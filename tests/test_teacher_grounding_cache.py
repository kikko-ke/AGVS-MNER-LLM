from __future__ import annotations

import base64
import json
from io import BytesIO

from PIL import Image
import pytest
import torch

from llm.client import LLMTeacherClient
from llm.schema import validate_teacher_output
from modules.llm_dataset import TeacherCacheDataset
from scripts.build_teacher_cache import (
    load_valid_sample_ids,
    validate_grounding_contract,
    validate_utility_contract,
)


def test_teacher_image_preparation_repairs_api_size_requirement(tmp_path):
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (10, 20), color="red").save(image_path)

    data_url = LLMTeacherClient.image_data_url(image_path)
    encoded = data_url.split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGB"
        assert min(image.size) >= 32


def test_teacher_image_preparation_rejects_non_image(tmp_path):
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="unreadable image"):
        LLMTeacherClient.image_data_url(image_path)


def test_resume_rejects_mixing_mner_and_grounding_caches(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "sample_id": 0,
                "teacher_task": "grounding",
                "token_probs": [[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
                "image_scores": [1.0, 0.0, 0.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_valid_sample_ids(cache_path, 2, 3, "grounding") == {0}
    with pytest.raises(ValueError):
        load_valid_sample_ids(cache_path, 2, 3, "mner")


def test_grounding_contract_requires_each_gold_entity_once():
    annotated = [{"start": 0, "end": 1, "label": "PER"}]
    teacher = validate_teacher_output(
        {
            "entities": [
                {
                    "start": 0,
                    "end": 1,
                    "label": "PER",
                    "image_ids": [],
                    "confidence": 0.9,
                }
            ],
            "image_scores": [1, 0, 0, 0],
        },
        token_count=1,
    )
    validate_grounding_contract(teacher, annotated)

    with pytest.raises(ValueError, match="changed the gold entity contract"):
        validate_grounding_contract(teacher, [])


def test_utility_contract_requires_explicit_consistent_fields():
    payload = {
        "entities": [
            {
                "start": 0,
                "end": 1,
                "label": "PER",
                "image_ids": [],
                "confidence": 0.9,
                "text_sufficient": True,
                "visual_utility": 0.0,
            }
        ],
        "image_scores": [1, 0, 0, 0],
    }
    teacher = validate_teacher_output(payload, token_count=1)
    validate_utility_contract(payload, teacher)

    payload["entities"][0]["image_ids"] = [0]
    contradictory = validate_teacher_output(payload, token_count=1)
    with pytest.raises(ValueError, match="contradicted"):
        validate_utility_contract(payload, contradictory)


class TinyTokenizer:
    def tokenize(self, word):
        return [word.lower()]


class TinyBaseDataset:
    label_mapping = {"O": 0, "B-PER": 1, "I-PER": 2}
    examples = {
        "words": [["Alice"]],
        "labels": [["B-PER"]],
        "images": [["alice.jpg"]],
    }
    tokenizer = TinyTokenizer()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return (
            torch.zeros(6, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
            torch.ones(6, dtype=torch.long),
            torch.zeros((4, 3, 224, 224)),
            torch.zeros(6, dtype=torch.long),
            torch.tensor([1, 0, 0, 0]),
        )


def test_teacher_cache_dataset_builds_five_way_targets_and_summary(tmp_path):
    cache_path = tmp_path / "MNER-UNI_train.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "sample_id": 0,
                "teacher_task": "grounding",
                "token_probs": [[1 / 3] * 3 for _ in range(6)],
                "image_scores": [1.0, 0.0, 0.0, 0.0],
                "entities": [
                    {
                        "start": 0,
                        "end": 1,
                        "label": "PER",
                        "image_ids": [0],
                        "confidence": 0.9,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TeacherCacheDataset(
        TinyBaseDataset(),
        cache_path,
        expected_teacher_task="grounding",
    )
    sample = dataset[0]

    assert len(sample) == 11
    assert sample[-2].shape == (6, 5)
    assert sample[-2][1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert dataset.grounding_summary()["entities"] == 1

    with pytest.raises(ValueError, match="expected 'mner'"):
        TeacherCacheDataset(
            TinyBaseDataset(), cache_path, expected_teacher_task="mner"
        )
