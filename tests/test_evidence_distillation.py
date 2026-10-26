from __future__ import annotations

import json

import torch
from torch import nn

from modules.evidence_distillation import (
    EVIDENCE_EMBEDDING_SCHEMA_VERSION,
    build_evidence_distillation_targets,
    evidence_representation_distillation_loss,
    load_evidence_embedding_cache,
    sha256_file,
)
from modules.llm_dataset import TeacherCacheDataset


class TinyTokenizer:
    def tokenize(self, word):
        return [str(word).lower()]


class TinyDataset:
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
            torch.zeros((4, 3, 8, 8)),
            torch.zeros(6, dtype=torch.long),
            torch.tensor([1, 0, 0, 0]),
        )


def artifact(embedding=(1.0, 0.0)):
    return {
        "schema_version": EVIDENCE_EMBEDDING_SCHEMA_VERSION,
        "teacher_cache_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "pooling": "mean",
        "sample_ids": torch.tensor([0]),
        "starts": torch.tensor([0]),
        "ends": torch.tensor([1]),
        "label_ids": torch.tensor([1]),  # PER
        "image_targets": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "confidences": torch.tensor([0.95]),
        "embeddings": torch.tensor([embedding], dtype=torch.float16),
    }


def test_evidence_cache_and_targets_require_gold_and_valid_images(tmp_path):
    path = tmp_path / "evidence.pt"
    torch.save(artifact(), path)
    index, loaded = load_evidence_embedding_cache(path)
    assert index == {0: [0]}

    token_masks, image_targets, embeddings, weights, accepted = (
        build_evidence_distillation_targets(
            words=["Alice"],
            gold_labels=["B-PER"],
            tokenizer=TinyTokenizer(),
            max_sequence_length=6,
            image_mask=torch.tensor([1, 0, 0, 0]),
            artifact=loaded,
            row_indices=index[0],
            max_entities=2,
        )
    )
    assert accepted == 1
    assert token_masks[0, 1] == 1
    assert image_targets[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert torch.allclose(embeddings[0], torch.tensor([1.0, 0.0]))
    assert torch.isclose(weights[0], torch.tensor(0.95))

    _, _, _, invalid_weights, invalid_count = build_evidence_distillation_targets(
        words=["Alice"],
        gold_labels=["O"],
        tokenizer=TinyTokenizer(),
        max_sequence_length=6,
        image_mask=torch.tensor([1, 0, 0, 0]),
        artifact=loaded,
        row_indices=index[0],
        max_entities=2,
    )
    assert invalid_count == 0
    assert invalid_weights.count_nonzero() == 0


def test_teacher_dataset_appends_evidence_tensors_only_when_enabled(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        json.dumps({"sample_id": 0, "teacher_task": "grounding", "entities": []})
        + "\n",
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.pt"
    cache_artifact = artifact()
    cache_artifact["teacher_cache_sha256"] = sha256_file(cache_path)
    torch.save(cache_artifact, evidence_path)
    dataset = TeacherCacheDataset(
        TinyDataset(),
        cache_path,
        expected_teacher_task="grounding",
        enable_grounding=False,
        enable_evidence_distillation=True,
        evidence_embedding_path=evidence_path,
        evidence_max_entities=2,
        evidence_embedding_dim=2,
    )
    sample = dataset[0]
    assert len(sample) == 15
    assert sample[-4].shape == (2, 6)
    assert sample[-3].shape == (2, 4)
    assert sample[-2].shape == (2, 2)
    assert torch.isclose(sample[-1][0], torch.tensor(0.95))
    assert sample[-1][1] == 0
    assert dataset.evidence_distillation_summary()["artifact_entities"] == 1


def test_teacher_dataset_rejects_stale_evidence_artifact(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        json.dumps({"sample_id": 0, "teacher_task": "grounding", "entities": []})
        + "\n",
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.pt"
    torch.save(artifact(), evidence_path)

    try:
        TeacherCacheDataset(
            TinyDataset(),
            cache_path,
            expected_teacher_task="grounding",
            enable_grounding=False,
            enable_evidence_distillation=True,
            evidence_embedding_path=evidence_path,
            evidence_max_entities=2,
            evidence_embedding_dim=2,
        )
    except ValueError as error:
        assert "teacher-cache hash" in str(error)
    else:
        raise AssertionError("A stale evidence artifact was accepted")


def test_evidence_loss_aligns_teacher_and_can_detach_text():
    text = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    images = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    token_masks = torch.tensor([[[1.0]]])
    image_targets = torch.tensor([[[1.0]]])
    weights = torch.tensor([[1.0]])
    projection = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[:, 2:] = torch.eye(2)

    aligned = evidence_representation_distillation_loss(
        text_features=text,
        image_features=images,
        teacher_embeddings=torch.tensor([[[1.0, 0.0]]]),
        token_masks=token_masks,
        image_targets=image_targets,
        weights=weights,
        student_projection=projection,
        temperature=0.1,
        contrastive_weight=0.0,
        detach_text=True,
        detach_image=False,
    )
    opposed = evidence_representation_distillation_loss(
        text_features=text,
        image_features=images,
        teacher_embeddings=torch.tensor([[[-1.0, 0.0]]]),
        token_masks=token_masks,
        image_targets=image_targets,
        weights=weights,
        student_projection=projection,
        temperature=0.1,
        contrastive_weight=0.0,
        detach_text=True,
        detach_image=False,
    )
    assert torch.isclose(aligned, torch.tensor(0.0))
    assert torch.isclose(opposed, torch.tensor(2.0))
    aligned.backward()
    assert text.grad is None
    assert images.grad is not None
