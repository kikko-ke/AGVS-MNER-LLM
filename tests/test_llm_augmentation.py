from __future__ import annotations

import ast
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION,
    AUGMENTATION_SCHEMA_VERSION,
    extract_protected_entities,
    labels_for_augmentation,
    validate_augmentation_output,
)
from llm.client import LLMTeacherClient
from scripts import build_augmented_dataset
from scripts.build_llm_augmentation_cache import (
    load_existing_records,
    select_source_records,
    sha256_file,
)


SOURCE_WORDS = ["watching", "Acme", "Corp", "at", "the", "arena"]
SOURCE_LABELS = ["O", "B-ORG", "I-ORG", "O", "O", "O"]


def accepted_payload() -> dict:
    return {
        "abstain": False,
        "context_segments": [
            ["watching"],
            ["beside", "the", "bright", "arena"],
        ],
        "used_image_ids": [0],
        "change_summary": "the visible stage changes the local setting",
        "abstain_reason": "",
    }


def test_valid_augmentation_preserves_entities_and_rebuilds_labels():
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    response = validate_augmentation_output(
        accepted_payload(), SOURCE_WORDS, entities, {0}
    )
    assert response.abstain is False
    assert response.tokens[1:3] == ("Acme", "Corp")
    assert labels_for_augmentation(
        len(response.tokens), entities, response.entity_spans
    ) == ["O", "B-ORG", "I-ORG", "O", "O", "O", "O"]
    assert 0.15 <= response.context_change_ratio <= 0.85


def test_abstention_is_valid_but_contains_no_training_example():
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    response = validate_augmentation_output(
        {
            "abstain": True,
            "context_segments": [],
            "used_image_ids": [],
            "change_summary": "",
            "abstain_reason": "image does not support a safe rewrite",
        },
        SOURCE_WORDS,
        entities,
        {0},
    )
    assert response.abstain is True
    assert response.tokens == ()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["context_segments"][1].extend(["Acme", "Corp"]),
            "duplicated protected entity",
        ),
        (
            lambda payload: payload["used_image_ids"].__setitem__(0, 3),
            "was not sent",
        ),
        (
            lambda payload: payload["context_segments"][1].__setitem__(2, "@new_account"),
            "suspicious named context",
        ),
    ],
)
def test_invalid_augmentations_are_rejected(mutator, message):
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    payload = accepted_payload()
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        validate_augmentation_output(payload, SOURCE_WORDS, entities, {0})


def test_bio_parser_rejects_orphan_inside_label():
    with pytest.raises(ValueError, match="invalid BIO"):
        extract_protected_entities(["Acme"], ["I-ORG"])


def test_context_segment_count_is_structurally_enforced():
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    payload = accepted_payload()
    payload["context_segments"] = [payload["context_segments"][0]]
    with pytest.raises(ValueError, match="exactly 2 lists"):
        validate_augmentation_output(payload, SOURCE_WORDS, entities, {0})


def test_generated_context_case_is_canonicalized_without_touching_entity_surface():
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    payload = accepted_payload()
    payload["context_segments"][1][2] = "BRIGHT"
    response = validate_augmentation_output(payload, SOURCE_WORDS, entities, {0})
    assert response.tokens[1:3] == ("Acme", "Corp")
    assert "bright" in response.tokens
    assert "BRIGHT" not in response.tokens


def test_client_validates_before_returning(monkeypatch):
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    client = LLMTeacherClient(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="model",
        max_retries=0,
    )
    monkeypatch.setattr(
        client,
        "_prepare_images",
        lambda paths: (list(paths), ["data:image/jpeg;base64,AA==", None]),
    )

    def fake_complete(
        system_prompt,
        user_text,
        prepared_images,
        *,
        validator,
        temperature,
        retry_temperature,
    ):
        assert AUGMENTATION_PROMPT_VERSION in system_prompt
        assert "Protected entities" in user_text
        assert len(prepared_images) == 2
        assert temperature == 0.1
        assert retry_temperature == 0.6
        return validator(accepted_payload())

    monkeypatch.setattr(client, "_complete_prepared_json", fake_complete)
    response = client.complete_augmentation_json(
        SOURCE_WORDS, [Path("image.jpg"), None], entities
    )
    assert response.used_image_ids == (0,)


def _selection_record(sample_id: int, label: str) -> dict:
    words = ["the", f"entity{sample_id}"]
    labels = ["O", f"B-{label}"]
    return {
        "sample_id": sample_id,
        "words": words,
        "labels": labels,
        "images": [f"{sample_id}.jpg"],
        "entities": extract_protected_entities(words, labels),
        "source_fingerprint": str(sample_id),
    }


def test_selection_is_reproducible_and_balances_target_types():
    records = [
        *[_selection_record(index, "MIS") for index in range(6)],
        *[_selection_record(index + 10, "ORG") for index in range(6)],
        _selection_record(99, "PER"),
    ]
    first = select_source_records(records, ["MIS", "ORG"], 6, 123)
    second = select_source_records(records, ["MIS", "ORG"], 6, 123)
    assert [item["sample_id"] for item in first] == [
        item["sample_id"] for item in second
    ]
    selected_types = [next(iter({e.label for e in item["entities"]})) for item in first]
    assert selected_types.count("MIS") == 3
    assert selected_types.count("ORG") == 3


def test_policy_rejection_is_a_resolved_non_training_cache_record(tmp_path):
    source = _selection_record(0, "MIS")
    cache = tmp_path / "augmentation.jsonl"
    cache.write_text(
        json.dumps(
            {
                "schema_version": AUGMENTATION_SCHEMA_VERSION,
                "prompt_version": AUGMENTATION_PROMPT_VERSION,
                "sample_id": 0,
                "source_fingerprint": source["source_fingerprint"],
                "llm_model": "model",
                "status": "rejected",
                "policy_rejection": "context was unchanged",
                "labels": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = Namespace(
        min_context_change_ratio=0.15,
        max_context_change_ratio=0.85,
        min_length_ratio=0.60,
        max_length_ratio=1.50,
        max_words=62,
    )
    assert load_existing_records(
        cache, {0: source}, model="model", args=args
    ) == {0}


def test_dataset_builder_writes_llm_arm_and_matched_control(tmp_path, monkeypatch):
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    source_record = {
        "text": SOURCE_WORDS,
        "label": SOURCE_LABELS,
        "images": ["image.jpg"],
    }
    train_path = input_dir / "MNER-UNI_train.txt"
    train_path.write_text(repr(source_record) + "\n", encoding="utf-8")
    validation = repr({"text": ["x"], "label": ["O"], "images": ["x.jpg"]}) + "\n"
    (input_dir / "MNER-UNI_val.txt").write_text(validation, encoding="utf-8")
    (input_dir / "MNER-UNI_test.txt").write_text(validation, encoding="utf-8")

    cache_path = tmp_path / "augmentation.jsonl"
    entities = extract_protected_entities(SOURCE_WORDS, SOURCE_LABELS)
    response = validate_augmentation_output(
        accepted_payload(), SOURCE_WORDS, entities, {0}
    )
    cache_record = {
        "schema_version": AUGMENTATION_SCHEMA_VERSION,
        "prompt_version": AUGMENTATION_PROMPT_VERSION,
        "sample_id": 0,
        "status": "accepted",
        "valid_image_ids": [0],
        "labels": labels_for_augmentation(
            len(response.tokens), entities, response.entity_spans
        ),
        **response.to_dict(),
    }
    cache_path.write_text(
        json.dumps(cache_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cache_path.with_name(cache_path.name + ".meta.json").write_text(
        json.dumps(
            {
                "schema_version": AUGMENTATION_SCHEMA_VERSION,
                "input_sha256": sha256_file(train_path),
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "llm"
    control_dir = tmp_path / "control"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_augmented_dataset.py",
            "--input-dir",
            str(input_dir),
            "--cache",
            str(cache_path),
            "--output-dir",
            str(output_dir),
            "--control-output-dir",
            str(control_dir),
        ],
    )
    build_augmented_dataset.main()
    llm_lines = (output_dir / "MNER-UNI_train.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    control_lines = (control_dir / "MNER-UNI_train.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(llm_lines) == len(control_lines) == 2
    assert ast.literal_eval(llm_lines[1])["text"] == list(response.tokens)
    assert control_lines[0] == control_lines[1]
    assert (output_dir / "MNER-UNI_val.txt").read_bytes() == (
        input_dir / "MNER-UNI_val.txt"
    ).read_bytes()
