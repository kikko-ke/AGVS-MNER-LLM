from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION
from scripts.build_llm_atomic_review_cache import (
    load_atomic_candidates,
    load_existing_atomic_reviews,
    validate_test_request_unlock,
)
from scripts.export_atomic_edit_candidates import (
    atomic_review_record_fingerprint,
    load_source_ids,
    select_low_margin_atomic_routes,
)


def _atomic_record(sample_id: int, margin: float = 0.1) -> dict:
    baseline = [
        {"start": 0, "end": 1, "label": "PER", "surface": "Alice"}
    ]
    record = {
        "schema_version": "atomic-edit-candidates-v1",
        "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
        "sample_id": sample_id,
        "source_sample_id": 100 + sample_id,
        "split": "meta",
        "official_test": False,
        "provenance": {"official_test": False},
        "checkpoint_sha256": "c" * 64,
        "input_sha256": "i" * 64,
        "source_ids_sha256": "s" * 64,
        "words": ["Alice", "works"],
        "retained_word_indices": [0, 1],
        "image_names": [],
        "valid_image_ids": [],
        "baseline_candidate_id": f"C_BASE{sample_id}",
        "baseline_entities": baseline,
        "candidates": [],
        "edits": [
            {
                "edit_id": f"E_EDIT{sample_id}",
                "candidate_id": f"C_EDIT{sample_id}",
                "edit_class": "deletion_spurious_removal",
                "removed_entity": {
                    "start": 0,
                    "end": 1,
                    "label": "PER",
                    "surface": "Alice",
                },
                "added_entity": None,
                "baseline_entities": baseline,
                "crf_rank": 2,
                "score_gap_per_word": margin,
                "display_position": 0,
            }
        ],
        "route_features": {"minimum_atomic_score_gap_per_word": margin},
        "review_route": {
            "selected": True,
            "strategy": "lowest_atomic_score_gap",
            "route_order": sample_id,
        },
    }
    record["review_fingerprint"] = atomic_review_record_fingerprint(record)
    return record


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_low_margin_route_uses_only_gold_free_atomic_margin() -> None:
    records = [
        _atomic_record(10, 0.3),
        _atomic_record(20, 0.1),
        _atomic_record(30, 0.2),
    ]
    routes = select_low_margin_atomic_routes(records, 2)

    assert [route["sample_id"] for route in routes] == [20, 30]
    assert [route["route_order"] for route in routes] == [0, 1]


def test_low_margin_route_keeps_but_does_not_route_empty_edit_records() -> None:
    empty = _atomic_record(1, 0.01)
    empty["edits"] = []
    empty["route_features"]["minimum_atomic_score_gap_per_word"] = None
    empty["review_route"]["selected"] = False
    empty["review_fingerprint"] = atomic_review_record_fingerprint(empty)

    assert select_low_margin_atomic_routes([empty, _atomic_record(2, 0.2)], 1) == [
        {
            "sample_id": 2,
            "route_order": 0,
            "minimum_atomic_score_gap_per_word": 0.2,
        }
    ]


def test_candidate_loader_rejects_any_fingerprint_tampering(tmp_path: Path) -> None:
    record = _atomic_record(0)
    artifact = tmp_path / "atomic.jsonl"
    _write_jsonl(artifact, [record])
    assert load_atomic_candidates(artifact)[0]["source_sample_id"] == 100

    tampered = deepcopy(record)
    tampered["edits"][0]["edit_class"] = "addition_missing_recovery"
    _write_jsonl(artifact, [tampered])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_atomic_candidates(artifact)


def test_resume_strictly_validates_review_provenance_and_schema(tmp_path: Path) -> None:
    record = _atomic_record(0)
    decision = {
        "schema_version": "atomic-edit-reviews-v1",
        "sample_id": 0,
        "source_sample_id": 100,
        "split": "meta",
        "official_test": False,
        "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
        "review_fingerprint": record["review_fingerprint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "input_sha256": record["input_sha256"],
        "source_ids_sha256": record["source_ids_sha256"],
        "llm_model": "teacher",
        "reviews": [
            {
                "edit_id": "E_EDIT0",
                "verdict": "KEEP_BASELINE",
                "confidence": 0.8,
                "text_support": "baseline",
                "visual_support": "none",
                "used_image_ids": [],
                "decisive_spans": [],
                "brief_reason": "The sentence supports the original person entity.",
            }
        ],
    }
    cache = tmp_path / "reviews.jsonl"
    _write_jsonl(cache, [decision])
    assert load_existing_atomic_reviews(cache, {0: record}, "teacher") == {0}

    stale = {**decision, "input_sha256": "different"}
    _write_jsonl(cache, [stale])
    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        load_existing_atomic_reviews(cache, {0: record}, "teacher")


def test_source_id_sidecar_requires_exact_unique_mapping(tmp_path: Path) -> None:
    sidecar = tmp_path / "ids.json"
    sidecar.write_text(
        json.dumps({"source_sample_ids": [7, 3, 9]}), encoding="utf-8"
    )
    values, digest = load_source_ids(sidecar, 3)
    assert values == [7, 3, 9]
    assert digest is not None and len(digest) == 64

    sidecar.write_text(
        json.dumps({"source_sample_ids": [7, 7, 9]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicates"):
        load_source_ids(sidecar, 3)


def test_official_test_lock_uses_input_derived_provenance() -> None:
    record = _atomic_record(0)
    record["official_test"] = True
    record["provenance"]["official_test"] = True
    record["review_fingerprint"] = atomic_review_record_fingerprint(record)

    with pytest.raises(ValueError, match="--allow-test"):
        validate_test_request_unlock([record], allow_test=False, scope="all")
    with pytest.raises(ValueError, match="--scope all"):
        validate_test_request_unlock([record], allow_test=True, scope="routed")
    validate_test_request_unlock([record], allow_test=True, scope="all")


def test_atomic_pipeline_scripts_support_direct_execution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for script in (
        "create_atomic_preflight_split.py",
        "export_atomic_edit_candidates.py",
        "build_llm_atomic_review_cache.py",
    ):
        result = subprocess.run(
            [sys.executable, "-I", str(project_root / "scripts" / script), "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
