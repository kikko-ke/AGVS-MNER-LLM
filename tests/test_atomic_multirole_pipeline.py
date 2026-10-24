from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path

import pytest

from llm.atomic_multirole_schema import (
    ATOMIC_MULTIROLE_PROMPT_VERSION,
    ROLE_CRITIC,
    ROLE_TEXT,
    ROLE_VISUAL,
)
from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION
from scripts.build_llm_atomic_multirole_cache import (
    SCHEMA_VERSION,
    build_metadata,
    expected_keys,
    load_existing_reviews,
    load_manifest_roles,
    select_atomic_edits,
    write_or_validate_metadata,
)
from scripts.export_atomic_edit_candidates import atomic_review_record_fingerprint


@pytest.fixture
def local_tmp_path() -> Path:
    """Workspace-local temp path for restricted Windows test environments."""
    root = Path(".pytest_atomic_multirole_cases")
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


def _candidate(sample_id: int, margin: float, *, source_id: int | None = None) -> dict:
    source_id = 100 + sample_id if source_id is None else source_id
    baseline = [{"start": 0, "end": 1, "label": "PER", "surface": "Alice"}]
    record = {
        "schema_version": "atomic-edit-candidates-v1",
        "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
        "sample_id": sample_id,
        "source_sample_id": source_id,
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
                "edit_id": f"E_HIGH{sample_id}",
                "candidate_id": f"C_HIGH{sample_id}",
                "edit_class": "deletion_spurious_removal",
                "removed_entity": baseline[0],
                "added_entity": None,
                "baseline_entities": baseline,
                "crf_rank": 3,
                "score_gap_per_word": margin + 0.2,
                "display_position": 1,
            },
            {
                "edit_id": f"E_LOW{sample_id}",
                "candidate_id": f"C_LOW{sample_id}",
                "edit_class": "deletion_spurious_removal",
                "removed_entity": baseline[0],
                "added_entity": None,
                "baseline_entities": baseline,
                "crf_rank": 2,
                "score_gap_per_word": margin,
                "display_position": 0,
            },
        ],
        "route_features": {
            "minimum_atomic_score_gap_per_word": margin,
            "top1_top2_gap_per_word": margin,
        },
        "review_route": {
            "selected": True,
            "strategy": "lowest_atomic_score_gap",
            "route_order": sample_id,
        },
    }
    record["review_fingerprint"] = atomic_review_record_fingerprint(record)
    return record


def _manifest(path: Path, records: list[dict], roles: list[str]) -> Path:
    payload = {
        "gold_labels_used_for_assignment": False,
        "official_validation_or_test_used": False,
        "assignments": [
            {
                "sample_id": record["sample_id"],
                "source_sample_id": record["source_sample_id"],
                "role": role,
            }
            for record, role in zip(records, roles, strict=True)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _selected(records: list[dict], manifest: Path, **kwargs) -> list[dict]:
    return select_atomic_edits(
        records,
        load_manifest_roles(manifest),
        roles=set(kwargs.get("roles", {"meta_train"})),
        route_samples=kwargs.get("route_samples", 20),
        max_edits_per_sample=kwargs.get("max_edits_per_sample", 1),
    )


def _cached_row(item: dict, *, role: str = ROLE_TEXT) -> dict:
    record = item["record"]
    verdict = "NO_OBJECTION" if role == ROLE_CRITIC else "SUPPORT_BASELINE"
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": item["sample_id"],
        "source_sample_id": record["source_sample_id"],
        "split": record["split"],
        "official_test": False,
        "meta_role": item["meta_role"],
        "route_order": item["route_order"],
        "prompt_version": ATOMIC_MULTIROLE_PROMPT_VERSION,
        "review_fingerprint": record["review_fingerprint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "input_sha256": record["input_sha256"],
        "source_ids_sha256": record["source_ids_sha256"],
        "llm_model": "teacher",
        "edit_id": item["edit_id"],
        "role": role,
        "verdict": verdict,
        "evidence_target": "none" if role == ROLE_CRITIC else "removed_entity",
        "reason_code": "baseline_supported",
        "used_image_ids": [],
        "brief_evidence": "The supplied edit is not justified.",
        "request_latency_seconds": 1.0,
        "request_attempts": 1,
        "usage": None,
    }


def test_manifest_is_gold_free_unique_and_locks_evaluation(
    local_tmp_path: Path,
) -> None:
    tmp_path = local_tmp_path
    records = [_candidate(0, 0.1), _candidate(1, 0.2)]
    path = _manifest(tmp_path / "manifest.json", records, ["meta_train", "meta_evaluation"])
    mapping = load_manifest_roles(path)
    assert mapping[0] == (100, "meta_train")
    assert mapping[1] == (101, "meta_evaluation")

    contaminated = json.loads(path.read_text(encoding="utf-8"))
    contaminated["gold_labels_used_for_assignment"] = True
    path.write_text(json.dumps(contaminated), encoding="utf-8")
    with pytest.raises(ValueError, match="gold labels"):
        load_manifest_roles(path)


def test_selection_uses_lowest_sample_and_edit_margin_without_gold(
    local_tmp_path: Path,
) -> None:
    tmp_path = local_tmp_path
    records = [_candidate(0, 0.3), _candidate(1, 0.1), _candidate(2, 0.2)]
    manifest = _manifest(
        tmp_path / "manifest.json",
        records,
        ["meta_train", "meta_train", "meta_calibration"],
    )
    selected = _selected(records, manifest, route_samples=1)
    assert [(row["sample_id"], row["edit_id"]) for row in selected] == [
        (1, "E_LOW1")
    ]
    assert selected[0]["route_order"] == 0

    calibration = _selected(
        records,
        manifest,
        roles={"meta_calibration"},
        route_samples=1,
        max_edits_per_sample=2,
    )
    assert [row["edit_id"] for row in calibration] == ["E_LOW2", "E_HIGH2"]


def test_selection_rejects_manifest_source_id_mismatch(local_tmp_path: Path) -> None:
    tmp_path = local_tmp_path
    record = _candidate(0, 0.1)
    manifest = _manifest(tmp_path / "manifest.json", [record], ["meta_train"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["assignments"][0]["source_sample_id"] = 999
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_sample_id mismatch"):
        _selected([record], manifest)


def test_expected_keys_cover_every_edit_role_and_metadata_is_frozen(
    local_tmp_path: Path,
) -> None:
    tmp_path = local_tmp_path
    record = _candidate(0, 0.1)
    manifest = _manifest(tmp_path / "manifest.json", [record], ["meta_train"])
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(record) + "\n", encoding="utf-8")
    selected = _selected([record], manifest)
    roles = [ROLE_TEXT, ROLE_VISUAL, ROLE_CRITIC]
    assert expected_keys(selected, roles) == [
        (0, "E_LOW0", ROLE_TEXT),
        (0, "E_LOW0", ROLE_VISUAL),
        (0, "E_LOW0", ROLE_CRITIC),
    ]
    metadata = build_metadata(
        candidates_path=candidates,
        manifest_path=manifest,
        selected=selected,
        reviewer_roles=roles,
        roles=["meta_train"],
        route_samples=20,
        max_edits_per_sample=1,
        scope="all",
    )
    assert metadata["gold_entity_labels_used"] is False
    assert metadata["expected_reviews"] == 3
    assert metadata["selected_edits"][0]["meta_role"] == "meta_train"

    path = tmp_path / "reviews.jsonl.meta.json"
    write_or_validate_metadata(path, metadata, resume=False)
    write_or_validate_metadata(path, metadata, resume=True)
    changed = deepcopy(metadata)
    changed["route_samples"] = 21
    with pytest.raises(ValueError, match="does not match"):
        write_or_validate_metadata(path, changed, resume=True)


def test_resume_uses_composite_key_and_rejects_duplicate_stale_or_missing_provenance(
    local_tmp_path: Path,
) -> None:
    tmp_path = local_tmp_path
    record = _candidate(0, 0.1)
    manifest = _manifest(tmp_path / "manifest.json", [record], ["meta_train"])
    selected = _selected([record], manifest)
    item = selected[0]
    selected_by_key = {(0, item["edit_id"]): item}
    cache = tmp_path / "reviews.jsonl"
    row = _cached_row(item)
    cache.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_existing_reviews(
        cache,
        selected_by_key,
        model="teacher",
        reviewer_roles={ROLE_TEXT, ROLE_VISUAL, ROLE_CRITIC},
    ) == {(0, "E_LOW0", ROLE_TEXT)}

    cache.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate review key"):
        load_existing_reviews(
            cache,
            selected_by_key,
            model="teacher",
            reviewer_roles={ROLE_TEXT, ROLE_VISUAL, ROLE_CRITIC},
        )

    stale = deepcopy(row)
    stale["review_fingerprint"] = "stale"
    cache.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale review fingerprint"):
        load_existing_reviews(
            cache,
            selected_by_key,
            model="teacher",
            reviewer_roles={ROLE_TEXT},
        )

    missing = deepcopy(row)
    del missing["checkpoint_sha256"]
    cache.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_existing_reviews(
            cache,
            selected_by_key,
            model="teacher",
            reviewer_roles={ROLE_TEXT},
        )


def test_multirole_script_supports_direct_execution() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_llm_atomic_multirole_cache.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--reviewer-roles" in completed.stdout
    assert "--route-samples" in completed.stdout
