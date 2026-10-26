from __future__ import annotations

import json
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

import pytest

from llm.atomic_multirole_schema import ATOMIC_MULTIROLE_PROMPT_VERSION
from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION
from modules.atomic_calibrator import sha256_file
from scripts.evaluate_atomic_multirole_pilot import (
    SCHEMA_VERSION,
    _selected_from_sidecar,
    evaluate_pilot,
    validate_development_role,
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


def _candidate(
    sample_id: int,
    source_id: int,
    baseline_bio: list[str],
    edited_bio: list[str],
) -> dict:
    baseline_entities = (
        [{"start": 0, "end": 1, "label": "PER", "surface": "Alice"}]
        if baseline_bio[0] == "B-PER"
        else []
    )
    edited_entities = (
        [{"start": 0, "end": 1, "label": "PER", "surface": "Alice"}]
        if edited_bio[0] == "B-PER"
        else []
    )
    removed = baseline_entities[0] if baseline_entities and not edited_entities else None
    added = edited_entities[0] if edited_entities and not baseline_entities else None
    edit_class = (
        "deletion_spurious_removal" if removed else "addition_missing_recovery"
    )
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
        "words": ["Alice"],
        "retained_word_indices": [0],
        "image_names": ["img.jpg"],
        "valid_image_ids": [0],
        "baseline_candidate_id": f"C_BASE{sample_id}",
        "baseline_entities": baseline_entities,
        "candidates": [
            {"candidate_id": f"C_BASE{sample_id}", "bio": baseline_bio},
            {"candidate_id": f"C_EDIT{sample_id}", "bio": edited_bio},
        ],
        "edits": [
            {
                "edit_id": f"E_EDIT{sample_id}",
                "candidate_id": f"C_EDIT{sample_id}",
                "edit_class": edit_class,
                "removed_entity": removed,
                "added_entity": added,
                "baseline_entities": baseline_entities,
                "crf_rank": 2,
                "score_gap_per_word": 0.1 + sample_id / 100.0,
                "display_position": 0,
            }
        ],
        "route_features": {"minimum_atomic_score_gap_per_word": 0.1},
        "review_route": {"selected": True, "route_order": sample_id},
    }
    record["review_fingerprint"] = atomic_review_record_fingerprint(record)
    return record


def _review(candidate: dict, role: str, verdict: str) -> dict:
    if role == "visual_verifier" and verdict != "INSUFFICIENT":
        image_ids = [0]
    else:
        image_ids = []
    if role == "adversarial_critic":
        target = "none" if verdict == "NO_OBJECTION" else "added_entity"
    else:
        target = "none" if verdict == "INSUFFICIENT" else (
            "added_entity"
            if candidate["edits"][0]["added_entity"] is not None
            else "removed_entity"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": candidate["sample_id"],
        "source_sample_id": candidate["source_sample_id"],
        "split": candidate["split"],
        "official_test": False,
        "meta_role": "meta_calibration",
        "route_order": candidate["sample_id"],
        "prompt_version": ATOMIC_MULTIROLE_PROMPT_VERSION,
        "review_fingerprint": candidate["review_fingerprint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "input_sha256": candidate["input_sha256"],
        "source_ids_sha256": candidate["source_ids_sha256"],
        "llm_model": "teacher",
        "edit_id": candidate["edits"][0]["edit_id"],
        "role": role,
        "verdict": verdict,
        "evidence_target": target,
        "reason_code": "local_evidence",
        "used_image_ids": image_ids,
        "brief_evidence": "The local evidence supports this role decision.",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _artifacts(tmp_path: Path, *, missing_critic: bool = False) -> dict[str, Path]:
    train = _candidate(0, 100, ["B-PER"], ["O"])
    beneficial = _candidate(1, 101, ["O"], ["B-PER"])
    harmful = _candidate(2, 102, ["B-PER"], ["O"])
    candidates = [train, beneficial, harmful]
    candidates_path = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates_path, candidates)

    input_path = tmp_path / "meta.txt"
    input_path.write_text(
        "".join(
            repr({"token": ["Alice"], "label": ["B-PER"], "images": ["img.jpg"]})
            + "\n"
            for _ in candidates
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "gold_labels_used_for_assignment": False,
        "official_validation_or_test_used": False,
        "assignments": [
            {"sample_id": 0, "source_sample_id": 100, "role": "meta_train"},
            {
                "sample_id": 1,
                "source_sample_id": 101,
                "role": "meta_calibration",
            },
            {
                "sample_id": 2,
                "source_sample_id": 102,
                "role": "meta_calibration",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reviews = [
        _review(beneficial, "text_verifier", "SUPPORT_EDIT"),
        _review(beneficial, "visual_verifier", "SUPPORT_EDIT"),
        _review(harmful, "text_verifier", "SUPPORT_BASELINE"),
        _review(harmful, "visual_verifier", "SUPPORT_EDIT"),
        _review(harmful, "adversarial_critic", "NO_OBJECTION"),
    ]
    if not missing_critic:
        reviews.append(
            _review(beneficial, "adversarial_critic", "NO_OBJECTION")
        )
    reviews_path = tmp_path / "reviews.jsonl"
    _write_jsonl(reviews_path, reviews)
    selected = [
        {
            "sample_id": candidate["sample_id"],
            "source_sample_id": candidate["source_sample_id"],
            "edit_id": candidate["edits"][0]["edit_id"],
            "meta_role": "meta_calibration",
            "route_order": index,
            "score_gap_per_word": 0.1,
        }
        for index, candidate in enumerate((beneficial, harmful))
    ]
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": ATOMIC_MULTIROLE_PROMPT_VERSION,
        "candidates_sha256": sha256_file(candidates_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gold_entity_labels_used": False,
        "roles": ["meta_calibration"],
        "reviewer_roles": [
            "text_verifier",
            "visual_verifier",
            "adversarial_critic",
        ],
        "selected_edits": selected,
    }
    sidecar_path = reviews_path.with_name(reviews_path.name + ".meta.json")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return {
        "input": input_path,
        "candidates": candidates_path,
        "reviews": reviews_path,
        "manifest": manifest_path,
        "sidecar": sidecar_path,
    }


def test_consensus_applies_at_most_one_supported_edit_and_reports_metrics(
    local_tmp_path: Path,
) -> None:
    paths = _artifacts(local_tmp_path)
    result = evaluate_pilot(
        input_path=paths["input"],
        candidates_path=paths["candidates"],
        reviews_path=paths["reviews"],
        manifest_path=paths["manifest"],
        meta_role="meta_calibration",
    )

    assert result["development_only"] is True
    assert result["baseline"]["counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert result["selected_edit_oracle"]["f1"] == 1.0
    assert result["consensus"]["f1"] == 1.0
    assert result["consensus"]["accepted"] == 1
    assert result["consensus"]["beneficial"] == 1
    assert result["consensus"]["harmful"] == 0
    assert result["consensus"]["top1_exact_preservation"] == 1.0
    assert result["role_diagnostics"]["valid_response_coverage"] == 1.0
    diagnostics = result["exploratory_outcome_diagnostics"]
    assert diagnostics["role_verdict_outcomes"]["visual_verifier"][
        "SUPPORT_EDIT"
    ] == {
        "total": 2,
        "beneficial": 1,
        "neutral": 0,
        "harmful": 1,
        "beneficial_precision": 0.5,
        "harmful_rate": 0.5,
    }
    assert diagnostics["diagnostic_policies"]["text_support_edit"][
        "beneficial"
    ] == 1
    assert diagnostics["diagnostic_policies"]["text_support_edit"]["harmful"] == 0
    assert result["go_no_go"]["passed"] is False  # accepted < locked minimum 20


def test_missing_role_safely_falls_back_to_top1(local_tmp_path: Path) -> None:
    paths = _artifacts(local_tmp_path, missing_critic=True)
    result = evaluate_pilot(
        input_path=paths["input"],
        candidates_path=paths["candidates"],
        reviews_path=paths["reviews"],
        manifest_path=paths["manifest"],
        meta_role="meta_calibration",
    )

    assert result["consensus"]["accepted"] == 0
    assert result["consensus"]["f1"] == result["baseline"]["f1"]
    assert result["role_diagnostics"]["complete_role_triplets"] == 1
    assert result["role_diagnostics"]["valid_response_coverage"] == 0.5


def test_stale_fingerprint_is_invalid_and_falls_back(local_tmp_path: Path) -> None:
    paths = _artifacts(local_tmp_path)
    rows = [json.loads(line) for line in paths["reviews"].read_text().splitlines()]
    for row in rows:
        if row["sample_id"] == 1 and row["role"] == "adversarial_critic":
            row["review_fingerprint"] = "stale"
    _write_jsonl(paths["reviews"], rows)

    result = evaluate_pilot(
        input_path=paths["input"],
        candidates_path=paths["candidates"],
        reviews_path=paths["reviews"],
        manifest_path=paths["manifest"],
        meta_role="meta_calibration",
    )
    assert result["consensus"]["accepted"] == 0
    assert result["role_diagnostics"]["invalid_reasons"] == {
        "fingerprint_mismatch": 1
    }


def test_mixed_llm_models_abort_instead_of_combining_reviewers(
    local_tmp_path: Path,
) -> None:
    paths = _artifacts(local_tmp_path)
    rows = [json.loads(line) for line in paths["reviews"].read_text().splitlines()]
    rows[0]["llm_model"] = "different-teacher"
    _write_jsonl(paths["reviews"], rows)

    with pytest.raises(ValueError, match="mixes multiple llm_model"):
        evaluate_pilot(
            input_path=paths["input"],
            candidates_path=paths["candidates"],
            reviews_path=paths["reviews"],
            manifest_path=paths["manifest"],
            meta_role="meta_calibration",
        )


def test_meta_evaluation_is_hard_locked() -> None:
    with pytest.raises(ValueError, match="meta_evaluation is permanently locked"):
        validate_development_role("meta_evaluation")


def test_sidecar_rejects_more_than_one_edit_per_sample(local_tmp_path: Path) -> None:
    paths = _artifacts(local_tmp_path)
    sidecar = json.loads(paths["sidecar"].read_text())
    duplicate = deepcopy(sidecar["selected_edits"][0])
    duplicate["edit_id"] = "E_OTHER"
    sidecar["selected_edits"].append(duplicate)

    with pytest.raises(ValueError, match="at most one selected edit"):
        _selected_from_sidecar(
            sidecar,
            role="meta_calibration",
            candidates_path=paths["candidates"],
            manifest_path=paths["manifest"],
        )
