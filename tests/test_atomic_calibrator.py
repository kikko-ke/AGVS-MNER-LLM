from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from modules.atomic_calibrator import (
    EXAMPLE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    OUTCOME_TO_INDEX,
    AtomicCalibrator,
    CountDelta,
    build_feature_vector,
    calibrator_loss,
    exact_atomic_oracle,
    outcome_from_gain,
    policy_grid,
    safe_select_edits,
    shuffle_review_features,
)
from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION


def _candidate() -> dict:
    return {
        "sample_id": 0,
        "words": ["Alice", "visits", "Paris"],
        "image_names": ["0.jpg"],
        "valid_image_ids": [0],
        "route_features": {"minimum_atomic_score_gap_per_word": 0.2},
    }


def _edit(edit_id: str = "E_TYPE") -> dict:
    return {
        "edit_id": edit_id,
        "candidate_id": "C_ALT",
        "edit_class": "entity_type_change",
        "removed_entity": {"start": 0, "end": 1, "label": "ORG"},
        "added_entity": {"start": 0, "end": 1, "label": "PER"},
        "crf_rank": 2,
        "score_gap_per_word": 0.2,
    }


def _review(edit_id: str = "E_TYPE") -> dict:
    return {
        "edit_id": edit_id,
        "verdict": "APPLY_EDIT",
        "confidence": 0.9,
        "text_support": "edit",
        "visual_support": "edit",
        "used_image_ids": [0],
        "decisive_spans": [{"start": 0, "end": 1}],
        "brief_reason": "free text must not become a feature",
    }


def _example(
    sample_id: int,
    edit_id: str,
    delta: tuple[int, int, int],
    outcome: str,
    *,
    baseline_exact: bool = False,
    edited_exact: bool = False,
) -> dict:
    features = [0.0] * len(FEATURE_NAMES)
    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "sample_id": sample_id,
        "edit_id": edit_id,
        "edit_class": "entity_type_change",
        "crf_rank": 2,
        "features": features,
        "expected_review_fingerprint": "same",
        "review_fingerprint": "same",
        "count_delta": {"tp": delta[0], "fp": delta[1], "fn": delta[2]},
        "outcome": outcome,
        "baseline_exact": baseline_exact,
        "edited_exact": edited_exact,
    }


def test_count_delta_dinkelbach_gain_has_corpus_f1_direction() -> None:
    q = 0.8
    assert CountDelta(0, -1, 0).dinkelbach_gain(q) == pytest.approx(0.8)
    assert CountDelta(1, -1, -1).dinkelbach_gain(q) == pytest.approx(2.0)
    harmful = CountDelta(-1, 0, 1).dinkelbach_gain(q)
    assert harmful == pytest.approx(-1.2)
    assert outcome_from_gain(harmful) == "harmful"
    assert outcome_from_gain(0.0) == "neutral"


def test_fixed_features_use_structured_review_but_ignore_reason_text() -> None:
    first = build_feature_vector(_candidate(), _edit(), _review())
    changed = _review()
    changed["brief_reason"] = "completely different unrestricted rationale"
    second = build_feature_vector(_candidate(), _edit(), changed)

    assert len(first) == len(FEATURE_NAMES)
    assert first == second
    assert all(torch.isfinite(torch.tensor(first)))


def test_joint_loss_is_differentiable_and_listwise_by_sample() -> None:
    model = AtomicCalibrator(len(FEATURE_NAMES), hidden_size=8)
    features = torch.randn(3, len(FEATURE_NAMES))
    utilities, logits = model(features)
    loss, components = calibrator_loss(
        utilities,
        logits,
        torch.tensor([1.0, -1.0, 0.5]),
        torch.tensor(
            [
                OUTCOME_TO_INDEX["beneficial"],
                OUTCOME_TO_INDEX["harmful"],
                OUTCOME_TO_INDEX["beneficial"],
            ]
        ),
        [10, 10, 20],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(components) == {
        "regression",
        "classification",
        "listwise",
        "harmful_penalty",
    }
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_policy_scan_accepts_benefit_and_blocks_high_harm_risk() -> None:
    benefit = _example(1, "E_GOOD", (0, -1, 0), "beneficial")
    harmful = _example(
        2,
        "E_BAD",
        (-1, 0, 1),
        "harmful",
        baseline_exact=True,
        edited_exact=False,
    )
    records = [benefit, harmful]
    metadata = {
        "roles": {
            "meta_calibration": {
                "baseline_counts": [10, 2, 2],
            }
        }
    }
    policy, summary = policy_grid(
        records,
        utilities=[0.8, 0.7],
        harmful_probabilities=[0.01, 0.99],
        metadata=metadata,
        role="meta_calibration",
        minimum_accepted=1,
    )

    assert policy["feasible_policy_found"] is True
    assert summary["selected"]["beneficial"] == 1
    assert summary["selected"]["harmful"] == 0
    assert summary["selected"]["f1"] > 20 / 24


def test_any_stale_review_forces_whole_sample_to_fallback() -> None:
    first = _example(1, "E_ONE", (0, -1, 0), "beneficial")
    second = _example(1, "E_TWO", (1, -1, -1), "beneficial")
    second["review_fingerprint"] = "stale"
    policy = {
        "schema_version": "atomic-calibrator-policy-v1",
        "score_threshold": 0.0,
        "harm_probability_cap": 0.5,
    }
    selections, fallbacks = safe_select_edits(
        [first, second], [1.0, 2.0], [0.0, 0.0], policy
    )

    assert selections == {1: None}
    assert fallbacks["review_fingerprint_mismatch"] == 1
    assert fallbacks["unsafe_sample"] == 1


def test_exact_oracle_uses_full_corpus_baseline_constants() -> None:
    beneficial = _example(1, "E_GOOD", (1, -1, -1), "beneficial")
    harmful = _example(2, "E_BAD", (-1, 0, 1), "harmful")
    metadata = {"roles": {"all": {"baseline_counts": [100, 20, 20]}}}
    oracle = exact_atomic_oracle(
        metadata,
        "all",
        {1: [beneficial], 2: [harmful]},
    )

    assert oracle["selections"][1] == "E_GOOD"
    assert oracle["selections"][2] is None
    assert oracle["f1"] > 200 / 240


def test_shuffled_review_ablation_preserves_structural_features() -> None:
    first = _example(1, "E_ONE", (0, -1, 0), "beneficial")
    second = _example(2, "E_TWO", (0, -1, 0), "beneficial")
    first["features"] = list(map(float, range(len(FEATURE_NAMES))))
    second["features"] = [1000.0 + value for value in range(len(FEATURE_NAMES))]
    shuffled = shuffle_review_features([first, second], seed=4)

    structural_count = len(FEATURE_NAMES) - 20
    # Derive the boundary by name instead of relying on the feature count above.
    review_start = FEATURE_NAMES.index("review_verdict_KEEP_BASELINE")
    assert shuffled[0][:review_start] == first["features"][:review_start]
    assert shuffled[1][:review_start] == second["features"][:review_start]
    assert {
        tuple(row[review_start:]) for row in shuffled
    } == {
        tuple(first["features"][review_start:]),
        tuple(second["features"][review_start:]),
    }
    assert structural_count > 0


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_gold_builder_freezes_q_from_meta_train_only(tmp_path: Path) -> None:
    gold = tmp_path / "MNER-UNI_meta.txt"
    gold.write_text(
        repr({"text": ["Alice"], "images": [], "label": ["B-PER"]})
        + "\n"
        + repr({"text": ["Paris"], "images": [], "label": ["B-LOC"]})
        + "\n",
        encoding="utf-8",
    )

    def candidate(sample_id: int, gold_type: str, wrong_type: str) -> dict:
        baseline_id = f"C_BASE{sample_id}"
        edit_id = f"E_EDIT{sample_id}"
        return {
            "schema_version": "atomic-edit-candidates-v1",
            "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
            "sample_id": sample_id,
            "source_sample_id": 100 + sample_id,
            "split": "meta",
            "official_test": False,
            "checkpoint_sha256": "c" * 64,
            "input_sha256": "i" * 64,
            "source_ids_sha256": "s" * 64,
            "review_fingerprint": f"fingerprint-{sample_id}",
            "words": ["Alice" if sample_id == 0 else "Paris"],
            "retained_word_indices": [0],
            "image_names": [],
            "valid_image_ids": [],
            "baseline_candidate_id": baseline_id,
            "baseline_entities": [
                {
                    "start": 0,
                    "end": 1,
                    "label": wrong_type,
                    "surface": "Alice" if sample_id == 0 else "Paris",
                }
            ],
            "candidates": [
                {
                    "candidate_id": baseline_id,
                    "crf_rank": 1,
                    "crf_score": 2.0,
                    "bio": [f"B-{wrong_type}"],
                },
                {
                    "candidate_id": f"C_EDIT{sample_id}",
                    "crf_rank": 2,
                    "crf_score": 1.0,
                    "bio": [f"B-{gold_type}"],
                },
            ],
            "edits": [
                {
                    "edit_id": edit_id,
                    "candidate_id": f"C_EDIT{sample_id}",
                    "edit_class": "entity_type_change",
                    "removed_entity": {
                        "start": 0,
                        "end": 1,
                        "label": wrong_type,
                    },
                    "added_entity": {"start": 0, "end": 1, "label": gold_type},
                    "crf_rank": 2,
                    "score_gap_per_word": 1.0,
                }
            ],
            "route_features": {"minimum_atomic_score_gap_per_word": 1.0},
        }

    candidates = [candidate(0, "PER", "ORG"), candidate(1, "LOC", "ORG")]
    reviews = []
    for item in candidates:
        edit = item["edits"][0]
        reviews.append(
            {
                "schema_version": "atomic-edit-reviews-v1",
                "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
                "sample_id": item["sample_id"],
                "source_sample_id": item["source_sample_id"],
                "split": "meta",
                "official_test": False,
                "checkpoint_sha256": item["checkpoint_sha256"],
                "input_sha256": item["input_sha256"],
                "source_ids_sha256": item["source_ids_sha256"],
                "llm_model": "test-teacher",
                "review_fingerprint": item["review_fingerprint"],
                "reviews": [
                    {
                        "edit_id": edit["edit_id"],
                        "verdict": "APPLY_EDIT",
                        "confidence": 0.9,
                        "text_support": "edit",
                        "visual_support": "none",
                        "used_image_ids": [],
                        "decisive_spans": [{"start": 0, "end": 1}],
                        "brief_reason": "type correction",
                    }
                ],
            }
        )
    candidate_path = tmp_path / "candidates.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    output = tmp_path / "examples.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(review_path, reviews)
    manifest.write_text(
        json.dumps({"meta_train": [0], "meta_calibration": [1]}),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(project_root / "scripts" / "build_atomic_calibrator_data.py"),
            "--input",
            str(gold),
            "--candidates",
            str(candidate_path),
            "--reviews",
            str(review_path),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(
        output.with_name(output.name + ".meta.json").read_text(encoding="utf-8")
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    # The meta-train baseline gets no entity correct, so q=0 even though the
    # calibration partition must remain invisible to q estimation.
    assert metadata["dinkelbach_q"] == 0.0
    assert metadata["dinkelbach_q_scope"] == "meta_train_top1_only"
    assert {row["source_sample_id"] for row in rows} == {100, 101}
    assert all(row["outcome"] == "beneficial" for row in rows)


def test_atomic_calibrator_scripts_support_direct_execution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for script in (
        "build_atomic_calibrator_data.py",
        "train_atomic_calibrator.py",
        "evaluate_atomic_calibrator.py",
    ):
        result = subprocess.run(
            [sys.executable, "-I", str(project_root / "scripts" / script), "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
