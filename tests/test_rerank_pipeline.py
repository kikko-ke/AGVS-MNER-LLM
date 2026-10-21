from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llm.prompts import RERANK_PROMPT_VERSION
from modules.reranker import candidate_record_fingerprint, opaque_candidate_ids
from scripts.evaluate_llm_reranker import (
    evaluate_threshold,
    explore_policy_grid,
    metrics,
    routed_oracle,
    wilson_interval,
)


def _record(sample_id: int, top1: str, alternative: str) -> dict:
    identifiers = opaque_candidate_ids(sample_id, 2, RERANK_PROMPT_VERSION)
    record = {
        "sample_id": sample_id,
        "words": ["name"],
        "retained_word_indices": [0],
        "pilot_route": {"selected": True, "arm": "uncertainty"},
        "candidates": [
            {
                "candidate_id": identifiers[0],
                "crf_rank": 1,
                "crf_score": 2.0,
                "bio": [top1],
            },
            {
                "candidate_id": identifiers[1],
                "crf_rank": 2,
                "crf_score": 1.0,
                "bio": [alternative],
            },
        ],
    }
    record["top1_candidate_id"] = identifiers[0]
    record["candidate_fingerprint"] = candidate_record_fingerprint(record)
    return record


def test_evaluator_applies_high_confidence_choice_and_falls_back_safely() -> None:
    first = _record(0, "B-ORG", "B-PER")
    second = _record(1, "B-LOC", "B-ORG")
    records = [first, second]
    gold = [["B-PER"], ["B-LOC"]]
    baseline = metrics(gold, [["B-ORG"], ["B-LOC"]])
    oracle, _ = routed_oracle(gold, records)
    decisions = {
        0: {
            "sample_id": 0,
            "candidate_fingerprint": first["candidate_fingerprint"],
            "prompt_version": RERANK_PROMPT_VERSION,
            "selected_candidate_id": first["candidates"][1]["candidate_id"],
            "confidence": 0.95,
            "uncertain": False,
            "baseline_error_type": "entity_type",
        },
        1: {
            "sample_id": 1,
            "candidate_fingerprint": second["candidate_fingerprint"],
            "prompt_version": RERANK_PROMPT_VERSION,
            "selected_candidate_id": second["candidates"][1]["candidate_id"],
            "confidence": 0.70,
            "uncertain": False,
            "baseline_error_type": "entity_type",
        },
    }

    result = evaluate_threshold(0.85, gold, records, decisions, baseline, oracle)

    assert result["metrics"]["f1"] == 1.0
    assert result["beneficial_changes"] == 1
    assert result["harmful_changes"] == 0
    assert result["fallbacks"]["below_confidence"] == 1
    assert result["top1_exact_preservation"] == 1.0
    diagnostics = result["accepted_change_diagnostics"]
    assert diagnostics["overall"]["count"] == 1
    assert diagnostics["overall"]["beneficial"] == 1
    assert diagnostics["overall"]["top1_exact_changed"] == 0
    assert diagnostics["by_baseline_error_type"]["entity_type"][
        "beneficial_rate"
    ] == 1.0
    assert diagnostics["by_selected_crf_rank"]["2"]["count"] == 1
    assert diagnostics["by_used_image_evidence"]["False"]["count"] == 1
    assert diagnostics["by_decisive_span"]["False"]["count"] == 1
    assert diagnostics["by_margin_bucket"]["missing"]["count"] == 1


def test_exploratory_grid_reports_intersections_without_freezing_policy() -> None:
    records = []
    gold = []
    decisions = {}
    for sample_id in range(8):
        beneficial = sample_id < 6
        top1 = "B-ORG" if beneficial else "B-LOC"
        alternative = "O" if beneficial else "B-ORG"
        record = _record(sample_id, top1, alternative)
        record["route_features"] = {"top1_top2_gap_per_word": 0.02}
        record["candidate_fingerprint"] = candidate_record_fingerprint(record)
        records.append(record)
        gold.append(["O"] if beneficial else ["B-LOC"])
        decisions[sample_id] = {
            "sample_id": sample_id,
            "candidate_fingerprint": record["candidate_fingerprint"],
            "prompt_version": RERANK_PROMPT_VERSION,
            "selected_candidate_id": record["candidates"][1]["candidate_id"],
            "confidence": 0.99,
            "uncertain": False,
            "baseline_error_type": (
                "spurious_entity" if beneficial else "missing_entity"
            ),
            "used_image_ids": ["IMG_1"] if sample_id % 2 == 0 else [],
        }

    baseline_predictions = [record["candidates"][0]["bio"] for record in records]
    baseline = metrics(gold, baseline_predictions)
    oracle, _ = routed_oracle(gold, records)
    result = explore_policy_grid(
        0.95,
        gold,
        records,
        decisions,
        baseline,
        oracle,
        min_accepted_changes=5,
        max_hypotheses=10,
    )

    assert result["status"] == "exploratory_only_not_a_frozen_policy"
    assert result["grid_hypotheses_evaluated"] == 7 * 4 * 6 * 3
    assert result["v2_pilot_changes_available"] == 8
    assert "disjoint V3" in result["required_next_step"]
    assert any(
        "error=spurious_entity" in key and "rank=2" in key
        for key in result["atomic_intersections"]
    )
    assert result["top_v3_hypotheses"]
    assert any(
        item["hypothesis"]["baseline_error_type"] == "spurious_entity"
        and item["changes"]["beneficial"] == 6
        and item["changes"]["harmful"] == 0
        and item["top1_exact_preservation"] == 1.0
        for item in result["top_v3_hypotheses"]
    )


def test_wilson_interval_handles_empty_and_small_samples() -> None:
    assert wilson_interval(0, 0) == [0.0, 1.0]
    lower, upper = wilson_interval(8, 9)
    assert 0.5 < lower < 8 / 9 < upper < 1.0


def test_exploratory_grid_cannot_freeze_a_policy() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(project_root / "scripts" / "evaluate_llm_reranker.py"),
            "--input",
            "unused-input",
            "--candidates",
            "unused-candidates",
            "--decisions",
            "unused-decisions",
            "--explore-policy-grid",
            "--freeze-policy",
            "forbidden-policy",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "cannot load/freeze a policy" in result.stderr


def test_rerank_scripts_support_direct_execution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for script in (
        "export_crf_rerank_candidates.py",
        "build_llm_rerank_cache.py",
        "evaluate_llm_reranker.py",
    ):
        result = subprocess.run(
            [sys.executable, "-I", str(project_root / "scripts" / script), "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
