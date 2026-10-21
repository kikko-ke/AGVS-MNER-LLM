from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llm.prompts import RERANK_PROMPT_VERSION
from modules.reranker import candidate_record_fingerprint, opaque_candidate_ids
from scripts.evaluate_llm_reranker import evaluate_threshold, metrics, routed_oracle


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
