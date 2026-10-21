"""Evaluate cached LLM candidate choices; this is the only stage that reads gold labels."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqeval.metrics import f1_score, precision_score, recall_score

from llm.prompts import RERANK_PROMPT_VERSION
from modules.crf_topk import select_corpus_micro_f1_oracle
from modules.llm_dataset import extract_bio_entities
from modules.reranker import (
    apply_reranker_decision,
    build_rerank_policy,
    candidate_record_fingerprint,
    sha256_file,
    sha256_inference_projection,
    validate_rerank_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Gold split used only in this evaluator")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--scope", choices=("pilot", "all"), default="pilot")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock one final test evaluation with a frozen threshold.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Use one frozen validation-selected policy threshold.",
    )
    parser.add_argument(
        "--freeze-policy",
        default=None,
        help="Write a policy manifest after a passing validation pilot.",
    )
    parser.add_argument("--frozen-threshold", type=float, default=None)
    parser.add_argument(
        "--explore-policy-grid",
        action="store_true",
        help=(
            "Run a validation-only, post-hoc policy-grid diagnostic. The result is "
            "exploratory evidence for a disjoint pilot and cannot be frozen."
        ),
    )
    parser.add_argument(
        "--grid-confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence floor used before the exploratory feature grid (default: 0.95).",
    )
    parser.add_argument(
        "--grid-min-accepted-changes",
        type=int,
        default=8,
        help="Minimum accepted changes for a grid hypothesis to pass screening.",
    )
    parser.add_argument(
        "--grid-max-hypotheses",
        type=int,
        default=20,
        help="Maximum number of exploratory hypotheses written to the summary.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_number}")
            records.append(record)
    return records


def indexed_records(records: list[dict[str, Any]], name: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if type(sample_id) is not int or sample_id < 0:
            raise ValueError(f"{name} contains an invalid sample_id")
        if sample_id in result:
            raise ValueError(f"{name} contains duplicate sample_id={sample_id}")
        result[sample_id] = record
    return result


def load_gold(path: Path, candidates: list[dict[str, Any]]) -> list[list[str]]:
    raw_records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                raw_records.append(ast.literal_eval(line))
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Invalid dataset record at line {line_number}") from error

    gold: list[list[str]] = []
    for candidate_record in candidates:
        sample_id = int(candidate_record["sample_id"])
        if sample_id >= len(raw_records):
            raise ValueError(f"Candidate sample_id={sample_id} is outside the dataset")
        raw_labels = raw_records[sample_id]["label"]
        labels = [str(label).replace("MISC", "MIS") for label in raw_labels]
        indices = candidate_record["retained_word_indices"]
        sequence = [labels[int(index)] for index in indices]
        if len(sequence) != len(candidate_record["words"]):
            raise ValueError(f"Gold/candidate length mismatch for sample_id={sample_id}")
        gold.append(sequence)
    return gold


def metrics(gold: list[list[str]], predictions: list[list[str]]) -> dict[str, float]:
    return {
        "f1": float(f1_score(gold, predictions, zero_division=0)),
        "precision": float(precision_score(gold, predictions, zero_division=0)),
        "recall": float(recall_score(gold, predictions, zero_division=0)),
    }


def entity_counts(gold: list[str], prediction: list[str]) -> tuple[int, int, int]:
    gold_entities = extract_bio_entities(gold)
    predicted_entities = extract_bio_entities(prediction)
    return (
        len(gold_entities & predicted_entities),
        len(predicted_entities - gold_entities),
        len(gold_entities - predicted_entities),
    )


def sample_f1(gold: list[str], prediction: list[str]) -> float:
    true_positive, false_positive, false_negative = entity_counts(gold, prediction)
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def summarize_change_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize accepted changes without exposing individual gold labels."""
    beneficial = sum(row["outcome"] == "beneficial" for row in rows)
    harmful = sum(row["outcome"] == "harmful" for row in rows)
    neutral = sum(row["outcome"] == "neutral" for row in rows)
    count = len(rows)
    top1_exact = sum(bool(row["top1_exact"]) for row in rows)
    directional_count = beneficial + harmful
    directional_win_rate = (
        beneficial / directional_count if directional_count else 0.0
    )
    return {
        "count": count,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "beneficial_rate": beneficial / count if count else 0.0,
        "harmful_rate": harmful / count if count else 0.0,
        "mean_sample_f1_delta": (
            sum(float(row["sample_f1_delta"]) for row in rows) / count
            if count
            else 0.0
        ),
        "top1_exact_changed": top1_exact,
        "directional_count": directional_count,
        "directional_win_rate": directional_win_rate,
        "directional_win_rate_wilson_95": wilson_interval(
            beneficial,
            directional_count,
        ),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a dependency-free Wilson score interval for a binomial rate."""
    if total <= 0:
        return [0.0, 1.0]
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def group_change_rows(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        group: summarize_change_rows(grouped[group])
        for group in sorted(grouped)
    }


def candidate_by_rank(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = sorted(record["candidates"], key=lambda item: int(item["crf_rank"]))
    if not candidates or int(candidates[0]["crf_rank"]) != 1:
        raise ValueError(f"sample_id={record['sample_id']} has no rank-one candidate")
    return candidates


def routed_oracle(
    gold: list[list[str]],
    records: list[dict[str, Any]],
    scope: str = "pilot",
) -> tuple[dict[str, float], list[list[str]]]:
    all_candidate_counts: list[list[tuple[int, int, int]]] = []
    all_candidate_scores: list[list[float]] = []
    all_candidates: list[list[list[str]]] = []
    for gold_sequence, record in zip(gold, records):
        ranked = candidate_by_rank(record)
        if scope == "pilot" and not record.get("pilot_route", {}).get("selected", False):
            ranked = ranked[:1]
        sequences = [list(candidate["bio"]) for candidate in ranked]
        all_candidates.append(sequences)
        all_candidate_counts.append(
            [entity_counts(gold_sequence, sequence) for sequence in sequences]
        )
        all_candidate_scores.append([float(candidate["crf_score"]) for candidate in ranked])
    indices, exact_f1 = select_corpus_micro_f1_oracle(
        all_candidate_counts,
        all_candidate_scores,
    )
    predictions = [row[index] for row, index in zip(all_candidates, indices)]
    result = metrics(gold, predictions)
    if abs(result["f1"] - exact_f1) > 1e-10:
        raise RuntimeError("Oracle entity counts disagree with seqeval")
    return result, predictions


def evaluate_threshold(
    threshold: float,
    gold: list[list[str]],
    records: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    baseline_metrics: dict[str, float],
    oracle_metrics: dict[str, float],
    scope: str = "pilot",
) -> dict[str, Any]:
    predictions: list[list[str]] = []
    accepted_changes = 0
    beneficial_changes = 0
    harmful_changes = 0
    neutral_changes = 0
    top1_exact_routed = 0
    top1_exact_preserved = 0
    change_rows: list[dict[str, Any]] = []
    fallbacks = {
        "not_routed": 0,
        "missing_decision": 0,
        "fingerprint_mismatch": 0,
        "uncertain": 0,
        "below_confidence": 0,
        "selected_top1": 0,
    }

    for gold_sequence, record in zip(gold, records):
        ranked = candidate_by_rank(record)
        top1 = list(ranked[0]["bio"])
        selected = top1
        route_selected = scope == "all" or bool(
            record.get("pilot_route", {}).get("selected", False)
        )
        top1_exact = extract_bio_entities(top1) == extract_bio_entities(gold_sequence)
        if route_selected and top1_exact:
            top1_exact_routed += 1

        if not route_selected:
            fallbacks["not_routed"] += 1
        else:
            decision = decisions.get(int(record["sample_id"]))
            applied = apply_reranker_decision(
                record,
                decision,
                prompt_version=RERANK_PROMPT_VERSION,
                confidence_threshold=threshold,
            )
            if not applied["accepted"]:
                reason = str(applied["fallback_reason"])
                reason = (
                    "fingerprint_mismatch"
                    if reason in {"candidate_fingerprint_mismatch", "prompt_version_mismatch"}
                    else reason
                )
                fallbacks.setdefault(reason, 0)
                fallbacks[reason] += 1
            elif not applied["changed_top1"]:
                fallbacks["selected_top1"] += 1
            else:
                selected = list(applied["candidate"]["bio"])
                accepted_changes += 1
                before = sample_f1(gold_sequence, top1)
                after = sample_f1(gold_sequence, selected)
                if after > before:
                    beneficial_changes += 1
                    outcome = "beneficial"
                elif after < before:
                    harmful_changes += 1
                    outcome = "harmful"
                else:
                    neutral_changes += 1
                    outcome = "neutral"
                raw_decision = decisions[int(record["sample_id"])]
                margin = record.get("route_features", {}).get(
                    "top1_top2_gap_per_word"
                )
                change_rows.append(
                    {
                        "outcome": outcome,
                        "sample_f1_delta": after - before,
                        "top1_exact": top1_exact,
                        "baseline_error_type": applied["baseline_error_type"],
                        "selected_crf_rank": int(
                            applied["candidate"].get("crf_rank", 0)
                        ),
                        "used_image_evidence": bool(
                            raw_decision.get("used_image_ids", [])
                        ),
                        "has_decisive_span": bool(
                            raw_decision.get("decisive_spans", [])
                        ),
                        "margin_bucket": margin_bucket(margin),
                    }
                )

        if route_selected and top1_exact and (
            extract_bio_entities(selected) == extract_bio_entities(gold_sequence)
        ):
            top1_exact_preserved += 1
        predictions.append(selected)

    result = metrics(gold, predictions)
    oracle_gain = oracle_metrics["f1"] - baseline_metrics["f1"]
    realised_gain = result["f1"] - baseline_metrics["f1"]
    return {
        "confidence_threshold": threshold,
        "metrics": result,
        "gain_over_top1": realised_gain,
        "routed_oracle_gain": oracle_gain,
        "normalised_oracle_gain": realised_gain / oracle_gain if oracle_gain > 0 else 0.0,
        "accepted_non_top1_changes": accepted_changes,
        "beneficial_changes": beneficial_changes,
        "harmful_changes": harmful_changes,
        "neutral_changes": neutral_changes,
        "top1_exact_preservation": (
            top1_exact_preserved / top1_exact_routed if top1_exact_routed else 1.0
        ),
        "top1_exact_routed": top1_exact_routed,
        "fallbacks": fallbacks,
        "accepted_change_diagnostics": {
            "overall": summarize_change_rows(change_rows),
            "by_baseline_error_type": group_change_rows(
                change_rows, "baseline_error_type"
            ),
            "by_selected_crf_rank": group_change_rows(
                change_rows, "selected_crf_rank"
            ),
            "by_used_image_evidence": group_change_rows(
                change_rows, "used_image_evidence"
            ),
            "by_decisive_span": group_change_rows(
                change_rows, "has_decisive_span"
            ),
            "by_margin_bucket": group_change_rows(
                change_rows, "margin_bucket"
            ),
        },
    }


def margin_bucket(value: Any) -> str:
    if value is None:
        return "missing"
    margin = float(value)
    if margin < 0.05:
        return "[0.00,0.05)"
    if margin < 0.10:
        return "[0.05,0.10)"
    if margin < 0.20:
        return "[0.10,0.20)"
    return "[0.20,inf)"


def _atomic_intersection_key(row: dict[str, Any]) -> str:
    image = "image" if row["used_image_evidence"] else "no_image"
    return (
        f"error={row['baseline_error_type']}|"
        f"margin={row['margin_bucket']}|"
        f"rank={row['selected_crf_rank']}|{image}"
    )


def _exploratory_change_cases(
    threshold: float,
    gold: list[list[str]],
    records: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
) -> tuple[list[list[str]], list[dict[str, Any]], int]:
    """Collect inference-observable features and gold-only outcomes for V2 changes."""
    baseline_predictions: list[list[str]] = []
    cases: list[dict[str, Any]] = []
    top1_exact_routed = 0
    for index, (gold_sequence, record) in enumerate(zip(gold, records)):
        ranked = candidate_by_rank(record)
        top1 = list(ranked[0]["bio"])
        baseline_predictions.append(top1)
        route_selected = bool(record.get("pilot_route", {}).get("selected", False))
        top1_exact = extract_bio_entities(top1) == extract_bio_entities(gold_sequence)
        if route_selected and top1_exact:
            top1_exact_routed += 1
        if not route_selected:
            continue

        sample_id = int(record["sample_id"])
        raw_decision = decisions.get(sample_id)
        applied = apply_reranker_decision(
            record,
            raw_decision,
            prompt_version=RERANK_PROMPT_VERSION,
            confidence_threshold=threshold,
        )
        if not applied["accepted"] or not applied["changed_top1"]:
            continue

        selected = list(applied["candidate"]["bio"])
        before = sample_f1(gold_sequence, top1)
        after = sample_f1(gold_sequence, selected)
        if after > before:
            outcome = "beneficial"
        elif after < before:
            outcome = "harmful"
        else:
            outcome = "neutral"
        margin = record.get("route_features", {}).get("top1_top2_gap_per_word")
        if margin is None:
            raise ValueError(
                f"sample_id={sample_id} lacks top1_top2_gap_per_word for grid audit"
            )
        decision_object = raw_decision if isinstance(raw_decision, dict) else {}
        cases.append(
            {
                "record_index": index,
                "selected_prediction": selected,
                "selected_exact": (
                    extract_bio_entities(selected)
                    == extract_bio_entities(gold_sequence)
                ),
                "outcome": outcome,
                "sample_f1_delta": after - before,
                "top1_exact": top1_exact,
                "baseline_error_type": str(applied["baseline_error_type"]),
                "selected_crf_rank": int(applied["candidate"].get("crf_rank", 0)),
                "used_image_evidence": bool(decision_object.get("used_image_ids", [])),
                "has_decisive_span": bool(decision_object.get("decisive_spans", [])),
                "margin": float(margin),
                "margin_bucket": margin_bucket(margin),
            }
        )
    return baseline_predictions, cases, top1_exact_routed


def _grid_dimensions() -> tuple[list[dict[str, Any]], ...]:
    all_error_types = {"boundary", "entity_type", "missing_entity", "spurious_entity"}
    error_policies = [
        {"name": "all", "values": all_error_types},
        *(
            {"name": name, "values": {name}}
            for name in sorted(all_error_types)
        ),
        {
            "name": "entity_type_or_spurious_entity",
            "values": {"entity_type", "spurious_entity"},
        },
        {
            "name": "non_additive_error",
            "values": {"boundary", "entity_type", "spurious_entity"},
        },
    ]
    margin_policies = [
        {"name": "margin_lt_0.05", "upper_exclusive": 0.05},
        {"name": "margin_lt_0.10", "upper_exclusive": 0.10},
        {"name": "margin_lt_0.20", "upper_exclusive": 0.20},
        {"name": "any_margin", "upper_exclusive": None},
    ]
    rank_policies = [
        {"name": "any_rank", "values": {2, 3, 4, 5}},
        {"name": "rank_2", "values": {2}},
        {"name": "rank_3", "values": {3}},
        {"name": "rank_4", "values": {4}},
        {"name": "rank_5", "values": {5}},
        {"name": "rank_2_or_3", "values": {2, 3}},
    ]
    image_policies = [
        {"name": "any_image_evidence", "value": None},
        {"name": "requires_image_evidence", "value": True},
        {"name": "requires_no_image_evidence", "value": False},
    ]
    return error_policies, margin_policies, rank_policies, image_policies


def explore_policy_grid(
    threshold: float,
    gold: list[list[str]],
    records: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    baseline_metrics: dict[str, float],
    oracle_metrics: dict[str, float],
    *,
    min_accepted_changes: int = 8,
    max_hypotheses: int = 20,
) -> dict[str, Any]:
    """Explore observable V2 feature intersections without producing a policy."""
    if min_accepted_changes < 1:
        raise ValueError("grid minimum accepted changes must be positive")
    if max_hypotheses < 1:
        raise ValueError("grid maximum hypotheses must be positive")
    baseline_predictions, cases, top1_exact_routed = _exploratory_change_cases(
        threshold,
        gold,
        records,
        decisions,
    )
    atomic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        atomic[_atomic_intersection_key(case)].append(case)

    error_policies, margin_policies, rank_policies, image_policies = (
        _grid_dimensions()
    )
    oracle_gain = oracle_metrics["f1"] - baseline_metrics["f1"]
    hypotheses: list[dict[str, Any]] = []
    for error_policy, margin_policy, rank_policy, image_policy in product(
        error_policies,
        margin_policies,
        rank_policies,
        image_policies,
    ):
        upper = margin_policy["upper_exclusive"]
        required_image = image_policy["value"]
        selected_cases = [
            case
            for case in cases
            if case["baseline_error_type"] in error_policy["values"]
            and (upper is None or case["margin"] < upper)
            and case["selected_crf_rank"] in rank_policy["values"]
            and (
                required_image is None
                or case["used_image_evidence"] is required_image
            )
        ]
        predictions = [list(sequence) for sequence in baseline_predictions]
        for case in selected_cases:
            predictions[case["record_index"]] = list(case["selected_prediction"])
        result_metrics = metrics(gold, predictions)
        summary = summarize_change_rows(selected_cases)
        preserved = top1_exact_routed - sum(
            bool(case["top1_exact"]) and not bool(case["selected_exact"])
            for case in selected_cases
        )
        preservation = (
            preserved / top1_exact_routed if top1_exact_routed else 1.0
        )
        gain = result_metrics["f1"] - baseline_metrics["f1"]
        screen_pass = (
            summary["count"] >= min_accepted_changes
            and summary["directional_count"] >= 5
            and summary["beneficial"] > summary["harmful"]
            and gain > 0.0
            and preservation >= 0.95
        )
        hypotheses.append(
            {
                "hypothesis": {
                    "baseline_error_type": error_policy["name"],
                    "margin": margin_policy["name"],
                    "selected_crf_rank": rank_policy["name"],
                    "image_evidence": image_policy["name"],
                },
                "metrics": result_metrics,
                "gain_over_top1": gain,
                "normalised_oracle_gain": gain / oracle_gain if oracle_gain > 0 else 0.0,
                "top1_exact_preservation": preservation,
                "changes": summary,
                "passes_exploratory_screen": screen_pass,
            }
        )

    screened = [item for item in hypotheses if item["passes_exploratory_screen"]]
    screened.sort(
        key=lambda item: (
            float(item["gain_over_top1"]),
            float(item["top1_exact_preservation"]),
            float(item["changes"]["directional_win_rate_wilson_95"][0]),
            int(item["changes"]["count"]),
        ),
        reverse=True,
    )
    return {
        "schema_version": "exploratory-rerank-policy-grid-v1",
        "status": "exploratory_only_not_a_frozen_policy",
        "confidence_threshold": threshold,
        "v2_pilot_changes_available": len(cases),
        "top1_exact_routed": top1_exact_routed,
        "grid_hypotheses_evaluated": len(hypotheses),
        "screened_hypotheses": len(screened),
        "feature_contract": {
            "inference_observable": [
                "LLM-declared baseline_error_type",
                "CRF top1/top2 score margin",
                "LLM-selected CRF candidate rank",
                "whether the LLM cited an input image",
            ],
            "gold_only_diagnostics": [
                "beneficial/harmful/neutral outcome",
                "micro F1 gain",
                "top1 exact preservation",
            ],
        },
        "screen": {
            "minimum_accepted_changes": min_accepted_changes,
            "minimum_directional_changes": 5,
            "beneficial_must_exceed_harmful": True,
            "corpus_f1_gain_must_be_positive": True,
            "minimum_top1_exact_preservation": 0.95,
        },
        "multiple_testing_warning": (
            "Every hypothesis and its ranking use the same 100-sample V2 validation "
            "pilot. Reported gains are in-sample and optimistically biased; they must "
            "not be used to freeze, deploy, or evaluate a test-set policy."
        ),
        "pilot_equivalence_warning": (
            "Different feature rules can accept the same cases in this small pilot yet "
            "behave differently out of sample; similar scores are not replication."
        ),
        "required_next_step": (
            "Choose at most a few simple hypotheses before querying a disjoint V3 "
            "validation pilot; exclude all V2 routed sample IDs and keep the test split locked."
        ),
        "atomic_intersections": {
            key: summarize_change_rows(atomic[key]) for key in sorted(atomic)
        },
        "top_v3_hypotheses": screened[:max_hypotheses],
    }


def main() -> None:
    cli = parse_args()
    if cli.policy and cli.freeze_policy:
        raise ValueError("Use either --policy or --freeze-policy, not both")
    if cli.frozen_threshold is not None and not cli.freeze_policy:
        raise ValueError("--frozen-threshold requires --freeze-policy")
    if cli.explore_policy_grid and (cli.policy or cli.freeze_policy or cli.allow_test):
        raise ValueError(
            "Exploratory policy-grid diagnostics cannot load/freeze a policy or unlock test"
        )
    if not 0.0 <= cli.grid_confidence_threshold <= 1.0:
        raise ValueError("Grid confidence threshold must be in [0, 1]")
    thresholds = (
        list(cli.confidence_thresholds)
        if cli.confidence_thresholds is not None
        else [0.80, 0.85, 0.90, 0.95]
    )
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("Confidence thresholds must be in [0, 1]")
    candidate_path = Path(cli.candidates)
    decision_path = Path(cli.decisions)
    records = load_jsonl(candidate_path)
    records.sort(key=lambda record: int(record["sample_id"]))
    if not records:
        raise ValueError("Candidate cache is empty")
    splits = {str(record.get("split", "")) for record in records}
    if len(splits) != 1 or "" in splits:
        raise ValueError("Candidate cache must contain exactly one declared split")
    split = next(iter(splits))
    if cli.explore_policy_grid and (split != "val" or cli.scope != "pilot"):
        raise ValueError(
            "Exploratory policy-grid diagnostics require --scope pilot on validation"
        )
    if split == "test":
        if not cli.allow_test:
            raise ValueError(
                "Test evaluation is locked; pass --allow-test only after freezing the "
                "prompt, route, and confidence threshold."
            )
        if cli.scope != "all":
            raise ValueError("Test evaluation requires --scope all")
    input_path = Path(cli.input)
    actual_input_hash = sha256_inference_projection(input_path)
    for record in records:
        sample_id = record.get("sample_id")
        if record.get("prompt_version") != RERANK_PROMPT_VERSION:
            raise ValueError(f"Prompt version mismatch for sample_id={sample_id}")
        if record.get("input_sha256") != actual_input_hash:
            raise ValueError(f"Dataset hash mismatch for sample_id={sample_id}")
        if record.get("candidate_fingerprint") != candidate_record_fingerprint(record):
            raise ValueError(f"Candidate fingerprint mismatch for sample_id={sample_id}")
    decision_records = load_jsonl(decision_path) if decision_path.exists() else []
    decisions = indexed_records(decision_records, "decision cache")
    policy = None
    if cli.policy:
        policy = json.loads(Path(cli.policy).read_text(encoding="utf-8"))
        decision_models = {
            str(decision.get("llm_model", ""))
            for decision in decision_records
            if decision.get("llm_model")
        }
        policy_model = str(policy.get("llm_model", ""))
        if not policy_model or decision_models - {policy_model}:
            raise ValueError("Decision cache LLM model disagrees with frozen policy")
        frozen_threshold = validate_rerank_policy(
            policy,
            records,
            llm_model=policy_model,
        )
        if cli.confidence_thresholds is not None and set(thresholds) != {
            frozen_threshold
        }:
            raise ValueError("CLI confidence thresholds disagree with frozen policy")
        thresholds = [frozen_threshold]
    elif split == "test":
        raise ValueError("Test evaluation requires --policy from the validation pilot")
    if split == "test" and len(set(thresholds)) != 1:
        raise ValueError("Test evaluation requires one frozen confidence threshold")
    candidate_sample_ids = {int(record["sample_id"]) for record in records}
    unknown_decisions = sorted(set(decisions) - candidate_sample_ids)
    if unknown_decisions:
        raise ValueError(f"Decision cache has unknown sample IDs: {unknown_decisions[:5]}")
    gold = load_gold(input_path, records)
    baseline_predictions = [list(candidate_by_rank(record)[0]["bio"]) for record in records]
    baseline = metrics(gold, baseline_predictions)
    oracle, _ = routed_oracle(gold, records, cli.scope)
    routed_count = sum(
        cli.scope == "all" or bool(record.get("pilot_route", {}).get("selected", False))
        for record in records
    )
    relevant_ids = {
        int(record["sample_id"])
        for record in records
        if cli.scope == "all" or bool(record.get("pilot_route", {}).get("selected", False))
    }
    relevant_decision_count = len(set(decisions) & relevant_ids)
    result: dict[str, Any] = {
        "scope": cli.scope,
        "split": split,
        "samples": len(records),
        "routed_samples": routed_count,
        "decision_records": relevant_decision_count,
        "decision_coverage": relevant_decision_count / max(routed_count, 1),
        "top1": baseline,
        "routed_top5_oracle": oracle,
        "routed_oracle_gain": oracle["f1"] - baseline["f1"],
        "policies": [],
        "go_no_go": {
            "target_normalised_oracle_gain": 0.60,
            "target_top1_exact_preservation": 0.95,
            "note": "Freeze route/prompt/threshold before any test-set reranking.",
        },
    }
    for threshold in sorted(set(thresholds)):
        result["policies"].append(
            evaluate_threshold(
                threshold,
                gold,
                records,
                decisions,
                baseline,
                oracle,
                cli.scope,
            )
        )

    if cli.explore_policy_grid:
        if relevant_decision_count != routed_count:
            raise ValueError(
                "Exploratory grid requires complete decisions for every routed pilot sample"
            )
        result["exploratory_policy_grid"] = explore_policy_grid(
            cli.grid_confidence_threshold,
            gold,
            records,
            decisions,
            baseline,
            oracle,
            min_accepted_changes=cli.grid_min_accepted_changes,
            max_hypotheses=cli.grid_max_hypotheses,
        )

    if cli.freeze_policy:
        if split != "val" or cli.scope != "pilot":
            raise ValueError("Policies can only be frozen from a validation pilot")
        if cli.frozen_threshold is None:
            raise ValueError("--freeze-policy requires --frozen-threshold")
        selected_policy = next(
            (
                item
                for item in result["policies"]
                if abs(item["confidence_threshold"] - cli.frozen_threshold) < 1e-12
            ),
            None,
        )
        if selected_policy is None:
            raise ValueError("Frozen threshold was not evaluated in this run")
        if result["decision_coverage"] < 1.0:
            raise ValueError("Cannot freeze policy before pilot decision coverage reaches 1")
        if (
            selected_policy["normalised_oracle_gain"]
            < result["go_no_go"]["target_normalised_oracle_gain"]
            or selected_policy["top1_exact_preservation"]
            < result["go_no_go"]["target_top1_exact_preservation"]
        ):
            raise ValueError("Selected pilot policy did not pass the go/no-go criteria")
        decision_models = {
            str(decisions[sample_id].get("llm_model", ""))
            for sample_id in relevant_ids
            if sample_id in decisions
        }
        if len(decision_models) != 1 or "" in decision_models:
            raise ValueError("Pilot decisions must contain exactly one LLM model")
        frozen = build_rerank_policy(
            records,
            llm_model=next(iter(decision_models)),
            confidence_threshold=cli.frozen_threshold,
            source_split=split,
        )
        frozen["validation_evidence"] = {
            "candidate_artifact_sha256": sha256_file(candidate_path),
            "decision_artifact_sha256": sha256_file(decision_path),
            "routed_samples": routed_count,
            "decision_coverage": result["decision_coverage"],
            "normalised_oracle_gain": selected_policy["normalised_oracle_gain"],
            "top1_exact_preservation": selected_policy[
                "top1_exact_preservation"
            ],
        }
        policy_output = Path(cli.freeze_policy)
        policy_output.parent.mkdir(parents=True, exist_ok=True)
        policy_output.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["frozen_policy"] = str(policy_output)

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(f"=== LLM reranker {cli.scope} evaluation ({split}) ===")
    print(rendered)
    if cli.output:
        output = Path(cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
