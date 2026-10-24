"""Evaluate the frozen V4 multi-role atomic-review development pilot.

This command is intentionally development-only.  It accepts only the existing
``meta_train`` and ``meta_calibration`` roles and permanently refuses
``meta_evaluation``.  Missing, stale, or incomplete role reviews always fall
back to the CRF top-1 prediction.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.atomic_multirole_schema import (
    ATOMIC_MULTIROLE_PROMPT_VERSION,
    ATOMIC_MULTIROLE_ROLES,
    ROLE_CRITIC,
    ROLE_TEXT,
    ROLE_VISUAL,
    validate_atomic_multirole_output,
)
from llm.atomic_review_schema import validate_atomic_edit_candidates
from modules.atomic_calibrator import (
    CountDelta,
    count_delta,
    entity_counts,
    f1_from_counts,
    outcome_from_gain,
    sha256_file,
)
from modules.llm_dataset import extract_bio_entities
from scripts.build_atomic_calibrator_data import atomic_edits, candidate_by_id
from scripts.build_llm_atomic_review_cache import load_atomic_candidates
from scripts.build_llm_atomic_multirole_cache import SCHEMA_VERSION, load_manifest_roles


ALLOWED_META_ROLES = frozenset({"meta_train", "meta_calibration"})
REQUIRED_REVIEWER_ROLES = (ROLE_TEXT, ROLE_VISUAL, ROLE_CRITIC)
REVIEW_FIELDS = (
    "edit_id",
    "role",
    "verdict",
    "evidence_target",
    "reason_code",
    "used_image_ids",
    "brief_evidence",
)
GO_NO_GO_THRESHOLDS = {
    "minimum_valid_response_coverage": 0.95,
    "minimum_accepted": 20,
    "minimum_beneficial_precision": 0.80,
    "maximum_harmful_rate": 0.10,
    "minimum_top1_exact_preservation": 0.98,
    "minimum_normalized_oracle_gain": 0.50,
    "minimum_f1_gain": 0.003,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Gold meta text file")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--meta-role",
        required=True,
        choices=sorted(ALLOWED_META_ROLES),
        help="Development role to evaluate; meta_evaluation is permanently locked.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_development_role(value: Any) -> str:
    role = str(value)
    if role not in ALLOWED_META_ROLES:
        raise ValueError(
            "--meta-role may be only meta_train or meta_calibration; "
            "meta_evaluation is permanently locked for V4 development"
        )
    return role


def _load_gold(path: Path, allowed_sample_ids: set[int]) -> dict[int, list[str]]:
    """Parse labels only for development IDs; locked rows remain opaque text."""
    result: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            sample_id = line_number - 1
            if sample_id not in allowed_sample_ids:
                continue
            try:
                item = ast.literal_eval(line)
                labels = item["label"]
            except (KeyError, SyntaxError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid gold record at {path}:{line_number}") from error
            if not isinstance(labels, list):
                raise ValueError(f"Gold labels must be a list at {path}:{line_number}")
            result[sample_id] = [
                str(label).replace("MISC", "MIS") for label in labels
            ]
    missing = allowed_sample_ids - set(result)
    if missing:
        raise ValueError(f"gold input is missing development sample IDs: {sorted(missing)[:5]}")
    return result


def _project_gold(
    all_gold: Mapping[int, Sequence[str]], candidate: Mapping[str, Any]
) -> list[str]:
    sample_id = int(candidate["sample_id"])
    if sample_id not in all_gold:
        raise ValueError(f"candidate sample_id={sample_id} is locked or outside the gold file")
    indices = candidate.get("retained_word_indices")
    if not isinstance(indices, list):
        raise ValueError("candidate is missing retained_word_indices")
    projected = [str(all_gold[sample_id][int(index)]) for index in indices]
    if len(projected) != len(candidate.get("words", [])):
        raise ValueError(f"gold/candidate length mismatch for sample_id={sample_id}")
    return projected


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _selected_from_sidecar(
    sidecar: Mapping[str, Any],
    *,
    role: str,
    candidates_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("multirole review sidecar schema mismatch")
    if sidecar.get("prompt_version") != ATOMIC_MULTIROLE_PROMPT_VERSION:
        raise ValueError("multirole review sidecar prompt version mismatch")
    if sidecar.get("gold_entity_labels_used") is not False:
        raise ValueError("multirole selection must prove that gold labels were unused")
    if sidecar.get("candidates_sha256") != sha256_file(candidates_path):
        raise ValueError("multirole sidecar candidate hash mismatch")
    if sidecar.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("multirole sidecar manifest hash mismatch")
    declared_roles = sidecar.get("roles")
    if not isinstance(declared_roles, list) or any(
        declared not in ALLOWED_META_ROLES for declared in declared_roles
    ):
        raise ValueError(
            "multirole sidecar may contain only meta_train/meta_calibration"
        )
    raw = sidecar.get("selected_edits")
    if not isinstance(raw, list):
        raise ValueError("multirole sidecar is missing selected_edits")
    selected: list[dict[str, Any]] = []
    seen_samples: set[int] = set()
    seen_pairs: set[tuple[int, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("selected_edits entries must be objects")
        meta_role = item.get("meta_role")
        if meta_role not in ALLOWED_META_ROLES:
            raise ValueError("selected_edits contains a locked or unsupported meta role")
        if meta_role != role:
            continue
        sample_id = item.get("sample_id")
        edit_id = item.get("edit_id")
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or sample_id < 0
            or not isinstance(edit_id, str)
            or not edit_id
        ):
            raise ValueError("selected_edits contains an invalid sample/edit key")
        pair = (sample_id, edit_id)
        if pair in seen_pairs or sample_id in seen_samples:
            raise ValueError("V4 pilot permits at most one selected edit per sample")
        seen_pairs.add(pair)
        seen_samples.add(sample_id)
        selected.append(dict(item))
    if not selected:
        raise ValueError(f"sidecar has no selected edits for meta role={role}")
    return selected


def _load_review_rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    models: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid review JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Review row must be an object at {path}:{line_number}")
            sample_id = row.get("sample_id")
            edit_id = row.get("edit_id")
            reviewer_role = row.get("role")
            if (
                isinstance(sample_id, bool)
                or not isinstance(sample_id, int)
                or not isinstance(edit_id, str)
                or reviewer_role not in ATOMIC_MULTIROLE_ROLES
            ):
                raise ValueError(f"Invalid review key at {path}:{line_number}")
            key = (sample_id, edit_id, str(reviewer_role))
            if key in seen:
                raise ValueError(f"Duplicate review key={key}")
            seen.add(key)
            model = str(row.get("llm_model", "")).strip()
            if not model:
                raise ValueError(f"Review row has no llm_model at {path}:{line_number}")
            models.add(model)
            result.append(row)
    if len(models) != 1:
        raise ValueError("multirole review cache mixes multiple llm_model values")
    return result


def _candidate_top1(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str]:
    candidate_map = candidate_by_id(candidate)
    top1_id = str(
        candidate.get("top1_candidate_id", candidate.get("baseline_candidate_id", ""))
    )
    if top1_id not in candidate_map:
        raise ValueError(
            f"sample_id={candidate.get('sample_id')} has an invalid top1 candidate"
        )
    return candidate_map, top1_id


def _counts_dict(counts: Sequence[int]) -> dict[str, int]:
    return {"tp": int(counts[0]), "fp": int(counts[1]), "fn": int(counts[2])}


def _consensus(reviews: Mapping[str, Mapping[str, Any]]) -> bool:
    if set(reviews) != set(REQUIRED_REVIEWER_ROLES):
        return False
    text = str(reviews[ROLE_TEXT]["verdict"])
    visual = str(reviews[ROLE_VISUAL]["verdict"])
    critic = str(reviews[ROLE_CRITIC]["verdict"])
    return (
        critic == "NO_OBJECTION"
        and text != "SUPPORT_BASELINE"
        and visual != "SUPPORT_BASELINE"
        and (text == "SUPPORT_EDIT" or visual == "SUPPORT_EDIT")
    )


def _metrics(
    baseline_counts: Sequence[int],
    baseline_exact_samples: set[int],
    rows: Sequence[Mapping[str, Any]],
    accepted_ids: set[tuple[int, str]],
) -> dict[str, Any]:
    counts = list(map(int, baseline_counts))
    accepted = [
        row
        for row in rows
        if (int(row["sample_id"]), str(row["edit_id"])) in accepted_ids
    ]
    for row in accepted:
        delta = row["count_delta"]
        counts[0] += int(delta["tp"])
        counts[1] += int(delta["fp"])
        counts[2] += int(delta["fn"])
    harmful_exact_changes = sum(
        int(row["sample_id"]) in baseline_exact_samples
        and not bool(row["edited_exact"])
        for row in accepted
    )
    beneficial = sum(row["outcome"] == "beneficial" for row in accepted)
    neutral = sum(row["outcome"] == "neutral" for row in accepted)
    harmful = sum(row["outcome"] == "harmful" for row in accepted)
    return {
        "counts": _counts_dict(counts),
        "f1": f1_from_counts(counts),
        "accepted": len(accepted),
        "beneficial": beneficial,
        "neutral": neutral,
        "harmful": harmful,
        "beneficial_precision": beneficial / len(accepted) if accepted else 0.0,
        "harmful_rate": harmful / len(accepted) if accepted else 0.0,
        "top1_exact_samples": len(baseline_exact_samples),
        "top1_exact_preservation": (
            (len(baseline_exact_samples) - harmful_exact_changes)
            / len(baseline_exact_samples)
            if baseline_exact_samples
            else 1.0
        ),
    }


def _selected_edit_oracle(
    baseline_counts: Sequence[int],
    baseline_exact_samples: set[int],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    q = f1_from_counts(baseline_counts)
    accepted: set[tuple[int, str]] = set()
    for iteration in range(1, 101):
        accepted = set()
        for row in rows:
            delta = row["count_delta"]
            gain = CountDelta(
                int(delta["tp"]), int(delta["fp"]), int(delta["fn"])
            ).dinkelbach_gain(q)
            if gain > 1e-12:
                accepted.add((int(row["sample_id"]), str(row["edit_id"])))
        result = _metrics(
            baseline_counts, baseline_exact_samples, rows, accepted
        )
        new_q = float(result["f1"])
        if abs(new_q - q) <= 1e-12:
            result["iterations"] = iteration
            return result
        q = new_q
    raise RuntimeError("selected-edit Dinkelbach oracle did not converge")


def evaluate_pilot(
    *,
    input_path: Path,
    candidates_path: Path,
    reviews_path: Path,
    manifest_path: Path,
    meta_role: str,
) -> dict[str, Any]:
    meta_role = validate_development_role(meta_role)
    candidates = load_atomic_candidates(candidates_path)
    if any(bool(record.get("official_test")) for record in candidates):
        raise ValueError("V4 development pilot cannot read official test candidates")
    candidate_by_sample = {int(record["sample_id"]): record for record in candidates}
    manifest_roles = load_manifest_roles(manifest_path)
    allowed_gold_ids = {
        sample_id
        for sample_id, (_, role) in manifest_roles.items()
        if role == "meta_train" or role == meta_role
    }
    gold = _load_gold(input_path, allowed_gold_ids)

    sidecar_path = reviews_path.with_name(reviews_path.name + ".meta.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"multirole review sidecar is missing: {sidecar_path}")
    sidecar = _read_json(sidecar_path)
    selected = _selected_from_sidecar(
        sidecar,
        role=meta_role,
        candidates_path=candidates_path,
        manifest_path=manifest_path,
    )
    selected_keys = {
        (int(item["sample_id"]), str(item["edit_id"])) for item in selected
    }
    all_sidecar_keys = {
        (int(item["sample_id"]), str(item["edit_id"]))
        for item in sidecar["selected_edits"]
    }

    baseline_counts_by_role: dict[str, list[int]] = {
        role: [0, 0, 0] for role in ALLOWED_META_ROLES
    }
    baseline_exact_by_role: dict[str, set[int]] = {
        role: set() for role in ALLOWED_META_ROLES
    }
    prepared: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        sample_id = int(candidate["sample_id"])
        assignment = manifest_roles.get(sample_id)
        if assignment is None:
            raise ValueError(f"manifest has no assignment for sample_id={sample_id}")
        source_id, role = assignment
        if role == "meta_evaluation":
            # The record remains in the candidate artifact, but no labels or reviews
            # from it are exposed through the V4 development result.
            continue
        if role not in ALLOWED_META_ROLES:
            raise ValueError(f"unsupported manifest meta role: {role!r}")
        if int(candidate.get("source_sample_id", -1)) != source_id:
            raise ValueError(f"manifest source mismatch for sample_id={sample_id}")
        candidate_map, top1_id = _candidate_top1(candidate)
        gold_entities = extract_bio_entities(_project_gold(gold, candidate))
        top1_entities = extract_bio_entities(candidate_map[top1_id]["bio"])
        counts = entity_counts(gold_entities, top1_entities)
        for index, value in enumerate(counts):
            baseline_counts_by_role[role][index] += int(value)
        if gold_entities == top1_entities:
            baseline_exact_by_role[role].add(sample_id)
        prepared[sample_id] = {
            "candidate": candidate,
            "candidate_map": candidate_map,
            "top1_entities": top1_entities,
            "gold_entities": gold_entities,
            "baseline_counts": counts,
            "role": role,
        }
    if not any(baseline_counts_by_role["meta_train"]):
        raise ValueError("meta_train baseline counts are required to freeze Dinkelbach q")
    dinkelbach_q = f1_from_counts(baseline_counts_by_role["meta_train"])

    selected_rows: list[dict[str, Any]] = []
    selected_edit_objects: dict[tuple[int, str], Any] = {}
    for item in selected:
        sample_id = int(item["sample_id"])
        edit_id = str(item["edit_id"])
        prepared_item = prepared.get(sample_id)
        if prepared_item is None or prepared_item["role"] != meta_role:
            raise ValueError(f"selected edit role mismatch for sample_id={sample_id}")
        candidate = prepared_item["candidate"]
        edits = atomic_edits(candidate)
        edit_map = {str(edit["edit_id"]): edit for edit in edits}
        if edit_id not in edit_map:
            raise ValueError(f"selected edit is absent from candidate: {(sample_id, edit_id)}")
        typed_map = {
            edit.edit_id: edit
            for edit in validate_atomic_edit_candidates(edits, candidate["words"])
        }
        selected_edit_objects[(sample_id, edit_id)] = typed_map[edit_id]
        edit = edit_map[edit_id]
        candidate_id = str(edit.get("candidate_id", ""))
        alternative = prepared_item["candidate_map"].get(candidate_id)
        if alternative is None:
            raise ValueError(f"selected edit references an unknown candidate: {candidate_id}")
        edited_entities = extract_bio_entities(alternative["bio"])
        edited_counts = entity_counts(prepared_item["gold_entities"], edited_entities)
        delta = count_delta(prepared_item["baseline_counts"], edited_counts)
        gain = delta.dinkelbach_gain(dinkelbach_q)
        selected_rows.append(
            {
                "sample_id": sample_id,
                "edit_id": edit_id,
                "count_delta": {
                    "tp": delta.delta_tp,
                    "fp": delta.delta_fp,
                    "fn": delta.delta_fn,
                },
                "dinkelbach_gain": gain,
                "outcome": outcome_from_gain(gain),
                "edited_exact": edited_entities == prepared_item["gold_entities"],
            }
        )

    review_rows = _load_review_rows(reviews_path)
    valid_reviews: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    invalid_reasons: Counter[str] = Counter()
    for row in review_rows:
        pair = (int(row["sample_id"]), str(row["edit_id"]))
        if pair not in all_sidecar_keys:
            raise ValueError(f"review row is not present in selected_edits: {pair}")
        if row.get("meta_role") == "meta_evaluation":
            raise ValueError("review cache contains locked meta_evaluation data")
        if pair not in selected_keys:
            continue
        candidate = candidate_by_sample[pair[0]]
        reviewer_role = str(row["role"])
        safe = True
        if row.get("schema_version") != SCHEMA_VERSION:
            invalid_reasons["schema_mismatch"] += 1
            safe = False
        if row.get("prompt_version") != ATOMIC_MULTIROLE_PROMPT_VERSION:
            invalid_reasons["prompt_version_mismatch"] += 1
            safe = False
        if row.get("review_fingerprint") != candidate.get("review_fingerprint"):
            invalid_reasons["fingerprint_mismatch"] += 1
            safe = False
        if row.get("meta_role") != meta_role:
            invalid_reasons["meta_role_mismatch"] += 1
            safe = False
        for field in (
            "source_sample_id",
            "split",
            "official_test",
            "checkpoint_sha256",
            "input_sha256",
            "source_ids_sha256",
        ):
            if row.get(field) != candidate.get(field):
                invalid_reasons[f"{field}_mismatch"] += 1
                safe = False
                break
        if not safe:
            continue
        payload = {field: row.get(field) for field in REVIEW_FIELDS}
        try:
            validated = validate_atomic_multirole_output(
                payload,
                selected_edit_objects[pair],
                reviewer_role,
                candidate.get("valid_image_ids", []),
            )
        except (TypeError, ValueError):
            invalid_reasons["invalid_role_response"] += 1
            continue
        valid_reviews.setdefault(pair, {})[reviewer_role] = validated.to_dict()

    patterns: Counter[str] = Counter()
    role_valid = Counter[str]()
    complete = 0
    consensus_ids: set[tuple[int, str]] = set()
    for pair in sorted(selected_keys):
        role_map = valid_reviews.get(pair, {})
        for reviewer_role in REQUIRED_REVIEWER_ROLES:
            if reviewer_role in role_map:
                role_valid[reviewer_role] += 1
        if set(role_map) == set(REQUIRED_REVIEWER_ROLES):
            complete += 1
        pattern = "|".join(
            f"{reviewer_role}={role_map.get(reviewer_role, {}).get('verdict', 'MISSING')}"
            for reviewer_role in REQUIRED_REVIEWER_ROLES
        )
        patterns[pattern] += 1
        if _consensus(role_map):
            consensus_ids.add(pair)

    baseline_counts = baseline_counts_by_role[meta_role]
    baseline_f1 = f1_from_counts(baseline_counts)
    baseline = {
        "counts": _counts_dict(baseline_counts),
        "f1": baseline_f1,
        "samples": sum(1 for item in prepared.values() if item["role"] == meta_role),
        "top1_exact_samples": len(baseline_exact_by_role[meta_role]),
    }
    oracle = _selected_edit_oracle(
        baseline_counts,
        baseline_exact_by_role[meta_role],
        selected_rows,
    )
    consensus = _metrics(
        baseline_counts,
        baseline_exact_by_role[meta_role],
        selected_rows,
        consensus_ids,
    )
    consensus["gain_over_top1"] = float(consensus["f1"] - baseline_f1)
    oracle_gain = float(oracle["f1"] - baseline_f1)
    consensus["normalized_oracle_gain"] = (
        consensus["gain_over_top1"] / oracle_gain if oracle_gain > 1e-12 else 0.0
    )
    selected_count = len(selected_rows)
    valid_response_coverage = complete / selected_count if selected_count else 0.0
    role_diagnostics = {
        "selected_edits": selected_count,
        "complete_role_triplets": complete,
        "valid_response_coverage": valid_response_coverage,
        "role_coverage": {
            reviewer_role: {
                "valid": role_valid[reviewer_role],
                "expected": selected_count,
                "coverage": role_valid[reviewer_role] / selected_count,
            }
            for reviewer_role in REQUIRED_REVIEWER_ROLES
        },
        "verdict_patterns": dict(sorted(patterns.items())),
        "invalid_review_rows": sum(invalid_reasons.values()),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
    }

    checks = {
        "valid_response_coverage_pass": valid_response_coverage
        >= GO_NO_GO_THRESHOLDS["minimum_valid_response_coverage"],
        "minimum_accepted_pass": consensus["accepted"]
        >= GO_NO_GO_THRESHOLDS["minimum_accepted"],
        "beneficial_precision_pass": consensus["beneficial_precision"]
        >= GO_NO_GO_THRESHOLDS["minimum_beneficial_precision"],
        "harmful_rate_pass": consensus["harmful_rate"]
        <= GO_NO_GO_THRESHOLDS["maximum_harmful_rate"],
        "top1_exact_preservation_pass": consensus["top1_exact_preservation"]
        >= GO_NO_GO_THRESHOLDS["minimum_top1_exact_preservation"],
        "normalized_oracle_gain_pass": consensus["normalized_oracle_gain"]
        >= GO_NO_GO_THRESHOLDS["minimum_normalized_oracle_gain"],
        "f1_gain_pass": consensus["gain_over_top1"]
        >= GO_NO_GO_THRESHOLDS["minimum_f1_gain"],
    }
    return {
        "schema_version": "atomic-multirole-pilot-evaluation-v1",
        "development_only": True,
        "meta_role": meta_role,
        "locked_roles": ["meta_evaluation"],
        "provenance": {
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "candidates": str(candidates_path),
            "candidates_sha256": sha256_file(candidates_path),
            "reviews": str(reviews_path),
            "reviews_sha256": sha256_file(reviews_path),
            "reviews_metadata": str(sidecar_path),
            "reviews_metadata_sha256": sha256_file(sidecar_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "dinkelbach_q": dinkelbach_q,
        "dinkelbach_q_scope": "meta_train_top1_counts_only",
        "consensus_rule": {
            "required_roles": list(REQUIRED_REVIEWER_ROLES),
            "critic": "NO_OBJECTION",
            "verifier_rejections": ["SUPPORT_BASELINE"],
            "minimum_support_edit_verifiers": 1,
            "fallback": "CRF top-1 on every missing, stale, invalid, or disagreeing review",
        },
        "baseline": baseline,
        "selected_edit_oracle": oracle,
        "consensus": consensus,
        "role_diagnostics": role_diagnostics,
        "go_no_go": {
            "thresholds": dict(GO_NO_GO_THRESHOLDS),
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    result = evaluate_pilot(
        input_path=Path(args.input),
        candidates_path=Path(args.candidates),
        reviews_path=Path(args.reviews),
        manifest_path=Path(args.manifest),
        meta_role=args.meta_role,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("=== Atomic multi-role development pilot ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
