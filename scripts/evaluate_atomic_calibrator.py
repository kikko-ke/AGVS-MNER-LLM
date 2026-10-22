"""Evaluate a frozen atomic calibrator and its LLM feature ablations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.atomic_calibrator import (
    EXAMPLE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    OUTCOME_TO_INDEX,
    POLICY_SCHEMA_VERSION,
    exact_atomic_oracle,
    feature_mode_vector,
    f1_from_counts,
    group_examples,
    load_checkpoint,
    safe_select_edits,
    selections_metrics,
    sha256_file,
    shuffle_review_features,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--structure-checkpoint",
        default=None,
        help="Independently trained structure-only calibrator checkpoint.",
    )
    parser.add_argument(
        "--structure-policy",
        default=None,
        help="Frozen policy paired with --structure-checkpoint.",
    )
    parser.add_argument("--role", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle-seed", type=int, default=20261022)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_examples(path: Path, role: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if record.get("schema_version") != EXAMPLE_SCHEMA_VERSION:
                raise ValueError(f"Example schema mismatch at {path}:{line_number}")
            if record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                raise ValueError(f"Feature schema mismatch at {path}:{line_number}")
            if tuple(record.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError(f"Feature names mismatch at {path}:{line_number}")
            if role == "all" or record.get("meta_role") == role:
                records.append(record)
    if not records:
        raise ValueError(f"no atomic examples found for role={role!r}")
    group_examples(records)
    return records


def predict(
    model: torch.nn.Module,
    raw_features: Sequence[Sequence[float]],
    payload: dict[str, Any],
    device: str,
) -> tuple[list[float], list[float]]:
    matrix = torch.tensor(raw_features, dtype=torch.float32, device=device)
    mean = payload["feature_mean"].to(device)
    std = payload["feature_std"].to(device)
    with torch.no_grad():
        utilities, logits = model(standardize(matrix, mean, std))
        harmful = torch.softmax(logits, dim=-1)[:, OUTCOME_TO_INDEX["harmful"]]
    return utilities.cpu().tolist(), harmful.cpu().tolist()


def evaluate_mode(
    name: str,
    records: list[dict[str, Any]],
    raw_features: Sequence[Sequence[float]],
    model: torch.nn.Module,
    payload: dict[str, Any],
    policy: dict[str, Any],
    metadata: dict[str, Any],
    role: str,
    device: str,
) -> dict[str, Any]:
    utilities, harmful = predict(model, raw_features, payload, device)
    selections, fallbacks = safe_select_edits(records, utilities, harmful, policy)
    result = selections_metrics(
        metadata, role, group_examples(records), selections
    )
    result.update(
        {
            "mode": name,
            "fallbacks": dict(fallbacks),
            "utility": {
                "minimum": min(utilities),
                "maximum": max(utilities),
                "mean": sum(utilities) / len(utilities),
            },
            "mean_harm_probability": sum(harmful) / len(harmful),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    data_path = Path(args.data)
    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else data_path.with_name(data_path.name + ".meta.json")
    )
    checkpoint_path = Path(args.checkpoint)
    policy_path = Path(args.policy)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported atomic calibrator policy")
    if policy.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("policy feature schema mismatch")
    if policy.get("feature_mode") != "full":
        raise ValueError("--policy must be calibrated in full LLM-feature mode")
    if policy.get("calibrator_sha256") != sha256_file(checkpoint_path):
        raise ValueError("policy is bound to a different calibrator checkpoint")
    if policy.get("data_sha256") != sha256_file(data_path):
        raise ValueError("policy is bound to a different atomic example artifact")
    if policy.get("metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("policy is bound to different atomic example metadata")
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("metadata feature schema mismatch")
    if args.role not in metadata.get("roles", {}):
        raise ValueError(f"metadata does not contain role={args.role!r}")

    records = load_examples(data_path, args.role)
    model, payload = load_checkpoint(checkpoint_path, args.device)
    if payload.get("feature_schema_version") != policy.get("feature_schema_version"):
        raise ValueError("checkpoint and policy feature schemas differ")
    if payload.get("data_sha256") != policy.get("data_sha256"):
        raise ValueError("checkpoint and policy data provenance differ")
    if payload.get("metadata_sha256") != policy.get("metadata_sha256"):
        raise ValueError("checkpoint and policy metadata provenance differ")
    if payload.get("feature_mode") != "full":
        raise ValueError("--checkpoint must be the full LLM-feature calibrator")

    full_features = [
        feature_mode_vector(record["features"], "full") for record in records
    ]
    results: dict[str, dict[str, Any]] = {
        "full": evaluate_mode(
            "full",
            records,
            full_features,
            model,
            payload,
            policy,
            metadata,
            args.role,
            args.device,
        )
    }
    if not args.no_ablations:
        zeroed_review_features = [
            feature_mode_vector(record["features"], "structure_only")
            for record in records
        ]
        results["zeroed_review"] = evaluate_mode(
            "zeroed_review",
            records,
            zeroed_review_features,
            model,
            payload,
            policy,
            metadata,
            args.role,
            args.device,
        )
        results["shuffled_review"] = evaluate_mode(
            "shuffled_review",
            records,
            shuffle_review_features(records, seed=args.shuffle_seed),
            model,
            payload,
            policy,
            metadata,
            args.role,
            args.device,
        )
    if bool(args.structure_checkpoint) != bool(args.structure_policy):
        raise ValueError(
            "--structure-checkpoint and --structure-policy must be provided together"
        )
    if args.structure_checkpoint:
        structure_checkpoint = Path(args.structure_checkpoint)
        structure_policy_path = Path(args.structure_policy)
        structure_policy = json.loads(
            structure_policy_path.read_text(encoding="utf-8")
        )
        if structure_policy.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported structure-only policy schema")
        if structure_policy.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("structure-only policy feature schema mismatch")
        if structure_policy.get("feature_mode") != "structure_only":
            raise ValueError("structure policy was not calibrated in structure-only mode")
        if structure_policy.get("calibrator_sha256") != sha256_file(
            structure_checkpoint
        ):
            raise ValueError("structure policy is bound to another checkpoint")
        for key, expected in (
            ("data_sha256", sha256_file(data_path)),
            ("metadata_sha256", sha256_file(metadata_path)),
        ):
            if structure_policy.get(key) != expected:
                raise ValueError(f"structure policy {key} provenance mismatch")
        structure_model, structure_payload = load_checkpoint(
            structure_checkpoint, args.device
        )
        if structure_payload.get("feature_mode") != "structure_only":
            raise ValueError(
                "--structure-checkpoint must be trained with --feature-mode structure-only"
            )
        if structure_payload.get("data_sha256") != structure_policy.get("data_sha256"):
            raise ValueError("structure checkpoint and policy data provenance differ")
        if structure_payload.get("metadata_sha256") != structure_policy.get(
            "metadata_sha256"
        ):
            raise ValueError("structure checkpoint and policy metadata provenance differ")
        structure_features = [
            feature_mode_vector(record["features"], "structure_only")
            for record in records
        ]
        results["structure_only"] = evaluate_mode(
            "structure_only",
            records,
            structure_features,
            structure_model,
            structure_payload,
            structure_policy,
            metadata,
            args.role,
            args.device,
        )
    grouped = group_examples(records)
    oracle = exact_atomic_oracle(metadata, args.role, grouped)
    baseline_counts = metadata["roles"][args.role]["baseline_counts"]
    baseline_f1 = f1_from_counts(baseline_counts)
    oracle_gain = oracle["f1"] - baseline_f1
    for result in results.values():
        result["gain_over_top1"] = result["f1"] - baseline_f1
        result["normalized_oracle_gain"] = (
            result["gain_over_top1"] / oracle_gain if oracle_gain > 0 else 0.0
        )
    full = results["full"]
    structure = results.get("structure_only")
    go_no_go = {
        "minimum_normalized_atomic_oracle_gain": 0.60,
        "minimum_top1_exact_preservation": 0.98,
        "minimum_full_minus_structure_f1": 0.003,
        "normalized_atomic_oracle_gain_pass": full["normalized_oracle_gain"] >= 0.60,
        "top1_exact_preservation_pass": full["top1_exact_preservation"] >= 0.98,
        "beneficial_exceeds_harmful_pass": full["beneficial"] > full["harmful"],
        "full_minus_structure_f1": None
        if structure is None
        else full["f1"] - structure["f1"],
        "full_beats_structure_pass": None
        if structure is None
        else full["f1"] - structure["f1"] >= 0.003,
    }
    go_no_go["passed"] = all(
        value is True
        for key, value in go_no_go.items()
        if key.endswith("_pass")
    ) and structure is not None
    summary = {
        "schema_version": "atomic-calibrator-evaluation-v1",
        "data": str(data_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "policy": str(policy_path),
        "role": args.role,
        "examples": len(records),
        "samples_with_edits": len(grouped),
        "baseline": {
            "counts": {
                "tp": int(baseline_counts[0]),
                "fp": int(baseline_counts[1]),
                "fn": int(baseline_counts[2]),
            },
            "f1": baseline_f1,
        },
        "atomic_oracle": {
            key: value for key, value in oracle.items() if key != "selections"
        },
        "policy_parameters": {
            "score_threshold": policy["score_threshold"],
            "harm_probability_cap": policy["harm_probability_cap"],
            "selected_on_role": policy.get("selected_on_role"),
        },
        "modes": results,
        "go_no_go": go_no_go,
    }
    rendered = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
    print("=== Atomic calibrator evaluation ===")
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
