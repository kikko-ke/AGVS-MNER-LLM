"""Train and calibrate a conservative supervised atomic-edit selector."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.atomic_calibrator import (
    EXAMPLE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    INDEX_TO_OUTCOME,
    OUTCOME_TO_INDEX,
    AtomicCalibrator,
    calibrator_loss,
    exact_atomic_oracle,
    feature_mode_vector,
    f1_from_counts,
    fit_standardizer,
    group_examples,
    policy_grid,
    sha256_file,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-output", default=None)
    parser.add_argument("--train-role", default="meta_train")
    parser.add_argument("--calibration-role", default="meta_calibration")
    parser.add_argument(
        "--feature-mode", choices=("full", "structure-only"), default="full"
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--harmful-class-multiplier", type=float, default=3.0)
    parser.add_argument("--harmful-penalty-weight", type=float, default=1.0)
    parser.add_argument("--minimum-top1-preservation", type=float, default=0.98)
    parser.add_argument("--minimum-accepted", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_examples(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Incompatible example at {path}:{line_number}")
            if record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                raise ValueError(f"Feature schema mismatch at {path}:{line_number}")
            if tuple(record.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError(f"Feature names mismatch at {path}:{line_number}")
            features = record.get("features")
            if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
                raise ValueError(f"Invalid feature vector at {path}:{line_number}")
            if not all(math.isfinite(float(value)) for value in features):
                raise ValueError(f"Non-finite feature at {path}:{line_number}")
            if record.get("outcome") not in OUTCOME_TO_INDEX:
                raise ValueError(f"Invalid outcome at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError("atomic calibrator data is empty")
    group_examples(records)
    return records


def tensorize(
    records: list[dict[str, Any]], mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    features = torch.tensor(
        [feature_mode_vector(record["features"], mode) for record in records],
        dtype=torch.float32,
    )
    gains = torch.tensor(
        [float(record["dinkelbach_gain"]) for record in records],
        dtype=torch.float32,
    )
    outcomes = torch.tensor(
        [OUTCOME_TO_INDEX[str(record["outcome"])] for record in records],
        dtype=torch.long,
    )
    sample_ids = [int(record["sample_id"]) for record in records]
    return features, gains, outcomes, sample_ids


def class_weights(outcomes: torch.Tensor, harmful_multiplier: float) -> torch.Tensor:
    counts = torch.bincount(outcomes, minlength=len(OUTCOME_TO_INDEX)).float()
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = outcomes.numel() / (
        len(OUTCOME_TO_INDEX) * counts[nonzero]
    )
    weights[OUTCOME_TO_INDEX["harmful"]] *= float(harmful_multiplier)
    positive = weights > 0
    if positive.any():
        weights[positive] /= weights[positive].mean()
    return weights


def predict(
    model: AtomicCalibrator,
    raw_features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
) -> tuple[list[float], list[float]]:
    with torch.no_grad():
        utilities, logits = model(
            standardize(raw_features.to(device), mean.to(device), std.to(device))
        )
        harmful = torch.softmax(logits, dim=-1)[:, OUTCOME_TO_INDEX["harmful"]]
    return utilities.cpu().tolist(), harmful.cpu().tolist()


def main() -> None:
    args = parse_args()
    if args.hidden_size <= 0 or args.epochs <= 0:
        raise ValueError("--hidden-size and --epochs must be positive")
    if args.lr <= 0 or args.harmful_class_multiplier <= 0:
        raise ValueError("learning rate and harmful multiplier must be positive")
    if not 0.0 <= args.minimum_top1_preservation <= 1.0:
        raise ValueError("--minimum-top1-preservation must be in [0, 1]")
    if args.minimum_accepted < 0:
        raise ValueError("--minimum-accepted must be non-negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")

    data_path = Path(args.data)
    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else data_path.with_name(data_path.name + ".meta.json")
    )
    output = Path(args.output)
    policy_output = (
        Path(args.policy_output)
        if args.policy_output
        else output.with_name(output.name + ".policy.json")
    )
    metrics_output = output.with_name(output.name + ".metrics.json")
    for path in (output, policy_output, metrics_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    records = load_examples(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("metadata feature schema mismatch")
    train_records = [
        record for record in records if record.get("meta_role") == args.train_role
    ]
    calibration_records = [
        record
        for record in records
        if record.get("meta_role") == args.calibration_role
    ]
    if not train_records or not calibration_records:
        raise ValueError("both train and calibration roles need reviewed atomic edits")

    mode = args.feature_mode.replace("-", "_")
    train_x, train_gains, train_outcomes, train_sample_ids = tensorize(
        train_records, mode
    )
    calibration_x, _, _, _ = tensorize(calibration_records, mode)
    mean, std = fit_standardizer(train_x)
    weights = class_weights(train_outcomes, args.harmful_class_multiplier)

    device = str(args.device)
    model = AtomicCalibrator(len(FEATURE_NAMES), args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    normalized_x = standardize(train_x.to(device), mean.to(device), std.to(device))
    gains_device = train_gains.to(device)
    outcomes_device = train_outcomes.to(device)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        utilities, logits = model(normalized_x)
        loss, components = calibrator_loss(
            utilities,
            logits,
            gains_device,
            outcomes_device,
            train_sample_ids,
            class_weights=weights,
            harmful_penalty_weight=args.harmful_penalty_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % 25 == 0:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                **{
                    name: float(value.cpu())
                    for name, value in components.items()
                },
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": "atomic-calibrator-checkpoint-v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_mode": mode,
        "hidden_size": args.hidden_size,
        "model_state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "seed": args.seed,
        "train_role": args.train_role,
        "calibration_role": args.calibration_role,
        "dinkelbach_q": float(metadata["dinkelbach_q"]),
        "data_sha256": sha256_file(data_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    torch.save(payload, output)
    checkpoint_sha256 = sha256_file(output)

    model.eval()
    calibration_utilities, calibration_harm = predict(
        model, calibration_x, mean, std, device
    )
    policy, calibration_summary = policy_grid(
        calibration_records,
        calibration_utilities,
        calibration_harm,
        metadata,
        args.calibration_role,
        minimum_top1_preservation=args.minimum_top1_preservation,
        minimum_accepted=args.minimum_accepted,
    )
    policy.update(
        {
            "calibrator_sha256": checkpoint_sha256,
            "feature_mode": mode,
            "data_sha256": sha256_file(data_path),
            "metadata_sha256": sha256_file(metadata_path),
            "seed": args.seed,
        }
    )
    policy_output.write_text(
        json.dumps(policy, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )

    calibration_grouped = group_examples(calibration_records)
    oracle = exact_atomic_oracle(
        metadata, args.calibration_role, calibration_grouped
    )
    baseline_counts = metadata["roles"][args.calibration_role]["baseline_counts"]
    baseline_f1 = f1_from_counts(baseline_counts)
    selected = calibration_summary["selected"]
    oracle_gain = oracle["f1"] - baseline_f1
    realized_gain = selected["f1"] - baseline_f1
    summary = {
        "checkpoint": str(output),
        "checkpoint_sha256": checkpoint_sha256,
        "policy": str(policy_output),
        "feature_mode": mode,
        "train_examples": len(train_records),
        "train_samples": len({int(row["sample_id"]) for row in train_records}),
        "calibration_examples": len(calibration_records),
        "calibration_samples": len(calibration_grouped),
        "train_outcomes": dict(Counter(row["outcome"] for row in train_records)),
        "calibration_outcomes": dict(
            Counter(row["outcome"] for row in calibration_records)
        ),
        "baseline_f1": baseline_f1,
        "atomic_oracle_f1": oracle["f1"],
        "selected_policy": selected,
        "normalized_oracle_gain": (
            realized_gain / oracle_gain if oracle_gain > 0 else 0.0
        ),
        "policy_scan": calibration_summary,
        "training_history": history,
        "outcome_indices": INDEX_TO_OUTCOME,
    }
    metrics_output.write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Atomic calibrator training complete ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
