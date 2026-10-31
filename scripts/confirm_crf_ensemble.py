"""Confirm a previously frozen two-model CRF policy on new training seeds (val only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from modules.dataset import MultimodalNerDataset
from scripts.evaluate_crf_ensemble import (
    _collect,
    decode_blend,
    score,
    sha256_file,
    validate_configs,
    word_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Policy JSON saved by the seed-1234 selection run.")
    parser.add_argument("--runs-root", default="outputs_v61")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--data-path", default="dataset/text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (policy.get("schema_version") != "crf-score-ensemble-policy-v1"
            or policy.get("selection_split") != "val"):
        raise ValueError("Expected a frozen validation-selection CRF ensemble policy")
    weights = policy.get("weights", {})
    if set(weights) != {"control", "baseline"}:
        raise ValueError("Confirmation requires the frozen control + baseline policy")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 < value < 1 for value in weights.values()):
        raise ValueError("Policy weights must be finite numbers strictly between 0 and 1")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Policy weights must sum to one")
    if not policy.get("validation_input_sha256"):
        raise ValueError("Policy lacks validation input SHA256")
    original = policy.get("checkpoints", {})
    if set(original) != {"control", "baseline"}:
        raise ValueError("Policy lacks source checkpoint metadata")
    validate_configs({name: original[name]["config"] for name in original})
    return policy


def find_checkpoint(root: Path, arm: str, seed: int) -> Path:
    directory = root / f"{arm}_seed{seed}"
    paths = sorted(directory.glob("*/best_model.pt"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint in {directory}, found {len(paths)}: "
            f"{[str(path) for path in paths]}. Use a runs root with one run per arm/seed."
        )
    return paths[0]


def _single_metrics(data: dict[str, Any], label_map: dict[str, int], checkpoint: Path):
    replay = decode_blend(
        data["emissions"], data["emissions"], data["crf"], data["crf"], data["mask"], 0.0,
    )
    gold, predicted = word_sequences(data["mask"], data["labels"], replay, label_map)
    _, native = word_sequences(data["mask"], data["labels"], data["native_paths"], label_map)
    if predicted != native:
        raise RuntimeError(f"Single-model CRF replay differs: {checkpoint}")
    metrics = score(gold, predicted)
    with (checkpoint.parent / "metrics.csv").open(encoding="utf-8", newline="") as source:
        recorded = [float(row["dev_f1"]) for row in csv.DictReader(source) if row.get("dev_f1")]
    if not recorded or any(not math.isfinite(value) for value in recorded):
        raise ValueError(f"Missing or non-finite recorded dev F1: {checkpoint}")
    metrics["recorded_best_dev_f1"] = max(recorded)
    if abs(metrics["f1"] - max(recorded)) > 0.0005:
        raise RuntimeError(
            f"F1 {metrics['f1']:.6f} does not reproduce recorded {max(recorded):.6f}: {checkpoint}"
        )
    return gold, metrics


def main() -> None:
    cli = parse_args()
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("Confirmation seeds must be unique")
    if cli.batch_size < 1 or cli.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if str(cli.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError("Confirmation output exists; choose a new --output path")
    policy_path = Path(cli.policy)
    policy = read_policy(policy_path)
    source_configs = {name: item["config"] for name, item in policy["checkpoints"].items()}
    selection_seed = int(source_configs["control"]["seed"])
    if selection_seed in cli.seeds:
        raise ValueError("The weight-selection seed cannot be a new confirmation seed")
    file_name = f"MNER-{source_configs['control']['dataset']}_val.txt"
    input_path = Path(cli.data_path) / file_name
    input_hash = sha256_file(input_path)
    if input_hash != policy["validation_input_sha256"]:
        raise ValueError("Validation SHA256 differs from the frozen selection policy")

    # Resolve and validate every run before doing any model inference. Never
    # choose a checkpoint by validation performance among repeated seed runs.
    runs = []
    for seed in cli.seeds:
        checkpoints = {
            "control": find_checkpoint(Path(cli.runs_root), "duplicate_control", seed),
            "baseline": find_checkpoint(Path(cli.runs_root), "strict_baseline", seed),
        }
        configs = {
            name: json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            for name, path in checkpoints.items()
        }
        validate_configs(configs)
        for name, config in configs.items():
            if int(config["seed"]) != seed:
                raise ValueError(f"Checkpoint seed does not match requested {seed}: {checkpoints[name]}")
            validate_configs({
                "control": source_configs[name],
                "confirmation": {**config, "seed": selection_seed},
            })
            if sha256_file(Path(config["data_path"]) / file_name) != input_hash:
                raise ValueError(f"Different validation data: {checkpoints[name]}")
            if not (checkpoints[name].parent / "metrics.csv").is_file():
                raise FileNotFoundError(checkpoints[name].parent / "metrics.csv")
        runs.append((seed, checkpoints, configs))

    dataset_config = {**source_configs["control"], "data_path": cli.data_path}
    dataset = MultimodalNerDataset(SimpleNamespace(**dataset_config), file_name)
    if not len(dataset):
        raise ValueError("Validation dataset is empty")
    label_map = dataset.processor.get_label_mapping(include_subword_label=True)
    rows = []
    for seed, checkpoints, configs in runs:
        collected, singles = {}, {}
        for name in ("control", "baseline"):
            data = _collect(checkpoints[name], configs[name], dataset, cli, label_map, f"seed{seed}/{name}")
            gold, singles[name] = _single_metrics(data, label_map, checkpoints[name])
            if collected and any(
                not torch.equal(data[key], collected["control"][key]) for key in ("mask", "labels")
            ):
                raise RuntimeError("Sample order, masks, or targets changed across models")
            collected[name] = data
            print(f"seed={seed} Single {name}: F1={singles[name]['f1']:.6f} (verified)", flush=True)
        control, baseline = collected["control"], collected["baseline"]
        paths = decode_blend(
            control["emissions"], baseline["emissions"], control["crf"], baseline["crf"],
            control["mask"], float(policy["weights"]["baseline"]),
        )
        gold, predicted = word_sequences(control["mask"], control["labels"], paths, label_map)
        fused = score(gold, predicted)
        row = {
            "seed": seed, "single_models": singles, "ensemble": fused,
            "gain_over_control": fused["f1"] - singles["control"]["f1"],
            "gain_over_baseline": fused["f1"] - singles["baseline"]["f1"],
            "gain_over_best_single": fused["f1"] - max(item["f1"] for item in singles.values()),
            "checkpoints": {name: {"path": str(path), "sha256": sha256_file(path), "config": configs[name]}
                            for name, path in checkpoints.items()},
        }
        rows.append(row)
        print(f"seed={seed} Fixed ensemble: F1={fused['f1']:.6f}; "
              f"gain over best single={row['gain_over_best_single']:+.6f}", flush=True)
    summary = {
        "schema_version": "fixed-crf-ensemble-seed-confirmation-v1",
        "split": "val", "weights_searched": False, "llm_api_requests": 0,
        "selection_seed": selection_seed, "confirmation_seeds": cli.seeds,
        "weights": policy["weights"], "source_policy": str(policy_path),
        "source_policy_sha256": sha256_file(policy_path),
        "source_checkpoints": policy["checkpoints"],
        "input": str(input_path), "input_sha256": input_hash, "samples": len(dataset),
        "per_seed": rows,
        "aggregate": {
            "mean_baseline_f1": statistics.mean(row["single_models"]["baseline"]["f1"] for row in rows),
            "mean_control_f1": statistics.mean(row["single_models"]["control"]["f1"] for row in rows),
            "mean_ensemble_f1": statistics.mean(row["ensemble"]["f1"] for row in rows),
            "mean_gain_over_best_single": statistics.mean(row["gain_over_best_single"] for row in rows),
            "positive_seeds_over_best_single": sum(row["gain_over_best_single"] > 1e-12 for row in rows),
            "confirmation_seed_count": len(rows),
        },
        "note": "Fixed-policy replication across training seeds on the same validation split; "
                "not a new held-out dataset, significance test, or evidence of LLM benefit.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n=== Fixed-weight seed confirmation (validation only) ===")
    print("seed       baseline      control     ensemble   gain_over_best_single")
    for row in rows:
        print(f"{row['seed']:<8} {row['single_models']['baseline']['f1']:.6f}     "
              f"{row['single_models']['control']['f1']:.6f}     {row['ensemble']['f1']:.6f}   "
              f"{row['gain_over_best_single']:+.6f}")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
