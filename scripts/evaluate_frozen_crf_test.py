"""Report all seeds on test using the policy frozen before seed confirmation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from modules.dataset import MultimodalNerDataset
from scripts.confirm_crf_ensemble import find_checkpoint, read_policy
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
    parser.add_argument("--policy", required=True)
    parser.add_argument("--confirmation-report", required=True)
    parser.add_argument("--runs-root", default="outputs_v61")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2026])
    parser.add_argument("--data-path", default="dataset/text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_confirmation(path: Path, policy_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema_version") != "fixed-crf-ensemble-seed-confirmation-v1"
            or report.get("split") != "val" or report.get("weights_searched") is not False):
        raise ValueError("Expected a completed fixed-weight validation confirmation report")
    if report.get("source_policy_sha256") != sha256_file(policy_path):
        raise ValueError("Policy SHA256 differs from validation confirmation")
    if report.get("weights") != policy["weights"]:
        raise ValueError("Weights differ from validation confirmation")
    if report.get("input_sha256") != policy["validation_input_sha256"]:
        raise ValueError("Validation hash differs between policy and confirmation")
    source_seed = int(policy["checkpoints"]["control"]["config"]["seed"])
    seeds = report.get("confirmation_seeds", [])
    if (not seeds or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or source_seed in seeds
            or report.get("selection_seed") != source_seed):
        raise ValueError("Confirmation seed provenance is inconsistent")
    rows = report.get("per_seed", [])
    if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
        raise ValueError("Confirmation report must contain every declared seed exactly once")
    return report


def _single_test_metrics(data: dict[str, Any], label_map: dict[str, int]):
    replay = decode_blend(
        data["emissions"], data["emissions"], data["crf"], data["crf"], data["mask"], 0.0,
    )
    gold, predicted = word_sequences(data["mask"], data["labels"], replay, label_map)
    _, native = word_sequences(data["mask"], data["labels"], data["native_paths"], label_map)
    if predicted != native:
        raise RuntimeError("Test single-model CRF replay differs from native decoding")
    # Test F1 must never be compared to recorded validation F1 or used to pick weights.
    return gold, score(gold, predicted)


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_count": len(rows)}
    for name in ("baseline", "control", "ensemble"):
        values = [row["metrics"][name]["f1"] for row in rows]
        result[name] = {
            "mean_f1": statistics.mean(values),
            "sample_std_f1": statistics.stdev(values) if len(values) > 1 else None,
        }
    for name in ("baseline", "control"):
        gains = [row[f"gain_over_{name}"] for row in rows]
        result[f"mean_gain_over_{name}"] = statistics.mean(gains)
        result[f"positive_seeds_over_{name}"] = sum(value > 1e-12 for value in gains)
    return result


def main() -> None:
    cli = parse_args()
    if cli.batch_size < 1 or cli.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("Test seeds must be unique")
    if str(cli.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError("Test report exists; choose a new output path")
    policy_path, confirmation_path = Path(cli.policy), Path(cli.confirmation_report)
    policy = read_policy(policy_path)
    confirmation = read_confirmation(confirmation_path, policy_path, policy)
    selection_seed = int(confirmation["selection_seed"])
    if set(cli.seeds) != {selection_seed, *confirmation["confirmation_seeds"]}:
        raise ValueError("Final test must report the selection seed and ALL confirmation seeds")
    source_configs = {name: row["config"] for name, row in policy["checkpoints"].items()}
    dataset_name = source_configs["control"]["dataset"]
    val_name, test_name = f"MNER-{dataset_name}_val.txt", f"MNER-{dataset_name}_test.txt"
    input_path = Path(cli.data_path) / test_name
    val_hash = sha256_file(Path(cli.data_path) / val_name)
    if val_hash != policy["validation_input_sha256"]:
        raise ValueError("Validation data changed since policy selection")
    test_hash = sha256_file(input_path)
    confirmed_rows = {row["seed"]: row for row in confirmation["per_seed"]}

    # Verify every model before reading test examples into the dataset. New seed
    # checkpoints must be the exact artifacts already used in confirmation.
    runs = []
    for seed in cli.seeds:
        paths = {
            "control": find_checkpoint(Path(cli.runs_root), "duplicate_control", seed),
            "baseline": find_checkpoint(Path(cli.runs_root), "strict_baseline", seed),
        }
        configs, hashes = {}, {}
        expected = policy["checkpoints"] if seed == selection_seed else confirmed_rows[seed]["checkpoints"]
        for name, checkpoint in paths.items():
            hashes[name] = sha256_file(checkpoint)
            if hashes[name] != expected[name]["sha256"]:
                raise ValueError(f"Checkpoint changed since validation: seed={seed}, arm={name}")
            config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
            if type(config["seed"]) is not int or config["seed"] != seed:
                raise ValueError(f"Checkpoint has incorrect seed: {checkpoint}")
            validate_configs({"control": expected[name]["config"], "current": config})
            validate_configs({
                "control": source_configs[name], "test": {**config, "seed": selection_seed},
            })
            for file_name, digest in ((val_name, val_hash), (test_name, test_hash)):
                if sha256_file(Path(config["data_path"]) / file_name) != digest:
                    raise ValueError(f"Different {file_name}: {checkpoint}")
            configs[name] = config
        validate_configs(configs)
        runs.append((seed, paths, configs, hashes))

    args = SimpleNamespace(**{**source_configs["control"], "data_path": cli.data_path})
    dataset = MultimodalNerDataset(args, test_name)
    if not len(dataset):
        raise ValueError("Test dataset is empty")
    label_map = dataset.processor.get_label_mapping(include_subword_label=True)
    rows = []
    for seed, paths, configs, hashes in runs:
        collected, singles = {}, {}
        for name in ("control", "baseline"):
            data = _collect(paths[name], configs[name], dataset, cli, label_map, f"test/seed{seed}/{name}")
            gold, singles[name] = _single_test_metrics(data, label_map)
            if collected and any(
                not torch.equal(data[key], collected["control"][key]) for key in ("mask", "labels")
            ):
                raise RuntimeError("Test sample order, masks, or targets changed across models")
            collected[name] = data
            print(f"seed={seed} Test {name}: F1={singles[name]['f1']:.6f}", flush=True)
        control, baseline = collected["control"], collected["baseline"]
        paths_decoded = decode_blend(
            control["emissions"], baseline["emissions"], control["crf"], baseline["crf"],
            control["mask"], float(policy["weights"]["baseline"]),
        )
        gold, predicted = word_sequences(control["mask"], control["labels"], paths_decoded, label_map)
        fused = score(gold, predicted)
        row = {
            "seed": seed, "metrics": {**singles, "ensemble": fused},
            "gain_over_baseline": fused["f1"] - singles["baseline"]["f1"],
            "gain_over_control": fused["f1"] - singles["control"]["f1"],
            "checkpoints": {name: {"path": str(path), "sha256": hashes[name], "config": configs[name]}
                            for name, path in paths.items()},
        }
        rows.append(row)
        print(f"seed={seed} Test fixed ensemble: F1={fused['f1']:.6f}", flush=True)
    report = {
        "schema_version": "frozen-crf-ensemble-final-test-v1", "split": "test",
        "seeds": cli.seeds, "weights": policy["weights"], "weights_searched": False,
        "models_selected_using_test": False, "llm_api_requests": 0,
        "policy": str(policy_path), "policy_sha256": sha256_file(policy_path),
        "confirmation_report": str(confirmation_path), "confirmation_report_sha256": sha256_file(confirmation_path),
        "input": str(input_path), "input_sha256": test_hash, "validation_input_sha256": val_hash,
        "samples": len(dataset), "label_mapping": label_map,
        "per_seed": rows, "aggregate": aggregate_results(rows),
        "note": "All seeds reported with weights frozen before final evaluation. Sample standard "
                "deviations describe training-seed variation, not a significance test. This dataset's "
                "test split was inspected in earlier project experiments; this is not a pristine holdout "
                "or evidence of LLM benefit.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n=== Frozen CRF ensemble final test ===")
    print("seed       baseline      control     ensemble   gain_baseline   gain_control")
    for row in rows:
        print(f"{row['seed']:<8} {row['metrics']['baseline']['f1']:.6f}     "
              f"{row['metrics']['control']['f1']:.6f}     {row['metrics']['ensemble']['f1']:.6f}   "
              f"{row['gain_over_baseline']:+.6f}       {row['gain_over_control']:+.6f}")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
