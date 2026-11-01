"""Replay validation, freeze all artifacts, then test four arms on MI and UNI.

All three fixed seeds and both datasets are mandatory. This script performs no
training, model/seed selection, fusion-weight search, or LLM requests.
"""

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
from scripts.confirm_crf_ensemble import _single_metrics, find_checkpoint
from scripts.evaluate_crf_ensemble import _collect, sha256_file, validate_configs
from scripts.evaluate_frozen_crf_test import _single_test_metrics
from scripts.evaluate_matched_resampling import audit_data, require_hash


DATASETS = ("MI", "UNI")
SEEDS = (42, 1234, 2026)
METHODS = ("baseline", "legacy_selected", "legacy_matched", "token_selected")
COMPARISONS = {
    "selection_minus_baseline": ("legacy_selected", "baseline"),
    "selection_minus_matched": ("legacy_selected", "legacy_matched"),
    "component_on_selected": ("token_selected", "legacy_selected"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="dataset/text")
    parser.add_argument("--mi-runs-root", default="outputs_mi_v63")
    parser.add_argument("--uni-runs-root", default="outputs_v61")
    parser.add_argument("--uni-matched-root", default="outputs_v62")
    parser.add_argument("--component-root", default="outputs_component_v1")
    parser.add_argument("--mi-matched-manifest", default="dataset/v63_mi_feature_matched_seed20261031/matched_resampling_manifest.json")
    parser.add_argument("--uni-matched-manifest", default="dataset/v62_feature_matched_seed20261031/matched_resampling_manifest.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_plan(cli: argparse.Namespace) -> dict[str, Any]:
    source = Path(cli.data_path)
    datasets, seen = {}, set()
    for dataset in DATASETS:
        manifest_path = Path(getattr(cli, f"{dataset.lower()}_matched_manifest"))
        manifest = audit_data(manifest_path, source, dataset=dataset)
        selected_dir = Path(manifest["reference_manifest"]).parent
        matched_dir = manifest_path.parent
        inputs = {split: {"path": str(source / f"MNER-{dataset}_{split}.txt"),
                          "sha256": sha256_file(source / f"MNER-{dataset}_{split}.txt")}
                  for split in ("train", "val", "test")}
        roots = {
            "baseline": (Path(getattr(cli, f"{dataset.lower()}_runs_root")), "strict_baseline"),
            "legacy_selected": (Path(getattr(cli, f"{dataset.lower()}_runs_root")),
                                "llm_selected" if dataset == "MI" else "duplicate_control"),
            "legacy_matched": (Path(cli.mi_runs_root if dataset == "MI" else cli.uni_matched_root), "matched_no_llm"),
            "token_selected": (Path(cli.component_root), f"{dataset.lower()}_token_evidence_selected"),
        }
        expected_dirs = {"baseline": source, "legacy_selected": selected_dir,
                         "legacy_matched": matched_dir, "token_selected": selected_dir}
        models, reference_protocol = [], None
        for seed in SEEDS:
            for method in METHODS:
                root, arm = roots[method]
                checkpoint = find_checkpoint(root, arm, seed)
                if checkpoint.resolve() in seen:
                    raise ValueError("Each dataset/seed/method must use a separate checkpoint")
                seen.add(checkpoint.resolve())
                config_path, metrics_path = checkpoint.parent / "config.json", checkpoint.parent / "metrics.csv"
                config = read_json(config_path)
                mode = "token_evidence" if method == "token_selected" else "legacy"
                if (config.get("dataset") != dataset or type(config.get("seed")) is not int
                        or config["seed"] != seed or config.get("visual_fusion", "legacy") != mode
                        or config.get("skip_test") is not True):
                    raise ValueError(f"Wrong dataset/seed/component or non-validation-only run: {checkpoint}")
                if Path(config["data_path"]).resolve() != expected_dirs[method].resolve():
                    raise ValueError(f"Wrong training data directory: {checkpoint}")
                # Exactly one architectural difference is allowed; all other
                # training settings must match, including across seeds.
                normalised = {**config, "seed": SEEDS[0], "visual_fusion": "legacy"}
                if reference_protocol is not None:
                    validate_configs({"control": reference_protocol, "current": normalised})
                reference_protocol = normalised
                training_data = Path(config["data_path"])
                expected_train = (inputs["train"]["sha256"] if method == "baseline" else
                                  manifest["output_files_sha256"][f"MNER-{dataset}_train.txt"] if method == "legacy_matched" else
                                  manifest["reference_train_sha256"])
                require_hash(training_data / f"MNER-{dataset}_train.txt", expected_train, "Training input")
                for split in ("val", "test"):
                    require_hash(training_data / f"MNER-{dataset}_{split}.txt", inputs[split]["sha256"], "Common evaluation input")
                models.append({
                    "seed": seed, "method": method, "checkpoint": str(checkpoint), "config": config,
                    "artifacts": {str(path): sha256_file(path) for path in (checkpoint, config_path, metrics_path)},
                })
                print(f"[preflight] {dataset} seed={seed} {method}: artifacts verified", flush=True)
        datasets[dataset] = {
            "inputs": inputs, "models": models,
            "matched_manifest": str(manifest_path), "matched_manifest_sha256": sha256_file(manifest_path),
            "audit": {key: manifest[key] for key in ("candidate_count", "added_examples", "balance_check", "overlap_check")},
        }
    repository = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__).resolve(), repository / "scripts" / "confirm_crf_ensemble.py",
                  repository / "scripts" / "evaluate_crf_ensemble.py",
                  repository / "scripts" / "evaluate_frozen_crf_test.py",
                  repository / "scripts" / "evaluate_matched_resampling.py"]
    code_files += list((repository / "models").glob("*.py")) + list((repository / "modules").glob("*.py"))
    return {
        "schema_version": "frozen-component-suite-v1", "datasets": datasets,
        "dataset_order": list(DATASETS), "seeds": list(SEEDS), "methods": list(METHODS),
        "comparisons": COMPARISONS, "checkpoints_selected_on": "val", "models_chosen_using_test": False,
        "evaluation_code": {str(path): sha256_file(path) for path in code_files},
        "llm_api_requests": 0, "training_runs": 0, "parameter_search": False,
        "note": "The original component on selected originals is the common candidate on both datasets. "
                "The new component remains an ablation, including its MI validation decline. "
                "UNI test was inspected earlier; this is post-hoc reporting, not independent confirmation "
                "or a significance test. Selection depends jointly on LLM and validators. "
                "No feature-matched arm has been trained with the new component; that causal comparison is unavailable.",
    }


def verify_plan(plan: dict[str, Any]) -> None:
    for name, digest in plan["evaluation_code"].items():
        require_hash(Path(name), digest, "Evaluation code")
    for dataset in DATASETS:
        entry = plan["datasets"][dataset]
        require_hash(Path(entry["matched_manifest"]), entry["matched_manifest_sha256"], "Matched manifest")
        for value in entry["inputs"].values():
            require_hash(Path(value["path"]), value["sha256"], "Frozen input")
        for model in entry["models"]:
            for name, digest in model["artifacts"].items():
                require_hash(Path(name), digest, "Frozen checkpoint/config/metrics")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_count": len(SEEDS), "methods": {}, "comparisons": {}}
    for method in METHODS:
        values = [row["metrics"][method]["f1"] for row in rows]
        result["methods"][method] = {"mean_f1": statistics.mean(values), "sample_std_f1": statistics.stdev(values)}
    for name, (left, right) in COMPARISONS.items():
        values = [row["metrics"][left]["f1"] - row["metrics"][right]["f1"] for row in rows]
        result["comparisons"][name] = {"mean_f1_difference": statistics.mean(values),
                                       "sample_std_difference": statistics.stdev(values),
                                       "positive_seeds": sum(value > 1e-12 for value in values)}
    return result


def evaluate_split(plan: dict[str, Any], cli: argparse.Namespace, split: str) -> dict[str, Any]:
    results = {}
    for dataset in DATASETS:
        entry = plan["datasets"][dataset]
        config = entry["models"][0]["config"]
        data = MultimodalNerDataset(SimpleNamespace(**{**config, "data_path": cli.data_path}), f"MNER-{dataset}_{split}.txt")
        if not len(data):
            raise ValueError(f"Empty {dataset}/{split}")
        label_map = data.processor.get_label_mapping(include_subword_label=True)
        rows = {seed: {"seed": seed, "metrics": {}} for seed in SEEDS}
        common_gold = None
        for model in entry["models"]:
            checkpoint = Path(model["checkpoint"])
            collected = _collect(checkpoint, model["config"], data, cli, label_map,
                                 f"{split}/{dataset}/seed{model['seed']}/{model['method']}")
            if split == "val":
                gold, metrics = _single_metrics(collected, label_map, checkpoint)
            else:
                gold, metrics = _single_test_metrics(collected, label_map)
            if common_gold is not None and gold != common_gold:
                raise RuntimeError("Evaluation example order/targets changed across arms")
            common_gold = gold
            rows[model["seed"]]["metrics"][model["method"]] = metrics
            del collected
            print(f"{split} {dataset} seed={model['seed']} {model['method']}: F1={metrics['f1']:.6f}", flush=True)
        values = [rows[seed] for seed in SEEDS]
        results[dataset] = {"samples": len(data), "label_mapping": label_map,
                            "per_seed": values, "aggregate": aggregate(values)}
    return {"schema_version": "component-suite-metrics-v1", "split": split, "datasets": results,
            "seeds": list(SEEDS), "methods": list(METHODS), "note": plan["note"]}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    output = Path(cli.output_dir)
    if output.exists():
        raise FileExistsError("Output directory exists; preserve reports and choose a new path")
    if cli.batch_size < 1 or cli.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use a GPU instance or --device cpu")
    plan = prepare_plan(cli)
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "frozen_manifest.json"
    write_json(frozen, plan)
    binding = sha256_file(frozen)
    validation = evaluate_split(plan, cli, "val")
    validation["frozen_manifest_sha256"] = binding
    write_json(output / "validation_metrics.json", validation)
    print("All 24 validation replays passed. Evaluating the same frozen models on test.", flush=True)
    require_hash(frozen, binding, "Frozen manifest")
    verify_plan(plan)
    test = evaluate_split(plan, cli, "test")
    test["frozen_manifest_sha256"] = binding
    write_json(output / "test_metrics.json", test)
    print("\n=== Fixed two-dataset test comparison: mean F1 +/- sample SD ===")
    for dataset in DATASETS:
        print(f"\n{dataset}:")
        for method, values in test["datasets"][dataset]["aggregate"]["methods"].items():
            print(f"{method:<20} {values['mean_f1']:.6f} +/- {values['sample_std_f1']:.6f}")
        print(json.dumps(test["datasets"][dataset]["aggregate"]["comparisons"], indent=2))
    print(f"Results: {output / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
