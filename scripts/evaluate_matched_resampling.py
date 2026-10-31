"""Compare accepted-source and feature-matched resampling with existing checkpoints.

This is a post-hoc ablation on a previously inspected test split. The standalone
comparison is primary; identical, already frozen CRF fusion is secondary.
"""

from __future__ import annotations

import argparse
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
from scripts.confirm_crf_ensemble import _single_metrics, find_checkpoint, read_policy
from scripts.evaluate_crf_ensemble import (
    _collect, decode_blend, score, sha256_file, validate_configs, word_sequences,
)
from scripts.evaluate_frozen_crf_test import _single_test_metrics, read_confirmation


SEEDS = (42, 1234, 2026)
METHODS = ("baseline", "llm_selected", "matched", "llm_selected_ensemble", "matched_ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", required=True, help="Existing frozen CRF final-test report.")
    parser.add_argument("--matched-manifest", required=True)
    parser.add_argument("--matched-runs-root", default="outputs_v62")
    parser.add_argument("--data-path", default="dataset/text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str, description: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{description} SHA256 changed: {path}")


def audit_data(manifest_path: Path, source: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != "feature-matched-resampling-v1"
            or manifest.get("arm") != "feature_matched_no_llm"
            or manifest.get("ready_for_training") is not True
            or manifest.get("candidate_scoring_uses_llm_outputs") is not False
            or manifest.get("selection_uses_validation_or_test_labels") is not False):
        raise ValueError("Expected a passed, original-feature-only matched manifest")
    balance, overlap = manifest["balance_check"], manifest["overlap_check"]
    difference = balance["max_abs_difference_in_pool_sd_units"]
    if (balance.get("passed") is not True or not math.isfinite(difference)
            or not 0 <= difference <= 0.10 or overlap.get("passed") is not True):
        raise ValueError("Matched data balance/overlap audit failed")
    selected, reference_ids = manifest["source_sample_ids"], manifest["reference_source_sample_ids"]
    count = manifest["added_examples"]
    if (type(count) is not int or count < 1 or len(selected) != count or len(reference_ids) != count
            or any(type(i) is not int for i in selected + reference_ids)
            or len(set(selected)) != count or len(set(reference_ids)) != count):
        raise ValueError("Matched and reference source IDs must be unique and equally numerous")
    fraction = len(set(selected) & set(reference_ids)) / count
    if (fraction > 0.80 or not math.isclose(fraction, overlap["fraction"], abs_tol=1e-12)
            or len(set(selected) & set(reference_ids)) != overlap["count"]):
        raise ValueError("Matched data overlap audit failed")
    metadata_path, reference_path = Path(manifest["candidate_metadata"]), Path(manifest["reference_manifest"])
    require_hash(metadata_path, manifest["candidate_metadata_sha256"], "Candidate metadata")
    require_hash(reference_path, manifest["reference_manifest_sha256"], "Reference manifest")
    metadata, reference = read_json(metadata_path), read_json(reference_path)
    candidate_ids = [row["sample_id"] for row in metadata["selected"]]
    if (len(set(candidate_ids)) != manifest["candidate_count"]
            or not set(selected + reference_ids).issubset(candidate_ids)
            or reference.get("arm") != "duplicate_control"
            or reference.get("source_sample_ids") != reference_ids
            or reference.get("accepted_augmentations") != count
            or reference.get("source_train_sha256") != manifest["source_train_sha256"]):
        raise ValueError("Candidate/reference sampling provenance differs")
    train_name = "MNER-UNI_train.txt"
    require_hash(source / train_name, manifest["source_train_sha256"], "Original training data")
    require_hash(reference_path.parent / train_name, manifest["reference_train_sha256"], "Reference training data")
    expected_names = {f"MNER-UNI_{split}.txt" for split in ("train", "val", "test")}
    if set(manifest["output_files_sha256"]) != expected_names:
        raise ValueError("Matched manifest must bind all three data files")
    for name, digest in manifest["output_files_sha256"].items():
        require_hash(manifest_path.parent / name, digest, "Matched data")
    originals = (source / train_name).read_text(encoding="utf-8").splitlines()
    if (len(originals) != manifest["original_examples"]
            or manifest["total_examples"] != len(originals) + count
            or reference.get("original_examples") != len(originals)
            or any(not 0 <= i < len(originals) for i in selected + reference_ids)):
        raise ValueError("Original/matched training counts differ")
    for directory, ids in ((reference_path.parent, reference_ids), (manifest_path.parent, selected)):
        actual = (directory / train_name).read_text(encoding="utf-8").splitlines()
        if actual != originals + [originals[i] for i in ids]:
            raise ValueError(f"Resampling file differs from its original-source IDs: {directory}")
        for split in ("val", "test"):
            name = f"MNER-UNI_{split}.txt"
            require_hash(directory / name, sha256_file(source / name), "Common evaluation data")
    return manifest


def prepare_runs(cli: argparse.Namespace):
    reference_path, manifest_path = Path(cli.reference_report), Path(cli.matched_manifest)
    reference = read_json(reference_path)
    if (reference.get("schema_version") != "frozen-crf-ensemble-final-test-v1"
            or reference.get("split") != "test" or reference.get("weights_searched") is not False
            or len(reference.get("seeds", [])) != len(SEEDS) or set(reference["seeds"]) != set(SEEDS)):
        raise ValueError("Reference report must include all three fixed seeds and frozen test policy")
    policy_path, confirmation_path = Path(reference["policy"]), Path(reference["confirmation_report"])
    require_hash(policy_path, reference["policy_sha256"], "Frozen policy")
    require_hash(confirmation_path, reference["confirmation_report_sha256"], "Confirmation report")
    policy = read_policy(policy_path)
    confirmation = read_confirmation(confirmation_path, policy_path, policy)
    if (reference["weights"] != policy["weights"]
            or reference["validation_input_sha256"] != policy["validation_input_sha256"]
            or {confirmation["selection_seed"], *confirmation["confirmation_seeds"]} != set(SEEDS)):
        raise ValueError("Frozen weights/seeds differ between reports")
    source = Path(cli.data_path)
    require_hash(source / "MNER-UNI_test.txt", reference["input_sha256"], "Previous test input")
    require_hash(source / "MNER-UNI_val.txt", policy["validation_input_sha256"], "Validation input")
    manifest = audit_data(manifest_path, source)
    previous = reference.get("per_seed", [])
    if len(previous) != len(SEEDS) or {row["seed"] for row in previous} != set(SEEDS):
        raise ValueError("Reference report must contain every seed exactly once")
    previous = {row["seed"]: row for row in previous}
    confirmed = {row["seed"]: row for row in confirmation["per_seed"]}
    runs, protocol = [], None
    all_paths = set()
    for seed in SEEDS:
        old = previous[seed]
        frozen = policy["checkpoints"] if seed == confirmation["selection_seed"] else confirmed[seed]["checkpoints"]
        paths = {
            "baseline": Path(old["checkpoints"]["baseline"]["path"]),
            "llm_selected": Path(old["checkpoints"]["control"]["path"]),
            "matched": find_checkpoint(Path(cli.matched_runs_root), "matched_no_llm", seed),
        }
        configs, hashes = {}, {}
        for name, checkpoint in paths.items():
            if checkpoint.resolve() in all_paths:
                raise ValueError("Different arms/seeds must use different checkpoints")
            all_paths.add(checkpoint.resolve())
            hashes[name] = sha256_file(checkpoint)
            config = read_json(checkpoint.parent / "config.json")
            if config.get("dataset") != "UNI" or type(config.get("seed")) is not int or config["seed"] != seed:
                raise ValueError(f"Incorrect dataset/seed in {checkpoint}")
            if name != "matched":
                old_name = "control" if name == "llm_selected" else "baseline"
                if hashes[name] != old["checkpoints"][old_name]["sha256"] or hashes[name] != frozen[old_name]["sha256"]:
                    raise ValueError(f"Checkpoint changed since frozen evaluation: {checkpoint}")
                validate_configs({"control": frozen[old_name]["config"], "current": config})
            expected_dir = source if name == "baseline" else (
                Path(manifest["reference_manifest"]).parent if name == "llm_selected" else manifest_path.parent
            )
            if Path(config["data_path"]).resolve() != expected_dir.resolve():
                raise ValueError(f"Checkpoint trained on an unexpected data directory: {checkpoint}")
            configs[name] = config
        validate_configs({"control": configs["llm_selected"], **configs})
        normalised = {**configs["baseline"], "seed": SEEDS[0]}
        if protocol is not None:
            validate_configs({"control": protocol, "current": normalised})
        protocol = normalised
        runs.append((seed, paths, configs, hashes))
    return reference, manifest, previous, runs


def require_replay(actual: dict[str, Any], expected: dict[str, Any], description: str) -> None:
    for name in ("f1", "precision", "recall"):
        if not math.isclose(actual[name], expected[name], abs_tol=1e-10, rel_tol=0):
            raise RuntimeError(f"Previous {description} metrics do not reproduce: {name}")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_count": len(rows)}
    for name in METHODS:
        values = [row["metrics"][name]["f1"] for row in rows]
        result[name] = {"mean_f1": statistics.mean(values), "sample_std_f1": statistics.stdev(values)}
    for name in ("primary_llm_selected_minus_matched", "secondary_ensemble_difference"):
        values = [row[name] for row in rows]
        result[name] = {"mean_f1_difference": statistics.mean(values),
                        "sample_std_difference": statistics.stdev(values),
                        "positive_seeds": sum(value > 1e-12 for value in values)}
    return result


def main() -> None:
    cli = parse_args()
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError("Report exists; preserve it or choose a new output path")
    if cli.batch_size < 1 or cli.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if str(cli.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    reference, manifest, previous, runs = prepare_runs(cli)
    common = SimpleNamespace(**{**runs[0][2]["baseline"], "data_path": cli.data_path})
    validation = MultimodalNerDataset(common, "MNER-UNI_val.txt")
    label_map = validation.processor.get_label_mapping(include_subword_label=True)
    validation_replays = []
    for seed, paths, configs, _ in runs:
        data = _collect(paths["matched"], configs["matched"], validation, cli, label_map, f"val/seed{seed}/matched")
        _, replay = _single_metrics(data, label_map, paths["matched"])
        validation_replays.append({"seed": seed, "metrics": replay})
        print(f"seed={seed} Matched validation F1={replay['f1']:.6f} (recorded dev replay verified)", flush=True)
        del data
    del validation
    dataset = MultimodalNerDataset(common, "MNER-UNI_test.txt")
    if not len(dataset):
        raise ValueError("Evaluation data is empty")
    if dataset.processor.get_label_mapping(include_subword_label=True) != label_map:
        raise RuntimeError("Validation/test label mappings differ")
    rows = []
    for seed, paths, configs, hashes in runs:
        data, metrics = {}, {}
        for name in ("baseline", "llm_selected", "matched"):
            current = _collect(paths[name], configs[name], dataset, cli, label_map, f"seed{seed}/{name}")
            if data and any(not torch.equal(current[key], data["baseline"][key]) for key in ("mask", "labels")):
                raise RuntimeError("Sample order/masks/targets differ between arms")
            gold, metrics[name] = _single_test_metrics(current, label_map)
            if name != "matched":
                old_name = "control" if name == "llm_selected" else "baseline"
                require_replay(metrics[name], previous[seed]["metrics"][old_name], f"seed{seed}/{name}")
            data[name] = current
            print(f"seed={seed} Test {name}: F1={metrics[name]['f1']:.6f}", flush=True)
        for name in ("llm_selected", "matched"):
            left, right = data[name], data["baseline"]
            paths_decoded = decode_blend(left["emissions"], right["emissions"], left["crf"], right["crf"],
                                        left["mask"], float(reference["weights"]["baseline"]))
            gold, predicted = word_sequences(left["mask"], left["labels"], paths_decoded, label_map)
            metrics[name + "_ensemble"] = score(gold, predicted)
        require_replay(metrics["llm_selected_ensemble"], previous[seed]["metrics"]["ensemble"], f"seed{seed}/ensemble")
        rows.append({
            "seed": seed, "metrics": metrics,
            "primary_llm_selected_minus_matched": metrics["llm_selected"]["f1"] - metrics["matched"]["f1"],
            "secondary_ensemble_difference": metrics["llm_selected_ensemble"]["f1"] - metrics["matched_ensemble"]["f1"],
            "checkpoints": {name: {"path": str(path), "sha256": hashes[name], "config": configs[name]}
                            for name, path in paths.items()},
        })
    report = {
        "schema_version": "matched-resampling-test-ablation-v1", "split": "test", "seeds": list(SEEDS),
        "post_hoc_analysis": True, "independent_holdout": False, "llm_api_requests": 0,
        "primary_comparison": "llm_selected minus matched standalone models",
        "secondary_comparison": "identically weighted resampled+baseline CRF ensembles",
        "weights": reference["weights"], "weights_searched": False,
        "reference_report": cli.reference_report, "reference_report_sha256": sha256_file(Path(cli.reference_report)),
        "matched_manifest": cli.matched_manifest, "matched_manifest_sha256": sha256_file(Path(cli.matched_manifest)),
        "input_sha256": reference["input_sha256"], "validation_input_sha256": reference["validation_input_sha256"],
        "data_audit": {name: manifest[name] for name in ("added_examples", "balance_check", "overlap_check", "sampling_seed")},
        "samples": len(dataset), "label_mapping": label_map, "matched_validation_replays": validation_replays,
        "per_seed": rows, "aggregate": aggregate(rows),
        "note": "The selected arm depends jointly on LLM generation and validators, not a pure semantic "
                "quality score. This compares one fixed resampling draw over three training seeds. "
                "Test was inspected earlier; this is exploratory corroboration, not independent confirmation "
                "or a significance test. All seeds and both comparisons are reported without retuning.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n=== Post-hoc matched-resampling test comparison ===")
    print("seed      baseline   LLM-selected   matched   primary_delta   selected_ens   matched_ens   secondary_delta")
    for row in rows:
        m = row["metrics"]
        print(f"{row['seed']:<8} {m['baseline']['f1']:.6f}   {m['llm_selected']['f1']:.6f}   "
              f"{m['matched']['f1']:.6f}   {row['primary_llm_selected_minus_matched']:+.6f}       "
              f"{m['llm_selected_ensemble']['f1']:.6f}       {m['matched_ensemble']['f1']:.6f}   "
              f"{row['secondary_ensemble_difference']:+.6f}")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
