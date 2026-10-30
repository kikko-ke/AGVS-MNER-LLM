"""Evaluate a fixed seven-policy CRF score ensemble on validation data only."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import DataLoader
from torchcrf import CRF
from tqdm.auto import tqdm

from models.agvs_mner import AgvsMnerModel
from modules.dataset import MultimodalNerDataset
from modules.trainer import NerTrainer, load_compatible_model_checkpoint
from run import set_seed


ALPHAS = (0.1, 0.2, 0.3)
CRF_KEYS = ("start_transitions", "end_transitions", "transitions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--data-path", help="Common validation directory; defaults to control config.")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configs(configs: dict[str, dict[str, Any]]) -> None:
    """Allow different training data/output locations, not different protocols."""
    ignored = {"data_path", "output_dir", "device", "num_workers"}
    reference = {key: value for key, value in configs["control"].items() if key not in ignored}
    for name, config in configs.items():
        comparison = {key: value for key, value in config.items() if key not in ignored}
        mismatches = [
            key for key in sorted(reference.keys() | comparison.keys())
            if reference.get(key) != comparison.get(key)
        ]
        if mismatches:
            raise ValueError(f"{name} training config differs from control: {mismatches}")


def decode_blend(
    control_emissions: torch.Tensor,
    other_emissions: torch.Tensor,
    control_crf: dict[str, torch.Tensor],
    other_crf: dict[str, torch.Tensor],
    mask: torch.Tensor,
    alpha: float,
    *,
    batch_size: int = 64,
) -> list[list[int]]:
    """Maximise the weighted sum of complete CRF sequence scores.

    Emissions AND start/end/pair transitions are blended. No gold labels,
    per-example routing, or BIO constraint changes are used in decoding.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if control_emissions.shape != other_emissions.shape or control_emissions.ndim != 3:
        raise ValueError("Both emission tensors must have the same [N, L, K] shape")
    if mask.shape != control_emissions.shape[:2]:
        raise ValueError("mask must match emission sample/length dimensions")

    num_tags = control_emissions.size(-1)
    decoder = CRF(num_tags, batch_first=True).to(control_emissions.device)
    expected_shapes = ((num_tags,), (num_tags,), (num_tags, num_tags))
    with torch.inference_mode():
        for key, shape in zip(CRF_KEYS, expected_shapes):
            left, right = control_crf[key], other_crf[key]
            if tuple(left.shape) != shape or tuple(right.shape) != shape:
                raise ValueError(f"CRF {key} has an incompatible shape")
            # Exact endpoints avoid changing even the last bit of single-model scores.
            mixed = left if alpha == 0 else right if alpha == 1 else (
                (1 - alpha) * left + alpha * right
            )
            getattr(decoder, key).copy_(mixed)
        predictions = []
        for start in range(0, len(mask), batch_size):
            left = control_emissions[start:start + batch_size]
            right = other_emissions[start:start + batch_size]
            mixed = left if alpha == 0 else right if alpha == 1 else (
                (1 - alpha) * left + alpha * right
            )
            predictions.extend(decoder.decode(mixed, mask=mask[start:start + batch_size].bool()))
    return predictions


def word_sequences(
    mask: torch.Tensor,
    labels: torch.Tensor,
    paths: list[list[int]],
    label_map: dict[str, int],
) -> tuple[list[list[str]], list[list[str]]]:
    # Reuse the training evaluator's exact X/WordPiece filtering convention.
    evaluator = object.__new__(NerTrainer)
    evaluator.label_map = label_map
    evaluator.index_to_label = {index: label for label, index in label_map.items()}
    gold: list[list[str]] = []
    predicted: list[list[str]] = []
    evaluator._append_predictions(mask, labels, paths, gold, predicted)
    return gold, predicted


def score(gold: list[list[str]], predicted: list[list[str]]) -> dict[str, Any]:
    return {
        "f1": float(f1_score(gold, predicted, zero_division=0)),
        "precision": float(precision_score(gold, predicted, zero_division=0)),
        "recall": float(recall_score(gold, predicted, zero_division=0)),
        "report": classification_report(gold, predicted, digits=6, zero_division=0),
    }


def complementarity(
    gold: list[list[str]], control: list[list[str]], llm: list[list[str]],
) -> dict[str, int]:
    counts = {
        "samples": len(gold), "different_entity_sets": 0,
        "control_wrong_llm_exact": 0, "control_exact_llm_wrong": 0,
        "both_wrong_different": 0,
    }
    for target, first, second in zip(gold, control, llm):
        target_set, first_set, second_set = (
            set(get_entities(sequence)) for sequence in (target, first, second)
        )
        if first_set != second_set:
            counts["different_entity_sets"] += 1
            if second_set == target_set:
                counts["control_wrong_llm_exact"] += 1
            elif first_set == target_set:
                counts["control_exact_llm_wrong"] += 1
            else:
                counts["both_wrong_different"] += 1
    return counts


def compare_policies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    control_f1 = rows[0]["metrics"]["f1"]
    best_without = max(
        (row for row in rows if row["member"] != "llm"),
        key=lambda row: row["metrics"]["f1"],
    )
    best_llm = max(
        (row for row in rows if row["member"] == "llm"),
        key=lambda row: row["metrics"]["f1"],
    )
    llm_f1 = best_llm["metrics"]["f1"]
    return {
        "best_without_llm": best_without["name"],
        "best_without_llm_f1": best_without["metrics"]["f1"],
        "best_llm": best_llm["name"],
        "best_llm_f1": llm_f1,
        "llm_gain_over_control": llm_f1 - control_f1,
        "llm_gain_over_best_without_llm": llm_f1 - best_without["metrics"]["f1"],
        "llm_increment_observed_on_validation": (
            llm_f1 > best_without["metrics"]["f1"] + 1e-12
        ),
        "independent_confirmation_required": True,
        "note": "Exploratory validation selection; not an independent significance test or test-set result.",
    }


def _collect(
    checkpoint: Path, config: dict[str, Any], dataset: MultimodalNerDataset,
    cli: argparse.Namespace, label_map: dict[str, int], name: str,
) -> dict[str, Any]:
    args = SimpleNamespace(**{**config, "device": cli.device})
    set_seed(int(args.seed))
    model = AgvsMnerModel(list(label_map), args)
    missing = load_compatible_model_checkpoint(model, checkpoint, map_location="cpu")
    if missing:
        print(f"[{name}] Optional head keys initialized: {missing}", flush=True)
    model.to(cli.device).eval()
    loader = DataLoader(
        dataset, batch_size=cli.batch_size, num_workers=cli.num_workers, shuffle=False,
    )
    emissions, masks, targets, native_paths = [], [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Forward {name}", unit="batch"):
            ids, types, mask, pixels, labels, image_mask = batch
            paths, _, features = model(
                input_ids=ids.to(cli.device), token_type_ids=types.to(cli.device),
                attention_mask=mask.to(cli.device), image_pixel_values=pixels.to(cli.device),
                image_mask=image_mask.to(cli.device), labels=None, return_features=True,
            )
            emissions.append(features["token_emissions"].detach().cpu())
            masks.append(mask.cpu())
            targets.append(labels.cpu())
            native_paths.extend(paths)
        crf_state = {key: getattr(model.crf, key).detach().cpu().clone() for key in CRF_KEYS}
    result = {
        "emissions": torch.cat(emissions), "mask": torch.cat(masks),
        "labels": torch.cat(targets), "native_paths": native_paths, "crf": crf_state,
    }
    if not torch.isfinite(result["emissions"]).all():
        raise RuntimeError(f"{name} produced non-finite emissions")
    del model, loader
    gc.collect()
    if str(cli.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    cli = parse_args()
    if cli.batch_size < 1 or cli.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if str(cli.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    output = Path(cli.output)
    policy_path = output.with_name(output.stem + ".policy.json")
    if output.exists() or policy_path.exists():
        raise FileExistsError("Result already exists; choose a new --output path")

    checkpoints = {name: Path(getattr(cli, name)) for name in ("baseline", "control", "llm")}
    configs = {}
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        configs[name] = json.loads(
            (checkpoint.parent / "config.json").read_text(encoding="utf-8")
        )
    validate_configs(configs)
    config = dict(configs["control"])
    config["data_path"] = cli.data_path or config["data_path"]
    file_name = f"MNER-{config['dataset']}_val.txt"
    input_path = Path(config["data_path"]) / file_name
    input_hash = sha256_file(input_path)
    for name, member_config in configs.items():
        member_val = Path(member_config["data_path"]) / file_name
        if sha256_file(member_val) != input_hash:
            raise ValueError(f"{name} validation file differs from common validation data")

    dataset = MultimodalNerDataset(SimpleNamespace(**config), file_name)
    if not len(dataset):
        raise ValueError("Validation dataset is empty")
    label_map = dataset.processor.get_label_mapping(include_subword_label=True)
    collected, singles, native_predictions = {}, {}, {}
    for name in ("control", "baseline", "llm"):
        data = _collect(checkpoints[name], configs[name], dataset, cli, label_map, name)
        replay = decode_blend(
            data["emissions"], data["emissions"], data["crf"], data["crf"],
            data["mask"], 0.0,
        )
        gold, predicted = word_sequences(data["mask"], data["labels"], replay, label_map)
        _, native = word_sequences(
            data["mask"], data["labels"], data["native_paths"], label_map,
        )
        if predicted != native:
            raise RuntimeError(f"{name} single-model CRF replay differs from native predictions")
        if collected:
            reference = collected["control"]
            if any(not torch.equal(data[key], reference[key]) for key in ("mask", "labels")):
                raise RuntimeError("Sample order, masks, or targets changed across models")
        singles[name] = score(gold, predicted)
        metrics_path = checkpoints[name].parent / "metrics.csv"
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8", newline="") as source:
                recorded = [float(row["dev_f1"]) for row in csv.DictReader(source) if row.get("dev_f1")]
            if recorded:
                singles[name]["recorded_best_dev_f1"] = max(recorded)
                if abs(singles[name]["f1"] - max(recorded)) > 0.0005:
                    raise RuntimeError(
                        f"{name} F1 {singles[name]['f1']:.6f} does not reproduce recorded "
                        f"best {max(recorded):.6f}; check checkpoint/data/environment"
                    )
        print(f"Single {name}: dev F1={singles[name]['f1']:.6f} (replay verified)", flush=True)
        collected[name], native_predictions[name] = data, predicted

    control = collected["control"]
    rows = [{"name": "control", "member": "control", "alpha": 0.0, "metrics": singles["control"]}]
    for name in ("baseline", "llm"):
        other = collected[name]
        for alpha in ALPHAS:
            paths = decode_blend(
                control["emissions"], other["emissions"], control["crf"], other["crf"],
                control["mask"], alpha,
            )
            gold, predicted = word_sequences(control["mask"], control["labels"], paths, label_map)
            metrics = score(gold, predicted)
            rows.append({"name": f"control_{1-alpha:.1f}_{name}_{alpha:.1f}",
                         "member": name, "alpha": alpha, "metrics": metrics})
    comparison = compare_policies(rows)
    summary = {
        "schema_version": "crf-score-ensemble-validation-v1", "split": "val",
        "exploratory": True, "samples": len(dataset), "llm_api_requests": 0,
        "input": str(input_path), "input_sha256": input_hash,
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256_file(path), "config": configs[name]}
            for name, path in checkpoints.items()
        },
        "label_mapping": label_map, "single_models": singles, "policies": rows,
        "complementarity": complementarity(gold, native_predictions["control"], native_predictions["llm"]),
        "comparison": comparison,
    }
    # A frozen candidate records weights and artifact hashes, never per-sample gold decisions.
    best_name = comparison["best_llm"] if comparison["llm_increment_observed_on_validation"] else comparison["best_without_llm"]
    chosen = next(row for row in rows if row["name"] == best_name)
    policy = {
        "schema_version": "crf-score-ensemble-policy-v1", "selection_split": "val",
        "name": chosen["name"], "weights": {"control": 1.0 - chosen["alpha"]},
        "blend": "raw emissions plus start/end/pair transitions; unchanged CRF decoding",
        "validation_input_sha256": input_hash, "independent_confirmation_required": True,
        "checkpoints": {"control": summary["checkpoints"]["control"]},
    }
    if chosen["member"] != "control":
        policy["weights"][chosen["member"]] = chosen["alpha"]
        policy["checkpoints"][chosen["member"]] = summary["checkpoints"][chosen["member"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((output, summary), (policy_path, policy)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n=== Fixed seven-policy validation comparison ===")
    for row in rows:
        delta = row["metrics"]["f1"] - singles["control"]["f1"]
        print(f"{row['name']:34s} F1={row['metrics']['f1']:.6f}  delta={delta:+.6f}")
    print(json.dumps(comparison, indent=2))
    print(f"Results: {output}\nCandidate policy: {policy_path}")


if __name__ == "__main__":
    main()
