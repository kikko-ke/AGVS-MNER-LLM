"""Compare a checkpoint on full-image and text-only validation views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from models.agvs_mner import AgvsMnerModel
from modules.dataset import MultimodalNerDataset
from modules.trainer import NerTrainer, load_compatible_model_checkpoint
from run_llm import set_seed


class ImageMaskOverrideDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Expose the same samples while disabling every visual input slot."""

    def __init__(self, base: Dataset[tuple[torch.Tensor, ...]]) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = list(self.base[index])
        sample[-1] = torch.zeros_like(sample[-1])
        return tuple(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def _checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _evaluator(
    model: torch.nn.Module,
    args: argparse.Namespace,
    label_mapping: dict[str, int],
) -> NerTrainer:
    evaluator = object.__new__(NerTrainer)
    evaluator.model = model
    evaluator.args = args
    evaluator.label_map = label_mapping
    evaluator.index_to_label = {
        index: label for label, index in label_mapping.items()
    }
    return evaluator


def main() -> None:
    cli = parse_args()
    checkpoint = Path(cli.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    config = _checkpoint_config(checkpoint)
    config["device"] = cli.device
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.num_workers is not None:
        config["num_workers"] = cli.num_workers
    args = SimpleNamespace(**config)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    set_seed(int(getattr(args, "seed", 1234)))

    dataset = MultimodalNerDataset(
        args,
        f"MNER-{args.dataset}_{cli.split}.txt",
    )
    label_mapping = dataset.processor.get_label_mapping(
        include_subword_label=True
    )
    model = AgvsMnerModel(list(label_mapping), args)
    missing = load_compatible_model_checkpoint(
        model,
        checkpoint,
        map_location="cpu",
    )
    if missing:
        print(f"Optional checkpoint keys initialized locally: {missing}")
    model.to(args.device)
    evaluator = _evaluator(model, args, label_mapping)

    loader_options = {
        "batch_size": int(getattr(args, "batch_size", 4)),
        "num_workers": int(getattr(args, "num_workers", 0)),
        "shuffle": False,
    }
    results: dict[str, dict[str, float]] = {}
    for mode, mode_dataset in (
        ("full", dataset),
        ("text_only", ImageMaskOverrideDataset(dataset)),
    ):
        metrics = evaluator._run_epoch(
            DataLoader(mode_dataset, **loader_options),
            training=False,
        )
        results[mode] = {
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "loss": float(metrics["loss"]),
        }
        print(f"\n=== {mode} ===")
        print(metrics["report"])

    results["comparison"] = {
        "full_minus_text_only_f1": (
            results["full"]["f1"] - results["text_only"]["f1"]
        )
    }
    print("\n=== summary ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
