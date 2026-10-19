"""Train AGVS-MNER with cached LLM teacher supervision."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.agvs_mner import AgvsMnerModel
from modules.dataset import MultimodalNerDataset
from modules.llm_dataset import TeacherCacheDataset
from modules.llm_trainer import LlmNerTrainer
from modules.trainer import load_compatible_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="./dataset/text")
    parser.add_argument("--image-path", default="./dataset/images")
    parser.add_argument("--twitter2017-image-path", default="./dataset/twitter2017_images")
    parser.add_argument("--bert-model", default="../pretrained_models/bert-base-uncased")
    parser.add_argument("--vit-model", default="../pretrained_models/ViTB-16")
    parser.add_argument("--dataset", default="UNI", choices=("MI", "UNI"))
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument(
        "--teacher-cache-task",
        choices=("mner", "grounding", "utility"),
        default="grounding",
        help="Expected cache semantics; old caches without metadata are mner",
    )
    parser.add_argument("--output-dir", default="./outputs_llm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bert-lr", type=float, default=1e-5)
    parser.add_argument("--vit-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--topk-img-patches", type=int, default=12)
    parser.add_argument("--image-dropout-prob", type=float, default=0.2)
    parser.add_argument("--output-dropout", type=float, default=0.2)
    # Raw teacher NER predictions are intentionally disabled by default. The
    # reliable LLM signal is entity-to-image evidence whose span/type agrees
    # with the gold training annotation.
    parser.add_argument("--llm-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--llm-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--llm-grounding-loss-weight", type=float, default=0.1)
    parser.add_argument("--llm-grounding-confidence", type=float, default=0.7)
    parser.add_argument(
        "--llm-utility-threshold",
        type=float,
        default=0.5,
        help="Minimum additional visual utility before an entity receives an image target",
    )
    parser.add_argument("--llm-grounding-temperature", type=float, default=0.1)
    parser.add_argument("--llm-grounding-size", type=int, default=256)
    parser.add_argument(
        "--llm-grounding-background-ratio",
        type=float,
        default=0.25,
        help="Total weak NONE supervision on background words relative to entity mass",
    )
    parser.add_argument(
        "--llm-grounding-detach-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep teacher grounding gradients out of the shared BERT/ViT encoders",
    )
    parser.add_argument(
        "--llm-grounding-fusion-init",
        type=float,
        default=0.0,
        help="Initial residual scale for grounding-guided visual fusion",
    )
    parser.add_argument(
        "--llm-grounding-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed the learned entity-image distribution into NER fusion",
    )
    parser.add_argument(
        "--llm-grounding-class-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--llm-utility-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use predicted NONE probability to suppress the original AGVS visual branch",
    )
    parser.add_argument(
        "--llm-utility-gate-init",
        type=float,
        default=0.0,
        help="Initial utility-gate strength; zero exactly reproduces the AGVS path",
    )
    parser.add_argument(
        "--llm-utility-gate-warmup-epochs",
        type=int,
        default=2,
        help="Train the utility selector before allowing it to alter NER visual features",
    )
    parser.add_argument(
        "--llm-routed-crf-loss-weight",
        type=float,
        default=0.0,
        help="CRF loss weight for the LLM evidence-preserving routed image view",
    )
    parser.add_argument(
        "--llm-consistency-kl-weight",
        type=float,
        default=0.0,
        help="Token-emission KL weight from the detached routed view to the full view",
    )
    parser.add_argument(
        "--llm-consistency-temperature",
        type=float,
        default=1.0,
        help="Temperature for routed-to-full token-emission consistency",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional baseline checkpoint used to initialize compatible model parameters",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate the test split after training; use for validation-only tuning",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    if args.llm_grounding_background_ratio < 0:
        raise ValueError("--llm-grounding-background-ratio must be non-negative")
    if args.llm_grounding_temperature <= 0:
        raise ValueError("--llm-grounding-temperature must be positive")
    if not 0 <= args.llm_utility_threshold <= 1:
        raise ValueError("--llm-utility-threshold must be in [0, 1]")
    if args.llm_utility_gate and args.teacher_cache_task != "utility":
        raise ValueError("--llm-utility-gate requires --teacher-cache-task utility")
    if args.llm_utility_gate_warmup_epochs < 0:
        raise ValueError("--llm-utility-gate-warmup-epochs must be non-negative")
    if args.llm_utility_gate_init < 0:
        raise ValueError("--llm-utility-gate-init must be non-negative")
    if args.llm_routed_crf_loss_weight < 0:
        raise ValueError("--llm-routed-crf-loss-weight must be non-negative")
    if args.llm_consistency_kl_weight < 0:
        raise ValueError("--llm-consistency-kl-weight must be non-negative")
    if args.llm_consistency_temperature <= 0:
        raise ValueError("--llm-consistency-temperature must be positive")
    evidence_intervention_enabled = (
        args.llm_routed_crf_loss_weight > 0
        or args.llm_consistency_kl_weight > 0
    )
    if evidence_intervention_enabled and args.teacher_cache_task not in {
        "grounding",
        "utility",
    }:
        raise ValueError(
            "Routed evidence losses require a grounding or utility teacher cache"
        )
    if evidence_intervention_enabled and args.llm_grounding_loss_weight > 0:
        raise ValueError(
            "Evidence intervention and the legacy grounding selector must be "
            "isolated; set --llm-grounding-loss-weight 0"
        )
    configure_logging(args.output_dir)
    set_seed(args.seed)

    train_base = MultimodalNerDataset(args, f"MNER-{args.dataset}_train.txt")
    dev_base = MultimodalNerDataset(args, f"MNER-{args.dataset}_val.txt")
    test_base = MultimodalNerDataset(args, f"MNER-{args.dataset}_test.txt")
    cache_dir = Path(args.teacher_cache)
    if args.llm_token_loss_weight > 0 and args.teacher_cache_task in {
        "grounding",
        "utility",
    }:
        raise ValueError(
            "Grounding caches receive gold entities and cannot be used for token KD; "
            "use a separate mner cache for the Raw token KD ablation"
        )
    train_data = TeacherCacheDataset(
        train_base,
        cache_dir / f"MNER-{args.dataset}_train.jsonl",
        grounding_confidence_threshold=args.llm_grounding_confidence,
        expected_teacher_task=args.teacher_cache_task,
        balance_grounding_classes=args.llm_grounding_class_balance,
        grounding_background_ratio=args.llm_grounding_background_ratio,
        visual_utility_threshold=args.llm_utility_threshold,
        enable_grounding=(
            args.llm_grounding_loss_weight > 0 or evidence_intervention_enabled
        ),
    )
    dev_data = TeacherCacheDataset(
        dev_base,
        cache_dir / f"MNER-{args.dataset}_val.jsonl",
        grounding_confidence_threshold=args.llm_grounding_confidence,
        enable_grounding=False,
    )
    test_data = TeacherCacheDataset(
        test_base,
        cache_dir / f"MNER-{args.dataset}_test.jsonl",
        grounding_confidence_threshold=args.llm_grounding_confidence,
        enable_grounding=False,
    )

    if args.llm_grounding_loss_weight > 0 or evidence_intervention_enabled:
        train_cache_path = cache_dir / f"MNER-{args.dataset}_train.jsonl"
        if not train_cache_path.exists():
            raise FileNotFoundError(
                f"Grounding loss is enabled but train cache is missing: {train_cache_path}"
            )
        summary = train_data.grounding_summary()
        logging.getLogger(__name__).info(
            "LLM grounding supervision: %s", json.dumps(summary, ensure_ascii=False)
        )
        if summary["entities"] == 0:
            raise ValueError(
                "Grounding loss is enabled but no teacher entity passed the cache task, "
                "gold-consistency, confidence, and image-evidence checks"
            )

    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    dev_loader = DataLoader(dev_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    label_mapping = train_base.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(label_mapping), args)
    if args.init_checkpoint:
        missing_keys = load_compatible_model_checkpoint(
            model,
            args.init_checkpoint,
            map_location="cpu",
        )
        logging.getLogger(__name__).info(
            "Initialized from %s; retained initialization for optional keys: %s",
            args.init_checkpoint,
            missing_keys or "none",
        )
    trainer = LlmNerTrainer(
        train_loader,
        dev_loader,
        test_loader,
        model,
        label_mapping,
        args,
        logging.getLogger(__name__),
    )
    trainer.train()


if __name__ == "__main__":
    main()
