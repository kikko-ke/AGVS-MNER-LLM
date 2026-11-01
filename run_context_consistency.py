"""Controlled training-internal experiment with existing LLM context views."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.agvs_mner import AgvsMnerModel
from modules.context_consistency import ContextConsistencyTrainer, ContextViewDataset, verify_internal_manifest
from modules.dataset import MultimodalNerDataset
from run import build_parser, configure_logging, set_seed


def parse_args():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--internal-manifest", required=True)
    parser.add_argument("--augmentation-cache", required=True)
    parser.add_argument("--consistency-mode", choices=("none", "dropout", "llm_context"), required=True)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--consistency-temperature", type=float, default=2.0)
    parser.add_argument("--consistency-warmup-epochs", type=int, default=2)
    parser.set_defaults(skip_test=True, visual_fusion="legacy")
    return parser.parse_args()


def main():
    args = parse_args()
    if (not math.isfinite(args.consistency_weight) or not math.isfinite(args.consistency_temperature)
            or args.consistency_weight < 0 or args.consistency_temperature <= 0 or args.consistency_warmup_epochs < 0):
        raise ValueError("Expected non-negative consistency weight/warmup and positive temperature")
    if not args.skip_test or args.visual_fusion != "legacy":
        raise ValueError("This controlled pilot requires the legacy component and no test evaluation")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; start with a GPU for this pilot")
    manifest = verify_internal_manifest(args.internal_manifest, args.dataset)
    if Path(args.data_path).resolve() != Path(manifest["partitions"]["train"]["path"]).parent.resolve():
        raise ValueError("data-path must be the declared internal split directory")
    configure_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    for key, value in vars(args).items():
        logger.info("%s = %s", key, value)
    base = MultimodalNerDataset(args, f"MNER-{args.dataset}_train.txt")
    train = ContextViewDataset(base, args.internal_manifest, args.augmentation_cache, args.dataset)
    dev = MultimodalNerDataset(args, f"MNER-{args.dataset}_val.txt")
    logger.info("Internal context views: %s", json.dumps(train.summary, sort_keys=True))
    # Explicit loader generators also separate shuffle/worker RNG from dropout.
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, generator=torch.Generator().manual_seed(args.seed + 1))
    labels = base.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(labels), args)
    trainer = ContextConsistencyTrainer(train_loader, dev_loader, None, model, labels, args, logger)
    trainer.train()


if __name__ == "__main__":
    main()
