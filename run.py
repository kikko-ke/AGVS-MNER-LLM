"""Command-line entry point for training AGVS-MNER."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.agvs_mner import AgvsMnerModel
from modules.dataset import MultimodalNerDataset
from modules.trainer import NerTrainer


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse training configuration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        "--data_path",
        default="./dataset/text",
        help="Directory containing split files.",
    )
    parser.add_argument(
        "--image-path",
        "--image_path",
        default="./dataset/images",
        help="Directory containing MNER-MI images.",
    )
    parser.add_argument(
        "--twitter2017-image-path",
        "--twitter2017_image_path",
        default="./dataset/twitter2017_images",
        help="Directory containing Twitter-2017 images.",
    )
    parser.add_argument(
        "--bert-model", "--bert_model", default="../pretrained_models/bert-base-uncased"
    )
    parser.add_argument("--vit-model", "--vit_model", default="../pretrained_models/ViTB-16")
    parser.add_argument(
        "--dataset", default="UNI", choices=("MI", "UNI"), help="MNER-MI or MNER-MI-Plus."
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="./outputs",
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Training device, for example cuda or cpu."
    )
    parser.add_argument("--num-epochs", "--num_epochs", default=15, type=int)
    parser.add_argument("--batch-size", "--batch_size", default=8, type=int)
    parser.add_argument("--num-workers", "--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate for task layers.")
    parser.add_argument("--bert-lr", "--bert_lr", default=1e-5, type=float)
    parser.add_argument("--vit-lr", "--vit_lr", default=5e-6, type=float)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", default=0.01, type=float)
    parser.add_argument("--weight-decay", "--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--patience", default=3, type=int, help="Use 0 to disable early stopping.")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--max-seq-length", "--max_seq_length", default=64, type=int)
    parser.add_argument("--topk-img-patches", "--topk_img_patches", default=12, type=int)
    parser.add_argument("--image-dropout-prob", "--image_dropout_prob", default=0.2, type=float)
    parser.add_argument("--output-dropout", "--output_dropout", default=0.2, type=float)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set all random-number generators used by the training process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(output_dir: str) -> None:
    """Configure console and persistent training logs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(Path(output_dir) / "training.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)


def main() -> None:
    """Build datasets and start model training."""
    args = parse_args()
    configure_logging(args.output_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu instead.")

    set_seed(args.seed)
    for key, value in vars(args).items():
        LOGGER.info("%s = %s", key, value)

    train_dataset = MultimodalNerDataset(args, f"MNER-{args.dataset}_train.txt")
    dev_dataset = MultimodalNerDataset(args, f"MNER-{args.dataset}_val.txt")
    test_dataset = MultimodalNerDataset(args, f"MNER-{args.dataset}_test.txt")
    data_loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_data = DataLoader(train_dataset, shuffle=True, **data_loader_kwargs)
    dev_data = DataLoader(dev_dataset, shuffle=False, **data_loader_kwargs)
    test_data = DataLoader(test_dataset, shuffle=False, **data_loader_kwargs)

    label_mapping = train_dataset.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(label_mapping), args)
    trainer = NerTrainer(train_data, dev_data, test_data, model, label_mapping, args, LOGGER)
    trainer.train()


if __name__ == "__main__":
    main()
