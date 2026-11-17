"""Original-word alignment and ordered image loading for the TPM-MI baseline."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import TAGS


def open_rgb(path):
    from PIL import Image
    content = Path(path).read_bytes()
    for candidate in (content, content + b"\xff\xd9"):
        try:
            with Image.open(BytesIO(candidate)) as image:
                result = image.convert("RGBA").convert("RGB")
                result.load()
                return result
        except OSError:
            continue
    raise ValueError(f"Cannot decode image: {path}")


def image_path(name, config):
    if "twitter2017" in name:
        root = Path(config["twitter_image_dir"])
        path = root / name
        return path if path.is_file() else root / name.split("-", 1)[-1]
    return Path(config["image_dir"]) / name


def encode_words(row, tokenizer, maximum):
    """Tokenize ONCE: encode_plus on already split WordPieces can shift labels."""
    if maximum < 3:
        raise ValueError("At least three text positions are required")
    pieces, labels, positions, indices = [], [], [], []
    budget = maximum - 2
    for index, word in enumerate(row["tokens"]):
        tokens = tokenizer.tokenize(word)
        if not tokens:
            continue
        remaining = budget - len(pieces)
        if remaining <= 0:
            break
        tokens = tokens[:remaining]
        positions.append(1 + len(pieces))
        indices.append(index)
        pieces.extend(tokens)
        tag = row.get("labels", ["O"] * len(row["tokens"]))[index]
        labels.extend([TAGS.index(tag), *([TAGS.index("X")] * (len(tokens) - 1))])
    ids = [tokenizer.cls_token_id, *tokenizer.convert_tokens_to_ids(pieces), tokenizer.sep_token_id]
    length = len(ids)
    labels = [TAGS.index("X"), *labels, TAGS.index("X")]
    return {"input_ids": torch.tensor(ids + [tokenizer.pad_token_id] * (maximum - length)),
            "attention_mask": torch.arange(maximum).lt(length),
            "token_type_ids": torch.zeros(maximum, dtype=torch.long),
            "labels": torch.tensor(labels + [0] * (maximum - length)),
            "positions": positions, "word_indices": indices}


class WordImageDataset(Dataset):
    def __init__(self, rows, config, image_audit, tokenizer=None, processor=None):
        from transformers import BertTokenizer, ViTImageProcessor
        self.rows, self.config, self.image_audit = rows, config, image_audit
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(config["bert_model"], local_files_only=True)
        self.processor = processor or ViTImageProcessor.from_pretrained(config["vit_model"], local_files_only=True)
        self.encoded = [encode_words(row, self.tokenizer, config["max_length"]) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        pixels = torch.zeros(4, 3, 224, 224)
        valid = torch.zeros(4, dtype=torch.bool)
        for slot, name in enumerate(self.rows[index]["images"]):
            if not self.image_audit[name]["valid"]:
                continue
            with open_rgb(image_path(name, self.config)) as image:
                pixels[slot] = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
            valid[slot] = True
        return {**self.encoded[index], "pixels": pixels, "image_mask": valid, "index": index}


def collate(items):
    tensor_keys = ("input_ids", "attention_mask", "token_type_ids", "labels", "pixels", "image_mask")
    return {**{key: torch.stack([item[key] for item in items]) for key in tensor_keys},
            **{key: [item[key] for item in items] for key in ("positions", "word_indices", "index")}}


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
