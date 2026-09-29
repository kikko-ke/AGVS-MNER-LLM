"""Dataset loading and preprocessing for multimodal named-entity recognition."""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor


NER_LABELS = (
    "O",
    "B-MIS",
    "I-MIS",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
)
SUBWORD_LABEL = "X"
MAX_IMAGES_PER_SAMPLE = 4


class MnerProcessor:
    """Loads MNER examples and provides the label vocabulary."""

    def __init__(self, args: Any) -> None:
        self.data_path = Path(args.data_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_model,
            do_lower_case=True,
        )

    def load_examples(self, file_name: str) -> dict[str, list[Any]]:
        """Load line-delimited Python/JSON-style MNER records safely."""
        words: list[list[str]] = []
        labels: list[list[str]] = []
        images: list[list[str]] = []
        file_path = self.data_path / file_name

        with file_path.open("r", encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                try:
                    record = ast.literal_eval(line)
                    text = record["text"]
                    raw_labels = record["label"]
                    image_names = record["images"]
                except (KeyError, SyntaxError, ValueError) as error:
                    raise ValueError(
                        f"Invalid record in {file_path} at line {line_number}."
                    ) from error

                if len(text) != len(raw_labels):
                    raise ValueError(
                        f"Text and label lengths differ in {file_path} at line {line_number}."
                    )

                normalized_labels = [
                    label.replace("MISC", "MIS") for label in raw_labels
                ]
                words.append(text)
                labels.append(normalized_labels)
                images.append(image_names)

        return {"words": words, "labels": labels, "images": images}

    @staticmethod
    def get_label_mapping(include_subword_label: bool = False) -> dict[str, int]:
        """Return the model label-to-index mapping."""
        labels = NER_LABELS + ((SUBWORD_LABEL,) if include_subword_label else ())
        return {label: index for index, label in enumerate(labels)}


class MultimodalNerDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Encodes text, token labels, and up to four associated images per sample."""

    def __init__(self, args: Any, file_name: str) -> None:
        self.max_sequence_length = args.max_seq_length
        self.processor = MnerProcessor(args)
        self.image_processor = ViTImageProcessor.from_pretrained(args.vit_model)
        self.examples = self.processor.load_examples(file_name)
        self.tokenizer = self.processor.tokenizer
        self.label_mapping = self.processor.get_label_mapping(include_subword_label=True)
        self.image_path = Path(args.image_path)
        self.twitter2017_image_path = Path(args.twitter2017_image_path)

    def __len__(self) -> int:
        return len(self.examples["words"])

    def _get_image_path(self, image_name: str) -> Path:
        if "twitter2017" in image_name:
            # Archives in the wild use both the prefixed and stripped filename.
            full_path = self.twitter2017_image_path / image_name
            if full_path.exists():
                return full_path
            return self.twitter2017_image_path / image_name.split("-", maxsplit=1)[1]
        return self.image_path / image_name

    def _encode_image(self, image_path: Path) -> torch.Tensor | None:
        """Return ViT pixels for an image, repairing a truncated JPEG when possible."""
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            try:
                image = Image.open(BytesIO(image_path.read_bytes() + b"\xff\xd9")).convert("RGB")
            except (FileNotFoundError, OSError, UnidentifiedImageError):
                return None

        return self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def _encode_placeholder_image(self) -> torch.Tensor | None:
        """Replicate the original fallback for unreadable image files."""
        placeholder_path = self.image_path / "inf.png"
        try:
            image = Image.open(placeholder_path).convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            return None
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def _encode_images(self, image_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = torch.zeros((MAX_IMAGES_PER_SAMPLE, 3, 224, 224))
        image_mask = torch.zeros(MAX_IMAGES_PER_SAMPLE, dtype=torch.long)

        for index, image_name in enumerate(image_names[:MAX_IMAGES_PER_SAMPLE]):
            image_tensor = self._encode_image(self._get_image_path(image_name))
            if image_tensor is not None:
                pixel_values[index] = image_tensor
                image_mask[index] = 1
            else:
                # Preserve the original placeholder pixels while marking them invalid.
                placeholder_tensor = self._encode_placeholder_image()
                if placeholder_tensor is not None:
                    pixel_values[index] = placeholder_tensor

        return pixel_values, image_mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        word_list = self.examples["words"][index]
        label_list = self.examples["labels"][index]
        image_pixel_values, image_mask = self._encode_images(self.examples["images"][index])

        tokens: list[str] = []
        label_ids: list[int] = []
        for word, label in zip(word_list, label_list):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if word_tokens:
                label_ids.append(self.label_mapping[label])
                label_ids.extend(
                    [self.label_mapping[SUBWORD_LABEL]] * (len(word_tokens) - 1)
                )

        max_token_count = self.max_sequence_length - 2
        tokens = tokens[:max_token_count]
        label_ids = label_ids[:max_token_count]

        encoded_text = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        labels = [self.label_mapping[SUBWORD_LABEL], *label_ids, self.label_mapping[SUBWORD_LABEL]]
        labels.extend([self.label_mapping["O"]] * (self.max_sequence_length - len(labels)))

        return (
            torch.tensor(encoded_text["input_ids"], dtype=torch.long),
            torch.tensor(encoded_text["token_type_ids"], dtype=torch.long),
            torch.tensor(encoded_text["attention_mask"], dtype=torch.long),
            image_pixel_values,
            torch.tensor(labels, dtype=torch.long),
            image_mask,
        )

