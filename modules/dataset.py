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

from modules.text_utils import normalize_unicode_text, normalize_words


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
                words.append(normalize_words(text))
                labels.append(normalized_labels)
                images.append(image_names)

        return {"words": words, "labels": labels, "images": images}

    def load_unlabeled_examples(self, file_name: str) -> dict[str, list[Any]]:
        """Load only fields available at inference time.

        Keeping this loader separate from :meth:`load_examples` makes it
        impossible for candidate generation to consume the NER labels by
        accident.  The source format may still contain a ``label`` member, but
        it is deliberately never indexed here.
        """
        words: list[list[str]] = []
        images: list[list[str]] = []
        file_path = self.data_path / file_name

        with file_path.open("r", encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                try:
                    record = ast.literal_eval(line)
                    text = record["text"]
                    image_names = record["images"]
                except (KeyError, SyntaxError, ValueError) as error:
                    raise ValueError(
                        f"Invalid inference record in {file_path} at line {line_number}."
                    ) from error
                if not isinstance(text, list) or not isinstance(image_names, list):
                    raise ValueError(
                        f"Text and images must be lists in {file_path} at line {line_number}."
                    )
                words.append(normalize_words(text))
                images.append([str(name) for name in image_names])

        return {"words": words, "images": images}

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

    def _encode_text(
        self,
        word_list: list[str],
        label_list: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize each word once and keep token IDs aligned with BIO labels."""
        tokens: list[str] = []
        label_ids: list[int] = []
        for word, label in zip(word_list, label_list):
            word_tokens = self.tokenizer.tokenize(normalize_unicode_text(word))
            if not word_tokens:
                continue
            tokens.extend(word_tokens)
            label_ids.append(self.label_mapping[label])
            label_ids.extend(
                [self.label_mapping[SUBWORD_LABEL]] * (len(word_tokens) - 1)
            )

        max_token_count = self.max_sequence_length - 2
        tokens = tokens[:max_token_count]
        label_ids = label_ids[:max_token_count]

        # ``tokens`` already contains WordPiece output. Calling the tokenizer with
        # ``is_split_into_words=True`` here would tokenize pieces such as ``##ing``
        # a second time and shift every following label. Convert the pieces directly
        # to IDs so inputs, gold labels, and cached teacher probabilities share the
        # exact same positions.
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_id_values = [self.tokenizer.cls_token_id, *token_ids, self.tokenizer.sep_token_id]
        attention_mask_values = [1] * len(input_id_values)
        padding_length = self.max_sequence_length - len(input_id_values)
        input_id_values.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask_values.extend([0] * padding_length)
        token_type_id_values = [0] * self.max_sequence_length

        labels = [
            self.label_mapping[SUBWORD_LABEL],
            *label_ids,
            self.label_mapping[SUBWORD_LABEL],
        ]
        labels.extend([self.label_mapping["O"]] * (self.max_sequence_length - len(labels)))

        input_ids = torch.tensor(input_id_values, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_id_values, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_values, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        expected_length = self.max_sequence_length
        if not all(
            tensor.shape == (expected_length,)
            for tensor in (input_ids, token_type_ids, attention_mask, label_tensor)
        ):
            raise ValueError("Encoded text and labels must have max_seq_length positions")

        return input_ids, token_type_ids, attention_mask, label_tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        word_list = self.examples["words"][index]
        label_list = self.examples["labels"][index]
        image_pixel_values, image_mask = self._encode_images(self.examples["images"][index])
        input_ids, token_type_ids, attention_mask, labels = self._encode_text(
            word_list,
            label_list,
        )

        return (
            input_ids,
            token_type_ids,
            attention_mask,
            image_pixel_values,
            labels,
            image_mask,
        )


class MultimodalInferenceDataset(MultimodalNerDataset):
    """Inference-only view that never loads NER labels from the source file."""

    def __init__(self, args: Any, file_name: str) -> None:
        self.max_sequence_length = args.max_seq_length
        self.processor = MnerProcessor(args)
        self.image_processor = ViTImageProcessor.from_pretrained(args.vit_model)
        self.examples = self.processor.load_unlabeled_examples(file_name)
        self.tokenizer = self.processor.tokenizer
        self.label_mapping = self.processor.get_label_mapping(include_subword_label=True)
        self.image_path = Path(args.image_path)
        self.twitter2017_image_path = Path(args.twitter2017_image_path)

    def token_structure(self, index: int) -> tuple[torch.Tensor, list[str], list[int]]:
        """Return special/continuation positions and retained input words.

        The structure is derived only from the tokenizer.  A word is retained
        when at least its first WordPiece fits in the model's input window,
        matching the flat-token truncation used by ``_encode_text``.
        """
        token_budget = self.max_sequence_length - 2
        used_tokens = 0
        structure = [True]  # CLS
        retained_words: list[str] = []
        retained_word_indices: list[int] = []
        for word_index, word in enumerate(self.examples["words"][index]):
            pieces = self.tokenizer.tokenize(normalize_unicode_text(word))
            if not pieces:
                continue
            remaining = token_budget - used_tokens
            if remaining <= 0:
                break
            retained_count = min(len(pieces), remaining)
            retained_words.append(word)
            retained_word_indices.append(word_index)
            structure.extend([False, *([True] * (retained_count - 1))])
            used_tokens += retained_count
            if used_tokens == token_budget:
                break
        structure.append(True)  # SEP

        padded = torch.zeros(self.max_sequence_length, dtype=torch.bool)
        padded[: len(structure)] = torch.tensor(structure, dtype=torch.bool)
        return padded, retained_words, retained_word_indices

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        word_list = self.examples["words"][index]
        image_pixel_values, image_mask = self._encode_images(self.examples["images"][index])
        input_ids, token_type_ids, attention_mask, _ = self._encode_text(
            word_list,
            ["O"] * len(word_list),
        )
        return (
            input_ids,
            token_type_ids,
            attention_mask,
            image_pixel_values,
            image_mask,
            torch.tensor(index, dtype=torch.long),
        )

