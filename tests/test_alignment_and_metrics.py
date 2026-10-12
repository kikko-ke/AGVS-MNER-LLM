from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

from models.agvs_mner import _teacher_token_mask
from modules.dataset import MnerProcessor, MultimodalNerDataset
from modules.text_utils import normalize_unicode_text, normalize_words
from modules.trainer import NerTrainer


class TrackingTokenizer:
    """Small tokenizer that fails if already-tokenized pieces are tokenized again."""

    piece_map = {
        "unaffordable": ["una", "##ff", "##ord", "##able"],
        "alice": ["alice"],
    }
    vocab = {
        "[PAD]": 0,
        "[CLS]": 101,
        "[SEP]": 102,
        "una": 1001,
        "##ff": 1002,
        "##ord": 1003,
        "##able": 1004,
        "alice": 1005,
    }
    pad_token_id = vocab["[PAD]"]
    cls_token_id = vocab["[CLS]"]
    sep_token_id = vocab["[SEP]"]

    def __call__(self, *args, **kwargs):
        raise AssertionError("WordPieces must not be passed through the tokenizer again")

    def tokenize(self, word: str) -> list[str]:
        return self.piece_map[word.lower()]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab[token] for token in tokens]


def test_dataset_converts_wordpieces_directly_to_ids():
    dataset = object.__new__(MultimodalNerDataset)
    dataset.max_sequence_length = 10
    dataset.tokenizer = TrackingTokenizer()
    dataset.label_mapping = MnerProcessor.get_label_mapping(include_subword_label=True)

    input_ids, token_type_ids, attention_mask, labels = dataset._encode_text(
        ["unaffordable", "Alice"],
        ["B-MIS", "B-PER"],
    )

    assert input_ids.tolist() == [101, 1001, 1002, 1003, 1004, 1005, 102, 0, 0, 0]
    assert token_type_ids.tolist() == [0] * 10
    assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert labels.tolist() == [9, 1, 9, 9, 9, 3, 9, 0, 0, 0]


def test_surrogate_pairs_are_normalized_before_tokenization():
    assert normalize_unicode_text("plain text") == "plain text"
    assert normalize_unicode_text("\ud83d\udc94") == "💔"
    assert normalize_words(["music", "\ud83c\udfbc"]) == ["music", "🎼"]


def test_teacher_loss_uses_only_available_non_subword_positions():
    mask = _teacher_token_mask(
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        labels=torch.tensor([[9, 3, 9, 0, 0], [9, 1, 0, 0, 0]]),
        subword_label_id=9,
        llm_available=torch.tensor([1.0, 0.0]),
    )

    assert mask.tolist() == [
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_predictions_map_internal_x_label_to_outside():
    trainer = object.__new__(NerTrainer)
    trainer.label_map = MnerProcessor.get_label_mapping(include_subword_label=True)
    trainer.index_to_label = {
        index: label for label, index in trainer.label_map.items()
    }

    true_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []
    trainer._append_predictions(
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
        labels=torch.tensor([[9, 3, 9, 0, 0]]),
        decoded_label_ids=[[9, 9, 5, 9]],
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    assert true_labels == [["B-PER", "O"]]
    assert predicted_labels == [["O", "O"]]
    assert all(label != "X" for sequence in predicted_labels for label in sequence)


class SingleBatchTrainer(NerTrainer):
    def _step(self, batch):
        return batch


def test_epoch_metrics_use_the_same_seqeval_f1_as_the_report():
    trainer = object.__new__(SingleBatchTrainer)
    trainer.model = SimpleNamespace(eval=lambda: None)
    trainer.label_map = MnerProcessor.get_label_mapping(include_subword_label=True)
    trainer.index_to_label = {
        index: label for label, index in trainer.label_map.items()
    }
    trainer.args = SimpleNamespace(device="cpu")
    trainer.logger = logging.getLogger(__name__)

    batch = (
        torch.tensor([[1, 1, 1, 0]]),
        torch.tensor([[9, 3, 0, 0]]),
        [[9, 3, 0]],
        torch.tensor(0.0),
    )
    metrics = trainer._run_epoch([batch], training=False)

    assert metrics["f1"] == 1.0
    assert "1.0000" in metrics["report"]
    assert "_" not in metrics["report"]
