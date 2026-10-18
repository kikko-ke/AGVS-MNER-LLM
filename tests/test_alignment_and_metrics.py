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


class GradientIsolationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Parameter(torch.zeros(2))
        self.grounding_text_proj = torch.nn.Linear(1, 1, bias=False)


def test_gradient_clipping_isolates_auxiliary_teacher_head():
    trainer = object.__new__(NerTrainer)
    trainer.model = GradientIsolationModel()
    trainer.args = SimpleNamespace(max_grad_norm=1.0)
    trainer.model.main.grad = torch.tensor([3.0, 4.0])
    trainer.model.grounding_text_proj.weight.grad = torch.tensor([[100.0]])

    trainer._clip_gradients()

    assert torch.isclose(trainer.model.main.grad.norm(), torch.tensor(1.0))
    assert torch.isclose(
        trainer.model.grounding_text_proj.weight.grad.norm(), torch.tensor(1.0)
    )


def test_utility_gate_activates_only_after_selector_warmup():
    trainer = object.__new__(NerTrainer)
    trainer.model = SimpleNamespace(
        llm_utility_gate=True,
        llm_utility_gate_active=True,
        utility_gate_scale=torch.nn.Parameter(torch.tensor(0.7)),
    )
    trainer.args = SimpleNamespace(
        llm_utility_gate_warmup_epochs=2,
        llm_utility_gate_init=0.1,
    )
    trainer.logger = logging.getLogger(__name__)

    trainer._configure_utility_gate_for_epoch(1)
    assert trainer.model.llm_utility_gate_active is False
    assert torch.isclose(
        trainer.model.utility_gate_scale.detach(), torch.tensor(0.7)
    )

    trainer._configure_utility_gate_for_epoch(3)
    assert trainer.model.llm_utility_gate_active is True
    assert torch.isclose(
        trainer.model.utility_gate_scale.detach(), torch.tensor(0.1)
    )

    trainer.best_dev_epoch = 2
    trainer._restore_utility_gate_for_best_checkpoint()
    assert trainer.model.llm_utility_gate_active is False

    trainer.best_dev_epoch = 3
    trainer._restore_utility_gate_for_best_checkpoint()
    assert trainer.model.llm_utility_gate_active is True


def test_utility_gate_projection_prevents_negative_gradient_dead_zone():
    trainer = object.__new__(NerTrainer)
    trainer.model = SimpleNamespace(
        llm_utility_gate=True,
        utility_gate_scale=torch.nn.Parameter(torch.tensor(-0.2)),
    )

    trainer._project_constrained_controls()

    assert float(trainer.model.utility_gate_scale.detach()) == 0.0

    with torch.no_grad():
        trainer.model.utility_gate_scale.fill_(0.3)
    trainer._project_constrained_controls()
    assert torch.isclose(
        trainer.model.utility_gate_scale.detach(), torch.tensor(0.3)
    )
