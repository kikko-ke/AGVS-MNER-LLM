from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import torch

from modules.llm_trainer import (
    build_evidence_route_mask,
    routed_emission_kl,
    routed_visual_strength,
)
from modules.trainer import NerTrainer, load_compatible_model_checkpoint


def test_evidence_route_keeps_union_and_ignores_background_supervision():
    image_mask = torch.tensor([[1, 1, 1, 0]])
    targets = torch.zeros((1, 5, 5))
    targets[0, 1, 1] = 1.0
    targets[0, 2, 2] = 1.0
    targets[0, 3, 0] = 1.0  # O-token supervision must not affect the route.
    weights = torch.tensor([[0.0, 0.8, 0.6, 0.9, 0.0]])
    labels = torch.tensor([[9, 3, 4, 0, 9]])

    routed, changed, token_weights = build_evidence_route_mask(
        image_mask,
        targets,
        weights,
        labels,
        torch.ones_like(labels),
        outside_label_id=0,
        subword_label_id=9,
    )

    assert routed.tolist() == [[0, 1, 1, 0]]
    assert changed.tolist() == [True]
    assert torch.equal(
        token_weights,
        torch.tensor([[0.0, 0.8, 0.6, 0.0, 0.0]]),
    )


def test_evidence_route_preserves_unsupervised_and_unchanged_samples():
    image_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    targets = torch.zeros((2, 3, 5))
    weights = torch.zeros((2, 3))
    labels = torch.tensor([[9, 3, 9], [9, 3, 9]])

    # Sample 0 has no teacher entity. Sample 1 selects every valid image, so
    # its routed view is identical to the full view.
    targets[1, 1, 0] = 0.5
    targets[1, 1, 1] = 0.5
    weights[1, 1] = 1.0

    routed, changed, token_weights = build_evidence_route_mask(
        image_mask,
        targets,
        weights,
        labels,
        torch.ones_like(labels),
        outside_label_id=0,
        subword_label_id=9,
    )

    assert torch.equal(routed, image_mask)
    assert changed.tolist() == [False, False]
    assert token_weights.count_nonzero() == 0


def test_none_only_entity_builds_text_only_route_and_discards_invalid_slots():
    image_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    targets = torch.zeros((2, 3, 5))
    weights = torch.tensor([[0.0, 0.9, 0.0], [0.0, 0.8, 0.0]])
    labels = torch.tensor([[9, 3, 9], [9, 5, 9]])
    targets[0, 1, 4] = 1.0  # Explicit NONE supervision.
    targets[1, 1, 2] = 1.0  # Defensive check: slot 2 is invalid.

    routed, changed, token_weights = build_evidence_route_mask(
        image_mask,
        targets,
        weights,
        labels,
        torch.ones_like(labels),
        outside_label_id=0,
        subword_label_id=9,
    )

    assert routed.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert changed.tolist() == [True, True]
    assert torch.equal(token_weights, weights)


def test_routed_emission_kl_detaches_teacher_and_masks_tokens():
    full = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]],
        requires_grad=True,
    )
    routed = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0], [3.0, -1.0]]],
        requires_grad=True,
    )
    token_weights = torch.tensor([[1.0, 0.0, 0.5]])

    loss = routed_emission_kl(full, routed, token_weights, temperature=2.0)
    loss.backward()

    assert loss.item() > 0
    assert full.grad is not None
    assert full.grad[0, 0].abs().sum() > 0
    assert full.grad[0, 1].abs().sum() == 0
    assert full.grad[0, 2].abs().sum() > 0
    assert routed.grad is None


def test_routed_visual_strength_preserves_scale_except_for_none_route():
    full = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
    )
    routed = torch.tensor(
        [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
    )

    strength = routed_visual_strength(full, routed)

    assert torch.equal(strength, torch.tensor([1.0, 0.0, 0.25]))


class _CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(2, 2)
        self.grounding_text_proj = torch.nn.Linear(2, 2, bias=False)
        self.utility_gate_scale = torch.nn.Parameter(torch.tensor(0.0))


def test_checkpoint_loader_accepts_missing_optional_teacher_heads():
    source = _CheckpointModel()
    destination = _CheckpointModel()
    shared_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if key.startswith("shared.")
    }
    checkpoint = Path.cwd() / f".checkpoint-loader-{uuid4().hex}.pt"
    try:
        torch.save({"model_state_dict": shared_state}, checkpoint)
        missing = load_compatible_model_checkpoint(destination, checkpoint)
    finally:
        checkpoint.unlink(missing_ok=True)

    assert set(missing) == {
        "grounding_text_proj.weight",
        "utility_gate_scale",
    }
    assert torch.equal(destination.shared.weight, source.shared.weight)
    assert torch.equal(destination.shared.bias, source.shared.bias)


def test_checkpoint_loader_rejects_missing_shared_parameters():
    destination = _CheckpointModel()
    checkpoint = Path.cwd() / f".checkpoint-loader-{uuid4().hex}.pt"
    try:
        torch.save({}, checkpoint)
        try:
            load_compatible_model_checkpoint(destination, checkpoint)
        except RuntimeError as error:
            assert "shared.weight" in str(error)
            assert "shared.bias" in str(error)
        else:
            raise AssertionError("Missing shared checkpoint parameters were accepted")
    finally:
        checkpoint.unlink(missing_ok=True)


def test_epoch_zero_checkpoint_wins_exact_validation_tie():
    trainer = object.__new__(NerTrainer)
    trainer.args = SimpleNamespace(init_checkpoint="baseline.pt")
    trainer.best_dev_f1 = 0.82
    trainer.best_dev_epoch = 0
    trainer.patience_counter = 0
    trainer.dev_data = object()
    trainer.checkpoint_path = Path("unused.pt")
    trainer.logger = logging.getLogger(__name__)
    trainer._run_epoch = lambda *_args, **_kwargs: {
        "loss": 1.0,
        "accuracy": 0.9,
        "f1": 0.82,
        "precision": 0.8,
        "recall": 0.84,
        "report": "report",
        "loss_components": {},
    }

    with patch("modules.trainer.torch.save") as save:
        trainer.evaluate(epoch=1)

    save.assert_not_called()
    assert trainer.best_dev_epoch == 0
    assert trainer.patience_counter == 1
