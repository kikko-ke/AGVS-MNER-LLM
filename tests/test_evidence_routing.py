from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import torch

from modules.llm_trainer import (
    LlmNerTrainer,
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


class _FakeCrf:
    def __call__(self, emissions, _labels, *, mask, reduction):
        log_likelihood = -(emissions[..., 0] * mask.float()).sum(dim=1)
        if reduction == "none":
            return log_likelihood
        return log_likelihood.mean()


class _TwoViewModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.crf = _FakeCrf()
        self.calls = []
        self.last_loss_components = {}

    def forward(
        self,
        *,
        input_ids,
        image_mask,
        labels=None,
        visual_strength_override=None,
        return_features=False,
        **_kwargs,
    ):
        batch_size, sequence_length = input_ids.shape
        noise = torch.rand(batch_size, 1, 1)
        if visual_strength_override is None:
            strength = image_mask.float().mean(dim=1)
        else:
            strength = visual_strength_override
        emissions = (
            self.scale * strength.reshape(batch_size, 1, 1) + noise
        ).expand(batch_size, sequence_length, 3)
        self.calls.append(
            {
                "mask": image_mask.detach().clone(),
                "noise": noise.detach().clone(),
                "strength": strength.detach().clone(),
            }
        )
        decoded = [[0] * sequence_length for _ in range(batch_size)]
        loss = None
        if labels is not None:
            loss = emissions.square().mean()
            self.last_loss_components = {"ner": loss.detach()}
        if return_features:
            return decoded, loss, {"token_emissions": emissions}
        return decoded, loss


def test_two_view_step_replays_dropout_and_preserves_visual_strength():
    trainer = object.__new__(LlmNerTrainer)
    trainer.model = _TwoViewModel()
    trainer.model.train()
    trainer.args = SimpleNamespace(
        llm_routed_crf_loss_weight=0.5,
        llm_consistency_kl_weight=0.2,
        llm_consistency_temperature=2.0,
    )
    trainer.label_map = {"O": 0, "B-PER": 1, "X": 2}

    input_ids = torch.tensor([[101, 10, 102]])
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    image_pixels = torch.zeros((1, 4, 3, 2, 2))
    labels = torch.tensor([[2, 1, 2]])
    image_mask = torch.tensor([[1, 1, 1, 0]])
    token_probs = torch.zeros((1, 3, 3))
    image_scores = torch.zeros((1, 4))
    available = torch.zeros(1)
    grounding_targets = torch.zeros((1, 3, 5))
    grounding_targets[0, 1, 1] = 1.0
    grounding_weights = torch.tensor([[0.0, 0.9, 0.0]])

    *_unused, loss = trainer._step(
        (
            input_ids,
            token_type_ids,
            attention_mask,
            image_pixels,
            labels,
            image_mask,
            token_probs,
            image_scores,
            available,
            grounding_targets,
            grounding_weights,
        )
    )
    loss.backward()

    assert len(trainer.model.calls) == 2
    full_call, routed_call = trainer.model.calls
    assert torch.equal(full_call["noise"], routed_call["noise"])
    assert routed_call["mask"].tolist() == [[0, 1, 0, 0]]
    assert torch.equal(routed_call["strength"], torch.tensor([0.75]))
    assert trainer.model.scale.grad is not None
    assert trainer.model.last_loss_components["evidence_route_fraction"] == 1
