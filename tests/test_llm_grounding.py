from __future__ import annotations

import torch

from models.agvs_mner import _grounding_features_for_loss, _masked_grounding_loss
from modules.llm_dataset import build_grounding_targets, extract_bio_entities


class GroundingTokenizer:
    def tokenize(self, word: str) -> list[str]:
        if word.lower() == "smith":
            return ["smi", "##th"]
        return [word.lower()]


def test_extract_bio_entities_repairs_orphan_inside_tag():
    entities = extract_bio_entities(["B-PER", "I-PER", "O", "I-LOC"])
    assert entities == {(0, 2, "PER"), (3, 4, "LOC")}


def test_grounding_targets_require_gold_agreement_and_valid_images():
    targets, weights = build_grounding_targets(
        words=["Alice", "Smith", "at", "Acme"],
        gold_labels=["B-PER", "I-PER", "B-LOC", "B-ORG"],
        teacher_entities=[
            {
                "start": 0,
                "end": 2,
                "label": "PER",
                "image_ids": [1],
                "confidence": 0.9,
            },
            {
                "start": 3,
                "end": 4,
                "label": "ORG",
                "image_ids": [0, 2],
                "confidence": 0.8,
            },
            {
                "start": 2,
                "end": 3,
                "label": "MIS",
                "image_ids": [0],
                "confidence": 0.99,
            },
            {
                "start": 2,
                "end": 3,
                "label": "LOC",
                "image_ids": [],
                "confidence": 0.75,
            },
            {
                "start": 0,
                "end": 2,
                "label": "PER",
                "image_ids": [3],
                "confidence": 0.99,
            },
        ],
        tokenizer=GroundingTokenizer(),
        max_sequence_length=10,
        image_mask=torch.tensor([1, 1, 0, 0]),
        confidence_threshold=0.7,
    )

    # [CLS], Alice, Smith, ##th, at, Acme, [SEP]
    assert targets[1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert targets[2].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert targets[3].sum() == 0.0
    assert targets[4].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert targets[5].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert torch.allclose(
        weights[[1, 2, 4, 5]], torch.tensor([0.45, 0.45, 0.75, 0.8])
    )


def test_grounding_loss_masks_padding_images_and_unused_tokens():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, 100.0, 100.0, -1.0],
                [0.0, 2.0, 100.0, 100.0, -1.0],
            ]
        ]
    )
    targets = torch.zeros((1, 2, 5))
    targets[0, 0, 0] = 1.0
    weights = torch.tensor([[1.0, 0.0]])

    loss = _masked_grounding_loss(
        logits,
        image_mask=torch.tensor([[1, 1, 0, 0]]),
        targets=targets,
        weights=weights,
    )
    expected = -torch.log_softmax(torch.tensor([2.0, 0.0, -1.0]), dim=0)[0]
    assert torch.isclose(loss, expected)

    empty_loss = _masked_grounding_loss(
        logits,
        image_mask=torch.tensor([[0, 0, 0, 0]]),
        targets=torch.zeros_like(targets),
        weights=torch.zeros_like(weights),
    )
    assert torch.isfinite(empty_loss)
    assert empty_loss.item() == 0.0


def test_grounding_targets_weakly_supervise_background_as_none():
    targets, weights = build_grounding_targets(
        words=["Alice", "visits", "Paris"],
        gold_labels=["B-PER", "O", "B-LOC"],
        teacher_entities=[
            {
                "start": 0,
                "end": 1,
                "label": "PER",
                "image_ids": [0],
                "confidence": 0.9,
            },
            {
                "start": 2,
                "end": 3,
                "label": "LOC",
                "image_ids": [],
                "confidence": 0.8,
            },
        ],
        tokenizer=GroundingTokenizer(),
        max_sequence_length=8,
        image_mask=torch.tensor([1, 1, 0, 0]),
        confidence_threshold=0.7,
        background_ratio=0.25,
    )

    assert targets[1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert targets[2].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert targets[3].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert torch.isclose(weights[1], torch.tensor(0.9))
    assert torch.isclose(weights[3], torch.tensor(0.8))
    assert torch.isclose(weights[2], torch.tensor((0.9 + 0.8) * 0.25))


def test_grounding_loss_sends_no_gradient_to_invalid_image_slots():
    logits = torch.zeros((1, 1, 5), requires_grad=True)
    targets = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]]])
    loss = _masked_grounding_loss(
        logits,
        image_mask=torch.tensor([[1, 1, 0, 0]]),
        targets=targets,
        weights=torch.tensor([[1.0]]),
    )
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0, 0] != 0
    assert logits.grad[0, 0, 1] != 0
    assert logits.grad[0, 0, 2] == 0
    assert logits.grad[0, 0, 3] == 0
    assert logits.grad[0, 0, 4] != 0


def test_grounding_loss_can_detach_shared_encoder_features():
    text = torch.randn(2, 3, requires_grad=True)
    images = torch.randn(2, 4, requires_grad=True)

    detached_text, detached_images = _grounding_features_for_loss(
        text, images, detach_backbone=True
    )
    assert not detached_text.requires_grad
    assert not detached_images.requires_grad

    live_text, live_images = _grounding_features_for_loss(
        text, images, detach_backbone=False
    )
    assert live_text is text
    assert live_images is images
