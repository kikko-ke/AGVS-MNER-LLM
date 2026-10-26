"""Training-only LLM evidence representation distillation utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from modules.text_utils import normalize_unicode_text


EVIDENCE_EMBEDDING_SCHEMA_VERSION = "llm-evidence-embeddings-v1"
EVIDENCE_ENTITY_TYPES = ("MIS", "PER", "ORG", "LOC")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for evidence-artifact provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_evidence_bio_entities(
    labels: Sequence[str],
) -> set[tuple[int, int, str]]:
    """Return normalized word-level BIO spans without importing the dataset wrapper."""
    entities: set[tuple[int, int, str]] = set()
    start: int | None = None
    current_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, current_type
        if start is not None and current_type is not None:
            entities.add((start, end, current_type))
        start = None
        current_type = None

    for index, raw_label in enumerate([*labels, "O"]):
        label = str(raw_label).upper().replace("MISC", "MIS")
        if label == "O" or "-" not in label:
            close(index)
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix not in {"B", "I"} or entity_type not in EVIDENCE_ENTITY_TYPES:
            close(index)
            continue
        if prefix == "B" or entity_type != current_type:
            close(index)
            start = index
            current_type = entity_type
    return entities


def _safe_torch_load(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Evidence embedding cache must contain a mapping")
    return payload


def load_evidence_embedding_cache(
    path: str | Path,
) -> tuple[dict[int, list[int]], Mapping[str, Any]]:
    """Validate a compact tensor cache and index its entities by sample ID."""
    payload = _safe_torch_load(path)
    if payload.get("schema_version") != EVIDENCE_EMBEDDING_SCHEMA_VERSION:
        raise ValueError("Unsupported evidence embedding cache schema")
    required = {
        "teacher_cache_sha256",
        "input_sha256",
        "sample_ids",
        "starts",
        "ends",
        "label_ids",
        "image_targets",
        "confidences",
        "embeddings",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Evidence embedding cache is missing: {sorted(missing)}")
    for name in ("teacher_cache_sha256", "input_sha256"):
        value = payload[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Evidence cache {name} must be a SHA-256 digest")

    tensor_names = required - {"teacher_cache_sha256", "input_sha256"}
    tensors = {name: payload[name] for name in tensor_names}
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("Evidence embedding cache fields must be tensors")
    count = int(tensors["sample_ids"].numel())
    for name in ("starts", "ends", "label_ids", "confidences"):
        if tensors[name].numel() != count:
            raise ValueError(f"Evidence cache tensor length mismatch: {name}")
    if tensors["image_targets"].shape != (count, 4):
        raise ValueError("Evidence image_targets must have shape [N, 4]")
    embeddings = tensors["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[0] != count:
        raise ValueError("Evidence embeddings must have shape [N, D]")
    if not torch.isfinite(embeddings.float()).all():
        raise ValueError("Evidence embeddings contain non-finite values")
    if not torch.isfinite(tensors["confidences"].float()).all():
        raise ValueError("Evidence confidences contain non-finite values")

    index: dict[int, list[int]] = defaultdict(list)
    for row, sample_id in enumerate(tensors["sample_ids"].tolist()):
        sample_id = int(sample_id)
        if sample_id < 0:
            raise ValueError("Evidence sample IDs must be non-negative")
        index[sample_id].append(row)
    return dict(index), payload


def build_evidence_distillation_targets(
    *,
    words: Sequence[str],
    gold_labels: Sequence[str],
    tokenizer: Any,
    max_sequence_length: int,
    image_mask: torch.Tensor,
    artifact: Mapping[str, Any],
    row_indices: Sequence[int],
    max_entities: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build fixed-shape, gold-consistent entity/rationale alignment targets."""
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    embeddings = artifact["embeddings"]
    embedding_dim = int(embeddings.shape[1])
    token_masks = torch.zeros((max_entities, max_sequence_length), dtype=torch.float)
    image_targets = torch.zeros((max_entities, 4), dtype=torch.float)
    teacher_embeddings = torch.zeros((max_entities, embedding_dim), dtype=torch.float)
    weights = torch.zeros(max_entities, dtype=torch.float)

    word_piece_positions: list[list[int]] = []
    cursor = 1
    maximum_position = max_sequence_length - 1
    for word in words:
        pieces = tokenizer.tokenize(normalize_unicode_text(str(word)))
        positions = [
            position
            for position in range(cursor, cursor + len(pieces))
            if position < maximum_position
        ]
        word_piece_positions.append(positions)
        cursor += len(pieces)

    gold_entities = extract_evidence_bio_entities(list(gold_labels))
    valid_images = image_mask.bool().float()
    candidates: list[tuple[float, int, int, int]] = []
    for row in row_indices:
        start = int(artifact["starts"][row])
        end = int(artifact["ends"][row])
        label_id = int(artifact["label_ids"][row])
        confidence = float(artifact["confidences"][row])
        if not 0 <= label_id < len(EVIDENCE_ENTITY_TYPES):
            continue
        if (start, end, EVIDENCE_ENTITY_TYPES[label_id]) not in gold_entities:
            continue
        if not 0 <= start < end <= len(words):
            continue
        candidates.append((-confidence, start, end, row))

    accepted = 0
    for negative_confidence, start, end, row in sorted(candidates):
        if accepted >= max_entities:
            break
        positions = [
            position
            for word_index in range(start, end)
            for position in word_piece_positions[word_index]
        ]
        target = artifact["image_targets"][row].float() * valid_images
        target_mass = float(target.sum())
        if not positions or target_mass <= 0:
            continue
        target = target / target_mass
        token_masks[accepted, positions] = 1.0 / len(positions)
        image_targets[accepted] = target
        teacher_embeddings[accepted] = embeddings[row].float()
        weights[accepted] = max(-negative_confidence, 0.0)
        accepted += 1

    return token_masks, image_targets, teacher_embeddings, weights, accepted


def evidence_representation_distillation_loss(
    *,
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    token_masks: torch.Tensor,
    image_targets: torch.Tensor,
    weights: torch.Tensor,
    student_projection: Callable[[torch.Tensor], torch.Tensor],
    temperature: float,
    contrastive_weight: float,
    detach_text: bool,
    detach_image: bool,
) -> torch.Tensor:
    """Align entity-image student states with frozen LLM rationale embeddings."""
    if temperature <= 0:
        raise ValueError("Evidence temperature must be positive")
    if contrastive_weight < 0:
        raise ValueError("Evidence contrastive weight must be non-negative")
    if token_masks.shape[:2] != teacher_embeddings.shape[:2]:
        raise ValueError("Evidence token masks and embeddings must share [B, E]")
    if image_targets.shape[:2] != teacher_embeddings.shape[:2]:
        raise ValueError("Evidence image targets and embeddings must share [B, E]")
    if weights.shape != teacher_embeddings.shape[:2]:
        raise ValueError("Evidence weights must have shape [B, E]")

    text = text_features.detach() if detach_text else text_features
    images = image_features.detach() if detach_image else image_features
    entity_text = torch.bmm(token_masks.to(text.dtype), text)
    entity_images = torch.bmm(image_targets.to(images.dtype), images)
    student = student_projection(torch.cat([entity_text, entity_images], dim=-1))

    active = weights.gt(0)
    if not bool(active.any()):
        return student.sum() * 0.0
    student = F.normalize(student[active].float(), dim=-1)
    teacher = F.normalize(teacher_embeddings[active].detach().float(), dim=-1)
    active_weights = weights[active].float()
    cosine_loss = ((1.0 - (student * teacher).sum(dim=-1)) * active_weights).sum()
    cosine_loss = cosine_loss / active_weights.sum().clamp_min(1e-12)

    if student.shape[0] <= 1 or contrastive_weight == 0:
        return cosine_loss
    logits = torch.matmul(student, teacher.transpose(0, 1)) / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    student_to_teacher = F.cross_entropy(logits, labels, reduction="none")
    teacher_to_student = F.cross_entropy(logits.transpose(0, 1), labels, reduction="none")
    contrastive = (
        ((student_to_teacher + teacher_to_student) * 0.5 * active_weights).sum()
        / active_weights.sum().clamp_min(1e-12)
    )
    return cosine_loss + contrastive_weight * contrastive
