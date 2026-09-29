"""Structured output contract and soft-label conversion for the LLM teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

ENTITY_TYPES = ("MIS", "PER", "ORG", "LOC")
MAX_IMAGES = 4


@dataclass(frozen=True)
class EntityEvidence:
    start: int
    end: int
    label: str
    image_ids: tuple[int, ...]
    confidence: float
    evidence: str = ""


@dataclass(frozen=True)
class TeacherOutput:
    entities: tuple[EntityEvidence, ...]
    image_scores: tuple[float, ...]
    raw_text: str = ""


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error


def _normalise_label(value: Any) -> str:
    label = str(value).upper().replace("MISC", "MIS")
    if label.startswith("B-") or label.startswith("I-"):
        label = label[2:]
    if label not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type: {label}")
    return label


def _normalise_scores(value: Any, num_images: int = MAX_IMAGES) -> tuple[float, ...]:
    if value is None:
        scores = [1.0 if index == 0 else 0.0 for index in range(num_images)]
    else:
        if not isinstance(value, (list, tuple)):
            raise ValueError("image_scores must be a list")
        scores = [float(item) for item in value[:num_images]]
        scores.extend([0.0] * (num_images - len(scores)))
    scores = [max(score, 0.0) for score in scores]
    total = sum(scores)
    if total <= 0.0:
        return tuple([1.0 / num_images] * num_images)
    return tuple(score / total for score in scores)


def validate_teacher_output(
    payload: dict[str, Any],
    token_count: int,
    num_images: int = MAX_IMAGES,
    raw_text: str = "",
) -> TeacherOutput:
    """Validate a teacher JSON object and return a typed representation."""
    if not isinstance(payload, dict):
        raise ValueError("teacher output must be a JSON object")
    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("entities must be a list")

    entities: list[EntityEvidence] = []
    for raw_entity in raw_entities:
        if not isinstance(raw_entity, dict):
            raise ValueError("each entity must be an object")
        start = _as_int(raw_entity.get("start"), "start")
        end = _as_int(raw_entity.get("end"), "end")
        if start < 0 or end <= start or end > token_count:
            raise ValueError(f"invalid entity span [{start}, {end}) for {token_count} tokens")
        image_ids_raw = raw_entity.get("image_ids", [])
        if not isinstance(image_ids_raw, list):
            raise ValueError("image_ids must be a list")
        image_ids = tuple(
            sorted(
                {
                    image_id
                    for image_id in (_as_int(item, "image_id") for item in image_ids_raw)
                    if 0 <= image_id < num_images
                }
            )
        )
        confidence = min(max(float(raw_entity.get("confidence", 1.0)), 0.0), 1.0)
        entities.append(
            EntityEvidence(
                start=start,
                end=end,
                label=_normalise_label(raw_entity.get("label")),
                image_ids=image_ids,
                confidence=confidence,
                evidence=str(raw_entity.get("evidence", "")),
            )
        )

    return TeacherOutput(
        entities=tuple(entities),
        image_scores=_normalise_scores(payload.get("image_scores"), num_images),
        raw_text=raw_text,
    )


def teacher_to_soft_labels(
    teacher: TeacherOutput,
    words: Sequence[str],
    tokenizer: Any,
    max_seq_length: int,
    label_mapping: dict[str, int],
) -> tuple[list[list[float]], list[float]]:
    """Convert word-level teacher spans to fixed-size token probabilities."""
    num_labels = len(label_mapping)
    base = 0.02
    probs = [[base] * num_labels for _ in range(max_seq_length)]
    for row in probs:
        row[label_mapping["O"]] += 0.90
        normaliser = sum(row)
        for index in range(len(row)):
            row[index] /= normaliser

    word_pieces: list[list[str]] = [tokenizer.tokenize(word) for word in words]
    token_positions: list[list[int]] = []
    cursor = 1
    for pieces in word_pieces:
        positions = list(range(cursor, cursor + len(pieces)))
        token_positions.append(positions)
        cursor += len(pieces)
    max_position = max_seq_length - 1

    for position in [p for group in token_positions for p in group]:
        if position >= max_position:
            continue
        for label_index in range(num_labels):
            probs[position][label_index] = base
        probs[position][label_mapping["O"]] = 0.1
        normaliser = sum(probs[position])
        for label_index in range(num_labels):
            probs[position][label_index] /= normaliser

    for entity in teacher.entities:
        for word_index in range(entity.start, min(entity.end, len(token_positions))):
            positions = token_positions[word_index]
            for piece_index, position in enumerate(positions):
                if position >= max_position:
                    continue
                prefix = "B" if word_index == entity.start and piece_index == 0 else "I"
                label_name = f"{prefix}-{entity.label}"
                if label_name not in label_mapping:
                    continue
                for label_index in range(num_labels):
                    probs[position][label_index] = base
                probs[position][label_mapping[label_name]] = max(entity.confidence, 0.5)
                normaliser = sum(probs[position])
                for label_index in range(num_labels):
                    probs[position][label_index] /= normaliser

    return probs, list(teacher.image_scores)
