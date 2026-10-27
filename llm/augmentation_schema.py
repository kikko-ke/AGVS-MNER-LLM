"""Strict contract for label-preserving multimodal training augmentation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


AUGMENTATION_PROMPT_VERSION = "multimodal-locked-context-segments-v2"
AUGMENTATION_SCHEMA_VERSION = "multimodal-locked-context-segments-cache-v2"
ENTITY_TYPES = frozenset({"LOC", "MIS", "ORG", "PER"})
_URL_RE = re.compile(r"(?i)(?:https?://|www\.|\S+\.com(?:/|$))")


@dataclass(frozen=True)
class ProtectedEntity:
    """One immutable gold entity from a training-only source example."""

    entity_id: str
    start: int
    end: int
    label: str
    tokens: tuple[str, ...]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "tokens": list(self.tokens),
        }


@dataclass(frozen=True)
class AugmentedEntitySpan:
    entity_id: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class AugmentationResponse:
    """Validated response; abstentions are valid cache records but not examples."""

    abstain: bool
    context_segments: tuple[tuple[str, ...], ...]
    tokens: tuple[str, ...]
    entity_spans: tuple[AugmentedEntitySpan, ...]
    used_image_ids: tuple[int, ...]
    change_summary: str
    abstain_reason: str
    context_change_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstain": self.abstain,
            "context_segments": [list(segment) for segment in self.context_segments],
            "tokens": list(self.tokens),
            "entity_spans": [span.to_dict() for span in self.entity_spans],
            "used_image_ids": list(self.used_image_ids),
            "change_summary": self.change_summary,
            "abstain_reason": self.abstain_reason,
            "context_change_ratio": self.context_change_ratio,
        }


def _normalise_entity_type(value: Any) -> str:
    label = str(value).upper().replace("MISC", "MIS")
    if label.startswith(("B-", "I-")):
        label = label[2:]
    if label not in ENTITY_TYPES:
        raise ValueError(f"unsupported entity type: {label}")
    return label


def extract_protected_entities(
    words: Sequence[str], labels: Sequence[str]
) -> tuple[ProtectedEntity, ...]:
    """Parse a strict word-level BIO sequence into immutable entities."""
    if len(words) != len(labels):
        raise ValueError("source words and labels must have equal length")
    entities: list[ProtectedEntity] = []
    index = 0
    while index < len(labels):
        raw_label = str(labels[index]).upper().replace("MISC", "MIS")
        if raw_label == "O":
            index += 1
            continue
        if not raw_label.startswith("B-"):
            raise ValueError(f"invalid BIO label {labels[index]!r} at word {index}")
        entity_type = _normalise_entity_type(raw_label)
        end = index + 1
        while end < len(labels):
            next_label = str(labels[end]).upper().replace("MISC", "MIS")
            if next_label != f"I-{entity_type}":
                break
            end += 1
        entities.append(
            ProtectedEntity(
                entity_id=f"E{len(entities)}",
                start=index,
                end=end,
                label=entity_type,
                tokens=tuple(str(word) for word in words[index:end]),
            )
        )
        index = end
    return tuple(entities)


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _validated_context_segment(value: Any, segment_index: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"context segment {segment_index} must be a list")
    tokens: list[str] = []
    for index, token in enumerate(value):
        if not isinstance(token, str):
            raise ValueError(
                f"context segment {segment_index} token {index} must be a string"
            )
        if not token or token.strip() != token or any(char.isspace() for char in token):
            raise ValueError(
                f"context segment {segment_index} token {index} must be one "
                "non-empty whitespace-free token"
            )
        if len(token) > 80:
            raise ValueError(
                f"context segment {segment_index} token {index} is unreasonably long"
            )
        # BERT is uncased in this project.  Canonicalising generated context avoids
        # treating ordinary sentence/title capitalisation as a new entity signal;
        # protected entities are inserted separately and retain their exact surface.
        tokens.append(token if token == "I" else token.casefold())
    return tuple(tokens)


def _context_tokens(
    words: Sequence[str], spans: Sequence[tuple[int, int]]
) -> tuple[str, ...]:
    protected = {
        index
        for start, end in spans
        for index in range(start, end)
    }
    return tuple(
        str(word).casefold() for index, word in enumerate(words) if index not in protected
    )


def _context_change_ratio(source: Sequence[str], augmented: Sequence[str]) -> float:
    return 1.0 - SequenceMatcher(a=tuple(source), b=tuple(augmented)).ratio()


def _suspicious_context_tokens(
    segments: Sequence[Sequence[str]],
) -> list[str]:
    return [
        token
        for segment in segments
        for token in segment
        if token.startswith(
            ("@", "#", "tokens_before_", "tokens_between_", "tokens_after_")
        )
        or _URL_RE.search(token)
    ]


def _contains_protected_surface(
    segments: Sequence[Sequence[str]], entities: Sequence[ProtectedEntity]
) -> str | None:
    """Prevent a context segment from duplicating any protected entity."""
    for entity in entities:
        needle = tuple(token.casefold() for token in entity.tokens)
        for segment in segments:
            haystack = tuple(token.casefold() for token in segment)
            width = len(needle)
            if width and any(
                haystack[index : index + width] == needle
                for index in range(len(haystack) - width + 1)
            ):
                return entity.entity_id
    return None


def validate_augmentation_output(
    payload: Mapping[str, Any],
    source_words: Sequence[str],
    protected_entities: Sequence[ProtectedEntity],
    valid_image_ids: set[int],
    *,
    min_context_change_ratio: float = 0.15,
    max_context_change_ratio: float = 0.85,
    min_length_ratio: float = 0.60,
    max_length_ratio: float = 1.50,
    max_words: int = 96,
) -> AugmentationResponse:
    """Build locked entities around validated, image-grounded context segments."""
    if not isinstance(payload, Mapping):
        raise ValueError("augmentation output must be a JSON object")
    abstain = payload.get("abstain")
    if not isinstance(abstain, bool):
        raise ValueError("abstain must be a boolean")
    abstain_reason = str(payload.get("abstain_reason", "")).strip()
    if abstain:
        if not abstain_reason:
            raise ValueError("an abstention must include abstain_reason")
        for field in ("context_segments", "used_image_ids"):
            if payload.get(field, []) not in (None, []):
                raise ValueError(f"an abstention must leave {field} empty")
        return AugmentationResponse(
            abstain=True,
            context_segments=(),
            tokens=(),
            entity_spans=(),
            used_image_ids=(),
            change_summary="",
            abstain_reason=abstain_reason[:240],
            context_change_ratio=None,
        )

    if not protected_entities:
        raise ValueError("accepted augmentation requires at least one protected entity")
    if not valid_image_ids:
        raise ValueError("accepted augmentation requires a decodable image")
    raw_segments = payload.get("context_segments")
    expected_segment_count = len(protected_entities) + 1
    if not isinstance(raw_segments, list) or len(raw_segments) != expected_segment_count:
        raise ValueError(
            f"context_segments must contain exactly {expected_segment_count} lists"
        )
    context_segments = tuple(
        _validated_context_segment(value, index)
        for index, value in enumerate(raw_segments)
    )
    duplicated_entity = _contains_protected_surface(
        context_segments, protected_entities
    )
    if duplicated_entity is not None:
        raise ValueError(
            f"context segments duplicated protected entity {duplicated_entity}"
        )

    tokens: list[str] = []
    spans: list[AugmentedEntitySpan] = []
    for segment, entity in zip(context_segments, protected_entities):
        tokens.extend(segment)
        start = len(tokens)
        tokens.extend(entity.tokens)
        spans.append(
            AugmentedEntitySpan(entity.entity_id, start, len(tokens))
        )
    tokens.extend(context_segments[-1])
    if not tokens:
        raise ValueError("accepted augmentation must not be empty")
    if len(tokens) > max_words:
        raise ValueError(f"augmentation exceeds max_words={max_words}")

    raw_image_ids = payload.get("used_image_ids")
    if not isinstance(raw_image_ids, list) or not raw_image_ids:
        raise ValueError("accepted augmentation must cite at least one used_image_id")
    image_ids = tuple(_strict_int(value, "used_image_id") for value in raw_image_ids)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("used_image_ids must not contain duplicates")
    if not set(image_ids).issubset(valid_image_ids):
        raise ValueError("used_image_ids contains an image that was not sent")

    length_ratio = len(tokens) / max(len(source_words), 1)
    if not math.isfinite(length_ratio) or not min_length_ratio <= length_ratio <= max_length_ratio:
        raise ValueError(
            f"augmentation length ratio {length_ratio:.3f} is outside "
            f"[{min_length_ratio}, {max_length_ratio}]"
        )
    source_context = _context_tokens(
        source_words,
        [(entity.start, entity.end) for entity in protected_entities],
    )
    augmented_context = _context_tokens(
        tokens,
        [(span.start, span.end) for span in spans],
    )
    change_ratio = _context_change_ratio(source_context, augmented_context)
    if not min_context_change_ratio <= change_ratio <= max_context_change_ratio:
        raise ValueError(
            f"context change ratio {change_ratio:.3f} is outside "
            f"[{min_context_change_ratio}, {max_context_change_ratio}]"
        )
    suspicious = _suspicious_context_tokens(context_segments)
    if suspicious:
        raise ValueError(
            "augmentation introduced suspicious named context tokens: "
            + ", ".join(repr(token) for token in suspicious[:5])
        )
    change_summary = str(payload.get("change_summary", "")).strip()
    if not change_summary:
        raise ValueError("accepted augmentation must include change_summary")
    if abstain_reason:
        raise ValueError("accepted augmentation must leave abstain_reason empty")
    return AugmentationResponse(
        abstain=False,
        context_segments=context_segments,
        tokens=tuple(tokens),
        entity_spans=tuple(spans),
        used_image_ids=tuple(sorted(image_ids)),
        change_summary=change_summary[:240],
        abstain_reason="",
        context_change_ratio=change_ratio,
    )


def labels_for_augmentation(
    token_count: int,
    protected_entities: Sequence[ProtectedEntity],
    spans: Sequence[AugmentedEntitySpan],
) -> list[str]:
    """Build word-level BIO labels exclusively from locked source entities."""
    if len(protected_entities) != len(spans):
        raise ValueError("protected entity and span counts differ")
    labels = ["O"] * token_count
    for entity, span in zip(protected_entities, spans):
        if span.entity_id != entity.entity_id:
            raise ValueError("entity ID mismatch while building labels")
        labels[span.start] = f"B-{entity.label}"
        for index in range(span.start + 1, span.end):
            labels[index] = f"I-{entity.label}"
    return labels
