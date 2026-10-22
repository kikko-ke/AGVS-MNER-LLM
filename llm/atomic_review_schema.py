"""Strict contracts for independent LLM reviews of atomic CRF edits."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from llm.schema import ENTITY_TYPES
from modules.crf_edit_decomposition import (
    ADDITION,
    BOUNDARY_CHANGE,
    DELETION,
    TYPE_CHANGE,
)


ATOMIC_EDIT_CLASSES = {DELETION, ADDITION, TYPE_CHANGE, BOUNDARY_CHANGE}
ATOMIC_REVIEW_VERDICTS = {"KEEP_BASELINE", "APPLY_EDIT", "ABSTAIN"}
ATOMIC_EVIDENCE_SUPPORT = {"baseline", "edit", "conflict", "none"}
_EDIT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class AtomicReviewEntity:
    start: int
    end: int
    label: str
    surface: str


@dataclass(frozen=True)
class AtomicEditCandidate:
    edit_id: str
    edit_class: str
    removed_entity: AtomicReviewEntity | None
    added_entity: AtomicReviewEntity | None

    @property
    def changed_spans(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (entity.start, entity.end)
            for entity in (self.removed_entity, self.added_entity)
            if entity is not None
        )


@dataclass(frozen=True)
class AtomicDecisiveSpan:
    start: int
    end: int
    evidence: str = ""


@dataclass(frozen=True)
class AtomicEditReview:
    edit_id: str
    verdict: str
    confidence: float
    text_support: str
    visual_support: str
    used_image_ids: tuple[int, ...]
    decisive_spans: tuple[AtomicDecisiveSpan, ...]
    brief_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "text_support": self.text_support,
            "visual_support": self.visual_support,
            "used_image_ids": list(self.used_image_ids),
            "decisive_spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "evidence": span.evidence,
                }
                for span in self.decisive_spans
            ],
            "brief_reason": self.brief_reason,
        }


@dataclass(frozen=True)
class AtomicReviewResponse:
    reviews: tuple[AtomicEditReview, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"reviews": [review.to_dict() for review in self.reviews]}


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _validate_entity(
    raw_entity: Any,
    words: Sequence[str],
    field: str,
) -> AtomicReviewEntity | None:
    if raw_entity is None:
        return None
    if not isinstance(raw_entity, Mapping):
        raise ValueError(f"{field} must be an entity object or null")
    start = _strict_int(raw_entity.get("start"), f"{field}.start")
    end = _strict_int(raw_entity.get("end"), f"{field}.end")
    if start < 0 or end <= start or end > len(words):
        raise ValueError(f"invalid {field} span [{start}, {end})")
    label = str(raw_entity.get("label", "")).upper().replace("MISC", "MIS")
    if label.startswith(("B-", "I-")):
        label = label[2:]
    if label not in ENTITY_TYPES:
        raise ValueError(f"unknown {field} entity type: {label}")
    return AtomicReviewEntity(
        start=start,
        end=end,
        label=label,
        surface=" ".join(str(word) for word in words[start:end]),
    )


def _overlaps(left: AtomicReviewEntity, right: AtomicReviewEntity) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def validate_atomic_baseline_entities(
    entities: Iterable[Mapping[str, Any]],
    words: Sequence[str],
) -> tuple[AtomicReviewEntity, ...]:
    """Validate the immutable top-1 entity set shown with atomic edits."""
    words = [str(word) for word in words]
    result: list[AtomicReviewEntity] = []
    previous_end = 0
    for raw_entity in entities:
        entity = _validate_entity(raw_entity, words, "baseline_entity")
        if entity is None:
            raise ValueError("baseline entities must be entity objects")
        if entity.start < previous_end:
            raise ValueError("baseline entities must be ordered and non-overlapping")
        result.append(entity)
        previous_end = entity.end
    return tuple(result)


def validate_atomic_edit_candidates(
    edits: Iterable[Mapping[str, Any]],
    words: Sequence[str],
) -> tuple[AtomicEditCandidate, ...]:
    """Validate strict single-entity edits and recompute all surface strings."""
    words = [str(word) for word in words]
    result: list[AtomicEditCandidate] = []
    seen_ids: set[str] = set()
    for raw_edit in edits:
        if not isinstance(raw_edit, Mapping):
            raise ValueError("each atomic edit must be an object")
        edit_id = raw_edit.get("edit_id")
        if not isinstance(edit_id, str) or not _EDIT_ID_RE.fullmatch(edit_id):
            raise ValueError("edit_id must be an opaque 3..64 character ID")
        if edit_id in seen_ids:
            raise ValueError(f"duplicate edit_id: {edit_id}")
        seen_ids.add(edit_id)

        edit_class = raw_edit.get("edit_class")
        if edit_class not in ATOMIC_EDIT_CLASSES:
            raise ValueError(f"unsupported atomic edit class: {edit_class}")
        removed = _validate_entity(raw_edit.get("removed_entity"), words, "removed_entity")
        added = _validate_entity(raw_edit.get("added_entity"), words, "added_entity")

        if edit_class == DELETION:
            valid = removed is not None and added is None
        elif edit_class == ADDITION:
            valid = removed is None and added is not None
        elif edit_class == TYPE_CHANGE:
            valid = (
                removed is not None
                and added is not None
                and (removed.start, removed.end) == (added.start, added.end)
                and removed.label != added.label
            )
        else:
            valid = (
                removed is not None
                and added is not None
                and removed.label == added.label
                and (removed.start, removed.end) != (added.start, added.end)
                and _overlaps(removed, added)
            )
        if not valid:
            raise ValueError(f"{edit_id} does not satisfy its strict {edit_class} contract")
        result.append(AtomicEditCandidate(edit_id, edit_class, removed, added))

    if not result:
        raise ValueError("at least one atomic edit is required")
    return tuple(result)


def validate_atomic_edits_against_baseline(
    edits: Sequence[AtomicEditCandidate],
    baseline_entities: Sequence[AtomicReviewEntity],
) -> None:
    """Require every edit to describe a real change to the supplied baseline."""
    baseline = {
        (entity.start, entity.end, entity.label) for entity in baseline_entities
    }
    for edit in edits:
        removed = edit.removed_entity
        if removed is not None:
            identity = (removed.start, removed.end, removed.label)
            if identity not in baseline:
                raise ValueError(
                    f"{edit.edit_id} removed_entity is not present in baseline_entities"
                )
        added = edit.added_entity
        if added is not None:
            identity = (added.start, added.end, added.label)
            if identity in baseline:
                raise ValueError(
                    f"{edit.edit_id} added_entity is already present in baseline_entities"
                )


def validate_atomic_review_output(
    payload: Mapping[str, Any],
    edits: Sequence[AtomicEditCandidate],
    valid_image_ids: Iterable[int],
    *,
    token_count: int,
) -> AtomicReviewResponse:
    """Validate one independent review per supplied edit without repairing output."""
    if not isinstance(payload, Mapping):
        raise ValueError("atomic review output must be a JSON object")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count must be a non-negative integer")
    expected = {edit.edit_id: edit for edit in edits}
    if len(expected) != len(edits) or not expected:
        raise ValueError("supplied atomic edits must have unique IDs")
    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, list):
        raise ValueError("reviews must be a list")

    valid_image_values = list(valid_image_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in valid_image_values
    ):
        raise ValueError("valid_image_ids must contain non-negative integers")
    valid_images = set(valid_image_values)
    reviews_by_id: dict[str, AtomicEditReview] = {}
    for raw_review in raw_reviews:
        if not isinstance(raw_review, Mapping):
            raise ValueError("each atomic edit review must be an object")
        edit_id = raw_review.get("edit_id")
        if not isinstance(edit_id, str) or edit_id not in expected:
            raise ValueError("review edit_id was not supplied")
        if edit_id in reviews_by_id:
            raise ValueError(f"duplicate review edit_id: {edit_id}")

        verdict = raw_review.get("verdict")
        if verdict not in ATOMIC_REVIEW_VERDICTS:
            raise ValueError(f"invalid verdict for edit_id={edit_id}")
        raw_confidence = raw_review.get("confidence")
        if isinstance(raw_confidence, bool):
            raise ValueError("confidence must be a finite number in [0, 1]")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as error:
            raise ValueError("confidence must be a finite number in [0, 1]") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite number in [0, 1]")

        text_support = raw_review.get("text_support")
        visual_support = raw_review.get("visual_support")
        if text_support not in ATOMIC_EVIDENCE_SUPPORT:
            raise ValueError(f"invalid text_support for edit_id={edit_id}")
        if visual_support not in ATOMIC_EVIDENCE_SUPPORT:
            raise ValueError(f"invalid visual_support for edit_id={edit_id}")

        raw_image_ids = raw_review.get("used_image_ids", [])
        if not isinstance(raw_image_ids, list):
            raise ValueError("used_image_ids must be a list")
        image_ids: list[int] = []
        for raw_image_id in raw_image_ids:
            image_id = _strict_int(raw_image_id, "used_image_id")
            if image_id not in valid_images:
                raise ValueError(f"used_image_id {image_id} was not supplied")
            if image_id in image_ids:
                raise ValueError(f"duplicate used_image_id: {image_id}")
            image_ids.append(image_id)
        if visual_support == "none" and image_ids:
            raise ValueError("visual_support=none cannot cite used_image_ids")
        if visual_support != "none" and not image_ids:
            raise ValueError("visual evidence support requires used_image_ids")

        raw_spans = raw_review.get("decisive_spans", [])
        if not isinstance(raw_spans, list):
            raise ValueError("decisive_spans must be a list")
        allowed_spans = expected[edit_id].changed_spans
        decisive_spans: list[AtomicDecisiveSpan] = []
        seen_spans: set[tuple[int, int]] = set()
        for raw_span in raw_spans:
            if not isinstance(raw_span, Mapping):
                raise ValueError("each decisive span must be an object")
            start = _strict_int(raw_span.get("start"), "decisive span start")
            end = _strict_int(raw_span.get("end"), "decisive span end")
            if start < 0 or end <= start or end > token_count:
                raise ValueError(f"invalid decisive span [{start}, {end})")
            if (start, end) not in allowed_spans:
                raise ValueError("decisive span is not part of the reviewed atomic edit")
            if (start, end) in seen_spans:
                raise ValueError(f"duplicate decisive span [{start}, {end})")
            seen_spans.add((start, end))
            decisive_spans.append(
                AtomicDecisiveSpan(
                    start,
                    end,
                    str(raw_span.get("evidence", "")).strip(),
                )
            )
        if verdict == "APPLY_EDIT" and not decisive_spans:
            raise ValueError("APPLY_EDIT requires a decisive changed span")

        reason = raw_review.get("brief_reason", raw_review.get("reason"))
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("brief_reason must be a non-empty string")
        reason = reason.strip()
        if len(reason) > 1000:
            raise ValueError("brief_reason is too long")

        reviews_by_id[edit_id] = AtomicEditReview(
            edit_id=edit_id,
            verdict=verdict,
            confidence=confidence,
            text_support=text_support,
            visual_support=visual_support,
            used_image_ids=tuple(image_ids),
            decisive_spans=tuple(decisive_spans),
            brief_reason=reason,
        )

    missing = set(expected) - set(reviews_by_id)
    if missing:
        raise ValueError(f"reviews do not cover every supplied edit: missing={sorted(missing)}")
    return AtomicReviewResponse(
        tuple(reviews_by_id[edit.edit_id] for edit in edits)
    )
