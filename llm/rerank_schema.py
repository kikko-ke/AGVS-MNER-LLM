"""Contracts for selecting one member of a fixed CRF candidate set."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from llm.schema import ENTITY_TYPES


_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,63}$")
BASELINE_ERROR_TYPES = {
    "none",
    "boundary",
    "entity_type",
    "missing_entity",
    "spurious_entity",
}


@dataclass(frozen=True)
class RerankEntity:
    start: int
    end: int
    label: str
    surface: str


@dataclass(frozen=True)
class RerankCandidate:
    candidate_id: str
    entities: tuple[RerankEntity, ...]


@dataclass(frozen=True)
class DecisiveSpan:
    start: int
    end: int
    evidence: str = ""


@dataclass(frozen=True)
class RerankDecision:
    selected_candidate_id: str
    confidence: float
    uncertain: bool
    baseline_error_type: str
    used_image_ids: tuple[int, ...]
    decisive_spans: tuple[DecisiveSpan, ...]
    brief_reason: str

    @property
    def reason(self) -> str:
        """Readable alias used by cache/reporting code."""
        return self.brief_reason

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON payload stored by resumable request scripts."""
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "confidence": self.confidence,
            "uncertain": self.uncertain,
            "baseline_error_type": self.baseline_error_type,
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


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_rerank_candidates(
    candidates: Iterable[Mapping[str, Any]],
    words: Sequence[str],
) -> tuple[RerankCandidate, ...]:
    """Validate and reduce request candidates to fields safe to show the LLM."""
    result: list[RerankCandidate] = []
    seen_ids: set[str] = set()
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("each rerank candidate must be an object")
        candidate_id = raw_candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(
            candidate_id
        ):
            raise ValueError(
                "candidate_id must be an opaque 3..64 character ID"
            )
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)

        raw_entities = raw_candidate.get("entities", [])
        if not isinstance(raw_entities, list):
            raise ValueError("candidate entities must be a list")
        entities: list[RerankEntity] = []
        previous_end = 0
        for raw_entity in raw_entities:
            if not isinstance(raw_entity, Mapping):
                raise ValueError("each candidate entity must be an object")
            start = _strict_int(raw_entity.get("start"), "entity start")
            end = _strict_int(raw_entity.get("end"), "entity end")
            if start < previous_end or end <= start or end > len(words):
                raise ValueError(
                    f"invalid or overlapping candidate span [{start}, {end})"
                )
            label = str(raw_entity.get("label", "")).upper().replace("MISC", "MIS")
            if label.startswith(("B-", "I-")):
                label = label[2:]
            if label not in ENTITY_TYPES:
                raise ValueError(f"unknown entity type: {label}")
            surface = " ".join(str(word) for word in words[start:end])
            entities.append(
                RerankEntity(
                    start=start,
                    end=end,
                    label=label,
                    surface=surface,
                )
            )
            previous_end = end
        result.append(RerankCandidate(candidate_id, tuple(entities)))

    if len(result) < 2:
        raise ValueError("reranking requires at least two candidates")
    return tuple(result)


def validate_rerank_output(
    payload: Mapping[str, Any],
    allowed_candidate_ids: Iterable[str],
    valid_image_ids: Iterable[int],
    *,
    token_count: int | None = None,
    candidate_spans: Mapping[str, Iterable[tuple[int, int]]] | None = None,
    baseline_candidate_id: str | None = None,
) -> RerankDecision:
    """Validate a model response without repairing an out-of-contract choice."""
    if not isinstance(payload, Mapping):
        raise ValueError("reranker output must be a JSON object")
    allowed_ids = set(allowed_candidate_ids)
    selected_id = payload.get("selected_candidate_id")
    if not isinstance(selected_id, str) or selected_id not in allowed_ids:
        raise ValueError("selected_candidate_id is not one of the supplied candidates")
    if baseline_candidate_id is None or baseline_candidate_id not in allowed_ids:
        raise ValueError("baseline_candidate_id is not one of the supplied candidates")

    raw_confidence = payload.get("confidence")
    if isinstance(raw_confidence, bool):
        raise ValueError("confidence must be a finite number in [0, 1]")
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("confidence must be a finite number in [0, 1]") from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a finite number in [0, 1]")

    uncertain = payload.get("uncertain")
    if not isinstance(uncertain, bool):
        raise ValueError("uncertain must be a boolean")

    baseline_error_type = payload.get("baseline_error_type")
    if baseline_error_type not in BASELINE_ERROR_TYPES:
        raise ValueError("baseline_error_type is invalid")
    if selected_id == baseline_candidate_id:
        if baseline_error_type != "none":
            raise ValueError("baseline selection requires baseline_error_type=none")
    else:
        if baseline_error_type == "none":
            raise ValueError("alternative selection requires a concrete baseline error")
        if uncertain:
            raise ValueError("an uncertain decision must keep the baseline candidate")

    raw_image_ids = payload.get("used_image_ids", [])
    if not isinstance(raw_image_ids, list):
        raise ValueError("used_image_ids must be a list")
    valid_images = set(valid_image_ids)
    image_ids: list[int] = []
    for raw_image_id in raw_image_ids:
        image_id = _strict_int(raw_image_id, "used_image_id")
        if image_id not in valid_images:
            raise ValueError(f"used_image_id {image_id} was not supplied")
        if image_id in image_ids:
            raise ValueError(f"duplicate used_image_id: {image_id}")
        image_ids.append(image_id)

    raw_spans = payload.get("decisive_spans", [])
    if not isinstance(raw_spans, list):
        raise ValueError("decisive_spans must be a list")
    allowed_spans = None
    if candidate_spans is not None:
        # A residual verifier may cite either the replacement span or the
        # baseline span that it removes/corrects.  Both are fixed candidate
        # content; arbitrary locations remain forbidden.
        allowed_spans = set(candidate_spans.get(selected_id, ()))
        allowed_spans.update(candidate_spans.get(baseline_candidate_id, ()))
    decisive_spans: list[DecisiveSpan] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("each decisive span must be an object")
        start = _strict_int(raw_span.get("start"), "decisive span start")
        end = _strict_int(raw_span.get("end"), "decisive span end")
        if start < 0 or end <= start or (
            token_count is not None and end > token_count
        ):
            raise ValueError(f"invalid decisive span [{start}, {end})")
        if allowed_spans is not None and (start, end) not in allowed_spans:
            raise ValueError(
                "decisive span is not in the baseline or selected candidate"
            )
        decisive_spans.append(
            DecisiveSpan(start, end, str(raw_span.get("evidence", "")).strip())
        )

    reason = payload.get("brief_reason", payload.get("reason"))
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("brief_reason must be a non-empty string")
    reason = reason.strip()
    if len(reason) > 1000:
        raise ValueError("brief_reason is too long")

    return RerankDecision(
        selected_candidate_id=selected_id,
        confidence=confidence,
        uncertain=uncertain,
        baseline_error_type=baseline_error_type,
        used_image_ids=tuple(image_ids),
        decisive_spans=tuple(decisive_spans),
        brief_reason=reason,
    )
