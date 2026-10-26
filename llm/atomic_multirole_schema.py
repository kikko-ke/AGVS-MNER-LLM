"""Strict V4 contracts for one-edit, one-role atomic reviews."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from llm.atomic_review_schema import AtomicEditCandidate


PROMPT_VERSION = "atomic-edit-multirole-v4"
ATOMIC_MULTIROLE_PROMPT_VERSION = PROMPT_VERSION

ROLE_TEXT = "text_verifier"
ROLE_VISUAL = "visual_verifier"
ROLE_CRITIC = "adversarial_critic"
ATOMIC_MULTIROLE_ROLES = frozenset({ROLE_TEXT, ROLE_VISUAL, ROLE_CRITIC})

VERIFIER_VERDICTS = frozenset(
    {"SUPPORT_EDIT", "SUPPORT_BASELINE", "INSUFFICIENT"}
)
CRITIC_VERDICTS = frozenset(
    {"OBJECTION_FOUND", "NO_OBJECTION", "INSUFFICIENT"}
)
EVIDENCE_TARGETS = frozenset(
    {"removed_entity", "added_entity", "both", "local_context", "none"}
)

_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_FORBIDDEN_FIELDS = frozenset(
    {"confidence", "decisive_span", "decisive_spans", "start", "end"}
)
_REVIEW_FIELDS = frozenset(
    {
        "edit_id",
        "role",
        "verdict",
        "evidence_target",
        "reason_code",
        "used_image_ids",
        "brief_evidence",
    }
)


@dataclass(frozen=True)
class AtomicMultiroleReview:
    """One validated role judgment for exactly one immutable atomic edit."""

    edit_id: str
    role: str
    verdict: str
    evidence_target: str
    reason_code: str
    used_image_ids: tuple[int, ...]
    brief_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "role": self.role,
            "verdict": self.verdict,
            "evidence_target": self.evidence_target,
            "reason_code": self.reason_code,
            "used_image_ids": list(self.used_image_ids),
            "brief_evidence": self.brief_evidence,
        }


def validate_atomic_multirole_role(role: Any) -> str:
    """Return a supported reviewer role without coercing arbitrary values."""
    if not isinstance(role, str) or role not in ATOMIC_MULTIROLE_ROLES:
        raise ValueError(f"unsupported atomic multirole reviewer role: {role!r}")
    return role


def _strict_image_ids(
    raw_image_ids: Any,
    valid_image_ids: Iterable[int],
) -> tuple[int, ...]:
    valid_values = list(valid_image_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in valid_values
    ):
        raise ValueError("valid_image_ids must contain non-negative integers")
    if len(valid_values) != len(set(valid_values)):
        raise ValueError("valid_image_ids must not contain duplicates")
    if not isinstance(raw_image_ids, list):
        raise ValueError("used_image_ids must be a list")

    valid = set(valid_values)
    result: list[int] = []
    for raw_value in raw_image_ids:
        # Some OpenAI-compatible providers serialize a JSON integer as a
        # decimal string even when the prompt shows an integer example. This
        # repair is deliberately narrow: only an unsigned ASCII decimal is
        # accepted, while names such as ``image_0`` and filenames remain
        # invalid. The normalized cache contract still stores integers.
        value = raw_value
        if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("used_image_id must be an integer")
        if value not in valid:
            raise ValueError(f"used_image_id {value} was not supplied")
        if value in result:
            raise ValueError(f"duplicate used_image_id: {value}")
        result.append(value)
    return tuple(result)


def _validate_evidence_target(
    target: Any,
    edit: AtomicEditCandidate,
) -> str:
    if not isinstance(target, str) or target not in EVIDENCE_TARGETS:
        raise ValueError(f"invalid evidence_target: {target!r}")
    if target in {"removed_entity", "both"} and edit.removed_entity is None:
        raise ValueError("evidence_target cites an absent removed_entity")
    if target in {"added_entity", "both"} and edit.added_entity is None:
        raise ValueError("evidence_target cites an absent added_entity")
    return target


def validate_atomic_multirole_output(
    payload: Mapping[str, Any],
    edit: AtomicEditCandidate,
    role: str,
    valid_image_ids: Iterable[int],
) -> AtomicMultiroleReview:
    """Validate a single role response for a single supplied atomic edit.

    The response intentionally contains no self-reported confidence and no free
    token spans. Evidence points only to immutable parts of the supplied edit.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("atomic multirole output must be a JSON object")
    role = validate_atomic_multirole_role(role)
    forbidden = _FORBIDDEN_FIELDS.intersection(payload)
    if forbidden:
        raise ValueError(
            "atomic multirole output contains forbidden fields: "
            f"{sorted(forbidden)}"
        )
    unexpected = set(payload) - _REVIEW_FIELDS
    if unexpected:
        raise ValueError(
            f"atomic multirole output contains unexpected fields: {sorted(unexpected)}"
        )
    missing = _REVIEW_FIELDS - set(payload)
    if missing:
        raise ValueError(
            f"atomic multirole output is missing fields: {sorted(missing)}"
        )

    if payload.get("edit_id") != edit.edit_id:
        raise ValueError("review edit_id does not match the supplied atomic edit")
    if payload.get("role") != role:
        raise ValueError("review role does not match the requested reviewer role")

    verdict = payload.get("verdict")
    allowed_verdicts = CRITIC_VERDICTS if role == ROLE_CRITIC else VERIFIER_VERDICTS
    if not isinstance(verdict, str) or verdict not in allowed_verdicts:
        raise ValueError(f"invalid verdict for role={role}: {verdict!r}")

    target = _validate_evidence_target(payload.get("evidence_target"), edit)
    reason_code = payload.get("reason_code")
    if not isinstance(reason_code, str) or not _REASON_CODE_RE.fullmatch(reason_code):
        raise ValueError(
            "reason_code must be a 2..64 character lowercase snake_case code"
        )

    image_ids = _strict_image_ids(payload.get("used_image_ids"), valid_image_ids)
    if role == ROLE_TEXT and image_ids:
        raise ValueError("text_verifier cannot cite images")
    if role == ROLE_VISUAL:
        if verdict in {"SUPPORT_EDIT", "SUPPORT_BASELINE"} and not image_ids:
            raise ValueError("visual support verdict requires used_image_ids")
        if verdict == "INSUFFICIENT" and image_ids:
            raise ValueError("visual INSUFFICIENT cannot cite used_image_ids")

    if role != ROLE_CRITIC and verdict == "INSUFFICIENT" and target != "none":
        raise ValueError("verifier INSUFFICIENT requires evidence_target=none")
    if role != ROLE_CRITIC and verdict != "INSUFFICIENT" and target == "none":
        raise ValueError("support verdict requires a non-none evidence_target")
    if role == ROLE_CRITIC and verdict == "OBJECTION_FOUND" and target == "none":
        raise ValueError("OBJECTION_FOUND requires a non-none evidence_target")
    if role == ROLE_CRITIC and verdict == "NO_OBJECTION" and target != "none":
        raise ValueError("NO_OBJECTION requires evidence_target=none")
    if role == ROLE_CRITIC and verdict == "INSUFFICIENT" and target != "none":
        raise ValueError("critic INSUFFICIENT requires evidence_target=none")

    brief_evidence = payload.get("brief_evidence")
    if not isinstance(brief_evidence, str) or not brief_evidence.strip():
        raise ValueError("brief_evidence must be a non-empty string")
    brief_evidence = brief_evidence.strip()
    if len(brief_evidence) > 1000:
        raise ValueError("brief_evidence is too long")

    return AtomicMultiroleReview(
        edit_id=edit.edit_id,
        role=role,
        verdict=verdict,
        evidence_target=target,
        reason_code=reason_code,
        used_image_ids=image_ids,
        brief_evidence=brief_evidence,
    )
