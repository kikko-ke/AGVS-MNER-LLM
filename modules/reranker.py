"""Deterministic, gold-free helpers for constrained CRF candidate reranking.

The helpers in this module deliberately operate on JSON-like values.  They do
not load a model, a dataset label, or an LLM client, which keeps candidate
generation/request preparation separate from gold-label evaluation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ENTITY_TYPES = {"LOC", "MIS", "ORG", "PER"}
FINGERPRINT_FIELDS = {"candidate_fingerprint", "fingerprint"}
RERANK_POLICY_SCHEMA_VERSION = "crf-rerank-policy-v1"


def _normalized_label(raw_label: Any) -> tuple[str, str] | None:
    label = str(raw_label).upper().replace("MISC", "MIS")
    if "-" not in label:
        return None
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"} or entity_type not in ENTITY_TYPES:
        return None
    return prefix, entity_type


def _entity_spans(labels: Sequence[str]) -> list[tuple[int, int, str]]:
    """Extract spans while treating a malformed ``I-X`` as a new ``B-X``."""
    entities: list[tuple[int, int, str]] = []
    start: int | None = None
    current_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, current_type
        if start is not None and current_type is not None:
            entities.append((start, end, current_type))
        start = None
        current_type = None

    for index, raw_label in enumerate([*labels, "O"]):
        normalized = _normalized_label(raw_label)
        if normalized is None:
            close(index)
            continue
        prefix, entity_type = normalized
        if prefix == "B" or entity_type != current_type:
            close(index)
            start = index
            current_type = entity_type

    return entities


def canonical_bio(labels: Sequence[str]) -> list[str]:
    """Project arbitrary BIO-like labels to a legal, normalized BIO sequence."""
    canonical = ["O"] * len(labels)
    for start, end, entity_type in _entity_spans(labels):
        canonical[start] = f"B-{entity_type}"
        for position in range(start + 1, end):
            canonical[position] = f"I-{entity_type}"
    return canonical


def serialize_entities(
    labels: Sequence[str],
    words: Sequence[str],
) -> list[dict[str, Any]]:
    """Serialize candidate spans with exact word offsets and surface strings."""
    if len(labels) != len(words):
        raise ValueError("Candidate BIO labels and words must have equal length")
    return [
        {
            "start": start,
            "end": end,
            "label": entity_type,
            "surface": " ".join(str(word) for word in words[start:end]),
        }
        for start, end, entity_type in _entity_spans(canonical_bio(labels))
    ]


def _stable_json_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            # Dataset records can contain escaped UTF-16 surrogate code units
            # (for example ``\\ud83d\\ude02``).  Python preserves those code
            # units after ``ast.literal_eval``, but they cannot be encoded as
            # standalone UTF-8 characters.  ASCII JSON escaping gives every
            # such value a deterministic byte representation without changing
            # the semantic fields included in the hash.
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Value is not canonical JSON") from error
    return serialized.encode("utf-8")


def _opaque_digest(
    sample_id: Any,
    candidate_index: int,
    prompt_version: str,
    *,
    salt: str,
) -> str:
    payload = {
        "candidate_index": candidate_index,
        "prompt_version": str(prompt_version),
        "sample_id": sample_id,
        "salt": salt,
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def opaque_candidate_ids(
    sample_id: Any,
    count: int,
    prompt_version: str,
) -> list[str]:
    """Return stable opaque IDs aligned with original candidate indices.

    IDs contain neither the CRF rank nor score.  The candidate display order is
    intentionally provided by :func:`candidate_display_order` instead of being
    encoded into these IDs.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if not str(prompt_version):
        raise ValueError("prompt_version must be non-empty")

    identifiers: list[str] = []
    used: set[str] = set()
    for candidate_index in range(count):
        digest = _opaque_digest(
            sample_id,
            candidate_index,
            prompt_version,
            salt="candidate-id-v1",
        )
        # Ten hexadecimal characters make accidental collisions vanishingly
        # unlikely; extend deterministically if a collision ever occurs.
        width = 10
        identifier = f"C_{digest[:width].upper()}"
        while identifier in used:
            width += 2
            identifier = f"C_{digest[:width].upper()}"
        used.add(identifier)
        identifiers.append(identifier)
    return identifiers


def candidate_display_order(
    sample_id: Any,
    count: int,
    prompt_version: str,
) -> list[int]:
    """Return a deterministic pseudorandom permutation of candidate indices."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return sorted(
        range(count),
        key=lambda candidate_index: _opaque_digest(
            sample_id,
            candidate_index,
            prompt_version,
            salt="display-order-v1",
        ),
    )


def prepare_candidate_display(
    record: Mapping[str, Any],
    prompt_version: str,
) -> list[dict[str, Any]]:
    """Build prompt-safe candidates with no rank, index, or CRF score fields."""
    words = list(record.get("words", []))
    candidates = list(record.get("candidates", []))
    identifiers = opaque_candidate_ids(
        record.get("sample_id"),
        len(candidates),
        prompt_version,
    )
    display: list[dict[str, Any]] = []
    for candidate_index in candidate_display_order(
        record.get("sample_id"),
        len(candidates),
        prompt_version,
    ):
        candidate = candidates[candidate_index]
        if not isinstance(candidate, Mapping):
            raise ValueError("Each candidate must be an object")
        bio = candidate.get("bio")
        if not isinstance(bio, Sequence) or isinstance(bio, (str, bytes)):
            raise ValueError("Each candidate must contain a BIO sequence")
        display.append(
            {
                "candidate_id": identifiers[candidate_index],
                "entities": serialize_entities(list(bio), words),
            }
        )
    return display


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_inference_projection(path: str | Path) -> str:
    """Hash ordered text/images fields while never indexing the gold label field."""
    digest = hashlib.sha256()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = ast.literal_eval(line)
                projection = {
                    "text": record["text"],
                    "images": record["images"],
                }
            except (KeyError, SyntaxError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid inference record at line {line_number}"
                ) from error
            digest.update(_stable_json_bytes(projection))
            digest.update(b"\n")
    return digest.hexdigest()


def build_candidate_provenance(
    *,
    checkpoint_sha256: str,
    input_sha256: str,
    split: str,
    candidate_count: int,
    decode_count: int,
    generator_version: str = "crf-topk-v1",
) -> dict[str, Any]:
    """Create the immutable provenance block stored with candidate records."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if decode_count < candidate_count:
        raise ValueError("decode_count must be at least candidate_count")
    if not split:
        raise ValueError("split must be non-empty")
    return {
        "schema_version": "crf-rerank-candidates-v1",
        "generator_version": str(generator_version),
        "checkpoint_sha256": str(checkpoint_sha256),
        "input_sha256": str(input_sha256),
        "split": str(split),
        "candidate_count": int(candidate_count),
        "decode_count": int(decode_count),
        "gold_entity_labels_used": False,
    }


def candidate_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Fingerprint a candidate record while ignoring self-referential fields."""
    payload = {
        key: value
        for key, value in record.items()
        if key not in FINGERPRINT_FIELDS
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def build_rerank_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    llm_model: str,
    confidence_threshold: float,
    source_split: str = "val",
) -> dict[str, Any]:
    """Freeze the validation-selected deployment policy without dataset labels."""
    if not records:
        raise ValueError("Cannot freeze a policy without candidate records")
    if not llm_model:
        raise ValueError("llm_model must be non-empty")
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    first = records[0]
    provenance = first.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Candidate provenance is missing")
    frozen = {
        "schema_version": RERANK_POLICY_SCHEMA_VERSION,
        "prompt_version": first.get("prompt_version"),
        "llm_model": str(llm_model),
        "confidence_threshold": float(confidence_threshold),
        "checkpoint_sha256": first.get("checkpoint_sha256"),
        "candidate_generator_version": provenance.get("generator_version"),
        "candidate_count": provenance.get("candidate_count"),
        "decode_count": provenance.get("decode_count"),
        "deployment_scope": "all",
        "selected_on_split": str(source_split),
    }
    validate_rerank_policy(frozen, records, llm_model=llm_model)
    return frozen


def validate_rerank_policy(
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    llm_model: str | None = None,
) -> float:
    """Validate a frozen policy against a candidate artifact and return its threshold."""
    if not isinstance(policy, Mapping):
        raise ValueError("Rerank policy must be a JSON object")
    if policy.get("schema_version") != RERANK_POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported rerank policy schema")
    if policy.get("deployment_scope") != "all":
        raise ValueError("Frozen rerank policy must use deployment_scope=all")
    if policy.get("selected_on_split") != "val":
        raise ValueError("Frozen rerank policy must be selected on validation data")
    if not records:
        raise ValueError("Cannot validate a policy without candidate records")
    threshold = policy.get("confidence_threshold")
    if isinstance(threshold, bool):
        raise ValueError("Policy confidence_threshold must be finite and in [0, 1]")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Policy confidence_threshold must be finite and in [0, 1]"
        ) from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Policy confidence_threshold must be finite and in [0, 1]")
    if llm_model is not None and policy.get("llm_model") != llm_model:
        raise ValueError("Frozen policy LLM model mismatch")

    expected_fields = {
        "prompt_version": policy.get("prompt_version"),
        "checkpoint_sha256": policy.get("checkpoint_sha256"),
    }
    expected_provenance = {
        "generator_version": policy.get("candidate_generator_version"),
        "candidate_count": policy.get("candidate_count"),
        "decode_count": policy.get("decode_count"),
    }
    for record in records:
        for field, expected in expected_fields.items():
            if not expected or record.get(field) != expected:
                raise ValueError(f"Frozen policy {field} mismatch")
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Candidate provenance is missing")
        for field, expected in expected_provenance.items():
            if expected is None or provenance.get(field) != expected:
                raise ValueError(f"Frozen policy provenance {field} mismatch")
    return threshold


def normalized_top1_top2_margin(record: Mapping[str, Any]) -> float:
    """Return ``(top1_score - top2_score) / retained_word_count``."""
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("record.candidates must be a sequence")
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required to compute a margin")
    words = record.get("words")
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        raise ValueError("record.words must be a sequence")
    if not words:
        raise ValueError("At least one retained word is required")
    try:
        top1_score = float(candidates[0]["crf_score"])
        top2_score = float(candidates[1]["crf_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The first two candidates need finite crf_score values") from error
    if not math.isfinite(top1_score) or not math.isfinite(top2_score):
        raise ValueError("The first two candidates need finite crf_score values")
    if top2_score > top1_score + 1e-8:
        raise ValueError("Candidates must be ordered by descending CRF score")
    return max(top1_score - top2_score, 0.0) / len(words)


def _candidate_signature(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    bio = candidate.get("bio")
    if not isinstance(bio, Sequence) or isinstance(bio, (str, bytes)):
        raise ValueError("Each candidate must contain a BIO sequence")
    return tuple(canonical_bio([str(label) for label in bio]))


def _random_route_key(record: Mapping[str, Any], seed: int | str) -> str:
    return hashlib.sha256(
        _stable_json_bytes(
            {
                "seed": seed,
                "sample_id": record.get("sample_id"),
            }
        )
    ).hexdigest()


def select_pilot_routes(
    records: Iterable[Mapping[str, Any]],
    *,
    uncertainty_count: int = 100,
    random_count: int = 100,
    seed: int | str = 20261020,
) -> list[dict[str, Any]]:
    """Select disjoint low-margin and seeded-random pilot routes.

    Records without a distinct alternative candidate are ineligible.  The
    random-control sample is drawn only from records not already selected by
    the uncertainty route.  Returned values are compact routing manifests;
    callers can join them back to candidate records by ``sample_id``.
    """
    if uncertainty_count < 0 or random_count < 0:
        raise ValueError("Route counts must be non-negative")

    eligible: list[tuple[float, str, Mapping[str, Any]]] = []
    seen_sample_ids: set[str] = set()
    for record in records:
        sample_id = record.get("sample_id")
        sample_key = json.dumps(sample_id, ensure_ascii=False, sort_keys=True)
        if sample_key in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in candidate records: {sample_id!r}")
        seen_sample_ids.add(sample_key)

        candidates = record.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        if len(candidates) < 2:
            continue
        try:
            signatures = {
                _candidate_signature(candidate)
                for candidate in candidates
                if isinstance(candidate, Mapping)
            }
            margin = normalized_top1_top2_margin(record)
        except ValueError:
            continue
        if len(signatures) < 2:
            continue
        eligible.append((margin, sample_key, record))

    eligible.sort(key=lambda item: (item[0], item[1]))
    uncertain = eligible[:uncertainty_count]
    uncertain_keys = {item[1] for item in uncertain}
    remaining = [item for item in eligible if item[1] not in uncertain_keys]
    controls = sorted(
        remaining,
        key=lambda item: (_random_route_key(item[2], seed), item[1]),
    )[:random_count]

    routes: list[dict[str, Any]] = []
    for route_group, selected in (
        ("uncertainty", uncertain),
        ("random_control", controls),
    ):
        for margin, _, record in selected:
            routes.append(
                {
                    "sample_id": record.get("sample_id"),
                    "route_group": route_group,
                    "route_order": len(routes),
                    "normalized_margin": margin,
                }
            )
    return routes


def validate_reranker_decision(
    decision: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate and normalize the decision fields that may affect prediction."""
    if not isinstance(decision, Mapping):
        raise ValueError("decision must be an object")
    selected = decision.get("selected_candidate_id")
    if not isinstance(selected, str) or selected not in candidate_ids:
        raise ValueError("selected_candidate_id is not one of the supplied candidates")
    confidence = decision.get("confidence")
    if isinstance(confidence, bool):
        raise ValueError("confidence must be a number")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("confidence must be a number") from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    uncertain = decision.get("uncertain")
    if not isinstance(uncertain, bool):
        raise ValueError("uncertain must be a boolean")
    return {
        "selected_candidate_id": selected,
        "confidence": confidence,
        "uncertain": uncertain,
    }


def apply_reranker_decision(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    *,
    prompt_version: str,
    confidence_threshold: float = 0.85,
) -> dict[str, Any]:
    """Apply a constrained decision or return a documented top-1 fallback."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("record.candidates must be a sequence")
    if not candidates:
        raise ValueError("At least one candidate is required for top-1 fallback")

    identifiers = opaque_candidate_ids(
        record.get("sample_id"),
        len(candidates),
        prompt_version,
    )
    selected_index = 0
    accepted = False
    fallback_reason: str | None = None
    normalized: dict[str, Any] | None = None

    if decision is None:
        fallback_reason = "missing_decision"
    elif not isinstance(decision, Mapping):
        fallback_reason = "invalid_decision"
    elif decision.get("candidate_fingerprint") != candidate_record_fingerprint(record):
        fallback_reason = "candidate_fingerprint_mismatch"
    elif decision.get("prompt_version") != prompt_version:
        fallback_reason = "prompt_version_mismatch"
    else:
        try:
            normalized = validate_reranker_decision(decision, identifiers)
        except ValueError:
            fallback_reason = "invalid_decision"
        else:
            if normalized["uncertain"]:
                fallback_reason = "uncertain"
            elif normalized["confidence"] < confidence_threshold:
                fallback_reason = "below_confidence"
            else:
                selected_index = identifiers.index(normalized["selected_candidate_id"])
                accepted = True

    return {
        "candidate_index": selected_index,
        "candidate_id": identifiers[selected_index],
        "candidate": candidates[selected_index],
        "accepted": accepted,
        "changed_top1": accepted and selected_index != 0,
        "fallback_reason": fallback_reason,
        "confidence": None if normalized is None else normalized["confidence"],
        "uncertain": None if normalized is None else normalized["uncertain"],
    }
