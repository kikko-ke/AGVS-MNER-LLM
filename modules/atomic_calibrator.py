"""Supervised calibration for conservative atomic CRF edits.

The module deliberately keeps gold-only label construction separate from
inference.  At inference time a calibrator sees fixed structural features and
validated, structured LLM review fields.  It may select at most one edit for a
sample; every validation or provenance failure falls back to CRF top-1.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_SCHEMA_VERSION = "atomic-calibrator-features-v1"
EXAMPLE_SCHEMA_VERSION = "atomic-calibrator-example-v1"
POLICY_SCHEMA_VERSION = "atomic-calibrator-policy-v1"

EDIT_CLASSES = (
    "deletion_spurious_removal",
    "addition_missing_recovery",
    "entity_type_change",
    "boundary_change",
)
ENTITY_LABELS = ("NONE", "MIS", "PER", "ORG", "LOC")
VERDICTS = ("KEEP_BASELINE", "APPLY_EDIT", "ABSTAIN")
SUPPORT_VALUES = ("baseline", "edit", "conflict", "none")
OUTCOME_TO_INDEX = {"beneficial": 0, "neutral": 1, "harmful": 2}
INDEX_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_INDEX.items()}

STRUCTURAL_FEATURE_NAMES = (
    "word_count_scaled",
    "image_count_scaled",
    "valid_image_fraction",
    "score_gap_per_word",
    "top1_top2_gap_per_word",
    *(f"rank_{rank}" for rank in range(2, 6)),
    *(f"edit_{name}" for name in EDIT_CLASSES),
    "removed_present",
    "added_present",
    "removed_start_norm",
    "removed_length_norm",
    "added_start_norm",
    "added_length_norm",
    "boundary_start_shift_norm",
    "boundary_end_shift_norm",
    *(f"source_label_{label}" for label in ENTITY_LABELS),
    *(f"target_label_{label}" for label in ENTITY_LABELS),
)
REVIEW_FEATURE_NAMES = (
    *(f"review_verdict_{value}" for value in VERDICTS),
    "review_confidence",
    *(f"text_support_{value}" for value in SUPPORT_VALUES),
    *(f"visual_support_{value}" for value in SUPPORT_VALUES),
    "review_used_image_fraction",
    "review_used_any_image",
    "review_decisive_span_count_scaled",
    "review_has_decisive_span",
    "review_apply_text_support",
    "review_apply_visual_support",
    "review_keep_baseline_support",
    "review_has_conflict",
)
FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + REVIEW_FEATURE_NAMES
REVIEW_FEATURE_START = len(STRUCTURAL_FEATURE_NAMES)


@dataclass(frozen=True)
class CountDelta:
    """Entity-count change made by one alternative relative to top-1."""

    delta_tp: int
    delta_fp: int
    delta_fn: int

    def dinkelbach_gain(self, q: float) -> float:
        if not 0.0 <= q <= 1.0 or not math.isfinite(q):
            raise ValueError("q must be a finite F1 value in [0, 1]")
        return 2.0 * self.delta_tp - q * (
            2.0 * self.delta_tp + self.delta_fp + self.delta_fn
        )


class AtomicCalibrator(nn.Module):
    """Small shared encoder with utility-regression and outcome heads."""

    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.utility_head = nn.Linear(hidden_size, 1)
        self.outcome_head = nn.Linear(hidden_size, len(OUTCOME_TO_INDEX))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.input_size:
            raise ValueError("features must have shape [N, input_size]")
        hidden = self.encoder(features)
        return self.utility_head(hidden).squeeze(-1), self.outcome_head(hidden)


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entity_counts(
    gold_entities: Iterable[tuple[int, int, str]],
    predicted_entities: Iterable[tuple[int, int, str]],
) -> tuple[int, int, int]:
    gold = set(gold_entities)
    prediction = set(predicted_entities)
    return (
        len(gold & prediction),
        len(prediction - gold),
        len(gold - prediction),
    )


def count_delta(
    baseline_counts: Sequence[int],
    edited_counts: Sequence[int],
) -> CountDelta:
    if len(baseline_counts) != 3 or len(edited_counts) != 3:
        raise ValueError("entity counts must contain TP, FP, and FN")
    return CountDelta(
        int(edited_counts[0]) - int(baseline_counts[0]),
        int(edited_counts[1]) - int(baseline_counts[1]),
        int(edited_counts[2]) - int(baseline_counts[2]),
    )


def f1_from_counts(counts: Sequence[int]) -> float:
    if len(counts) != 3:
        raise ValueError("counts must contain TP, FP, and FN")
    tp, fp, fn = map(int, counts)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator <= 0 else 2.0 * tp / denominator


def outcome_from_gain(gain: float, epsilon: float = 1e-12) -> str:
    if not math.isfinite(gain):
        raise ValueError("Dinkelbach gain must be finite")
    if gain > epsilon:
        return "beneficial"
    if gain < -epsilon:
        return "harmful"
    return "neutral"


def _one_hot(value: Any, vocabulary: Sequence[str]) -> list[float]:
    text = str(value)
    return [float(text == item) for item in vocabulary]


def _entity(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("atomic edit entities must be objects or null")
    start = raw.get("start")
    end = raw.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("entity boundaries must be integers")
    start = int(start)
    end = int(end)
    label = str(raw.get("label", "")).upper().replace("MISC", "MIS")
    if start < 0 or end <= start or label not in ENTITY_LABELS[1:]:
        raise ValueError("invalid atomic edit entity")
    return {"start": start, "end": end, "label": label}


def normalize_edit_class(value: Any) -> str:
    aliases = {
        "deletion": EDIT_CLASSES[0],
        "spurious_entity": EDIT_CLASSES[0],
        "addition": EDIT_CLASSES[1],
        "missing_entity": EDIT_CLASSES[1],
        "entity_type": EDIT_CLASSES[2],
        "type_change": EDIT_CLASSES[2],
        "boundary": EDIT_CLASSES[3],
    }
    result = aliases.get(str(value), str(value))
    if result not in EDIT_CLASSES:
        raise ValueError(f"unsupported atomic edit class: {value!r}")
    return result


def normalize_verdict(value: Any) -> str:
    aliases = {
        "KEEP": "KEEP_BASELINE",
        "BASELINE": "KEEP_BASELINE",
        "APPLY": "APPLY_EDIT",
        "EDIT": "APPLY_EDIT",
        "UNCERTAIN": "ABSTAIN",
    }
    result = aliases.get(str(value).strip().upper(), str(value).strip().upper())
    if result not in VERDICTS:
        raise ValueError(f"invalid review verdict: {value!r}")
    return result


def normalize_support(value: Any) -> str:
    aliases = {
        "supports_baseline": "baseline",
        "keep": "baseline",
        "supports_edit": "edit",
        "alternative": "edit",
        "ambiguous": "conflict",
        "neither": "none",
    }
    text = str(value if value is not None else "none").strip().lower()
    result = aliases.get(text, text)
    if result not in SUPPORT_VALUES:
        raise ValueError(f"invalid review support value: {value!r}")
    return result


def validate_review(
    review: Mapping[str, Any],
    edit: Mapping[str, Any],
    valid_image_ids: Iterable[int],
) -> dict[str, Any]:
    """Validate one structured review and discard unrestricted rationale text."""
    if str(review.get("edit_id", "")) != str(edit.get("edit_id", "")):
        raise ValueError("review edit_id does not match its atomic edit")
    verdict = normalize_verdict(review.get("verdict"))
    raw_confidence = review.get("confidence")
    if isinstance(raw_confidence, bool):
        raise ValueError("review confidence must be a finite number")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("review confidence must be in [0, 1]")
    text_support = normalize_support(review.get("text_support"))
    visual_support = normalize_support(review.get("visual_support"))

    valid_images = {int(value) for value in valid_image_ids}
    raw_image_ids = review.get("used_image_ids", [])
    if not isinstance(raw_image_ids, list):
        raise ValueError("used_image_ids must be a list")
    image_ids: list[int] = []
    for raw in raw_image_ids:
        if isinstance(raw, bool):
            raise ValueError("used_image_ids must contain integers")
        image_id = int(raw)
        if image_id not in valid_images or image_id in image_ids:
            raise ValueError("review contains an invalid or duplicate image id")
        image_ids.append(image_id)

    allowed_spans = {
        (entity["start"], entity["end"])
        for entity in (
            _entity(edit.get("removed_entity")),
            _entity(edit.get("added_entity")),
        )
        if entity is not None
    }
    raw_spans = review.get("decisive_spans", review.get("decisive_span", []))
    if isinstance(raw_spans, Mapping):
        raw_spans = [raw_spans]
    if not isinstance(raw_spans, list):
        raise ValueError("decisive_spans must be a list")
    spans: list[dict[str, int]] = []
    for raw in raw_spans:
        if not isinstance(raw, Mapping):
            raise ValueError("each decisive span must be an object")
        start = int(raw.get("start"))
        end = int(raw.get("end"))
        if (start, end) not in allowed_spans:
            raise ValueError("decisive span must be a removed or added entity span")
        spans.append({"start": start, "end": end})

    return {
        "edit_id": str(edit["edit_id"]),
        "verdict": verdict,
        "confidence": confidence,
        "text_support": text_support,
        "visual_support": visual_support,
        "used_image_ids": image_ids,
        "decisive_spans": spans,
    }


def build_feature_vector(
    candidate_record: Mapping[str, Any],
    edit: Mapping[str, Any],
    review: Mapping[str, Any],
) -> list[float]:
    """Create the fixed inference-safe V1 feature vector."""
    words = candidate_record.get("words", [])
    if not isinstance(words, list) or not words:
        raise ValueError("candidate words must be a non-empty list")
    word_count = len(words)
    image_names = candidate_record.get("image_names", [])
    valid_image_ids = candidate_record.get("valid_image_ids", [])
    if not isinstance(image_names, list) or not isinstance(valid_image_ids, list):
        raise ValueError("candidate image fields must be lists")

    edit_class = normalize_edit_class(edit.get("edit_class"))
    removed = _entity(edit.get("removed_entity"))
    added = _entity(edit.get("added_entity"))
    if removed is None and added is None:
        raise ValueError("atomic edit must remove or add one entity")
    rank = int(edit.get("crf_rank", 0))
    if rank < 2 or rank > 5:
        raise ValueError("atomic edit crf_rank must be in 2..5")
    score_gap = float(edit.get("score_gap_per_word"))
    top_gap = float(
        candidate_record.get("route_features", {}).get(
            "top1_top2_gap_per_word",
            candidate_record.get("route_features", {}).get(
                "minimum_atomic_score_gap_per_word", score_gap
            ),
        )
    )
    if not math.isfinite(score_gap) or not math.isfinite(top_gap):
        raise ValueError("CRF margins must be finite")

    checked_review = validate_review(review, edit, map(int, valid_image_ids))
    denominator = float(max(word_count, 1))
    source_label = removed["label"] if removed else "NONE"
    target_label = added["label"] if added else "NONE"
    removed_start = float(removed["start"] / denominator) if removed else 0.0
    removed_length = (
        float((removed["end"] - removed["start"]) / denominator)
        if removed
        else 0.0
    )
    added_start = float(added["start"] / denominator) if added else 0.0
    added_length = (
        float((added["end"] - added["start"]) / denominator) if added else 0.0
    )
    start_shift = (
        abs(float(added["start"] - removed["start"])) / denominator
        if removed and added
        else 0.0
    )
    end_shift = (
        abs(float(added["end"] - removed["end"])) / denominator
        if removed and added
        else 0.0
    )

    verdict = checked_review["verdict"]
    text_support = checked_review["text_support"]
    visual_support = checked_review["visual_support"]
    used_images = checked_review["used_image_ids"]
    decisive_spans = checked_review["decisive_spans"]
    structural = [
        min(word_count, 64) / 64.0,
        min(len(image_names), 4) / 4.0,
        min(len(valid_image_ids), 4) / 4.0,
        score_gap,
        top_gap,
        *_one_hot(str(rank), tuple(map(str, range(2, 6)))),
        *_one_hot(edit_class, EDIT_CLASSES),
        float(removed is not None),
        float(added is not None),
        removed_start,
        removed_length,
        added_start,
        added_length,
        start_shift,
        end_shift,
        *_one_hot(source_label, ENTITY_LABELS),
        *_one_hot(target_label, ENTITY_LABELS),
    ]
    review_features = [
        *_one_hot(verdict, VERDICTS),
        float(checked_review["confidence"]),
        *_one_hot(text_support, SUPPORT_VALUES),
        *_one_hot(visual_support, SUPPORT_VALUES),
        min(len(used_images), 4) / 4.0,
        float(bool(used_images)),
        min(len(decisive_spans), 2) / 2.0,
        float(bool(decisive_spans)),
        float(verdict == "APPLY_EDIT" and text_support == "edit"),
        float(verdict == "APPLY_EDIT" and visual_support == "edit"),
        float(verdict == "KEEP_BASELINE" and text_support == "baseline"),
        float(text_support == "conflict" or visual_support == "conflict"),
    ]
    result = [float(value) for value in (*structural, *review_features)]
    if len(result) != len(FEATURE_NAMES) or not all(map(math.isfinite, result)):
        raise RuntimeError("atomic calibrator feature construction is inconsistent")
    return result


def feature_mode_vector(features: Sequence[float], mode: str) -> list[float]:
    result = [float(value) for value in features]
    if len(result) != len(FEATURE_NAMES):
        raise ValueError("feature vector has the wrong length")
    if mode == "full":
        return result
    if mode == "structure_only":
        result[REVIEW_FEATURE_START:] = [0.0] * len(REVIEW_FEATURE_NAMES)
        return result
    raise ValueError(f"unknown feature mode: {mode}")


def fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def standardize(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if features.shape[-1] != mean.numel() or mean.shape != std.shape:
        raise ValueError("standardizer shape does not match features")
    return (features - mean.to(features.device)) / std.to(features.device)


def calibrator_loss(
    utilities: torch.Tensor,
    outcome_logits: torch.Tensor,
    gains: torch.Tensor,
    outcomes: torch.Tensor,
    sample_ids: Sequence[int],
    *,
    class_weights: torch.Tensor | None = None,
    harmful_penalty_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine utility, outcome, and per-sample listwise objectives."""
    if utilities.ndim != 1 or gains.shape != utilities.shape:
        raise ValueError("utilities and gains must be aligned vectors")
    if outcome_logits.shape != (len(utilities), len(OUTCOME_TO_INDEX)):
        raise ValueError("outcome_logits has the wrong shape")
    if outcomes.shape != utilities.shape or len(sample_ids) != len(utilities):
        raise ValueError("outcomes and sample_ids must align with utilities")
    regression = F.smooth_l1_loss(utilities, gains)
    classification = F.cross_entropy(
        outcome_logits,
        outcomes,
        weight=None if class_weights is None else class_weights.to(outcome_logits.device),
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped[int(sample_id)].append(index)
    listwise_terms = []
    for indices in grouped.values():
        index_tensor = torch.tensor(indices, device=utilities.device)
        scores = torch.cat([utilities.new_zeros(1), utilities.index_select(0, index_tensor)])
        targets = torch.cat([gains.new_zeros(1), gains.index_select(0, index_tensor)])
        target = targets.argmax().reshape(1)
        listwise_terms.append(F.cross_entropy(scores.reshape(1, -1), target))
    listwise = torch.stack(listwise_terms).mean()
    harmful_mask = outcomes == OUTCOME_TO_INDEX["harmful"]
    harmful_penalty = (
        F.softplus(utilities[harmful_mask]).mean()
        if harmful_mask.any()
        else utilities.new_zeros(())
    )
    total = (
        regression
        + 0.5 * classification
        + 0.5 * listwise
        + float(harmful_penalty_weight) * harmful_penalty
    )
    return total, {
        "regression": regression.detach(),
        "classification": classification.detach(),
        "listwise": listwise.detach(),
        "harmful_penalty": harmful_penalty.detach(),
    }


def group_examples(records: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen_edit_ids: set[tuple[int, str]] = set()
    for record in records:
        if record.get("schema_version") != EXAMPLE_SCHEMA_VERSION:
            raise ValueError("incompatible atomic calibrator example schema")
        sample_id = int(record["sample_id"])
        edit_id = str(record["edit_id"])
        key = (sample_id, edit_id)
        if key in seen_edit_ids:
            raise ValueError(f"duplicate calibrator edit example: {key}")
        seen_edit_ids.add(key)
        grouped[sample_id].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["crf_rank"]), str(row["edit_id"])))
    return dict(grouped)


def _baseline_counts(metadata: Mapping[str, Any], role: str) -> list[int]:
    roles = metadata.get("roles", {})
    section = roles.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"metadata does not contain role={role!r}")
    counts = section.get("baseline_counts")
    if not isinstance(counts, list) or len(counts) != 3:
        raise ValueError(f"metadata role={role!r} has invalid baseline counts")
    return list(map(int, counts))


def selections_metrics(
    metadata: Mapping[str, Any],
    role: str,
    grouped: Mapping[int, Sequence[Mapping[str, Any]]],
    selections: Mapping[int, str | None],
) -> dict[str, Any]:
    counts = _baseline_counts(metadata, role)
    accepted: list[Mapping[str, Any]] = []
    for sample_id, edit_id in selections.items():
        if edit_id is None:
            continue
        rows = grouped.get(int(sample_id), ())
        matches = [row for row in rows if str(row["edit_id"]) == str(edit_id)]
        if len(matches) != 1:
            raise ValueError("selection does not identify exactly one atomic edit")
        row = matches[0]
        delta = row["count_delta"]
        counts[0] += int(delta["tp"])
        counts[1] += int(delta["fp"])
        counts[2] += int(delta["fn"])
        accepted.append(row)
    beneficial = sum(row["outcome"] == "beneficial" for row in accepted)
    neutral = sum(row["outcome"] == "neutral" for row in accepted)
    harmful = sum(row["outcome"] == "harmful" for row in accepted)
    exact_baseline_samples = {
        sample_id
        for sample_id, rows in grouped.items()
        if rows and bool(rows[0].get("baseline_exact", False))
    }
    exact_preserved = sum(
        selections.get(sample_id) is None
        or bool(
            next(
                row
                for row in grouped[sample_id]
                if str(row["edit_id"]) == str(selections[sample_id])
            ).get("edited_exact", False)
        )
        for sample_id in exact_baseline_samples
    )
    return {
        "counts": {"tp": counts[0], "fp": counts[1], "fn": counts[2]},
        "f1": f1_from_counts(counts),
        "accepted": len(accepted),
        "beneficial": beneficial,
        "neutral": neutral,
        "harmful": harmful,
        "top1_exact_samples": len(exact_baseline_samples),
        "top1_exact_preservation": (
            exact_preserved / len(exact_baseline_samples)
            if exact_baseline_samples
            else 1.0
        ),
    }


def exact_atomic_oracle(
    metadata: Mapping[str, Any],
    role: str,
    grouped: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    tolerance: float = 1e-12,
    maximum_iterations: int = 100,
) -> dict[str, Any]:
    """Solve the independent-edit fractional objective with Dinkelbach updates."""
    baseline_counts = _baseline_counts(metadata, role)
    q = f1_from_counts(baseline_counts)
    selections: dict[int, str | None] = {sample_id: None for sample_id in grouped}
    for _ in range(maximum_iterations):
        for sample_id, rows in grouped.items():
            best_gain = 0.0
            best_edit: str | None = None
            best_rank = 10**9
            for row in rows:
                delta = row["count_delta"]
                gain = CountDelta(
                    int(delta["tp"]), int(delta["fp"]), int(delta["fn"])
                ).dinkelbach_gain(q)
                rank = int(row["crf_rank"])
                if gain > best_gain + tolerance or (
                    abs(gain - best_gain) <= tolerance
                    and gain > tolerance
                    and rank < best_rank
                ):
                    best_gain = gain
                    best_edit = str(row["edit_id"])
                    best_rank = rank
            selections[sample_id] = best_edit
        result = selections_metrics(metadata, role, grouped, selections)
        new_q = float(result["f1"])
        if abs(new_q - q) <= tolerance:
            result["iterations"] = _ + 1
            result["selections"] = selections
            return result
        q = new_q
    raise RuntimeError("atomic Dinkelbach oracle did not converge")


def safe_select_edits(
    records: Sequence[Mapping[str, Any]],
    utilities: Sequence[float],
    harmful_probabilities: Sequence[float],
    policy: Mapping[str, Any],
) -> tuple[dict[int, str | None], Counter[str]]:
    """Apply a frozen policy with conservative per-sample fallback semantics."""
    if not (len(records) == len(utilities) == len(harmful_probabilities)):
        raise ValueError("records and predictions must align")
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported atomic calibrator policy schema")
    threshold = float(policy["score_threshold"])
    harm_cap = float(policy["harm_probability_cap"])
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[int(record["sample_id"])].append(index)
    selections: dict[int, str | None] = {}
    fallbacks: Counter[str] = Counter()
    for sample_id, indices in grouped_indices.items():
        valid: list[int] = []
        for index in indices:
            record = records[index]
            features = record.get("features")
            expected = record.get("expected_review_fingerprint")
            actual = record.get("review_fingerprint")
            if record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                fallbacks["feature_schema_mismatch"] += 1
                continue
            if expected is None or actual != expected:
                fallbacks["review_fingerprint_mismatch"] += 1
                continue
            if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
                fallbacks["invalid_features"] += 1
                continue
            if not all(math.isfinite(float(value)) for value in features):
                fallbacks["invalid_features"] += 1
                continue
            if not (
                math.isfinite(float(utilities[index]))
                and math.isfinite(float(harmful_probabilities[index]))
            ):
                fallbacks["invalid_prediction"] += 1
                continue
            valid.append(index)
        if len(valid) != len(indices):
            selections[sample_id] = None
            fallbacks["unsafe_sample"] += 1
            continue
        best = max(
            valid,
            key=lambda index: (
                float(utilities[index]),
                -int(records[index]["crf_rank"]),
                str(records[index]["edit_id"]),
            ),
        )
        if float(utilities[best]) < threshold:
            selections[sample_id] = None
            fallbacks["below_score_threshold"] += 1
        elif float(harmful_probabilities[best]) > harm_cap:
            selections[sample_id] = None
            fallbacks["harm_risk"] += 1
        else:
            selections[sample_id] = str(records[best]["edit_id"])
    return selections, fallbacks


def policy_grid(
    records: Sequence[Mapping[str, Any]],
    utilities: Sequence[float],
    harmful_probabilities: Sequence[float],
    metadata: Mapping[str, Any],
    role: str,
    *,
    minimum_top1_preservation: float = 0.98,
    minimum_accepted: int = 5,
    harm_caps: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a conservative policy using only an independent calibration role."""
    if not records:
        raise ValueError("policy calibration requires examples")
    grouped = group_examples(records)
    finite_scores = sorted({float(value) for value in utilities if math.isfinite(float(value))})
    if not finite_scores:
        raise ValueError("no finite utility predictions")
    thresholds = [finite_scores[-1] + 1e-6]
    if len(finite_scores) <= 200:
        thresholds.extend(finite_scores)
    else:
        thresholds.extend(
            finite_scores[round(index * (len(finite_scores) - 1) / 199)]
            for index in range(200)
        )
    candidates: list[dict[str, Any]] = []
    for threshold in sorted(set(thresholds), reverse=True):
        for harm_cap in harm_caps:
            temporary = {
                "schema_version": POLICY_SCHEMA_VERSION,
                "score_threshold": threshold,
                "harm_probability_cap": float(harm_cap),
            }
            selections, fallbacks = safe_select_edits(
                records, utilities, harmful_probabilities, temporary
            )
            result = selections_metrics(metadata, role, grouped, selections)
            result.update(
                {
                    "score_threshold": threshold,
                    "harm_probability_cap": float(harm_cap),
                    "fallbacks": dict(fallbacks),
                    "feasible": (
                        result["accepted"] >= minimum_accepted
                        and result["top1_exact_preservation"]
                        >= minimum_top1_preservation
                        and result["beneficial"] > result["harmful"]
                    ),
                }
            )
            candidates.append(result)
    feasible = [item for item in candidates if item["feasible"]]
    pool = feasible or [item for item in candidates if item["accepted"] == 0]
    best = max(
        pool,
        key=lambda item: (
            item["f1"],
            item["top1_exact_preservation"],
            -item["harmful"],
            -item["accepted"],
            item["score_threshold"],
        ),
    )
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selected_on_role": role,
        "score_threshold": best["score_threshold"],
        "harm_probability_cap": best["harm_probability_cap"],
        "constraints": {
            "minimum_top1_exact_preservation": minimum_top1_preservation,
            "minimum_accepted": minimum_accepted,
            "beneficial_must_exceed_harmful": True,
        },
        "feasible_policy_found": bool(feasible),
    }
    summary = {
        "selected": best,
        "policies_evaluated": len(candidates),
        "feasible_policies": len(feasible),
    }
    return policy, summary


def shuffle_review_features(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[list[float]]:
    """Permute review blocks within edit class for a deterministic negative control."""
    result = [[float(value) for value in record["features"]] for record in records]
    generator = torch.Generator().manual_seed(int(seed))
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_class[str(record["edit_class"])].append(index)
    for indices in by_class.values():
        if len(indices) < 2:
            continue
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        source_blocks = [result[index][REVIEW_FEATURE_START:] for index in indices]
        for target_position, record_index in enumerate(indices):
            result[record_index][REVIEW_FEATURE_START:] = source_blocks[
                permutation[target_position]
            ]
    return result


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[AtomicCalibrator, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("calibrator checkpoint feature schema mismatch")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("calibrator checkpoint feature names mismatch")
    model = AtomicCalibrator(
        input_size=len(FEATURE_NAMES),
        hidden_size=int(payload["hidden_size"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
