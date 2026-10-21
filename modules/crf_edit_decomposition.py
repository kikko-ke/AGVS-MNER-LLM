"""Gold-aware oracle audit grouped by gold-free edits from CRF top-1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from modules.crf_topk import EntityCounts, select_corpus_micro_f1_oracle


Entity = tuple[int, int, str]
EntitySet = frozenset[Entity]

DELETION = "deletion_spurious_removal"
ADDITION = "addition_missing_recovery"
TYPE_CHANGE = "entity_type_change"
BOUNDARY_CHANGE = "boundary_change"
MIXED = "mixed"
UNCHANGED = "unchanged"
EDIT_CLASSES = (DELETION, ADDITION, TYPE_CHANGE, BOUNDARY_CHANGE, MIXED)
EDIT_GROUPS = {
    "deletion_or_entity_type": (DELETION, TYPE_CHANGE),
}


def _normalize_entities(entities: Iterable[tuple[Any, Any, Any]]) -> EntitySet:
    normalized: set[Entity] = set()
    for raw_entity in entities:
        if len(raw_entity) != 3:
            raise ValueError("Each entity must contain start, end, and type")
        start, end, entity_type = raw_entity
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Entity boundaries must be integers")
        start = int(start)
        end = int(end)
        entity_type = str(entity_type)
        if start < 0 or end <= start or not entity_type:
            raise ValueError(f"Invalid entity ({start}, {end}, {entity_type!r})")
        normalized.add((start, end, entity_type))
    return frozenset(normalized)


def _overlaps(left: Entity, right: Entity) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _is_pure_boundary_change(top1: EntitySet, alternative: EntitySet) -> bool:
    removed = list(top1 - alternative)
    added = list(alternative - top1)
    if not removed or len(removed) != len(added):
        return False

    adjacency = [
        [
            added_index
            for added_index, added_entity in enumerate(added)
            if removed_entity[2] == added_entity[2]
            and _overlaps(removed_entity, added_entity)
        ]
        for removed_entity in removed
    ]
    if any(not matches for matches in adjacency):
        return False

    # The candidate lists are tiny (K <= 10 and usually one changed entity), so
    # a deterministic augmenting-path match is clearer than another dependency.
    matched_removed_by_added: dict[int, int] = {}

    def augment(removed_index: int, seen_added: set[int]) -> bool:
        for added_index in adjacency[removed_index]:
            if added_index in seen_added:
                continue
            seen_added.add(added_index)
            previous = matched_removed_by_added.get(added_index)
            if previous is None or augment(previous, seen_added):
                matched_removed_by_added[added_index] = removed_index
                return True
        return False

    order = sorted(range(len(removed)), key=lambda index: len(adjacency[index]))
    return all(augment(index, set()) for index in order)


def classify_entity_edit(
    top1_entities: Iterable[tuple[Any, Any, Any]],
    alternative_entities: Iterable[tuple[Any, Any, Any]],
) -> str:
    """Classify one candidate using only its entity delta from CRF top-1.

    The names ``spurious_removal`` and ``missing_recovery`` describe their
    potential role. Whether an edit actually repairs a gold error is measured
    separately by the oracle audit.
    """
    top1 = _normalize_entities(top1_entities)
    alternative = _normalize_entities(alternative_entities)
    if alternative == top1:
        return UNCHANGED
    if alternative < top1:
        return DELETION
    if top1 < alternative:
        return ADDITION

    top1_boundaries = {(start, end) for start, end, _ in top1}
    alternative_boundaries = {(start, end) for start, end, _ in alternative}
    if len(top1) == len(alternative) and top1_boundaries == alternative_boundaries:
        return TYPE_CHANGE
    if len(top1) == len(alternative) and _is_pure_boundary_change(top1, alternative):
        return BOUNDARY_CHANGE
    return MIXED


def _entity_counts(gold: EntitySet, prediction: EntitySet) -> EntityCounts:
    return (
        len(gold & prediction),
        len(prediction - gold),
        len(gold - prediction),
    )


def _metrics_from_counts(counts: Sequence[EntityCounts]) -> dict[str, float]:
    true_positive = sum(item[0] for item in counts)
    false_positive = sum(item[1] for item in counts)
    false_negative = sum(item[2] for item in counts)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "f1": 0.0 if not f1_denominator else 2 * true_positive / f1_denominator,
        "precision": (
            0.0 if not precision_denominator else true_positive / precision_denominator
        ),
        "recall": 0.0 if not recall_denominator else true_positive / recall_denominator,
    }


def _sample_f1(counts: EntityCounts) -> float:
    true_positive, false_positive, false_negative = counts
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def decompose_topk_edit_oracles(
    gold_entities: Sequence[Iterable[tuple[Any, Any, Any]]],
    candidate_entities: Sequence[Sequence[Iterable[tuple[Any, Any, Any]]]],
    candidate_scores: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Compute exact corpus micro-F1 under restricted top-1 edit sets."""
    if not (
        len(gold_entities) == len(candidate_entities) == len(candidate_scores)
    ):
        raise ValueError("Gold, candidate, and score rows must have equal length")

    normalized_gold = [_normalize_entities(row) for row in gold_entities]
    normalized_candidates: list[list[EntitySet]] = []
    normalized_scores: list[list[float]] = []
    edit_labels: list[list[str]] = []
    unchanged_alternatives = 0
    for candidates, scores in zip(candidate_entities, candidate_scores):
        if not candidates:
            raise ValueError("Every sample must contain a top-1 candidate")
        if len(candidates) != len(scores):
            raise ValueError("Every candidate row must match its score row")
        candidate_row = [_normalize_entities(candidate) for candidate in candidates]
        score_row = [float(score) for score in scores]
        labels = [UNCHANGED]
        for candidate in candidate_row[1:]:
            label = classify_entity_edit(candidate_row[0], candidate)
            labels.append(label)
            unchanged_alternatives += int(label == UNCHANGED)
        normalized_candidates.append(candidate_row)
        normalized_scores.append(score_row)
        edit_labels.append(labels)

    top1_counts = [
        _entity_counts(gold, candidates[0])
        for gold, candidates in zip(normalized_gold, normalized_candidates)
    ]
    top1_metrics = _metrics_from_counts(top1_counts)
    top1_error_samples = sum(
        candidates[0] != gold
        for gold, candidates in zip(normalized_gold, normalized_candidates)
    )

    audit_specs = {
        **{edit_class: (edit_class,) for edit_class in EDIT_CLASSES},
        **EDIT_GROUPS,
    }
    audit_results: dict[str, Any] = {}
    for audit_name, allowed_edit_classes in audit_specs.items():
        allowed_counts: list[list[EntityCounts]] = []
        allowed_scores: list[list[float]] = []
        original_indices: list[list[int]] = []
        alternative_candidates = 0
        samples_with_alternative = 0
        error_samples_with_alternative = 0
        locally_improvable_errors = 0
        exactly_recoverable_errors = 0

        for gold, candidates, scores, labels, baseline_counts in zip(
            normalized_gold,
            normalized_candidates,
            normalized_scores,
            edit_labels,
            top1_counts,
        ):
            indices = [
                0,
                *[
                    index
                    for index, label in enumerate(labels[1:], start=1)
                    if label in allowed_edit_classes
                ],
            ]
            alternatives = indices[1:]
            alternative_candidates += len(alternatives)
            if alternatives:
                samples_with_alternative += 1
            top1_is_error = candidates[0] != gold
            if top1_is_error and alternatives:
                error_samples_with_alternative += 1
                alternative_count_values = [
                    _entity_counts(gold, candidates[index]) for index in alternatives
                ]
                if any(
                    _sample_f1(counts) > _sample_f1(baseline_counts)
                    for counts in alternative_count_values
                ):
                    locally_improvable_errors += 1
                if any(candidates[index] == gold for index in alternatives):
                    exactly_recoverable_errors += 1

            original_indices.append(indices)
            allowed_counts.append(
                [_entity_counts(gold, candidates[index]) for index in indices]
            )
            allowed_scores.append([scores[index] for index in indices])

        selected_local, oracle_f1 = select_corpus_micro_f1_oracle(
            allowed_counts,
            allowed_scores,
        )
        selected_original = [
            indices[local_index]
            for indices, local_index in zip(original_indices, selected_local)
        ]
        selected_counts = [
            _entity_counts(gold, candidates[index])
            for gold, candidates, index in zip(
                normalized_gold, normalized_candidates, selected_original
            )
        ]
        oracle_metrics = _metrics_from_counts(selected_counts)
        if abs(oracle_metrics["f1"] - oracle_f1) > 1e-12:
            raise RuntimeError("Edit oracle totals disagree with the exact optimizer")
        rank_histogram = Counter(index + 1 for index in selected_original)
        audit_results[audit_name] = {
            "allowed_edit_classes": list(allowed_edit_classes),
            "alternative_candidates": alternative_candidates,
            "samples_with_alternative": samples_with_alternative,
            "top1_error_samples_with_alternative": error_samples_with_alternative,
            "locally_improvable_top1_errors": locally_improvable_errors,
            "exactly_recoverable_top1_errors": exactly_recoverable_errors,
            "exact_recoverable_rate_over_all_top1_errors": exactly_recoverable_errors
            / max(top1_error_samples, 1),
            "exact_recoverable_rate_given_class_alternative": (
                exactly_recoverable_errors / max(error_samples_with_alternative, 1)
            ),
            "oracle_selected_alternatives": sum(index != 0 for index in selected_original),
            "oracle_rank_histogram": {
                str(rank): count for rank, count in sorted(rank_histogram.items())
            },
            "exact_corpus_micro_f1_oracle": oracle_metrics,
            "oracle_gain_over_top1": oracle_metrics["f1"] - top1_metrics["f1"],
        }

    return {
        "classification_basis": (
            "gold-free exact entity-set delta from constrained top1; overlap and "
            "same type are required for a pure boundary change"
        ),
        "top1_entity_error_samples": top1_error_samples,
        "unchanged_alternatives_ignored": unchanged_alternatives,
        "classes": {
            edit_class: audit_results[edit_class] for edit_class in EDIT_CLASSES
        },
        "groups": {
            group_name: audit_results[group_name] for group_name in EDIT_GROUPS
        },
    }
