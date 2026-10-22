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
ATOMIC_DELETION_OR_TYPE = "single_deletion_or_single_entity_type"
ATOMIC_ALL_PURE = "single_pure_entity_edit"


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


def is_atomic_entity_edit(
    top1_entities: Iterable[tuple[Any, Any, Any]],
    alternative_entities: Iterable[tuple[Any, Any, Any]],
    edit_class: str | None = None,
) -> bool:
    """Return whether an alternative makes exactly one supported entity edit.

    Supported operations are one complete-entity deletion, one addition, one
    same-span type replacement, or one overlapping same-type boundary
    replacement.  This deliberately excludes candidates that happen to share
    a coarse edit class but modify several entities at once.
    """
    top1 = _normalize_entities(top1_entities)
    alternative = _normalize_entities(alternative_entities)
    label = edit_class or classify_entity_edit(top1, alternative)
    removed = top1 - alternative
    added = alternative - top1
    if label == DELETION:
        return len(removed) == 1 and not added
    if label == ADDITION:
        return len(added) == 1 and not removed
    if label == TYPE_CHANGE:
        return (
            len(removed) == 1
            and len(added) == 1
            and next(iter(removed))[:2] == next(iter(added))[:2]
        )
    if label == BOUNDARY_CHANGE:
        return (
            len(removed) == 1
            and len(added) == 1
            and next(iter(removed))[2] == next(iter(added))[2]
            and _overlaps(next(iter(removed)), next(iter(added)))
        )
    return False


def describe_atomic_entity_edit(
    top1_entities: Iterable[tuple[Any, Any, Any]],
    alternative_entities: Iterable[tuple[Any, Any, Any]],
) -> dict[str, Any] | None:
    """Describe one strict atomic alternative using only candidate content.

    The returned object is deliberately JSON-compatible so it can be written
    directly to a gold-free candidate artifact and shown to an LLM.  ``None``
    means that the alternative is unchanged, mixed, or changes more than one
    entity.  Gold labels are neither accepted nor consulted here.
    """
    top1 = _normalize_entities(top1_entities)
    alternative = _normalize_entities(alternative_entities)
    edit_class = classify_entity_edit(top1, alternative)
    if not is_atomic_entity_edit(top1, alternative, edit_class):
        return None

    removed = sorted(top1 - alternative)
    added = sorted(alternative - top1)

    def serialize(entity: Entity) -> dict[str, Any]:
        start, end, label = entity
        return {"start": start, "end": end, "label": label}

    return {
        "edit_class": edit_class,
        "removed_entity": serialize(removed[0]) if removed else None,
        "added_entity": serialize(added[0]) if added else None,
        "is_atomic": True,
    }


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


def _target_analysis(
    top1_f1: float,
    oracle_f1: float,
    target_f1: float,
) -> dict[str, Any]:
    gain_needed = max(target_f1 - top1_f1, 0.0)
    available_gain = max(oracle_f1 - top1_f1, 0.0)
    return {
        "target_f1": target_f1,
        "target_reachable": oracle_f1 + 1e-12 >= target_f1,
        "gain_needed_over_top1": gain_needed,
        "oracle_gain_available": available_gain,
        "fraction_of_oracle_gain_required": (
            0.0
            if gain_needed == 0.0
            else None
            if available_gain == 0.0
            else gain_needed / available_gain
        ),
    }


def decompose_topk_edit_oracles(
    gold_entities: Sequence[Iterable[tuple[Any, Any, Any]]],
    candidate_entities: Sequence[Sequence[Iterable[tuple[Any, Any, Any]]]],
    candidate_scores: Sequence[Sequence[float]],
    *,
    normalization_lengths: Sequence[int] | None = None,
    target_f1: float = 0.86,
    atomic_route_budgets: Sequence[int] = (50, 100, 200, 400, 800, 1200, 1600),
) -> dict[str, Any]:
    """Compute exact corpus micro-F1 under restricted top-1 edit sets."""
    if not (
        len(gold_entities) == len(candidate_entities) == len(candidate_scores)
    ):
        raise ValueError("Gold, candidate, and score rows must have equal length")
    if not 0.0 <= target_f1 <= 1.0:
        raise ValueError("target_f1 must be in [0, 1]")
    if normalization_lengths is not None:
        if len(normalization_lengths) != len(gold_entities):
            raise ValueError("normalization_lengths must match the number of rows")
        if any(int(length) <= 0 for length in normalization_lengths):
            raise ValueError("normalization_lengths must contain positive integers")
    if any(int(budget) <= 0 for budget in atomic_route_budgets):
        raise ValueError("atomic_route_budgets must contain positive integers")

    normalized_gold = [_normalize_entities(row) for row in gold_entities]
    normalized_candidates: list[list[EntitySet]] = []
    normalized_scores: list[list[float]] = []
    edit_labels: list[list[str]] = []
    atomic_edits: list[list[bool]] = []
    unchanged_alternatives = 0
    for candidates, scores in zip(candidate_entities, candidate_scores):
        if not candidates:
            raise ValueError("Every sample must contain a top-1 candidate")
        if len(candidates) != len(scores):
            raise ValueError("Every candidate row must match its score row")
        candidate_row = [_normalize_entities(candidate) for candidate in candidates]
        score_row = [float(score) for score in scores]
        labels = [UNCHANGED]
        atomic = [False]
        for candidate in candidate_row[1:]:
            label = classify_entity_edit(candidate_row[0], candidate)
            labels.append(label)
            atomic.append(is_atomic_entity_edit(candidate_row[0], candidate, label))
            unchanged_alternatives += int(label == UNCHANGED)
        normalized_candidates.append(candidate_row)
        normalized_scores.append(score_row)
        edit_labels.append(labels)
        atomic_edits.append(atomic)

    top1_counts = [
        _entity_counts(gold, candidates[0])
        for gold, candidates in zip(normalized_gold, normalized_candidates)
    ]
    top1_metrics = _metrics_from_counts(top1_counts)
    top1_error_samples = sum(
        candidates[0] != gold
        for gold, candidates in zip(normalized_gold, normalized_candidates)
    )

    def compute_audit(
        allowed_edit_classes: Sequence[str],
        *,
        require_atomic: bool = False,
        maximum_rank: int | None = None,
        allowed_samples: set[int] | None = None,
    ) -> dict[str, Any]:
        allowed_counts: list[list[EntityCounts]] = []
        allowed_scores: list[list[float]] = []
        original_indices: list[list[int]] = []
        alternative_candidates = 0
        samples_with_alternative = 0
        error_samples_with_alternative = 0
        locally_improvable_errors = 0
        exactly_recoverable_errors = 0

        for sample_index, (
            gold,
            candidates,
            scores,
            labels,
            atomic,
            baseline_counts,
        ) in enumerate(
            zip(
                normalized_gold,
                normalized_candidates,
                normalized_scores,
                edit_labels,
                atomic_edits,
                top1_counts,
            )
        ):
            indices = [
                0,
                *[
                    index
                    for index, label in enumerate(labels[1:], start=1)
                    if label in allowed_edit_classes
                    and (not require_atomic or atomic[index])
                    and (maximum_rank is None or index + 1 <= maximum_rank)
                    and (allowed_samples is None or sample_index in allowed_samples)
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
        result = {
            "allowed_edit_classes": list(allowed_edit_classes),
            "atomic_single_edit_required": require_atomic,
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
            "target_analysis": _target_analysis(
                top1_metrics["f1"], oracle_metrics["f1"], target_f1
            ),
        }
        return result

    audit_specs = {
        **{edit_class: (edit_class,) for edit_class in EDIT_CLASSES},
        **EDIT_GROUPS,
    }
    audit_results = {
        audit_name: compute_audit(allowed_edit_classes)
        for audit_name, allowed_edit_classes in audit_specs.items()
    }

    maximum_available_rank = max(len(row) for row in normalized_candidates)

    def compute_atomic_group(atomic_classes: Sequence[str]) -> dict[str, Any]:
        atomic_result = compute_audit(atomic_classes, require_atomic=True)
        atomic_result["cumulative_rank_ceiling_oracles"] = {
            str(rank): compute_audit(
                atomic_classes,
                require_atomic=True,
                maximum_rank=rank,
            )
            for rank in range(2, maximum_available_rank + 1)
        }

        if normalization_lengths is not None:
            atomic_eligible_samples = {
                sample_index
                for sample_index, (labels, atomic) in enumerate(
                    zip(edit_labels, atomic_edits)
                )
                if any(
                    label in atomic_classes and atomic[index]
                    for index, label in enumerate(labels[1:], start=1)
                )
            }
            routed_by_margin: list[tuple[float, int]] = []
            for sample_index in atomic_eligible_samples:
                scores = normalized_scores[sample_index]
                if len(scores) < 2:
                    continue
                margin = max(scores[0] - scores[1], 0.0) / int(
                    normalization_lengths[sample_index]
                )
                routed_by_margin.append((margin, sample_index))
            routed_by_margin.sort(key=lambda item: (item[0], item[1]))

            route_oracles: dict[str, Any] = {}
            for requested_budget in sorted(
                {int(value) for value in atomic_route_budgets}
            ):
                routed = routed_by_margin[:requested_budget]
                allowed_samples = {sample_index for _, sample_index in routed}
                section = compute_audit(
                    atomic_classes,
                    require_atomic=True,
                    allowed_samples=allowed_samples,
                )
                section["requested_route_budget"] = requested_budget
                section["routed_samples"] = len(routed)
                section["maximum_normalized_top1_top2_margin"] = (
                    routed[-1][0] if routed else None
                )
                route_oracles[str(requested_budget)] = section
            atomic_result["low_margin_route_audit"] = {
                "routing_basis": (
                    "lowest normalized top1-top2 CRF margin among samples that "
                    "contain an alternative in this atomic edit group"
                ),
                "eligible_samples": len(routed_by_margin),
                "budgets": route_oracles,
            }
        return atomic_result

    atomic_deletion_or_type = (DELETION, TYPE_CHANGE)
    atomic_all_pure = (DELETION, ADDITION, TYPE_CHANGE, BOUNDARY_CHANGE)

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
        "atomic_classes": {
            edit_class: compute_audit((edit_class,), require_atomic=True)
            for edit_class in (
                DELETION,
                ADDITION,
                TYPE_CHANGE,
                BOUNDARY_CHANGE,
            )
        },
        "atomic_groups": {
            ATOMIC_DELETION_OR_TYPE: compute_atomic_group(
                atomic_deletion_or_type
            ),
            ATOMIC_ALL_PURE: compute_atomic_group(atomic_all_pure),
        },
    }
