from __future__ import annotations

import pytest

from modules.crf_edit_decomposition import (
    ADDITION,
    ATOMIC_ALL_PURE,
    ATOMIC_DELETION_OR_TYPE,
    BOUNDARY_CHANGE,
    DELETION,
    EDIT_GROUPS,
    MIXED,
    TYPE_CHANGE,
    UNCHANGED,
    classify_entity_edit,
    decompose_topk_edit_oracles,
    describe_atomic_entity_edit,
    is_atomic_entity_edit,
)


def test_classifies_mutually_exclusive_entity_edits() -> None:
    top1 = {(0, 1, "PER"), (3, 5, "ORG")}

    assert classify_entity_edit(top1, {(0, 1, "PER")}) == DELETION
    assert classify_entity_edit(top1, {*top1, (6, 7, "LOC")}) == ADDITION
    assert (
        classify_entity_edit(top1, {(0, 1, "LOC"), (3, 5, "ORG")})
        == TYPE_CHANGE
    )
    assert (
        classify_entity_edit(top1, {(0, 2, "PER"), (3, 5, "ORG")})
        == BOUNDARY_CHANGE
    )
    assert classify_entity_edit(top1, top1) == UNCHANGED


@pytest.mark.parametrize(
    "alternative",
    [
        {(0, 2, "LOC"), (3, 5, "ORG")},  # type and boundary both change
        {(0, 1, "PER"), (7, 8, "ORG")},  # same type moved without overlap
        {(0, 2, "PER")},  # boundary change plus deletion
        {(0, 1, "PER"), (3, 5, "ORG"), (8, 9, "MIS"), (9, 10, "LOC")},
    ],
)
def test_mixed_edits_are_not_mislabeled_as_boundary_changes(alternative) -> None:
    top1 = {(0, 1, "PER"), (3, 5, "ORG"), (6, 7, "LOC")}
    assert classify_entity_edit(top1, alternative) == MIXED


def test_pure_boundary_matching_handles_multiple_same_type_entities() -> None:
    top1 = {(0, 2, "PER"), (4, 6, "PER")}
    alternative = {(0, 1, "PER"), (3, 6, "PER")}
    assert classify_entity_edit(top1, alternative) == BOUNDARY_CHANGE


def test_atomic_entity_edits_reject_multi_entity_changes() -> None:
    top1 = {(0, 1, "PER"), (3, 4, "ORG")}

    assert is_atomic_entity_edit(top1, {(0, 1, "PER")})
    assert not is_atomic_entity_edit(top1, set())
    assert is_atomic_entity_edit(top1, {*top1, (6, 7, "MIS")})
    assert is_atomic_entity_edit(
        top1,
        {(0, 1, "LOC"), (3, 4, "ORG")},
    )
    assert not is_atomic_entity_edit(
        top1,
        {(0, 1, "LOC"), (3, 4, "MIS")},
    )
    assert is_atomic_entity_edit(
        top1,
        {(0, 2, "PER"), (3, 4, "ORG")},
    )
    assert not is_atomic_entity_edit(
        top1,
        {(0, 2, "PER"), (3, 5, "ORG")},
    )


@pytest.mark.parametrize(
    ("alternative", "edit_class", "removed", "added"),
    [
        (
            {(0, 1, "PER")},
            DELETION,
            {"start": 3, "end": 4, "label": "ORG"},
            None,
        ),
        (
            {(0, 1, "PER"), (3, 4, "ORG"), (6, 7, "MIS")},
            ADDITION,
            None,
            {"start": 6, "end": 7, "label": "MIS"},
        ),
        (
            {(0, 1, "LOC"), (3, 4, "ORG")},
            TYPE_CHANGE,
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 0, "end": 1, "label": "LOC"},
        ),
        (
            {(0, 2, "PER"), (3, 4, "ORG")},
            BOUNDARY_CHANGE,
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 0, "end": 2, "label": "PER"},
        ),
    ],
)
def test_describes_each_strict_atomic_entity_edit(
    alternative,
    edit_class,
    removed,
    added,
) -> None:
    top1 = {(0, 1, "PER"), (3, 4, "ORG")}

    assert describe_atomic_entity_edit(top1, alternative) == {
        "edit_class": edit_class,
        "removed_entity": removed,
        "added_entity": added,
        "is_atomic": True,
    }


@pytest.mark.parametrize(
    "alternative",
    [
        {(0, 1, "PER"), (3, 4, "ORG")},
        set(),
        {(0, 1, "LOC"), (3, 4, "MIS")},
        {(0, 2, "PER"), (3, 5, "ORG")},
    ],
)
def test_atomic_edit_description_rejects_unchanged_or_multi_edits(alternative) -> None:
    top1 = {(0, 1, "PER"), (3, 4, "ORG")}
    assert describe_atomic_entity_edit(top1, alternative) is None


def test_edit_oracles_allow_only_one_class_plus_top1() -> None:
    gold = [
        {(0, 1, "PER")},
        {(1, 2, "LOC")},
        {(3, 4, "ORG")},
        {(5, 7, "MIS")},
        {(8, 9, "PER")},
    ]
    candidates = [
        [
            {(0, 1, "PER"), (2, 3, "ORG")},
            {(0, 1, "PER")},  # deletion fixes a false positive
        ],
        [
            set(),
            {(1, 2, "LOC")},  # addition recovers a missing entity
        ],
        [
            {(3, 4, "PER")},
            {(3, 4, "ORG")},  # type correction
        ],
        [
            {(5, 6, "MIS")},
            {(5, 7, "MIS")},  # boundary correction
        ],
        [
            {(8, 9, "ORG"), (10, 11, "LOC")},
            {(8, 9, "PER")},  # type correction and deletion
        ],
    ]
    scores = [[2.0, 1.0] for _ in candidates]

    result = decompose_topk_edit_oracles(gold, candidates, scores)

    assert result["top1_entity_error_samples"] == 5
    for edit_class in (DELETION, ADDITION, TYPE_CHANGE, BOUNDARY_CHANGE, MIXED):
        section = result["classes"][edit_class]
        assert section["alternative_candidates"] == 1
        assert section["exactly_recoverable_top1_errors"] == 1
        assert section["oracle_selected_alternatives"] == 1
        assert section["exact_corpus_micro_f1_oracle"]["f1"] > 0

    # Each restricted oracle can repair exactly one of five independent errors;
    # it must not silently borrow candidates assigned to another edit class.
    assert result["classes"][DELETION]["exact_corpus_micro_f1_oracle"]["f1"] < 1.0
    assert result["classes"][MIXED]["exact_corpus_micro_f1_oracle"]["f1"] < 1.0

    safe_group = result["groups"]["deletion_or_entity_type"]
    assert safe_group["allowed_edit_classes"] == list(
        EDIT_GROUPS["deletion_or_entity_type"]
    )
    assert safe_group["alternative_candidates"] == 2
    assert safe_group["exactly_recoverable_top1_errors"] == 2
    assert safe_group["oracle_selected_alternatives"] == 2
    assert safe_group["oracle_gain_over_top1"] > result["classes"][DELETION][
        "oracle_gain_over_top1"
    ]
    atomic_groups = result["atomic_groups"]
    assert atomic_groups[ATOMIC_DELETION_OR_TYPE]["alternative_candidates"] == 2
    assert atomic_groups[ATOMIC_ALL_PURE]["alternative_candidates"] == 4
    assert atomic_groups[ATOMIC_ALL_PURE]["exactly_recoverable_top1_errors"] == 4
    assert result["atomic_classes"][ADDITION]["alternative_candidates"] == 1
    assert result["atomic_classes"][BOUNDARY_CHANGE]["alternative_candidates"] == 1


def test_decomposition_uses_exact_corpus_micro_f1_not_sentence_average() -> None:
    gold = [
        {(0, 1, "PER")},
        {(index, index + 1, "ORG") for index in range(100)},
    ]
    noisy_top1 = {(0, 1, "PER")} | {
        (index + 2, index + 3, "LOC") for index in range(9)
    }
    candidates = [
        [
            noisy_top1,
            set(),
        ],
        [gold[1]],
    ]
    scores = [[2.0, 1.0], [1.0]]

    result = decompose_topk_edit_oracles(gold, candidates, scores)

    deletion = result["classes"][DELETION]
    assert deletion["oracle_selected_alternatives"] == 1
    assert deletion["exact_corpus_micro_f1_oracle"]["f1"] == pytest.approx(200 / 201)


def test_atomic_oracle_reports_rank_margin_and_target_headroom() -> None:
    gold = [
        {(0, 1, "PER")},
        {(0, 1, "LOC"), (2, 3, "ORG")},
        {(0, 1, "MIS")},
    ]
    candidates = [
        [
            {(0, 1, "ORG")},
            {(0, 1, "LOC")},
            {(0, 1, "PER")},
        ],
        [
            {(0, 1, "LOC"), (2, 3, "ORG"), (4, 5, "PER")},
            {(0, 1, "LOC"), (2, 3, "ORG")},
            {(0, 1, "LOC")},
        ],
        [
            {(0, 1, "PER"), (2, 3, "ORG")},
            set(),  # multi-deletion must not enter the atomic oracle
        ],
    ]
    scores = [
        [10.0, 9.5, 9.0],
        [8.0, 7.8, 7.0],
        [6.0, 5.9],
    ]

    result = decompose_topk_edit_oracles(
        gold,
        candidates,
        scores,
        normalization_lengths=[2, 2, 2],
        target_f1=0.65,
        atomic_route_budgets=[1, 2],
    )
    atomic = result["atomic_groups"][ATOMIC_DELETION_OR_TYPE]
    all_pure = result["atomic_groups"][ATOMIC_ALL_PURE]

    assert atomic["atomic_single_edit_required"] is True
    assert atomic["alternative_candidates"] == 3
    assert atomic["exactly_recoverable_top1_errors"] == 2
    assert atomic["cumulative_rank_ceiling_oracles"]["2"][
        "exactly_recoverable_top1_errors"
    ] == 1
    assert atomic["cumulative_rank_ceiling_oracles"]["3"][
        "exactly_recoverable_top1_errors"
    ] == 2
    assert atomic["target_analysis"]["target_reachable"] is True
    assert all_pure["alternative_candidates"] == atomic["alternative_candidates"]

    route = atomic["low_margin_route_audit"]
    assert route["eligible_samples"] == 2
    assert route["budgets"]["1"]["routed_samples"] == 1
    assert route["budgets"]["1"][
        "maximum_normalized_top1_top2_margin"
    ] == pytest.approx(0.1)
    assert route["budgets"]["2"]["routed_samples"] == 2
