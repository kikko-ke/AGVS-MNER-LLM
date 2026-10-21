from __future__ import annotations

import pytest

from modules.crf_edit_decomposition import (
    ADDITION,
    BOUNDARY_CHANGE,
    DELETION,
    EDIT_GROUPS,
    MIXED,
    TYPE_CHANGE,
    UNCHANGED,
    classify_entity_edit,
    decompose_topk_edit_oracles,
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
