from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from torchcrf import CRF

from modules.crf_topk import select_corpus_micro_f1_oracle, topk_viterbi_decode


def _score_path(crf, emissions: torch.Tensor, path: tuple[int, ...]) -> float:
    score = crf.start_transitions[path[0]] + emissions[0, path[0]]
    for position in range(1, len(path)):
        score = (
            score
            + crf.transitions[path[position - 1], path[position]]
            + emissions[position, path[position]]
        )
    return float(score + crf.end_transitions[path[-1]])


def test_topk_viterbi_matches_brute_force_with_allowed_tags() -> None:
    crf = SimpleNamespace(
        start_transitions=torch.tensor([0.1, -0.2, 0.3]),
        end_transitions=torch.tensor([0.0, 0.4, -0.1]),
        transitions=torch.tensor(
            [
                [0.2, -0.3, 0.1],
                [0.5, 0.0, -0.4],
                [-0.2, 0.3, 0.2],
            ]
        ),
    )
    emissions = torch.tensor(
        [
            [0.4, -0.1, 0.2],
            [0.0, 0.3, -0.2],
            [0.1, -0.4, 0.5],
        ]
    )
    allowed = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
        ]
    )
    expected = []
    for path in itertools.product(range(3), repeat=3):
        if all(allowed[position, tag] for position, tag in enumerate(path)):
            expected.append((_score_path(crf, emissions, path), list(path)))
    expected.sort(reverse=True)

    paths, scores = topk_viterbi_decode(
        crf,
        emissions.unsqueeze(0),
        torch.ones((1, 3), dtype=torch.bool),
        k=5,
        allowed_tags=allowed.unsqueeze(0),
    )

    assert torch.allclose(
        torch.tensor(scores[0]),
        torch.tensor([score for score, _ in expected[:5]]),
    )
    assert len({tuple(path) for path in paths[0]}) == 5
    for path, score in zip(paths[0], scores[0]):
        assert all(allowed[position, tag] for position, tag in enumerate(path))
        expected_score = _score_path(crf, emissions, tuple(path))
        assert torch.isclose(torch.tensor(score), torch.tensor(expected_score))


def test_topk_viterbi_respects_variable_lengths() -> None:
    crf = SimpleNamespace(
        start_transitions=torch.zeros(2),
        end_transitions=torch.zeros(2),
        transitions=torch.zeros((2, 2)),
    )
    emissions = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 4.0], [9.0, 9.0]],
            [[0.0, 2.0], [5.0, 0.0], [0.0, 6.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    paths, _ = topk_viterbi_decode(crf, emissions, mask, k=1)

    assert paths == [[[0, 1]], [[1, 0, 1]]]


def test_top1_matches_torchcrf_decode() -> None:
    torch.manual_seed(17)
    crf = CRF(4, batch_first=True)
    emissions = torch.randn(3, 6, 4)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )

    expected = crf.decode(emissions, mask=mask)
    actual, _ = topk_viterbi_decode(crf, emissions, mask, k=1)

    assert [paths[0] for paths in actual] == expected


def test_topk_rejects_non_prefix_mask() -> None:
    crf = SimpleNamespace(
        start_transitions=torch.zeros(2),
        end_transitions=torch.zeros(2),
        transitions=torch.zeros((2, 2)),
    )
    emissions = torch.zeros((1, 3, 2))
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    with pytest.raises(ValueError, match="contiguous active prefix"):
        topk_viterbi_decode(crf, emissions, mask, k=1)


def test_corpus_micro_f1_oracle_matches_cartesian_brute_force() -> None:
    # Sentence-local F1 picks sample 1 candidate 0 (2/11 > 0), but doing so
    # hurts corpus micro F1 once the large fixed second sample is included.
    counts = [
        [(1, 9, 0), (0, 0, 1)],
        [(100, 0, 0)],
    ]
    scores = [[2.0, 1.0], [3.0]]
    selected, score = select_corpus_micro_f1_oracle(counts, scores)

    brute_force: list[tuple[float, tuple[int, ...]]] = []
    for choices in itertools.product(*(range(len(row)) for row in counts)):
        true_positive = sum(counts[i][choice][0] for i, choice in enumerate(choices))
        false_positive = sum(counts[i][choice][1] for i, choice in enumerate(choices))
        false_negative = sum(counts[i][choice][2] for i, choice in enumerate(choices))
        denominator = 2 * true_positive + false_positive + false_negative
        micro_f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        brute_force.append((micro_f1, choices))

    expected_score, expected_choices = max(brute_force)
    assert tuple(selected) == expected_choices == (1, 0)
    assert score == pytest.approx(expected_score)

