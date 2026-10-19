from __future__ import annotations

import itertools
from types import SimpleNamespace

import torch
from torchcrf import CRF

from modules.crf_topk import topk_viterbi_decode


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
        assert torch.isclose(torch.tensor(score), torch.tensor(_score_path(crf, emissions, tuple(path))))


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

