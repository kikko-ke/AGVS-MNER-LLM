"""Top-k Viterbi decoding utilities for the linear-chain CRF."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


EntityCounts = tuple[int, int, int]


def select_corpus_micro_f1_oracle(
    candidate_counts: Sequence[Sequence[EntityCounts]],
    candidate_scores: Sequence[Sequence[float]] | None = None,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> tuple[list[int], float]:
    """Select one candidate per sample to maximize corpus entity micro F1.

    Micro F1 is a ratio of corpus totals, so maximizing sentence-level F1
    independently is not an exact oracle.  This uses Dinkelbach's method for
    the finite fractional program

    ``2 * sum(tp) / (2 * sum(tp) + sum(fp) + sum(fn))``.

    CRF scores only break ties in the transformed objective and therefore do
    not change the oracle upper bound.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if candidate_scores is not None and len(candidate_scores) != len(candidate_counts):
        raise ValueError("candidate_scores must have one row per sample")
    if not candidate_counts:
        return [], 0.0

    normalized_counts: list[list[EntityCounts]] = []
    normalized_scores: list[list[float]] = []
    for sample_index, counts_row in enumerate(candidate_counts):
        if not counts_row:
            raise ValueError("Every sample must contain at least one candidate")
        row: list[EntityCounts] = []
        for counts in counts_row:
            if len(counts) != 3:
                raise ValueError("Each count tuple must contain tp, fp, and fn")
            tp, fp, fn = (int(value) for value in counts)
            if min(tp, fp, fn) < 0:
                raise ValueError("Entity counts must be non-negative")
            row.append((tp, fp, fn))
        normalized_counts.append(row)

        if candidate_scores is None:
            normalized_scores.append([0.0] * len(row))
        else:
            score_row = [float(value) for value in candidate_scores[sample_index]]
            if len(score_row) != len(row):
                raise ValueError("Each score row must match its candidate count row")
            if not all(math.isfinite(value) for value in score_row):
                raise ValueError("Candidate scores must be finite")
            normalized_scores.append(score_row)

    ratio = 0.0
    previous_selection: list[int] | None = None
    for _ in range(max_iterations):
        selection: list[int] = []
        for counts_row, score_row in zip(normalized_counts, normalized_scores):
            best_index = max(
                range(len(counts_row)),
                key=lambda index: (
                    2 * counts_row[index][0]
                    - ratio
                    * (
                        2 * counts_row[index][0]
                        + counts_row[index][1]
                        + counts_row[index][2]
                    ),
                    score_row[index],
                    -index,
                ),
            )
            selection.append(best_index)

        true_positive = sum(
            normalized_counts[sample_index][candidate_index][0]
            for sample_index, candidate_index in enumerate(selection)
        )
        false_positive = sum(
            normalized_counts[sample_index][candidate_index][1]
            for sample_index, candidate_index in enumerate(selection)
        )
        false_negative = sum(
            normalized_counts[sample_index][candidate_index][2]
            for sample_index, candidate_index in enumerate(selection)
        )
        numerator = 2 * true_positive
        denominator = numerator + false_positive + false_negative
        if denominator == 0:
            return selection, 0.0

        residual = numerator - ratio * denominator
        updated_ratio = numerator / denominator
        if abs(residual) <= tolerance or selection == previous_selection:
            return selection, updated_ratio
        previous_selection = selection
        ratio = updated_ratio

    raise RuntimeError("Corpus micro-F1 oracle did not converge")


def _topk_single(
    emissions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    transitions: torch.Tensor,
    k: int,
    allowed_tags: torch.Tensor | None,
) -> tuple[list[list[int]], list[float]]:
    """Decode one unpadded sequence while retaining k paths per tag."""
    if emissions.ndim != 2:
        raise ValueError("Single-sequence emissions must have shape [length, tags]")
    if emissions.shape[0] == 0:
        return [], []
    if k <= 0:
        raise ValueError("k must be positive")

    length, num_tags = emissions.shape
    if allowed_tags is not None:
        if allowed_tags.shape != emissions.shape:
            raise ValueError("allowed_tags must match the emission shape")
        if not allowed_tags.any(dim=-1).all():
            raise ValueError("Every active token must allow at least one tag")
        emissions = emissions.masked_fill(~allowed_tags.bool(), float("-inf"))

    # scores[tag, rank]. At the first position there is one path per tag.
    scores = (start_transitions + emissions[0]).unsqueeze(1)
    history: list[tuple[torch.Tensor, int]] = []

    for position in range(1, length):
        previous_k = scores.shape[1]
        candidates = (
            scores[:, :, None]
            + transitions[:, None, :]
            + emissions[position][None, None, :]
        )
        candidates = candidates.permute(2, 0, 1).reshape(num_tags, -1)
        retained_k = min(k, candidates.shape[1])
        scores, backpointers = candidates.topk(retained_k, dim=1)
        history.append((backpointers, previous_k))

    final_scores = scores + end_transitions[:, None]
    flattened_scores = final_scores.reshape(-1)
    retained_k = min(k, flattened_scores.numel())
    best_scores, best_indices = flattened_scores.topk(retained_k)
    final_rank_count = scores.shape[1]

    paths: list[list[int]] = []
    path_scores: list[float] = []
    for score, flattened_index in zip(best_scores, best_indices):
        if not torch.isfinite(score):
            continue
        tag = int(flattened_index // final_rank_count)
        rank = int(flattened_index % final_rank_count)
        reversed_path = [tag]
        for backpointers, previous_k in reversed(history):
            previous_flattened = int(backpointers[tag, rank])
            tag = previous_flattened // previous_k
            rank = previous_flattened % previous_k
            reversed_path.append(tag)
        paths.append(list(reversed(reversed_path)))
        path_scores.append(float(score))
    return paths, path_scores


def topk_viterbi_decode(
    crf: Any,
    emissions: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    allowed_tags: torch.Tensor | None = None,
) -> tuple[list[list[list[int]]], list[list[float]]]:
    """Return top-k CRF paths and unnormalised scores for every batch item.

    The implementation follows ``torchcrf.CRF.decode`` transition semantics.
    Computation is moved to CPU because this audit favours deterministic,
    inspectable decoding over training throughput.
    """
    if emissions.ndim != 3:
        raise ValueError("emissions must have shape [batch, length, tags]")
    if emissions.shape[1] == 0 or emissions.shape[2] == 0:
        raise ValueError("emissions must contain at least one token and one tag")
    if mask.shape != emissions.shape[:2]:
        raise ValueError("mask must match the batch and sequence dimensions")
    if allowed_tags is not None and allowed_tags.shape != emissions.shape:
        raise ValueError("allowed_tags must match emissions")
    if k <= 0:
        raise ValueError("k must be positive")

    emissions_cpu = emissions.detach().float().cpu()
    mask_cpu = mask.detach().bool().cpu()
    allowed_cpu = allowed_tags.detach().bool().cpu() if allowed_tags is not None else None
    start = crf.start_transitions.detach().float().cpu()
    end = crf.end_transitions.detach().float().cpu()
    transitions = crf.transitions.detach().float().cpu()

    batch_size, _, num_tags = emissions_cpu.shape
    if start.shape != (num_tags,) or end.shape != (num_tags,):
        raise ValueError("CRF start/end transitions must match the emission tag dimension")
    if transitions.shape != (num_tags, num_tags):
        raise ValueError("CRF transitions must have shape [tags, tags]")
    if not mask_cpu[:, 0].all():
        raise ValueError("Every mask must start with an active token")
    if ((~mask_cpu[:, :-1]) & mask_cpu[:, 1:]).any():
        raise ValueError("Each mask must be one contiguous active prefix")
    if not torch.isfinite(emissions_cpu[mask_cpu]).all():
        raise ValueError("Active emissions must be finite")
    if not all(torch.isfinite(value).all() for value in (start, end, transitions)):
        raise ValueError("CRF transition parameters must be finite")

    all_paths: list[list[list[int]]] = []
    all_scores: list[list[float]] = []
    for batch_index in range(batch_size):
        length = int(mask_cpu[batch_index].sum())
        sample_allowed = (
            allowed_cpu[batch_index, :length] if allowed_cpu is not None else None
        )
        paths, scores = _topk_single(
            emissions_cpu[batch_index, :length],
            start,
            end,
            transitions,
            k,
            sample_allowed,
        )
        all_paths.append(paths)
        all_scores.append(scores)
    return all_paths, all_scores

