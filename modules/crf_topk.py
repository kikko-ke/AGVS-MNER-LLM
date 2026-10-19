"""Top-k Viterbi decoding utilities for the linear-chain CRF."""

from __future__ import annotations

from typing import Any

import torch


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

    all_paths: list[list[list[int]]] = []
    all_scores: list[list[float]] = []
    for batch_index in range(emissions_cpu.shape[0]):
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

