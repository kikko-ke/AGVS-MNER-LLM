"""Gold-free conversion of CRF paths into word-level reranking candidates."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from modules.reranker import canonical_bio, serialize_entities


def word_level_prediction(
    prediction_ids: Sequence[int],
    structural_mask: torch.Tensor,
    active_length: int,
    index_to_label: dict[int, str],
    outside_label_id: int,
    subword_label_id: int,
) -> list[str]:
    """Remove CLS/SEP/WordPiece continuation positions from one CRF path."""
    retained_ids = [
        int(prediction_ids[position])
        for position in range(1, active_length)
        if not bool(structural_mask[position])
    ]
    return [
        index_to_label[
            outside_label_id if prediction_id == subword_label_id else prediction_id
        ]
        for prediction_id in retained_ids
    ]


def build_allowed_tag_mask(
    emissions: torch.Tensor,
    attention_mask: torch.Tensor,
    structural_mask: torch.Tensor,
    subword_label_id: int,
) -> torch.Tensor:
    """Allow NER tags only on words and X only on structural positions."""
    if emissions.shape[:2] != attention_mask.shape:
        raise ValueError("attention_mask must match emissions batch/length dimensions")
    if structural_mask.shape != attention_mask.shape:
        raise ValueError("structural_mask must match attention_mask")
    active = attention_mask.bool()
    word_positions = ~structural_mask.bool() & active
    allowed = word_positions.unsqueeze(-1).expand_as(emissions).clone()
    allowed[..., subword_label_id] = structural_mask.bool()
    allowed[~active] = True
    return allowed


def deduplicate_topk_paths(
    paths: Sequence[Sequence[int]],
    scores: Sequence[float],
    *,
    structural_mask: torch.Tensor,
    active_length: int,
    index_to_label: dict[int, str],
    outside_label_id: int,
    subword_label_id: int,
    maximum_candidates: int,
) -> list[dict[str, Any]]:
    """Convert score-ordered token paths into unique canonical word BIO paths."""
    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for path, score in zip(paths, scores):
        bio = canonical_bio(
            word_level_prediction(
                path,
                structural_mask,
                active_length,
                index_to_label,
                outside_label_id,
                subword_label_id,
            )
        )
        key = tuple(bio)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"bio": bio, "crf_score": float(score)})
        if len(candidates) == maximum_candidates:
            break
    if not candidates:
        raise RuntimeError("Top-k decoding produced no finite candidate")
    return candidates
