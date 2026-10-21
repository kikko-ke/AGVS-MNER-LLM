"""Audit whether CRF top-k candidates leave enough headroom for LLM reranking."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.agvs_mner import AgvsMnerModel
from modules.crf_edit_decomposition import decompose_topk_edit_oracles
from modules.crf_topk import select_corpus_micro_f1_oracle, topk_viterbi_decode
from modules.dataset import MultimodalNerDataset
from modules.llm_dataset import extract_bio_entities
from modules.text_utils import normalize_unicode_text
from modules.trainer import load_compatible_model_checkpoint
from run_llm import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=10,
        help="Decode this multiple of max(K) before entity-equivalent deduplication.",
    )
    return parser.parse_args()


def _checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _word_level_prediction(
    prediction_ids: list[int],
    structural_mask: torch.Tensor,
    length: int,
    index_to_label: dict[int, str],
    outside_label_id: int,
    subword_label_id: int,
) -> list[str]:
    retained_prediction_ids = [
        prediction_ids[position]
        for position in range(1, length)
        if not bool(structural_mask[position])
    ]
    return [
        index_to_label[
            outside_label_id if prediction_id == subword_label_id else prediction_id
        ]
        for prediction_id in retained_prediction_ids
    ]


def _word_level_gold(
    gold_ids: torch.Tensor,
    structural_mask: torch.Tensor,
    length: int,
    index_to_label: dict[int, str],
) -> list[str]:
    return [
        index_to_label[int(gold_ids[position])]
        for position in range(1, length)
        if not bool(structural_mask[position])
    ]


def _entity_counts(
    gold: list[str],
    prediction: list[str],
) -> tuple[int, int, int]:
    gold_entities = extract_bio_entities(gold)
    predicted_entities = extract_bio_entities(prediction)
    true_positive = len(gold_entities & predicted_entities)
    false_positive = len(predicted_entities - gold_entities)
    false_negative = len(gold_entities - predicted_entities)
    return true_positive, false_positive, false_negative


def _canonical_bio(labels: list[str]) -> list[str]:
    """Project a CRF path to a legal BIO sequence with the same entity spans."""
    canonical = ["O"] * len(labels)
    for start, end, entity_type in sorted(extract_bio_entities(labels)):
        canonical[start] = f"B-{entity_type}"
        for position in range(start + 1, end):
            canonical[position] = f"I-{entity_type}"
    return canonical


def _structural_mask(
    dataset: MultimodalNerDataset,
    sample_index: int,
    active_length: int,
) -> torch.Tensor:
    """Derive CLS/SEP/continuation positions without reading gold NER labels."""
    structural = [True]
    for word in dataset.examples["words"][sample_index]:
        pieces = dataset.tokenizer.tokenize(normalize_unicode_text(word))
        if pieces:
            structural.extend([False, *([True] * (len(pieces) - 1))])
    structural = structural[: dataset.max_sequence_length - 1]
    structural.append(True)
    if len(structural) != active_length:
        raise RuntimeError(
            "Tokenizer-derived structure does not match the encoded attention mask: "
            f"sample={sample_index}, derived={len(structural)}, active={active_length}"
        )
    padded = torch.zeros(dataset.max_sequence_length, dtype=torch.bool)
    padded[:active_length] = torch.tensor(structural, dtype=torch.bool)
    return padded


def _metrics(gold: list[list[str]], predictions: list[list[str]]) -> dict[str, float]:
    return {
        "f1": float(f1_score(gold, predictions, zero_division=0)),
        "precision": float(precision_score(gold, predictions, zero_division=0)),
        "recall": float(recall_score(gold, predictions, zero_division=0)),
    }


def main() -> None:
    cli = parse_args()
    if not cli.ks or min(cli.ks) <= 0:
        raise ValueError("--ks must contain positive integers")
    if cli.oversample_factor <= 0:
        raise ValueError("--oversample-factor must be positive")
    requested_ks = sorted(set(cli.ks))
    maximum_k = max(requested_ks)
    decode_k = maximum_k * cli.oversample_factor

    checkpoint = Path(cli.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _checkpoint_config(checkpoint)
    config["device"] = cli.device
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.num_workers is not None:
        config["num_workers"] = cli.num_workers
    args = SimpleNamespace(**config)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    set_seed(int(getattr(args, "seed", 1234)))

    dataset = MultimodalNerDataset(args, f"MNER-{args.dataset}_{cli.split}.txt")
    label_mapping = dataset.processor.get_label_mapping(include_subword_label=True)
    index_to_label = {index: label for label, index in label_mapping.items()}
    outside_label_id = label_mapping["O"]
    subword_label_id = label_mapping["X"]

    model = AgvsMnerModel(list(label_mapping), args)
    missing = load_compatible_model_checkpoint(model, checkpoint, map_location="cpu")
    if missing:
        print(f"Optional checkpoint keys initialized locally: {missing}")
    model.to(args.device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(getattr(args, "batch_size", 4)),
        num_workers=int(getattr(args, "num_workers", 0)),
        shuffle=False,
    )
    gold_sequences: list[list[str]] = []
    model_top1: list[list[str]] = []
    candidate_sequences: list[list[list[str]]] = []
    candidate_scores: list[list[float]] = []
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="CRF top-k audit", unit="batch"):
            (
                input_ids,
                token_type_ids,
                attention_mask,
                image_pixel_values,
                labels,
                image_mask,
            ) = (value.to(args.device) for value in batch)
            decoded, _, features = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                image_pixel_values=image_pixel_values,
                image_mask=image_mask,
                labels=None,
                return_features=True,
            )
            emissions = features["token_emissions"]
            structural = torch.stack(
                [
                    _structural_mask(
                        dataset,
                        sample_offset + batch_index,
                        int(attention_mask[batch_index].sum()),
                    )
                    for batch_index in range(attention_mask.shape[0])
                ]
            ).to(args.device)
            active = attention_mask.bool()
            word_positions = ~structural & attention_mask.bool()
            allowed_tags = word_positions.unsqueeze(-1).expand_as(emissions).clone()
            allowed_tags[..., subword_label_id] = structural
            allowed_tags[~active] = True
            topk_paths, topk_scores = topk_viterbi_decode(
                model.crf,
                emissions,
                active,
                decode_k,
                allowed_tags=allowed_tags,
            )

            for batch_index, paths in enumerate(topk_paths):
                length = int(attention_mask[batch_index].sum())
                sample_structure = structural[batch_index].cpu()
                gold = _word_level_gold(
                    labels[batch_index].cpu(),
                    sample_structure,
                    length,
                    index_to_label,
                )
                baseline_prediction = _word_level_prediction(
                    decoded[batch_index],
                    sample_structure,
                    length,
                    index_to_label,
                    outside_label_id,
                    subword_label_id,
                )
                converted_candidates: list[list[str]] = []
                converted_scores: list[float] = []
                seen_candidates: set[tuple[str, ...]] = set()
                for path, path_score in zip(paths, topk_scores[batch_index]):
                    candidate = _canonical_bio(
                        _word_level_prediction(
                            path,
                            sample_structure,
                            length,
                            index_to_label,
                            outside_label_id,
                            subword_label_id,
                        )
                    )
                    key = tuple(candidate)
                    if key in seen_candidates:
                        continue
                    seen_candidates.add(key)
                    converted_candidates.append(candidate)
                    converted_scores.append(path_score)
                    if len(converted_candidates) == maximum_k:
                        break
                if not converted_candidates:
                    raise RuntimeError("Top-k decoding produced no finite candidate")
                gold_sequences.append(gold)
                model_top1.append(baseline_prediction)
                candidate_sequences.append(converted_candidates)
                candidate_scores.append(converted_scores)
            sample_offset += attention_mask.shape[0]

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "split": cli.split,
        "samples": len(gold_sequences),
        "candidate_generation": {
            "gold_entity_labels_used": False,
            "structural_constraints": "tokenizer-derived CLS/SEP/WordPiece positions",
            "decode_k": decode_k,
            "maximum_reported_k": maximum_k,
            "deduplication": "canonical word-level BIO sequence",
        },
        "model_top1": _metrics(gold_sequences, model_top1),
        "topk": {},
    }
    total_gold_entities = sum(
        len(extract_bio_entities(sequence)) for sequence in gold_sequences
    )
    for k in requested_ks:
        constrained_top1 = [candidates[0] for candidates in candidate_sequences]
        exact_sequence_matches = 0
        exact_entity_set_matches = 0
        union_matched_entities = 0
        retained_counts: list[int] = []
        retained_candidate_sequences: list[list[list[str]]] = []
        retained_candidate_scores: list[list[float]] = []
        retained_candidate_counts: list[list[tuple[int, int, int]]] = []
        normalized_score_gaps: list[float] = []
        top1_entity_errors = 0
        recoverable_top1_entity_errors = 0
        for gold, candidates, scores in zip(
            gold_sequences, candidate_sequences, candidate_scores
        ):
            retained_candidates = candidates[:k]
            retained_scores = scores[:k]
            retained_candidate_sequences.append(retained_candidates)
            retained_candidate_scores.append(retained_scores)
            retained_candidate_counts.append(
                [_entity_counts(gold, candidate) for candidate in retained_candidates]
            )
            retained_counts.append(len(retained_candidates))
            if any(candidate == gold for candidate in retained_candidates):
                exact_sequence_matches += 1
            gold_entities = extract_bio_entities(gold)
            candidate_entity_sets = [
                extract_bio_entities(candidate) for candidate in retained_candidates
            ]
            if any(entities == gold_entities for entities in candidate_entity_sets):
                exact_entity_set_matches += 1
            if candidate_entity_sets[0] != gold_entities:
                top1_entity_errors += 1
                if any(entities == gold_entities for entities in candidate_entity_sets[1:]):
                    recoverable_top1_entity_errors += 1
            union_entities = set().union(*candidate_entity_sets)
            union_matched_entities += len(gold_entities & union_entities)
            if len(retained_scores) >= 2:
                normalized_score_gaps.append(
                    (retained_scores[0] - retained_scores[1]) / max(len(gold), 1)
                )

        oracle_indices, oracle_micro_f1 = select_corpus_micro_f1_oracle(
            retained_candidate_counts,
            retained_candidate_scores,
        )
        oracle_predictions = [
            candidates[index]
            for candidates, index in zip(retained_candidate_sequences, oracle_indices)
        ]
        oracle_metrics = _metrics(gold_sequences, oracle_predictions)
        if abs(oracle_metrics["f1"] - oracle_micro_f1) > 1e-10:
            raise RuntimeError(
                "Entity-count oracle and seqeval disagree: "
                f"{oracle_micro_f1} vs {oracle_metrics['f1']}"
            )
        constrained_metrics = _metrics(gold_sequences, constrained_top1)
        rank_histogram = Counter(index + 1 for index in oracle_indices)

        summary["topk"][str(k)] = {
            "structurally_constrained_top1": constrained_metrics,
            "exact_corpus_micro_f1_oracle": oracle_metrics,
            "oracle_gain_over_constrained_top1": oracle_metrics["f1"]
            - constrained_metrics["f1"],
            "oracle_rank_histogram": {
                str(rank): count for rank, count in sorted(rank_histogram.items())
            },
            "exact_sequence_recall": exact_sequence_matches / max(len(gold_sequences), 1),
            "exact_entity_set_recall": exact_entity_set_matches
            / max(len(gold_sequences), 1),
            "gold_entity_union_recall": union_matched_entities
            / max(total_gold_entities, 1),
            "recoverable_top1_entity_errors": recoverable_top1_entity_errors,
            "top1_entity_error_samples": top1_entity_errors,
            "recoverable_top1_entity_error_rate": recoverable_top1_entity_errors
            / max(top1_entity_errors, 1),
            "mean_candidates": sum(retained_counts) / max(len(retained_counts), 1),
            "mean_top1_top2_score_gap_per_word": sum(normalized_score_gaps)
            / max(len(normalized_score_gaps), 1),
            "edit_decomposition": decompose_topk_edit_oracles(
                [extract_bio_entities(gold) for gold in gold_sequences],
                [
                    [extract_bio_entities(candidate) for candidate in candidates]
                    for candidates in retained_candidate_sequences
                ],
                retained_candidate_scores,
            ),
        }

    print("=== CRF top-k oracle summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

