"""Audit whether CRF top-k candidates leave enough headroom for LLM reranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from models.agvs_mner import AgvsMnerModel
from modules.crf_topk import topk_viterbi_decode
from modules.dataset import MultimodalNerDataset
from modules.llm_dataset import extract_bio_entities
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
    return parser.parse_args()


def _checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _word_level_sequence(
    gold_ids: torch.Tensor,
    prediction_ids: list[int],
    length: int,
    index_to_label: dict[int, str],
    outside_label_id: int,
    subword_label_id: int,
) -> tuple[list[str], list[str]]:
    gold = gold_ids[:length].tolist()[1:length]
    prediction = prediction_ids[1:length]
    retained = [
        (gold_id, prediction_id)
        for gold_id, prediction_id in zip(gold, prediction)
        if gold_id != subword_label_id
    ]
    gold_labels = [index_to_label[gold_id] for gold_id, _ in retained]
    predicted_labels = [
        index_to_label[
            outside_label_id if prediction_id == subword_label_id else prediction_id
        ]
        for _, prediction_id in retained
    ]
    return gold_labels, predicted_labels


def _oracle_key(
    gold: list[str],
    prediction: list[str],
    sequence_score: float,
) -> tuple[float, int, int, float]:
    gold_entities = extract_bio_entities(gold)
    predicted_entities = extract_bio_entities(prediction)
    true_positive = len(gold_entities & predicted_entities)
    false_positive = len(predicted_entities - gold_entities)
    false_negative = len(gold_entities - predicted_entities)
    denominator = 2 * true_positive + false_positive + false_negative
    sentence_f1 = 1.0 if denominator == 0 else 2 * true_positive / denominator
    return sentence_f1, true_positive, -false_positive, sequence_score


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
    requested_ks = sorted(set(cli.ks))
    maximum_k = max(requested_ks)

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

    with torch.no_grad():
        for batch in loader:
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
            allowed_tags = torch.ones_like(emissions, dtype=torch.bool)
            structural = labels.eq(subword_label_id) & attention_mask.bool()
            word_positions = ~structural & attention_mask.bool()
            allowed_tags[structural] = False
            allowed_tags[..., subword_label_id] = ~word_positions
            topk_paths, topk_scores = topk_viterbi_decode(
                model.crf,
                emissions,
                attention_mask.bool(),
                maximum_k,
                allowed_tags=allowed_tags,
            )

            for batch_index, paths in enumerate(topk_paths):
                length = int(attention_mask[batch_index].sum())
                gold, baseline_prediction = _word_level_sequence(
                    labels[batch_index].cpu(),
                    decoded[batch_index],
                    length,
                    index_to_label,
                    outside_label_id,
                    subword_label_id,
                )
                converted_candidates = [
                    _word_level_sequence(
                        labels[batch_index].cpu(),
                        path,
                        length,
                        index_to_label,
                        outside_label_id,
                        subword_label_id,
                    )[1]
                    for path in paths
                ]
                gold_sequences.append(gold)
                model_top1.append(baseline_prediction)
                candidate_sequences.append(converted_candidates)
                candidate_scores.append(topk_scores[batch_index])

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "split": cli.split,
        "samples": len(gold_sequences),
        "model_top1": _metrics(gold_sequences, model_top1),
        "topk": {},
    }
    total_gold_entities = sum(
        len(extract_bio_entities(sequence)) for sequence in gold_sequences
    )
    for k in requested_ks:
        constrained_top1 = [candidates[0] for candidates in candidate_sequences]
        oracle_predictions: list[list[str]] = []
        exact_sequence_matches = 0
        union_matched_entities = 0
        retained_counts: list[int] = []
        for gold, candidates, scores in zip(
            gold_sequences, candidate_sequences, candidate_scores
        ):
            retained_candidates = candidates[:k]
            retained_scores = scores[:k]
            retained_counts.append(len(retained_candidates))
            if any(candidate == gold for candidate in retained_candidates):
                exact_sequence_matches += 1
            gold_entities = extract_bio_entities(gold)
            union_entities = set().union(
                *(extract_bio_entities(candidate) for candidate in retained_candidates)
            )
            union_matched_entities += len(gold_entities & union_entities)
            best_index = max(
                range(len(retained_candidates)),
                key=lambda index: _oracle_key(
                    gold,
                    retained_candidates[index],
                    retained_scores[index],
                ),
            )
            oracle_predictions.append(retained_candidates[best_index])

        summary["topk"][str(k)] = {
            "constrained_top1": _metrics(gold_sequences, constrained_top1),
            "oracle": _metrics(gold_sequences, oracle_predictions),
            "exact_sequence_recall": exact_sequence_matches / max(len(gold_sequences), 1),
            "gold_entity_union_recall": union_matched_entities
            / max(total_gold_entities, 1),
            "mean_candidates": sum(retained_counts) / max(len(retained_counts), 1),
        }

    print("=== CRF top-k oracle summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

