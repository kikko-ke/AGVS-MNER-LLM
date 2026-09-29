"""Entity-level evaluation utilities for BIO-style NER labels."""

from __future__ import annotations

import numpy as np

def get_chunks(label_ids: list[int], label_mapping: dict[str, int]) -> list[tuple[str, int, int]]:
    """Extract ``(entity_type, start, end)`` spans from a label-id sequence."""
    outside_label_id = label_mapping["O"]
    index_to_label = {index: label for label, index in label_mapping.items()}
    chunks: list[tuple[str, int, int]] = []
    entity_type: str | None = None
    start_index: int | None = None

    for index, label_id in enumerate(label_ids):
        if label_id == outside_label_id:
            if entity_type is not None:
                chunks.append((entity_type, start_index, index))
                entity_type, start_index = None, None
            continue

        prefix, current_type = get_chunk_type(label_id, index_to_label)
        if entity_type is None:
            entity_type, start_index = current_type, index
        elif current_type != entity_type or prefix == "B":
            chunks.append((entity_type, start_index, index))
            entity_type, start_index = current_type, index

    if entity_type is not None:
        chunks.append((entity_type, start_index, len(label_ids)))
    return chunks


def get_chunk_type(label_id: int, index_to_label: dict[int, str]) -> tuple[str, str]:
    """Split a BIO label id into its prefix and entity type."""
    label = index_to_label[label_id]
    return label.split("-")[0], label.split("-")[-1]


def evaluate(
    predicted_label_ids: list[list[int]],
    target_label_ids: list[list[int]],
    label_mapping: dict[str, int],
) -> tuple[float, float, float, float]:
    """Return token accuracy and entity-level F1, precision, and recall."""
    token_matches: list[bool] = []
    correct_predictions = 0
    predicted_entities = 0
    target_entities = 0

    for target_ids, prediction_ids in zip(target_label_ids, predicted_label_ids):
        token_matches.extend(
            target_id == prediction_id
            for target_id, prediction_id in zip(target_ids, prediction_ids)
        )
        target_chunks = set(get_chunks(target_ids, label_mapping))
        predicted_chunks = set(get_chunks(prediction_ids, label_mapping))
        correct_predictions += len(target_chunks & predicted_chunks)
        predicted_entities += len(predicted_chunks)
        target_entities += len(target_chunks)

    precision = correct_predictions / predicted_entities if predicted_entities else 0.0
    recall = correct_predictions / target_entities if target_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(token_matches))
    return accuracy, f1, precision, recall
