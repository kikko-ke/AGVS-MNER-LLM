"""Dataset wrapper that adds cached LLM teacher signals to AGVS samples."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from modules.evidence_distillation import (
    build_evidence_distillation_targets,
    load_evidence_embedding_cache,
    sha256_file,
)
from modules.text_utils import normalize_unicode_text


ENTITY_TYPES = {"MIS", "PER", "ORG", "LOC"}


def extract_bio_entities(labels: list[str]) -> set[tuple[int, int, str]]:
    """Return ``(start, end, type)`` spans from a word-level BIO sequence."""
    entities: set[tuple[int, int, str]] = set()
    start: int | None = None
    current_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, current_type
        if start is not None and current_type is not None:
            entities.add((start, end, current_type))
        start = None
        current_type = None

    for index, raw_label in enumerate([*labels, "O"]):
        label = str(raw_label).upper().replace("MISC", "MIS")
        if label == "O" or "-" not in label:
            close(index)
            continue

        prefix, entity_type = label.split("-", 1)
        if entity_type not in ENTITY_TYPES:
            close(index)
            continue
        if prefix not in {"B", "I"}:
            close(index)
            continue
        if prefix == "B" or current_type != entity_type:
            close(index)
            start = index
            current_type = entity_type

    return entities


def build_grounding_targets(
    *,
    words: list[str],
    gold_labels: list[str],
    teacher_entities: list[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
    image_mask: torch.Tensor,
    confidence_threshold: float,
    class_weights: dict[str, float] | None = None,
    background_ratio: float = 0.0,
    teacher_task: str = "grounding",
    visual_utility_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build gold-consistent entity-to-image targets from LLM evidence.

    The LLM is trusted only for the information absent from the MNER labels:
    which valid image(s) support an entity. Its span and type must exactly agree
    with a gold training entity, so an incorrect teacher NER prediction cannot
    change the CRF target.
    """
    num_images = int(image_mask.numel())
    # Four image slots plus a NONE class meaning that visual evidence should
    # not be used for this entity.
    targets = torch.zeros((max_sequence_length, num_images + 1), dtype=torch.float)
    weights = torch.zeros(max_sequence_length, dtype=torch.float)
    priorities = torch.zeros(max_sequence_length, dtype=torch.float)
    gold_entities = extract_bio_entities(gold_labels)
    valid_images = image_mask.bool().tolist()

    # The base dataset tokenizes each word exactly once and inserts [CLS] at 0.
    # Supervise the first WordPiece of every entity word so long words do not
    # receive more weight merely because BERT splits them into more pieces.
    first_piece_positions: list[int | None] = []
    cursor = 1
    maximum_position = max_sequence_length - 1
    for word in words:
        pieces = tokenizer.tokenize(normalize_unicode_text(word))
        first_piece_positions.append(cursor if pieces and cursor < maximum_position else None)
        cursor += len(pieces)

    for entity in teacher_entities:
        try:
            start = int(entity["start"])
            end = int(entity["end"])
            entity_type = str(entity["label"]).upper().replace("MISC", "MIS")
            if entity_type.startswith(("B-", "I-")):
                entity_type = entity_type[2:]
            confidence = min(max(float(entity.get("confidence", 1.0)), 0.0), 1.0)
            visual_utility = min(
                max(float(entity.get("visual_utility", 1.0)), 0.0), 1.0
            )
            text_sufficient = entity.get("text_sufficient", False)
            if not isinstance(text_sufficient, bool):
                continue
        except (KeyError, TypeError, ValueError):
            continue

        if confidence < confidence_threshold:
            continue
        if (start, end, entity_type) not in gold_entities:
            continue

        raw_image_ids = entity.get("image_ids", [])
        if not isinstance(raw_image_ids, (list, tuple)):
            continue
        image_ids: list[int] = []
        for raw_image_id in raw_image_ids:
            try:
                image_id = int(raw_image_id)
            except (TypeError, ValueError):
                continue
            if 0 <= image_id < num_images and valid_images[image_id]:
                image_ids.append(image_id)
        image_ids = sorted(set(image_ids))
        # Utility supervision asks a counterfactual question: does vision add
        # entity-type information beyond text? Low-utility/text-sufficient
        # entities are explicit NONE targets even when an image depicts them.
        target_distribution = torch.zeros(num_images + 1, dtype=torch.float)
        if teacher_task == "utility":
            if text_sufficient or visual_utility < visual_utility_threshold:
                target_distribution[num_images] = 1.0
            elif not raw_image_ids:
                target_distribution[num_images] = 1.0
            elif not image_ids:
                # Selected slots existed in the response but none is a valid
                # decodable image; do not turn corrupted evidence into NONE.
                continue
            else:
                target_distribution[image_ids] = visual_utility / len(image_ids)
                target_distribution[num_images] = 1.0 - visual_utility
        else:
            # An explicitly empty list is a useful negative target (NONE). If
            # selected slots are invalid, skip rather than changing semantics.
            if not raw_image_ids:
                target_distribution[num_images] = 1.0
            elif not image_ids:
                continue
            else:
                target_distribution[image_ids] = 1.0 / len(image_ids)
        supervised_positions = [
            first_piece_positions[word_index]
            for word_index in range(start, min(end, len(first_piece_positions)))
            if first_piece_positions[word_index] is not None
        ]
        if not supervised_positions:
            continue
        class_weight = (class_weights or {}).get(entity_type, 1.0)
        token_weight = confidence * class_weight / len(supervised_positions)
        for position in supervised_positions:
            assert position is not None
            if confidence < float(priorities[position]):
                continue
            targets[position] = target_distribution
            weights[position] = token_weight
            priorities[position] = confidence

    # The image selector is used at inference for every token, while the
    # teacher only annotates gold entities. Give background words a weak NONE
    # target so unsupervised O tokens do not inject arbitrary visual context.
    # The total background mass is a fraction of the accepted entity mass,
    # keeping long sentences from overwhelming the teacher supervision.
    if background_ratio > 0:
        entity_word_indices = {
            word_index
            for start, end, _ in gold_entities
            for word_index in range(start, end)
        }
        background_positions = [
            position
            for word_index, position in enumerate(first_piece_positions)
            if position is not None and word_index not in entity_word_indices
        ]
        if background_positions:
            positive_mass = float(weights.sum())
            reference_mass = positive_mass if positive_mass > 0 else 1.0
            background_weight = (
                float(background_ratio) * reference_mass / len(background_positions)
            )
            for position in background_positions:
                assert position is not None
                targets[position].zero_()
                targets[position, num_images] = 1.0
                weights[position] = background_weight

    return targets, weights


class TeacherCacheDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        base_dataset: Any,
        cache_path: str | Path,
        grounding_confidence_threshold: float = 0.7,
        enable_grounding: bool = True,
        expected_teacher_task: str | None = None,
        balance_grounding_classes: bool = False,
        grounding_background_ratio: float = 0.0,
        visual_utility_threshold: float = 0.5,
        enable_evidence_distillation: bool = False,
        evidence_embedding_path: str | Path | None = None,
        evidence_max_entities: int = 8,
        evidence_embedding_dim: int = 768,
    ) -> None:
        self.base_dataset = base_dataset
        self.cache_path = Path(cache_path)
        self.grounding_confidence_threshold = grounding_confidence_threshold
        self.enable_grounding = enable_grounding
        self.expected_teacher_task = expected_teacher_task
        self.grounding_background_ratio = grounding_background_ratio
        self.visual_utility_threshold = visual_utility_threshold
        self.enable_evidence_distillation = enable_evidence_distillation
        self.evidence_max_entities = int(evidence_max_entities)
        self.evidence_embedding_dim = int(evidence_embedding_dim)
        if self.evidence_max_entities <= 0:
            raise ValueError("evidence_max_entities must be positive")
        if self.evidence_embedding_dim <= 0:
            raise ValueError("evidence_embedding_dim must be positive")
        self.evidence_rows_by_sample: dict[int, list[int]] = {}
        self.evidence_artifact: dict[str, Any] | None = None
        if evidence_embedding_path is not None:
            evidence_index, artifact = load_evidence_embedding_cache(
                evidence_embedding_path
            )
            actual_cache_hash = sha256_file(self.cache_path)
            if artifact["teacher_cache_sha256"] != actual_cache_hash:
                raise ValueError(
                    "Evidence artifact teacher-cache hash does not match the active cache"
                )
            source_path = getattr(self.base_dataset, "source_path", None)
            if source_path is not None:
                actual_input_hash = sha256_file(source_path)
                if artifact["input_sha256"] != actual_input_hash:
                    raise ValueError(
                        "Evidence artifact input hash does not match the active dataset"
                    )
            artifact_dim = int(artifact["embeddings"].shape[1])
            if artifact_dim != self.evidence_embedding_dim:
                raise ValueError(
                    "Evidence embedding dimension mismatch: "
                    f"artifact={artifact_dim}, expected={self.evidence_embedding_dim}"
                )
            self.evidence_rows_by_sample = evidence_index
            self.evidence_artifact = dict(artifact)
        self.grounding_class_weights = self._grounding_class_weights(
            balance_grounding_classes
        )
        self.records: dict[int, dict[str, Any]] = {}
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as cache_file:
                for line_number, line in enumerate(cache_file):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    sample_id = int(record.get("sample_id", line_number))
                    self.records[sample_id] = record
        self.teacher_tasks = Counter(
            str(record.get("teacher_task", "mner"))
            for record in self.records.values()
        )
        if expected_teacher_task is not None:
            unexpected = set(self.teacher_tasks) - {expected_teacher_task}
            if unexpected:
                raise ValueError(
                    f"Teacher cache contains {sorted(unexpected)}, expected "
                    f"{expected_teacher_task!r}; use a separate cache directory"
                )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _grounding_class_weights(self, enabled: bool) -> dict[str, float]:
        if not enabled:
            return {}
        counts: Counter[str] = Counter()
        for labels in self.base_dataset.examples["labels"]:
            counts.update(entity_type for _, _, entity_type in extract_bio_entities(labels))
        if not counts:
            return {}
        largest = max(counts.values())
        return {
            entity_type: min(math.sqrt(largest / count), 2.0)
            for entity_type, count in counts.items()
            if count > 0
        }

    def _default_teacher(self, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        num_labels = len(self.base_dataset.label_mapping)
        token_probs = torch.full((sequence_length, num_labels), 1.0 / num_labels)
        image_scores = torch.full((4,), 0.25)
        return token_probs, image_scores

    def grounding_summary(self) -> dict[str, Any]:
        """Count candidate supervision without decoding every image twice."""
        accepted_types: Counter[str] = Counter()
        accepted_samples: set[int] = set()
        target_modes: Counter[str] = Counter()
        entity_count = 0
        token_count = 0
        utility_sum = 0.0

        for sample_id, record in self.records.items():
            if not 0 <= sample_id < len(self.base_dataset):
                continue
            gold_entities = extract_bio_entities(
                self.base_dataset.examples["labels"][sample_id]
            )
            listed_image_count = min(
                len(self.base_dataset.examples["images"][sample_id]), 4
            )
            for entity in record.get("entities", []):
                try:
                    start = int(entity["start"])
                    end = int(entity["end"])
                    entity_type = str(entity["label"]).upper().replace("MISC", "MIS")
                    if entity_type.startswith(("B-", "I-")):
                        entity_type = entity_type[2:]
                    confidence = float(entity.get("confidence", 1.0))
                    raw_image_ids = entity.get("image_ids", [])
                    visual_utility = min(
                        max(float(entity.get("visual_utility", 0.0)), 0.0), 1.0
                    )
                    text_sufficient = entity.get("text_sufficient", False)
                except (KeyError, TypeError, ValueError):
                    continue
                if not isinstance(raw_image_ids, (list, tuple)):
                    continue
                if confidence < self.grounding_confidence_threshold:
                    continue
                if (start, end, entity_type) not in gold_entities:
                    continue
                listed_ids = {
                    int(image_id)
                    for image_id in raw_image_ids
                    if str(image_id).lstrip("-").isdigit()
                    and 0 <= int(image_id) < listed_image_count
                }
                if raw_image_ids and not listed_ids:
                    continue
                entity_count += 1
                token_count += end - start
                accepted_types[entity_type] += 1
                accepted_samples.add(sample_id)
                if self.expected_teacher_task == "utility":
                    utility_sum += visual_utility
                    if text_sufficient:
                        target_modes["NONE_text_sufficient"] += 1
                    elif visual_utility < self.visual_utility_threshold:
                        target_modes["NONE_low_utility"] += 1
                    elif raw_image_ids:
                        target_modes["useful_images"] += 1
                    else:
                        target_modes["NONE_no_useful_image"] += 1
                else:
                    target_modes["image" if raw_image_ids else "NONE"] += 1

        return {
            "cache_records": len(self.records),
            "teacher_tasks": dict(self.teacher_tasks),
            "entities": entity_count,
            "samples": len(accepted_samples),
            "entity_words": token_count,
            "types": dict(accepted_types),
            "targets": dict(target_modes),
            "mean_visual_utility": (
                utility_sum / entity_count
                if self.expected_teacher_task == "utility" and entity_count
                else None
            ),
            "class_weights": self.grounding_class_weights,
        }

    def evidence_distillation_summary(self) -> dict[str, Any]:
        """Summarize the frozen rationale-embedding artifact before training."""
        if self.evidence_artifact is None:
            return {
                "enabled": self.enable_evidence_distillation,
                "artifact_entities": 0,
                "artifact_samples": 0,
                "embedding_dim": self.evidence_embedding_dim,
                "max_entities_per_sample": self.evidence_max_entities,
            }
        return {
            "enabled": self.enable_evidence_distillation,
            "schema_version": self.evidence_artifact.get("schema_version"),
            "artifact_entities": int(self.evidence_artifact["embeddings"].shape[0]),
            "artifact_samples": len(self.evidence_rows_by_sample),
            "embedding_dim": self.evidence_embedding_dim,
            "max_entities_per_sample": self.evidence_max_entities,
            "teacher_cache_sha256": self.evidence_artifact.get(
                "teacher_cache_sha256"
            ),
            "input_sha256": self.evidence_artifact.get("input_sha256"),
            "pooling": self.evidence_artifact.get("pooling"),
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.base_dataset[index]
        token_probs, image_scores = self._default_teacher(sample[0].shape[0])
        grounding_targets = torch.zeros(
            (sample[0].shape[0], sample[5].shape[0] + 1), dtype=torch.float
        )
        grounding_weights = torch.zeros(sample[0].shape[0], dtype=torch.float)
        available = 0.0
        record = self.records.get(index)
        if record is not None:
            cached_probs = torch.tensor(record.get("token_probs", []), dtype=torch.float)
            cached_images = torch.tensor(record.get("image_scores", []), dtype=torch.float)
            if (
                cached_probs.shape == token_probs.shape
                and cached_images.shape == image_scores.shape
            ):
                token_probs, image_scores = cached_probs, cached_images
                available = 1.0
            if self.enable_grounding:
                grounding_targets, grounding_weights = build_grounding_targets(
                    words=self.base_dataset.examples["words"][index],
                    gold_labels=self.base_dataset.examples["labels"][index],
                    teacher_entities=record.get("entities", []),
                    tokenizer=self.base_dataset.tokenizer,
                    max_sequence_length=sample[0].shape[0],
                    image_mask=sample[5],
                    confidence_threshold=self.grounding_confidence_threshold,
                    class_weights=self.grounding_class_weights,
                    background_ratio=self.grounding_background_ratio,
                    teacher_task=self.expected_teacher_task or "grounding",
                    visual_utility_threshold=self.visual_utility_threshold,
                )
        base_result = (
            *sample,
            token_probs,
            image_scores,
            torch.tensor(available, dtype=torch.float),
            grounding_targets,
            grounding_weights,
        )
        if not self.enable_evidence_distillation:
            return base_result

        evidence_token_masks = torch.zeros(
            (self.evidence_max_entities, sample[0].shape[0]), dtype=torch.float
        )
        evidence_image_targets = torch.zeros(
            (self.evidence_max_entities, sample[5].shape[0]), dtype=torch.float
        )
        evidence_embeddings = torch.zeros(
            (self.evidence_max_entities, self.evidence_embedding_dim),
            dtype=torch.float,
        )
        evidence_weights = torch.zeros(self.evidence_max_entities, dtype=torch.float)
        if self.evidence_artifact is not None:
            (
                evidence_token_masks,
                evidence_image_targets,
                evidence_embeddings,
                evidence_weights,
                _,
            ) = build_evidence_distillation_targets(
                words=self.base_dataset.examples["words"][index],
                gold_labels=self.base_dataset.examples["labels"][index],
                tokenizer=self.base_dataset.tokenizer,
                max_sequence_length=sample[0].shape[0],
                image_mask=sample[5],
                artifact=self.evidence_artifact,
                row_indices=self.evidence_rows_by_sample.get(index, []),
                max_entities=self.evidence_max_entities,
            )
        return (
            *base_result,
            evidence_token_masks,
            evidence_image_targets,
            evidence_embeddings,
            evidence_weights,
        )
