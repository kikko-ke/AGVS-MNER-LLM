"""Dataset wrapper that adds cached LLM teacher signals to AGVS samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class TeacherCacheDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, base_dataset: Any, cache_path: str | Path) -> None:
        self.base_dataset = base_dataset
        self.cache_path = Path(cache_path)
        self.records: dict[int, dict[str, Any]] = {}
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as cache_file:
                for line_number, line in enumerate(cache_file):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    sample_id = int(record.get("sample_id", line_number))
                    self.records[sample_id] = record

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _default_teacher(self, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        num_labels = len(self.base_dataset.label_mapping)
        token_probs = torch.full((sequence_length, num_labels), 1.0 / num_labels)
        image_scores = torch.full((4,), 0.25)
        return token_probs, image_scores

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.base_dataset[index]
        token_probs, image_scores = self._default_teacher(sample[0].shape[0])
        available = 0.0
        record = self.records.get(index)
        if record is not None:
            cached_probs = torch.tensor(record.get("token_probs", []), dtype=torch.float)
            cached_images = torch.tensor(record.get("image_scores", []), dtype=torch.float)
            if cached_probs.shape == token_probs.shape and cached_images.shape == image_scores.shape:
                token_probs, image_scores = cached_probs, cached_images
                available = 1.0
        return (*sample, token_probs, image_scores, torch.tensor(available, dtype=torch.float))
