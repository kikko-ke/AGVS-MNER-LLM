"""Training adapter that feeds cached teacher signals into the AGVS model."""

from __future__ import annotations

from typing import Any

import torch

from modules.trainer import NerTrainer


class LlmNerTrainer(NerTrainer):
    def _step(
        self, batch: tuple[Any, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor]:
        (
            input_ids,
            token_type_ids,
            attention_mask,
            image_pixel_values,
            labels,
            image_mask,
            llm_token_probs,
            llm_image_scores,
            llm_available,
        ) = batch
        decoded_label_ids, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_pixel_values=image_pixel_values,
            image_mask=image_mask,
            labels=labels,
            llm_token_probs=llm_token_probs,
            llm_image_scores=llm_image_scores,
            llm_available=llm_available,
        )
        return attention_mask, labels, decoded_label_ids, loss
