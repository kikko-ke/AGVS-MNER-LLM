"""Token-specific patch retrieval with an explicit zero-valued visual null.

The global AGVS anchor supplies a prior, while each text position retrieves its
own patch subset. Selected scores remain in the softmax, so the query/key
retriever receives gradients despite hard top-k indices.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TokenEvidenceFusion(nn.Module):
    def __init__(self, hidden_size: int = 768, retrieval_size: int = 256,
                 topk: int = 12, temperature: float = 0.1,
                 anchor_weight: float = 0.25, dropout: float = 0.1):
        super().__init__()
        if hidden_size < 1 or retrieval_size < 1 or topk < 1 or temperature <= 0:
            raise ValueError("Hidden/retrieval sizes, topk and temperature must be positive")
        if anchor_weight < 0 or not 0 <= dropout < 1:
            raise ValueError("anchor_weight must be non-negative and dropout in [0,1)")
        self.topk = topk
        self.temperature = temperature
        self.anchor_weight = anchor_weight
        self.text_norm = nn.LayerNorm(hidden_size)
        self.patch_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, retrieval_size, bias=False)
        self.key = nn.Linear(hidden_size, retrieval_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.value.weight)
        self.null_score = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.null_score.weight)
        nn.init.zeros_(self.null_score.bias)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, patches: torch.Tensor,
                patch_mask: torch.Tensor, anchor_scores: torch.Tensor,
                text_mask: torch.Tensor):
        if (text.ndim != 3 or patches.ndim != 3 or text.shape[0] != patches.shape[0]
                or text.shape[-1] != patches.shape[-1] or patches.shape[1] < 1):
            raise ValueError("Expected compatible text [B,L,H] and patches [B,P,H]")
        if patch_mask.shape != patches.shape[:2] or anchor_scores.shape != patch_mask.shape:
            raise ValueError("Patch mask and anchor scores must have shape [B,P]")
        if text_mask.shape != text.shape[:2]:
            raise ValueError("Text mask must have shape [B,L]")
        patch_mask = patch_mask.bool()
        text_mask = text_mask.bool()
        # Mask before projection: invalid image pixels cannot become evidence.
        patches = patches.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        normal_text = self.text_norm(text)
        normal_patches = self.patch_norm(patches)
        query = F.normalize(self.query(normal_text).float(), dim=-1)
        key = F.normalize(self.key(normal_patches).float(), dim=-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / self.temperature
        # A finite placeholder prior also handles a sample with no valid image.
        prior = F.log_softmax(anchor_scores.float().masked_fill(~patch_mask, -1e4), dim=-1)
        scores = scores + self.anchor_weight * prior.unsqueeze(1)
        scores = scores.masked_fill(~patch_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        selected_scores, indices = scores.topk(min(self.topk, patches.shape[1]), dim=-1)
        batch = torch.arange(text.shape[0], device=text.device).view(-1, 1, 1)
        values = self.value(normal_patches).masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        selected_values = values[batch, indices]
        null_score = self.null_score(normal_text).float()
        probabilities = F.softmax(torch.cat([selected_scores, null_score], dim=-1), dim=-1)
        context = (probabilities[..., :-1].to(values.dtype).unsqueeze(-1) * selected_values).sum(dim=-2)
        gate = torch.sigmoid(self.gate(torch.cat([normal_text, context], dim=-1)))
        delta = self.dropout(context) * gate
        delta = delta.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        return delta, {
            "null_probability": probabilities[..., -1],
            "gate_mean": gate.mean(dim=-1),
            "selected_indices": indices,
        }
