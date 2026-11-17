"""TPM-MI topology, with explicit batch axes, padding masks and word alignment.

Reference: Huang et al., LREC-COLING 2024, https://github.com/JinFish/MNER-MI
This is an audited reproduction, not a claim to reproduce the published score.
"""
from __future__ import annotations

import torch
from torch import nn
from torchcrf import CRF

from .common import TAGS


class TPMMI(nn.Module):
    def __init__(self, bert, vit, prompt_width=1024, image_heads=8, image_ffn=2048):
        super().__init__()
        self.bert, self.vit = bert, vit
        self.width = bert.config.hidden_size
        if vit.config.hidden_size != self.width:
            raise ValueError("TPM-MI requires equal BERT and ViT hidden widths")
        self.layers = bert.config.num_hidden_layers
        self.heads = bert.config.num_attention_heads
        self.head_width = self.width // self.heads
        self.img_prompt_encoder = nn.Sequential(
            nn.Linear(self.width, prompt_width), nn.Tanh(),
            nn.Linear(prompt_width, self.layers * 2 * self.width))
        self.temporalEmbedding = nn.Embedding(4, self.width)
        # Upstream feeds (batch, image, hidden) into a sequence-first layer.
        # batch_first=True prevents attention between unrelated posts.
        self.transformer = nn.TransformerEncoderLayer(
            self.width, image_heads, dim_feedforward=image_ffn, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(self.width, len(TAGS))
        self.crf = CRF(len(TAGS), batch_first=True)

    @classmethod
    def from_pretrained(cls, config):
        from transformers import BertModel, ViTModel
        bert = BertModel.from_pretrained(config["bert_model"], local_files_only=True,
                                         attn_implementation="eager")
        vit = ViTModel.from_pretrained(config["vit_model"], local_files_only=True,
                                      attn_implementation="eager")
        return cls(bert, vit)

    def forward(self, batch):
        pixels = batch["pixels"]
        batch_size, image_count, channels, height, width = pixels.shape
        if image_count != 4:
            raise ValueError("Expected four ordered image slots")
        features = self.vit(pixel_values=pixels.reshape(-1, channels, height, width)).pooler_output
        features = features.reshape(batch_size, image_count, self.width)
        valid = batch["image_mask"].bool()
        features = features + self.temporalEmbedding(torch.arange(image_count, device=pixels.device))
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        # Prevent an all-masked softmax; its dummy output is subsequently removed.
        safe = valid.clone()
        safe[~safe.any(1), 0] = True
        features = self.transformer(features, src_key_padding_mask=~safe)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        prompt = self.img_prompt_encoder(features).reshape(
            batch_size, image_count, self.layers, 2, self.heads, self.head_width)
        prompt = prompt.permute(2, 3, 0, 4, 1, 5)
        past = tuple((prompt[layer, 0], prompt[layer, 1]) for layer in range(self.layers))
        attention = torch.cat((valid, batch["attention_mask"].bool()), dim=1)
        hidden = self.bert(input_ids=batch["input_ids"], attention_mask=attention,
                           token_type_ids=batch["token_type_ids"], past_key_values=past,
                           return_dict=True).last_hidden_state
        emissions = self.fc(hidden)
        if not torch.isfinite(emissions).all():
            raise ValueError("Non-finite TPM-MI emissions")
        return emissions

    def loss(self, emissions, batch):
        return -self.crf(emissions, batch["labels"], mask=batch["attention_mask"].bool(), reduction="mean")
