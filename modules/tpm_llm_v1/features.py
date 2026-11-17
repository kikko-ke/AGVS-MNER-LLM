"""Frozen word features and gold-only auxiliary heads; no AGVS dependencies."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .common import BIO, TAGS, seed_everything

WIDTH = 768


def word_offsets(words):
    offsets, cursor = [], 0
    for word in words:
        offsets.append((cursor, cursor + len(word)))
        cursor += len(word) + 1
    return " ".join(words), offsets


def last_overlapping(words, subtokens):
    return [max((index for index, (a, b) in enumerate(subtokens)
                 if a < b and start < end and a < end and start < b), default=-1)
            for start, end in words]


def extract(rows, config, encoder):
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, BertModel
    location = config[f"{encoder}_model"]
    tokenizer = AutoTokenizer.from_pretrained(location, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("Frozen word extraction needs tokenizer character offsets")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if encoder == "qwen":
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                  llm_int8_skip_modules=["visual", "lm_head"])
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            location, local_files_only=True, torch_dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa").eval()
        model = vlm.get_decoder()
        del vlm
    elif encoder == "bert":
        model = BertModel.from_pretrained(location, local_files_only=True).to("cuda:0").eval()
    else:
        raise ValueError(f"Unknown feature encoder: {encoder}")
    model.requires_grad_(False)
    features, aligned, projection = [], [], None
    batch_size = config["feature_batch_size"]
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc=f"{encoder} frozen word features"):
            batch = rows[start:start + batch_size]
            texts, bounds = zip(*(word_offsets(row["tokens"]) for row in batch))
            encoded = tokenizer(list(texts), padding=True, return_tensors="pt", return_offsets_mapping=True)
            offsets = encoded.pop("offset_mapping").tolist()
            if encoded["input_ids"].shape[1] > 512:
                raise ValueError("Auxiliary text exceeds 512 subtokens; no silent truncation is allowed")
            inputs = {key: encoded[key].to("cuda:0") for key in ("input_ids", "attention_mask")}
            hidden = model(**inputs, **({"use_cache": False} if encoder == "qwen" else {}),
                           return_dict=True).last_hidden_state
            if hidden.size(-1) != WIDTH and projection is None:
                generator = torch.Generator().manual_seed(config["projection_seed"])
                projection = (torch.randn(hidden.size(-1), WIDTH, generator=generator) /
                              WIDTH ** 0.5).to(hidden.device)
            for index, spans in enumerate(bounds):
                positions = torch.tensor(last_overlapping(spans, offsets[index]), device=hidden.device)
                valid = positions.ge(0)
                values = hidden[index, positions.clamp_min(0)].float()
                if projection is not None:
                    values = values @ projection
                values = values.masked_fill(~valid.unsqueeze(-1), 0)
                if not torch.isfinite(values).all():
                    raise ValueError("Non-finite frozen word feature")
                features.append(values.to(device="cpu", dtype=torch.float16))
                aligned.append(valid.cpu())
    del model, hidden
    torch.cuda.empty_cache()
    return {"features": features, "aligned": aligned,
            "source_ids": [row["source_id"] for row in rows], "encoder": encoder}


class WordHead(nn.Module):
    def __init__(self, architecture="bigru"):
        super().__init__()
        if architecture not in ("bigru", "linear"):
            raise ValueError("Expected bigru or linear")
        self.architecture = architecture
        self.norm = nn.LayerNorm(WIDTH)
        if architecture == "bigru":
            self.context = nn.GRU(WIDTH, 128, batch_first=True, bidirectional=True)
        self.out = nn.Linear(256 if architecture == "bigru" else WIDTH, len(BIO))

    def forward(self, words, lengths):
        values = self.norm(words)
        if self.architecture == "bigru":
            packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
            values, _ = self.context(packed)
            values, _ = pad_packed_sequence(values, batch_first=True)
        return self.out(values)


def infer_head(head, features, batch_size=16, device="cuda:0"):
    head.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size]
            lengths = [len(values) for values in batch]
            logits = head(pad_sequence(batch, batch_first=True).to(device, torch.float32), lengths)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite auxiliary logits")
            scores.extend(values[:length].cpu().clone() for values, length in zip(logits, lengths))
    return scores


def fit_head(features, training_rows, config, seed, architecture="bigru", device="cuda:0"):
    """The only accepted gold labels are from training_rows, never dev/test."""
    seed_everything(seed)
    if len(features) != len(training_rows) or any(len(x) != len(row["tokens"])
                                                for x, row in zip(features, training_rows)):
        raise ValueError("Training feature/word alignment mismatch")
    head = WordHead(architecture).to(device)
    gold = [torch.tensor([BIO.index(tag) for tag in row["labels"]]) for row in training_rows]
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_lr"], weight_decay=0.01)
    batch_size = config["feature_batch_size"]
    history = []
    for epoch in range(config["head_epochs"]):
        head.train()
        order = torch.randperm(len(features), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = [features[index] for index in indices]
            words = pad_sequence(batch, batch_first=True).to(device, torch.float32)
            labels = pad_sequence([gold[index] for index in indices], batch_first=True,
                                  padding_value=-100).to(device)
            logits = head(words, [len(values) for values in batch])
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite word-head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach().item())
        mean = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "mean_loss": mean})
        print(f"{architecture} head seed={seed} epoch={epoch+1}/{config['head_epochs']} loss={mean:.6f}",
              flush=True)
    return head.eval(), history


def fuse(emissions, rows, mappings, logits, aligned, coefficient, clip=True):
    """Nine-way centering BEFORE X->O lookup; first pieces only, no gold access."""
    if not (len(rows) == len(mappings) == len(logits) == len(aligned) == emissions.shape[0]):
        raise ValueError("Fusion row counts differ")
    residual = torch.zeros_like(emissions)
    lookup = [BIO.index(tag if tag != "X" else "O") for tag in TAGS]
    for index, (row, mapping, scores, valid) in enumerate(zip(rows, mappings, logits, aligned)):
        if scores.shape != (len(row["tokens"]), len(BIO)) or len(valid) != len(row["tokens"]):
            raise ValueError("Auxiliary scores do not cover every original word")
        centered = scores - scores.mean(1, keepdim=True)
        if clip:
            centered = centered.clamp(-8, 8)
        centered = centered * valid.unsqueeze(-1)
        if mapping["positions"]:
            residual[index, mapping["positions"]] = centered[mapping["word_indices"]][:, lookup].to(emissions)
    result = emissions + coefficient * residual
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite fused emissions")
    return result
