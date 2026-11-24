"""Frozen native Qwen VL forwards, aligned to the unchanged original words."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn

from modules.tpm_llm_v1.data import open_rgb
from modules.tpm_llm_v1.features import WIDTH, last_overlapping, word_offsets

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"


def prepare_input(row, processor, audit, maximum):
    """Align using the verified text-token suffix, NOT expanded-placeholder offsets.

    The native processor expands each image placeholder to a variable number of
    tokens. Its offsets therefore refer to an expanded string. Tokenizing just
    the original text gives the same offsets as the old text-only branch; an
    exact suffix check then locates those tokens in the multimodal sequence.
    """
    text, bounds = word_offsets(row["tokens"])
    if any(token in text for token in (VISION_START, IMAGE_PAD, VISION_END)):
        raise ValueError(f"Reserved vision token in source text: {row['source_id']}")
    plain = processor.tokenizer(text, return_offsets_mapping=True)
    ids = plain["input_ids"]
    if not ids or len(ids) > 512:
        raise ValueError(f"Original text must contain 1..512 subtokens: {row['source_id']}")
    images, slots, missing = [], [], []
    try:
        for slot, name in enumerate(row["images"]):
            info = audit[name]
            if not info["valid"]:
                missing.append({"slot": slot, "name": name})
                continue
            # A previously valid file becoming unreadable is an error, not a new fallback.
            images.append(open_rgb(Path(info["path"])))
            slots.append(slot)
        prefix = (VISION_START + IMAGE_PAD + VISION_END) * len(images)
        batch = processor(text=[prefix + text], return_tensors="pt", padding=False,
                          truncation=False, **({"images": images} if images else {}))
        observed = batch["input_ids"][0].tolist()
        if len(observed) > maximum:
            raise ValueError(
                f"Multimodal sequence exceeds {maximum} tokens for {row['source_id']} "
                f"({len(observed)} tokens, {len(images)} images); use a NEW run with "
                "a larger --max-input-tokens or smaller --max-pixels. No input was truncated.")
        if observed[-len(ids):] != ids:
            raise ValueError(f"Multimodal text suffix differs from text-only tokenization: {row['source_id']}")
        prefix_tokens = len(observed) - len(ids)
        positions = last_overlapping(bounds, plain["offset_mapping"])
        positions = [position + prefix_tokens if position >= 0 else -1 for position in positions]
        if images:
            if "pixel_values" not in batch or len(batch["image_grid_thw"]) != len(images):
                raise ValueError("Processor did not preserve all valid images")
            image_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD)
            merge = processor.image_processor.merge_size ** 2
            expected = int((batch["image_grid_thw"].prod(dim=1) // merge).sum())
            if observed[:prefix_tokens].count(image_id) != expected:
                raise ValueError("Expanded image tokens and image grids disagree")
        usage = {"source_id": row["source_id"], "original_images": len(row["images"]),
                 "valid_images": len(images), "used_slots": slots, "missing": missing,
                 "sequence_tokens": len(observed), "text_tokens": len(ids),
                 "prefix_tokens": prefix_tokens}
        # Metadata is never forwarded to Qwen, especially offset_mapping or gold labels.
        inputs = {key: batch[key] for key in
                  ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") if key in batch}
        return inputs, positions, usage
    finally:
        for image in images:
            image.close()


def feature_forward(model, inputs):
    """Native multimodal forward with an identity readout instead of a vocab head.

    Transformers 4.51.3 otherwise allocates vocab logits at EVERY image/text
    position. Replacing only lm_head with Identity preserves the exact decoder
    states and native image injection / M-RoPE, without storing all layer states.
    No labels, generation, or language-model loss are used.
    """
    result = model(**inputs, use_cache=False, output_hidden_states=False,
                   output_attentions=False, return_dict=True)
    return result.logits


def project_words(hidden, positions, projection_seed, projection=None):
    positions = torch.tensor(positions, device=hidden.device)
    valid = positions.ge(0)
    if hidden.size(-1) != WIDTH and projection is None:
        generator = torch.Generator().manual_seed(projection_seed)
        projection = (torch.randn(hidden.size(-1), WIDTH, generator=generator) /
                      WIDTH ** 0.5).to(hidden.device)
    values = hidden[0, positions.clamp_min(0)].float()
    if projection is not None:
        values = values @ projection
    values = values.masked_fill(~valid.unsqueeze(-1), 0)
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite image-conditioned word features")
    return values.to(device="cpu", dtype=torch.float16), valid.cpu(), projection


def usage_summary(usage):
    return {"samples": len(usage),
            "valid_image_count_histogram": dict(sorted(Counter(str(x["valid_images"]) for x in usage).items())),
            "text_only_fallback_samples": sum(x["valid_images"] == 0 for x in usage),
            "skipped_image_slots": sum(len(x["missing"]) for x in usage),
            "max_sequence_tokens": max((x["sequence_tokens"] for x in usage), default=0)}


class FrozenVisualExtractor:
    def __init__(self, config):
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        if transformers.__version__ != "4.51.3":
            raise RuntimeError("Use the existing mner-vlm environment with transformers==4.51.3")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Qwen visual extraction requires a BF16-capable CUDA GPU")
        self.config, self.projection = config, None
        self.processor = AutoProcessor.from_pretrained(
            config["qwen_model"], local_files_only=True,
            min_pixels=config["min_pixels"], max_pixels=config["max_pixels"])
        if not self.processor.tokenizer.is_fast:
            raise ValueError("Character alignment requires a fast Qwen tokenizer")
        self.processor.tokenizer.padding_side = "right"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  llm_int8_skip_modules=["visual", "lm_head"])
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["qwen_model"], local_files_only=True, torch_dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa")
        self.model.eval().requires_grad_(False)
        self.model.set_output_embeddings(nn.Identity())
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Qwen must be completely frozen")

    def row(self, row, audit, check_visual=False):
        inputs, positions, usage = prepare_input(
            row, self.processor, audit, self.config["max_input_tokens"])
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = feature_forward(self.model, inputs)
            values, valid, self.projection = project_words(
                hidden, positions, self.config["projection_seed"], self.projection)
            if check_visual:
                if "pixel_values" not in inputs or not valid.any():
                    raise ValueError("Visual smoke probe needs a valid image and an aligned word")
                changed = {**inputs, "pixel_values": torch.zeros_like(inputs["pixel_values"])}
                altered = feature_forward(self.model, changed)
                selected = torch.tensor(positions, device=hidden.device)[valid.to(hidden.device)]
                delta = (hidden[0, selected].float() - altered[0, selected].float()).abs().max().item()
                if not torch.isfinite(torch.tensor(delta)) or delta <= 0:
                    raise ValueError("Changing image pixels did not change any aligned text state")
                usage["visual_probe_max_hidden_delta"] = delta
        return values, valid, usage

    def rows(self, rows, audit):
        from tqdm.auto import tqdm
        features, aligned, usage = [], [], []
        for row in tqdm(rows, desc="Qwen text + original images", unit="post"):
            values, valid, record = self.row(row, audit)
            features.append(values)
            aligned.append(valid)
            usage.append(record)
        return {"features": features, "aligned": aligned, "image_usage": usage,
                "source_ids": [row["source_id"] for row in rows], "encoder": "qwen_multimodal"}

    def close(self):
        del self.model
        torch.cuda.empty_cache()
