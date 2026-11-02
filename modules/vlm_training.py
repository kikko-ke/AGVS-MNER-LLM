"""Small-batch Qwen2.5-VL QLoRA helpers; GPU/model imports are lazy."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from modules.internal_split import sha256_file
from modules.vlm_ner import messages, open_image, target, parse_entities, span_metrics

VERSIONS = {"torch":"2.6.0", "transformers":"4.51.3", "peft":"0.15.2",
            "accelerate":"1.6.0", "bitsandbytes":"0.45.5"}


def model_binding(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file()
                   and p.suffix in {".json", ".safetensors", ".txt", ".model", ".jinja"})
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local safetensors model weights are missing")
    return {p.name:sha256_file(p) for p in files}


def check_runtime():
    versions = {name:importlib.metadata.version(name) for name in VERSIONS}
    if any(versions[k].split("+")[0] != v for k,v in VERSIONS.items()):
        raise RuntimeError(f"Use the isolated, pinned VLM environment; found {versions}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This pilot requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The fixed pilot requires BF16 support")
    # Check the actual quantization binary before downloading/loading 3B weights.
    import bitsandbytes as bnb
    probe = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda:0")
    with torch.no_grad():
        result = probe(torch.ones(1, 64, device="cuda:0", dtype=torch.bfloat16))
    if not torch.isfinite(result).all():
        raise RuntimeError("The NF4 CUDA kernel preflight produced non-finite output")
    del probe, result
    torch.cuda.empty_cache()
    return versions


def text_lora_targets(model):
    import torch
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)
               and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
               and not any(part in {"visual", "vision_model", "vision_tower"} for part in name.split("."))]
    if not targets:
        raise ValueError("No language attention projections found; inspect model version")
    return targets


def load_model(directory, seed, min_pixels, max_pixels):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(seed)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=["visual", "lm_head"])
    processor = AutoProcessor.from_pretrained(directory, min_pixels=min_pixels, max_pixels=max_pixels,
                                               local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        directory, local_files_only=True, torch_dtype=torch.bfloat16,
        quantization_config=quant, device_map={"":0}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant":False})
    targets = text_lora_targets(model)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                            target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or ".visual." in n for n in trainable):
        raise RuntimeError("Only language LoRA parameters may be trainable")
    model.print_trainable_parameters()
    return model, processor


def train_mode(model):
    model.train()
    for name, module in model.named_modules():
        if name.split(".")[-1] == "visual":
            module.eval()


def encode(processor, row, mode, training, max_length):
    import torch
    images = []
    try:
        if mode == "multi_image":
            for image in row["images"]:
                if sha256_file(image["path"]) != image["sha256"]:
                    raise ValueError("Source image changed after preparation")
                images.append(open_image(Path(image["path"]).read_bytes()))
        chat = messages(row, mode)
        prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        kwargs = {"images":images} if images else {}
        prefix = processor(text=[prompt], return_tensors="pt", padding=False, **kwargs)
        if not training:
            if prefix["input_ids"].shape[1] > max_length:
                raise ValueError(f"Prompt exceeds fixed token budget: source_id={row['source_id']}")
            return dict(prefix)
        full_chat = chat + [{"role":"assistant", "content":[{"type":"text", "text":target(row)}]}]
        text = processor.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)
        full = processor(text=[text], return_tensors="pt", padding=False, **kwargs)
        length = prefix["input_ids"].shape[1]
        if not torch.equal(full["input_ids"][:, :length], prefix["input_ids"]):
            raise ValueError("Assistant loss mask does not match the exact multimodal prompt prefix")
        if full["input_ids"].shape[1] > max_length:
            raise ValueError(f"Full target exceeds fixed token budget: source_id={row['source_id']}")
        labels = full["input_ids"].clone()
        labels[:, :length] = -100
        if not labels.ne(-100).any():
            raise ValueError("No supervised assistant tokens")
        return {**full, "labels":labels}
    finally:
        for image in images:
            image.close()


def move_to_cuda(batch):
    return {k:v.to("cuda:0") for k,v in batch.items()}


def accumulation_divisor(index, count, accumulation):
    return min(accumulation, count - (index // accumulation) * accumulation)


def optimizer_update(model, optimizer, scheduler):
    import torch
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise RuntimeError("LoRA gradient must be finite and nonzero")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def smoke_rows(rows, count=8):
    # Input-only stress selection, including every available image-count bucket.
    ordered = sorted(rows, key=lambda r:(-len(r["images"]), -sum(map(len,r["tokens"])), r["source_id"]))
    result = []
    for number in sorted({len(r["images"]) for r in ordered}, reverse=True):
        result.append(next(r for r in ordered if len(r["images"]) == number))
    ids = {r["source_id"] for r in result}
    result.extend(r for r in ordered if r["source_id"] not in ids)
    return result[:count]


def generate_evaluation(model, processor, rows, mode, max_length, max_new_tokens, output):
    import torch
    from tqdm.auto import tqdm
    gold, predicted, records = [], [], []
    malformed = 0
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in tqdm(rows, desc="Generate internal dev", unit="sample"):
            batch = move_to_cuda(encode(processor, row, mode, False, max_length-max_new_tokens))
            prefix_length = batch["input_ids"].shape[1]
            generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                       num_beams=1, use_cache=True)
            text = processor.tokenizer.decode(generated[0, prefix_length:], skip_special_tokens=True)
            error = None
            try:
                entities = parse_entities(text, len(row["tokens"]))
            except (ValueError, TypeError, KeyError) as exc:
                entities, error = [], str(exc)
                malformed += 1
            gold.append(parse_entities(target(row), len(row["tokens"])))
            predicted.append(entities)
            records.append({"source_id":row["source_id"], "response":text, "error":error,
                            "predicted":[list(e) for e in entities]})
            del batch, generated
    Path(output).write_text("".join(json.dumps(r, ensure_ascii=True)+"\n" for r in records), encoding="utf-8")
    return {**span_metrics(gold, predicted), "malformed_responses":malformed,
            "malformed_rate":malformed/len(rows) if rows else 0.,
            "scope":"fixed training-internal dev panel; full original word spans"}
