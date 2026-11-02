"""Gold-supervised Qwen2.5-VL QLoRA: explicit GPU smoke gate before long training."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.internal_split import sha256_file
from modules.vlm_ner import load_prepared
from modules.vlm_training import (
    check_runtime, model_binding, load_model, encode, move_to_cuda, train_mode,
    smoke_rows, accumulation_divisor, optimizer_update, generate_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "train"), required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", default="dataset/vlm_gold_mi_v1/manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-report")
    parser.add_argument("--input-mode", choices=("text", "multi_image"), default="multi_image")
    args = parser.parse_args()
    if args.stage == "train" and not args.smoke_report:
        parser.error("--stage train requires --smoke-report from the real GPU smoke run")
    return args


def main():
    args = parse_args()
    versions = check_runtime()
    manifest, rows = load_prepared(args.data)
    profile = {"schema_version":"vlm-qlora-profile-v1", "model_files_sha256":model_binding(args.model),
               "implementation_sha256":{name:sha256_file(Path(__file__).resolve().parents[1]/name)
                  for name in ("modules/internal_split.py", "modules/vlm_ner.py", "modules/vlm_training.py", "scripts/run_vlm_ner.py")},
               "data_sha256":sha256_file(args.data), "input_mode":args.input_mode, "seed":1234,
               "min_pixels":4*28*28, "max_pixels":128*28*28, "max_length":2048,
               "max_new_tokens":256, "lora_r":8, "lora_alpha":16, "lora_dropout":0.05,
               "quantization":"nf4-double-quant-bf16", "batch_size":1,
               "epochs":2, "gradient_accumulation":8, "lr":1e-4, "warmup_ratio":0.03,
               "weight_decay":0.01, "versions":versions}
    if args.stage == "train":
        smoke = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
        if smoke.get("schema_version") != "vlm-ner-smoke-v1" or smoke.get("passed") is not True or smoke.get("profile") != profile:
            raise ValueError("A passing GPU smoke report with identical model/data/settings is required")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty; preserve and inspect the previous attempt")
    output.mkdir(parents=True, exist_ok=True)
    config = {"stage":args.stage, "profile":profile, "data":str(args.data), "model":str(args.model),
              "official_val_or_test_read":False, "llm_api_requests":0,
              "gold_scope":manifest["gold_scope"], "panel_source_ids":manifest["panel_source_ids"]}
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    import torch
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        model, processor = load_model(args.model, 1234, profile["min_pixels"], profile["max_pixels"])
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"error":str(exc), "stage":"model_loading",
                 "peak_allocated_GiB":torch.cuda.max_memory_allocated()/1024**3}, indent=2)+"\n", encoding="utf-8")
        raise
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=profile["lr"], weight_decay=profile["weight_decay"])
    selected = smoke_rows(rows["train"]) if args.stage == "smoke" else rows["train"]
    accumulation = 1 if args.stage == "smoke" else profile["gradient_accumulation"]
    epochs = 1 if args.stage == "smoke" else profile["epochs"]
    steps = math.ceil(len(selected)/accumulation)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*profile["warmup_ratio"]), steps)
    history, best, best_epoch, gradients, token_max = [], -1., None, [], 0
    try:
        if args.stage == "train":
            # Fail before optimizer updates rather than truncating/dropping hard rows.
            dev_ids = {r["source_id"] for r in rows["dev_panel"]}
            for row in tqdm(rows["train"]+rows["dev_panel"], desc="Audit token budgets", unit="sample"):
                value = encode(processor, row, args.input_mode, True, profile["max_length"])
                token_max = max(token_max, value["input_ids"].shape[1])
                del value
                if row["source_id"] in dev_ids:
                    value = encode(processor, row, args.input_mode, False, profile["max_length"]-profile["max_new_tokens"])
                    del value
            metrics = generate_evaluation(model, processor, rows["dev_panel"], args.input_mode,
                       profile["max_length"], profile["max_new_tokens"], output / "dev_epoch0.jsonl")
            history.append({"epoch":0, "metrics":metrics})
            best, best_epoch = metrics["f1"], 0
            model.save_pretrained(output / "best_adapter", safe_serialization=True, save_embedding_layers=False)
            processor.save_pretrained(output / "best_adapter")
            print("Epoch 0 dev panel:", json.dumps(metrics), flush=True)
        for epoch in range(1, epochs+1):
            train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(selected), generator=torch.Generator().manual_seed(1234+epoch)).tolist()
            losses = []
            for index, position in enumerate(tqdm(order, desc=f"QLoRA epoch {epoch}/{epochs}", unit="sample")):
                row = selected[position]
                batch = move_to_cuda(encode(processor, row, args.input_mode, True, profile["max_length"]))
                token_max = max(token_max, batch["input_ids"].shape[1])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch, use_cache=False)
                    loss = result.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at source_id={row['source_id']}")
                losses.append(float(loss.detach()))
                (loss/accumulation_divisor(index, len(selected), accumulation)).backward()
                if (index+1) % accumulation == 0 or index+1 == len(selected):
                    gradients.append(optimizer_update(model, optimizer, scheduler))
                if index % 20 == 0 or args.stage == "smoke":
                    print(f"[step] epoch={epoch} sample={index+1}/{len(selected)} loss={losses[-1]:.4f} "
                          f"peak_allocated_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)
                del batch, result, loss
            if args.stage == "train":
                metrics = generate_evaluation(model, processor, rows["dev_panel"], args.input_mode,
                           profile["max_length"], profile["max_new_tokens"], output / f"dev_epoch{epoch}.jsonl")
                history.append({"epoch":epoch, "mean_train_loss":sum(losses)/len(losses), "metrics":metrics})
                if metrics["f1"] > best:
                    best, best_epoch = metrics["f1"], epoch
                    model.save_pretrained(output / "best_adapter", safe_serialization=True, save_embedding_layers=False)
                print(f"Epoch {epoch} dev panel:", json.dumps(metrics), flush=True)
                (output / "history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
        if args.stage == "smoke":
            # This checks generation/memory only, not held-out accuracy.
            generated = generate_evaluation(model, processor, selected[:2], args.input_mode,
                         profile["max_length"], 32, output / "smoke_generations.jsonl")
            report = {"schema_version":"vlm-ner-smoke-v1", "passed":True, "profile":profile,
                      "samples":len(selected), "optimizer_steps":len(gradients), "maximum_tokens":token_max,
                      "finite_nonzero_gradients":True, "generation_completed":True,
                      "malformed_smoke_outputs":generated["malformed_responses"], "heldout_accuracy_evaluated":False,
                      "GPU":torch.cuda.get_device_name(0),
                      "peak_allocated_GiB":torch.cuda.max_memory_allocated()/1024**3,
                      "peak_reserved_GiB":torch.cuda.max_memory_reserved()/1024**3,
                      "elapsed_seconds":time.monotonic()-started,
                      "note":"Hardware feasibility on selected training inputs; no claim of NER improvement."}
            name = "smoke_report.json"
        else:
            report = {"schema_version":"vlm-ner-pilot-v1", "profile":profile, "best_epoch":best_epoch,
                      "best_dev_panel_f1":best, "history":history, "maximum_tokens":token_max,
                      "official_val_or_test_read":False, "not_comparable_to_old_truncated_f1":True,
                      "note":"Plain supervised VLM baseline, not evidence of an AGVS component contribution."}
            name = "pilot_report.json"
        (output / name).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print("Report:", output / name, flush=True)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"error":str(exc), "stage":args.stage,
                 "peak_allocated_GiB":torch.cuda.max_memory_allocated()/1024**3}, indent=2)+"\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
