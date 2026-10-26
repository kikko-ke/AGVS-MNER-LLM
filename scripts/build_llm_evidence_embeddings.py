#!/usr/bin/env python
"""Encode cached LLM visual-evidence rationales for training-only distillation."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.evidence_distillation import (
    EVIDENCE_EMBEDDING_SCHEMA_VERSION,
    EVIDENCE_ENTITY_TYPES,
    extract_evidence_bio_entities,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--bert-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_gold(path: Path) -> list[set[tuple[int, int, str]]]:
    result = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = ast.literal_eval(line)
            labels = [str(label).replace("MISC", "MIS") for label in record["label"]]
            result.append(extract_evidence_bio_entities(labels))
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    if args.batch_size <= 0 or args.max_length <= 2:
        raise ValueError("batch-size must be positive and max-length must exceed 2")
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence-threshold must be in [0, 1]")

    input_path = Path(args.input)
    cache_path = Path(args.teacher_cache)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    gold = _load_gold(input_path)

    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, str]] = set()
    rejected = Counter()
    with cache_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = int(record.get("sample_id", line_number))
            if not 0 <= sample_id < len(gold):
                rejected["sample_id"] += 1
                continue
            if str(record.get("teacher_task", "")) != "grounding":
                raise ValueError("Evidence embeddings require a grounding teacher cache")
            for entity in record.get("entities", []):
                try:
                    start = int(entity["start"])
                    end = int(entity["end"])
                    label = str(entity["label"]).upper().replace("MISC", "MIS")
                    if label.startswith(("B-", "I-")):
                        label = label[2:]
                    confidence = float(entity.get("confidence", 0.0))
                    evidence = str(entity.get("evidence", "")).strip()
                    raw_image_ids = entity.get("image_ids", [])
                except (KeyError, TypeError, ValueError):
                    rejected["malformed"] += 1
                    continue
                if label not in EVIDENCE_ENTITY_TYPES:
                    rejected["label"] += 1
                    continue
                if confidence < args.confidence_threshold:
                    rejected["confidence"] += 1
                    continue
                if (start, end, label) not in gold[sample_id]:
                    rejected["gold_mismatch"] += 1
                    continue
                if not evidence:
                    rejected["empty_evidence"] += 1
                    continue
                if not isinstance(raw_image_ids, list):
                    rejected["image_ids"] += 1
                    continue
                image_ids = sorted(
                    {
                        int(value)
                        for value in raw_image_ids
                        if str(value).lstrip("-").isdigit() and 0 <= int(value) < 4
                    }
                )
                if not image_ids:
                    rejected["no_visual_evidence"] += 1
                    continue
                key = (sample_id, start, end, label)
                if key in seen:
                    raise ValueError(f"Duplicate evidence entity: {key}")
                seen.add(key)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "start": start,
                        "end": end,
                        "label_id": EVIDENCE_ENTITY_TYPES.index(label),
                        "confidence": confidence,
                        "image_ids": image_ids,
                        "evidence": evidence,
                    }
                )
    rows.sort(key=lambda row: (row["sample_id"], row["start"], row["end"]))
    if not rows:
        raise ValueError("No usable LLM evidence rationales were found")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertModel.from_pretrained(args.bert_model).to(args.device)
    model.eval()
    encoded_batches = []
    with torch.no_grad():
        for offset in range(0, len(rows), args.batch_size):
            texts = [str(row["evidence"]) for row in rows[offset : offset + args.batch_size]]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = encoded.pop("special_tokens_mask").to(args.device)
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool() & ~special.bool()
            fallback = encoded["attention_mask"].bool()
            empty = ~mask.any(dim=1)
            if empty.any():
                mask[empty] = fallback[empty]
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
            encoded_batches.append(pooled.cpu().to(torch.float16))
            print(f"[progress] encoded {min(offset + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    count = len(rows)
    image_targets = torch.zeros((count, 4), dtype=torch.float32)
    for index, row in enumerate(rows):
        image_ids = list(row["image_ids"])
        image_targets[index, image_ids] = 1.0 / len(image_ids)
    metadata = {
        "schema_version": EVIDENCE_EMBEDDING_SCHEMA_VERSION,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "teacher_cache": str(cache_path),
        "teacher_cache_sha256": sha256_file(cache_path),
        "teacher_task": "grounding",
        "bert_model": str(args.bert_model),
        "pooling": "last_hidden_mean_without_special_tokens",
        "max_length": args.max_length,
        "confidence_threshold": args.confidence_threshold,
        "entities": count,
        "samples": len({int(row["sample_id"]) for row in rows}),
        "embedding_dim": int(encoded_batches[0].shape[1]),
        "rejected": dict(rejected),
    }
    artifact = {
        **metadata,
        "sample_ids": torch.tensor([row["sample_id"] for row in rows], dtype=torch.long),
        "starts": torch.tensor([row["start"] for row in rows], dtype=torch.long),
        "ends": torch.tensor([row["end"] for row in rows], dtype=torch.long),
        "label_ids": torch.tensor([row["label_id"] for row in rows], dtype=torch.long),
        "image_targets": image_targets,
        "confidences": torch.tensor([row["confidence"] for row in rows], dtype=torch.float32),
        "embeddings": torch.cat(encoded_batches, dim=0),
    }
    temporary_output = output.with_name(output.name + ".tmp")
    torch.save(artifact, temporary_output)
    temporary_output.replace(output)
    meta_path = output.with_name(output.name + ".meta.json")
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== LLM evidence embedding cache complete ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Artifact: {output}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
