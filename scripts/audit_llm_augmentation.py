"""Audit coverage and validity of an LLM augmentation cache without an API call."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.augmentation_schema import AUGMENTATION_SCHEMA_VERSION
from scripts.build_llm_augmentation_cache import (
    load_existing_records,
    load_source_records,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache", required=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    input_path = Path(cli.input)
    cache_path = Path(cli.cache)
    metadata_path = cache_path.with_name(cache_path.name + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != AUGMENTATION_SCHEMA_VERSION:
        raise ValueError("augmentation metadata schema mismatch")
    if metadata.get("input_sha256") != sha256_file(input_path):
        raise ValueError("input hash differs from augmentation metadata")
    all_records = {record["sample_id"]: record for record in load_source_records(input_path)}
    selected: dict[int, dict] = {}
    for item in metadata.get("selected", []):
        sample_id = int(item["sample_id"])
        record = all_records.get(sample_id)
        if record is None or record["source_fingerprint"] != item["source_fingerprint"]:
            raise ValueError(f"selected source mismatch for sample_id={sample_id}")
        selected[sample_id] = record
    validation_args = argparse.Namespace(
        min_context_change_ratio=float(metadata["min_context_change_ratio"]),
        max_context_change_ratio=float(metadata["max_context_change_ratio"]),
        min_length_ratio=float(metadata["min_length_ratio"]),
        max_length_ratio=float(metadata["max_length_ratio"]),
        max_words=int(metadata["max_words"]),
    )
    valid_ids = load_existing_records(
        cache_path,
        selected,
        model=str(metadata["llm_model"]),
        args=validation_args,
    )
    status = Counter()
    entity_types = Counter()
    abstain_reasons = Counter()
    rejection_reasons = Counter()
    ratios: list[float] = []
    word_counts: list[int] = []
    image_counts: list[int] = []
    with cache_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            state = str(item.get("status"))
            status[state] += 1
            if state == "accepted":
                entity_types.update(map(str, item.get("source_entity_types", [])))
                ratios.append(float(item["context_change_ratio"]))
                word_counts.append(len(item.get("tokens", [])))
                image_counts.append(len(item.get("used_image_ids", [])))
            elif state == "abstained":
                abstain_reasons[str(item.get("abstain_reason", ""))] += 1
            elif state == "rejected":
                rejection_reasons[str(item.get("policy_rejection", ""))] += 1
    selected_count = len(selected)
    accepted_count = status["accepted"]
    generated_count = accepted_count + status["abstained"]
    summary = {
        "schema_version": AUGMENTATION_SCHEMA_VERSION,
        "selected_requests": selected_count,
        "resolved_cached_requests": len(valid_ids),
        "valid_generation_responses": generated_count,
        "remaining_selected_requests": selected_count - len(valid_ids),
        "resolved_coverage": len(valid_ids) / selected_count if selected_count else 0.0,
        "status": dict(status),
        "acceptance_rate_over_resolved": (
            accepted_count / len(valid_ids) if valid_ids else 0.0
        ),
        "acceptance_rate_over_valid_generation": (
            accepted_count / generated_count if generated_count else 0.0
        ),
        "accepted_entity_type_coverage": dict(entity_types),
        "mean_context_change_ratio": statistics.mean(ratios) if ratios else None,
        "mean_augmented_words": statistics.mean(word_counts) if word_counts else None,
        "mean_cited_images": statistics.mean(image_counts) if image_counts else None,
        "abstain_reasons": dict(abstain_reasons),
        "policy_rejection_reasons": dict(rejection_reasons),
        "input_sha256": metadata["input_sha256"],
        "cache_sha256": sha256_file(cache_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
