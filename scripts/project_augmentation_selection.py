"""Reuse existing augmentation outcomes only for identical target training records.

No API calls are made. A target candidate is retained only when its full original
text, canonical BIO labels, and ordered image names match the source record.
The output training arm duplicates accepted ORIGINAL target records.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.augmentation_schema import AUGMENTATION_PROMPT_VERSION, AUGMENTATION_SCHEMA_VERSION
from scripts.build_augmented_dataset import _load_source, _write_lines
from scripts.build_llm_augmentation_cache import (
    _stable_hash, load_existing_records, load_source_records, sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--target-input-dir", required=True)
    parser.add_argument("--target-dataset", choices=("MI", "UNI"), default="MI")
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unlabeled_fingerprint(words, images) -> str:
    return _stable_hash({"text": words, "images": images})


def main() -> None:
    cli = parse_args()
    source_input, cache = Path(cli.source_input), Path(cli.source_cache)
    target_dir, output_dir, output_cache = Path(cli.target_input_dir), Path(cli.output_dir), Path(cli.output_cache)
    metadata_path = cache.with_name(cache.name + ".meta.json")
    output_metadata = output_cache.with_name(output_cache.name + ".meta.json")
    if output_dir.exists() or output_cache.exists() or output_metadata.exists():
        raise FileExistsError("Projection output already exists; preserve it and verify before reuse")
    metadata = read_json(metadata_path)
    if (metadata.get("schema_version") != AUGMENTATION_SCHEMA_VERSION
            or metadata.get("prompt_version") != AUGMENTATION_PROMPT_VERSION
            or metadata.get("input_sha256") != sha256_file(source_input)):
        raise ValueError("Source augmentation metadata/schema/input SHA256 mismatch")
    source_records = {row["sample_id"]: row for row in load_source_records(source_input)}
    selected = {}
    for entry in metadata["selected"]:
        sample_id = entry["sample_id"]
        if type(sample_id) is not int or sample_id in selected:
            raise ValueError("Source candidate IDs must be unique integers")
        record = source_records.get(sample_id)
        if record is None or record["source_fingerprint"] != entry["source_fingerprint"]:
            raise ValueError(f"Source candidate fingerprint mismatch: {sample_id}")
        selected[sample_id] = record
    valid = load_existing_records(cache, selected, model=metadata["llm_model"], args=argparse.Namespace(**metadata))
    if valid != set(selected):
        raise ValueError("Source cache must resolve every selected candidate before projection")
    cached = {row["sample_id"]: row for row in (
        json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()
    )}
    target_train = target_dir / f"MNER-{cli.target_dataset}_train.txt"
    target_lines, raw_targets = _load_source(target_train)
    targets = load_source_records(target_train)
    if [row["sample_id"] for row in targets] != list(range(len(target_lines))):
        raise ValueError("Target training file has blank/non-contiguous rows; inspect its format")
    by_fingerprint = defaultdict(list)
    for record in targets:
        by_fingerprint[record["source_fingerprint"]].append(record)
    # Read only text/images from held-out files, solely to check split overlap.
    # Their entity labels are never accessed, scored, or sent to an LLM.
    heldout = set()
    split_hashes = {}
    for split in ("val", "test"):
        path = target_dir / f"MNER-{cli.target_dataset}_{split}.txt"
        split_hashes[path.name] = sha256_file(path)
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                raw = ast.literal_eval(line)
                heldout.add(unlabeled_fingerprint([str(word) for word in raw["text"]],
                                                 [str(name) for name in raw["images"]]))
    overlap = [row["sample_id"] for row in targets
               if unlabeled_fingerprint(row["words"], row["images"]) in heldout]
    if overlap:
        raise ValueError(f"Target train overlaps held-out text+images; inspect splits before training: {overlap[:20]}")
    source_overlap = [row["sample_id"] for row in selected.values()
                      if unlabeled_fingerprint(row["words"], row["images"]) in heldout]
    if source_overlap:
        raise ValueError(f"Source cached candidates overlap target held-out text+images; inspect split compatibility: {source_overlap[:20]}")
    projected, entries, mapping, used_target_ids = [], [], [], set()
    for source_id, record in selected.items():
        matches = by_fingerprint.get(record["source_fingerprint"], [])
        if not matches:
            continue
        if len(matches) != 1 or matches[0]["sample_id"] in used_target_ids:
            raise ValueError(f"Ambiguous many-to-one or one-to-many mapping for source candidate {source_id}")
        target = matches[0]
        target_id = target["sample_id"]
        used_target_ids.add(target_id)
        projected.append({**cached[source_id], "sample_id": target_id,
                          "origin_source_sample_id": source_id})
        entries.append({"sample_id": target_id, "source_fingerprint": target["source_fingerprint"],
                        "entity_types": sorted({entity.label for entity in target["entities"]})})
        mapping.append({"source_sample_id": source_id, "target_sample_id": target_id,
                        "source_fingerprint": target["source_fingerprint"], "status": cached[source_id]["status"]})
    accepted = [row["sample_id"] for row in projected if row["status"] == "accepted"]
    if not accepted:
        raise ValueError("No accepted source candidates match target training records; no selection arm can be built")
    projection = {
        "schema_version": "exact-training-cache-projection-v1", "llm_api_requests": 0,
        "source_input": str(source_input), "source_input_sha256": sha256_file(source_input),
        "source_cache": str(cache), "source_cache_sha256": sha256_file(cache),
        "source_metadata": str(metadata_path), "source_metadata_sha256": sha256_file(metadata_path),
        "source_candidate_count": len(selected), "projected_candidate_count": len(projected),
        "projected_status_counts": dict(Counter(row["status"] for row in projected)),
        "heldout_labels_read": False, "train_heldout_text_images_overlap": 0,
        "source_candidates_target_heldout_text_images_overlap": 0,
        "target_evaluation_sha256": split_hashes, "mapping": mapping,
        "note": "Inherited source-pool selection, not a new target-specific 1000-query experiment. "
                "Only target training matches are reused; images/order and canonical gold labels match. "
                "All resolved statuses are retained. Accepted original records, not rewrites, are duplicated.",
    }
    projected_metadata = {
        **metadata, "input": str(target_train), "input_sha256": sha256_file(target_train),
        "dataset": cli.target_dataset, "route_samples": len(projected), "selected": entries,
        "candidate_selection": "exact target-training subset of the existing source candidate pool",
        "projection": projection,
    }
    # All validation above precedes output creation.
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    output_cache.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in projected), encoding="utf-8")
    output_metadata.write_text(json.dumps(projected_metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=False)
    train_name = target_train.name
    _write_lines(output_dir / train_name, target_lines + [raw_targets[i]["raw_line"] for i in accepted])
    for name in split_hashes:
        shutil.copyfile(target_dir / name, output_dir / name)
    manifest = {
        "schema_version": "llm-augmentation-dataset-v1", "arm": "duplicate_control", "dataset": cli.target_dataset,
        "source_train": str(target_train), "source_train_sha256": sha256_file(target_train),
        "augmentation_cache": str(output_cache), "augmentation_cache_sha256": sha256_file(output_cache),
        "original_examples": len(target_lines), "accepted_augmentations": len(accepted),
        "control_examples": len(target_lines) + len(accepted), "source_sample_ids": accepted,
        "control": "duplicate each accepted outcome's exact target-training original",
        "llm_api_requests": 0, "projection_metadata": str(output_metadata),
        "projection_metadata_sha256": sha256_file(output_metadata),
    }
    (output_dir / "augmentation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8",
    )
    print("=== Exact training-cache projection complete ===")
    print(f"Target dataset: {cli.target_dataset}; original training examples: {len(target_lines)}")
    print(f"Source candidates: {len(selected)}; target-training matches: {len(projected)}")
    print(f"Matched statuses: {projection['projected_status_counts']}")
    print(f"Added original examples: {len(accepted)}; total: {len(target_lines) + len(accepted)}")
    print(f"Projected cache: {output_cache}")
    print(f"Selected-original training data: {output_dir}")


if __name__ == "__main__":
    main()
