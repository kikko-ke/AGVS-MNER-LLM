"""Build an LLM-augmented training directory and a matched duplication control."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION,
    AUGMENTATION_SCHEMA_VERSION,
    extract_protected_entities,
    labels_for_augmentation,
    validate_augmentation_output,
)
from scripts.build_llm_augmentation_cache import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-output-dir", required=True)
    parser.add_argument("--dataset", default="UNI")
    parser.add_argument("--max-augmentations", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_source(path: Path) -> tuple[list[str], dict[int, dict[str, Any]]]:
    raw_lines: list[str] = []
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for sample_id, line in enumerate(source):
            if not line.strip():
                continue
            raw_lines.append(line.rstrip("\r\n"))
            raw = ast.literal_eval(line)
            words = [str(item) for item in raw["text"]]
            labels = [str(item).upper().replace("MISC", "MIS") for item in raw["label"]]
            images = [str(item) for item in raw["images"]]
            records[sample_id] = {
                "text": words,
                "label": labels,
                "images": images,
                "entities": extract_protected_entities(words, labels),
                "raw_line": line.rstrip("\r\n"),
            }
    return raw_lines, records


def _load_accepted(
    cache_path: Path,
    source_records: dict[int, dict[str, Any]],
    *,
    max_augmentations: int,
) -> list[tuple[int, dict[str, Any]]]:
    accepted: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    with cache_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            cached = json.loads(line)
            if cached.get("schema_version") != AUGMENTATION_SCHEMA_VERSION:
                raise ValueError(f"cache schema mismatch at line {line_number}")
            if cached.get("prompt_version") != AUGMENTATION_PROMPT_VERSION:
                raise ValueError(f"prompt mismatch at line {line_number}")
            sample_id = cached.get("sample_id")
            if isinstance(sample_id, bool) or not isinstance(sample_id, int):
                raise ValueError(f"invalid sample_id at cache line {line_number}")
            if sample_id in seen:
                raise ValueError(f"duplicate cache sample_id={sample_id}")
            seen.add(sample_id)
            if cached.get("status") != "accepted":
                continue
            source_record = source_records.get(sample_id)
            if source_record is None:
                raise ValueError(f"cache sample_id={sample_id} is absent from source")
            payload = {
                field: cached.get(field)
                for field in (
                    "abstain",
                    "context_segments",
                    "used_image_ids",
                    "change_summary",
                    "abstain_reason",
                )
            }
            response = validate_augmentation_output(
                payload,
                source_record["text"],
                source_record["entities"],
                set(map(int, cached.get("valid_image_ids", []))),
                min_context_change_ratio=0.0,
                max_context_change_ratio=1.0,
                min_length_ratio=0.01,
                max_length_ratio=100.0,
                max_words=max(len(cached.get("tokens", [])), 1),
            )
            labels = labels_for_augmentation(
                len(response.tokens), source_record["entities"], response.entity_spans
            )
            if cached.get("labels") != labels:
                raise ValueError(f"cache labels mismatch at line {line_number}")
            accepted.append(
                (
                    sample_id,
                    {
                        "text": list(response.tokens),
                        "label": labels,
                        "images": source_record["images"],
                    },
                )
            )
            if max_augmentations and len(accepted) >= max_augmentations:
                break
    return accepted


def _write_lines(path: Path, lines: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; use --overwrite or a new path")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.max_augmentations < 0:
        raise ValueError("--max-augmentations must be non-negative")
    input_dir = Path(args.input_dir)
    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir)
    control_dir = Path(args.control_output_dir)
    if output_dir.resolve() == control_dir.resolve():
        raise ValueError("augmented and control output directories must differ")
    train_name = f"MNER-{args.dataset}_train.txt"
    val_name = f"MNER-{args.dataset}_val.txt"
    test_name = f"MNER-{args.dataset}_test.txt"
    train_path = input_dir / train_name
    metadata_path = cache_path.with_name(cache_path.name + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != AUGMENTATION_SCHEMA_VERSION:
        raise ValueError("augmentation metadata schema mismatch")
    if metadata.get("input_sha256") != sha256_file(train_path):
        raise ValueError("source training file hash differs from augmentation metadata")

    original_lines, source_records = _load_source(train_path)
    accepted = _load_accepted(
        cache_path,
        source_records,
        max_augmentations=args.max_augmentations,
    )
    if not accepted:
        raise ValueError("cache contains no accepted augmentations")
    _prepare_output(output_dir, overwrite=args.overwrite)
    _prepare_output(control_dir, overwrite=args.overwrite)

    augmented_lines = original_lines + [
        repr(record) for _, record in accepted
    ]
    control_lines = original_lines + [
        source_records[sample_id]["raw_line"] for sample_id, _ in accepted
    ]
    _write_lines(output_dir / train_name, augmented_lines)
    _write_lines(control_dir / train_name, control_lines)
    for name in (val_name, test_name):
        source_path = input_dir / name
        shutil.copyfile(source_path, output_dir / name)
        shutil.copyfile(source_path, control_dir / name)

    report = {
        "schema_version": "llm-augmentation-dataset-v1",
        "source_train": str(train_path),
        "source_train_sha256": sha256_file(train_path),
        "augmentation_cache": str(cache_path),
        "augmentation_cache_sha256": sha256_file(cache_path),
        "original_examples": len(original_lines),
        "accepted_augmentations": len(accepted),
        "augmented_examples": len(augmented_lines),
        "control_examples": len(control_lines),
        "control": "duplicate each accepted augmentation's original source example",
        "source_sample_ids": [sample_id for sample_id, _ in accepted],
    }
    for directory, arm in ((output_dir, "llm_augmentation"), (control_dir, "duplicate_control")):
        arm_report = {**report, "arm": arm}
        (directory / "augmentation_manifest.json").write_text(
            json.dumps(arm_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("=== Augmented dataset build complete ===")
    print(f"Original examples: {len(original_lines)}")
    print(f"Accepted augmentations: {len(accepted)}")
    print(f"LLM arm: {output_dir}")
    print(f"Matched duplicate control: {control_dir}")


if __name__ == "__main__":
    main()
