"""Build resumable, training-only multimodal augmentation records.

Gold BIO labels are used only to lock the entities of training examples.  The
LLM may rewrite surrounding context, while a strict validator reconstructs the
labels and rejects entity drift or ungrounded/suspicious output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import BertTokenizer

from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION,
    AUGMENTATION_SCHEMA_VERSION,
    AugmentationResponse,
    ProtectedEntity,
    extract_protected_entities,
    labels_for_augmentation,
    validate_augmentation_output,
)
from llm.client import LLMTeacherClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--twitter-image-root", required=True)
    parser.add_argument("--bert-model", required=True)
    parser.add_argument("--target-types", nargs="+", default=["MIS", "ORG"])
    parser.add_argument("--route-samples", type=int, default=20)
    parser.add_argument("--route-seed", type=int, default=20261027)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=62)
    parser.add_argument("--min-context-change-ratio", type=float, default=0.15)
    parser.add_argument("--max-context-change-ratio", type=float, default=0.85)
    parser.add_argument("--min-length-ratio", type=float, default=0.60)
    parser.add_argument("--max-length-ratio", type=float, default=1.50)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def image_path(name: str, image_root: Path, twitter_root: Path) -> Path:
    if "twitter2017" in name:
        full_path = twitter_root / name
        if full_path.exists():
            return full_path
        parts = name.split("-", maxsplit=1)
        return twitter_root / (parts[1] if len(parts) == 2 else name)
    return image_root / name


def _normalise_record(raw: Any, sample_id: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"sample_id={sample_id} must be an object")
    words = raw.get("text")
    labels = raw.get("label")
    images = raw.get("images")
    if not all(isinstance(value, list) for value in (words, labels, images)):
        raise ValueError(f"sample_id={sample_id} text, label, and images must be lists")
    words = [str(word) for word in words]
    labels = [str(label).upper().replace("MISC", "MIS") for label in labels]
    images = [str(name) for name in images]
    entities = extract_protected_entities(words, labels)
    projection = {"text": words, "label": labels, "images": images}
    return {
        "sample_id": sample_id,
        "words": words,
        "labels": labels,
        "images": images,
        "entities": entities,
        "source_fingerprint": _stable_hash(projection),
    }


def load_source_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for sample_id, line in enumerate(source):
            if not line.strip():
                continue
            try:
                raw = ast.literal_eval(line)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"invalid source record at sample_id={sample_id}") from error
            records.append(_normalise_record(raw, sample_id))
    return records


def _entity_types(record: Mapping[str, Any]) -> set[str]:
    return {entity.label for entity in record["entities"]}


def select_source_records(
    records: Sequence[dict[str, Any]],
    target_types: Sequence[str],
    route_samples: int,
    route_seed: int,
) -> list[dict[str, Any]]:
    """Deterministically balance the requested entity types before filling."""
    target_types = list(dict.fromkeys(target_types))
    if not target_types or any(label not in {"MIS", "ORG", "LOC", "PER"} for label in target_types):
        raise ValueError("--target-types must contain supported entity types")
    eligible = [
        record
        for record in records
        if record["entities"]
        and record["images"]
        and _entity_types(record).intersection(target_types)
    ]
    generator = random.Random(route_seed)
    shuffled = list(eligible)
    generator.shuffle(shuffled)
    buckets = {
        label: [record for record in shuffled if label in _entity_types(record)]
        for label in target_types
    }
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    cursors = {label: 0 for label in target_types}
    while len(selected) < min(route_samples, len(eligible)):
        progress = False
        for label in target_types:
            bucket = buckets[label]
            while cursors[label] < len(bucket):
                record = bucket[cursors[label]]
                cursors[label] += 1
                if record["sample_id"] in seen:
                    continue
                selected.append(record)
                seen.add(record["sample_id"])
                progress = True
                break
            if len(selected) >= route_samples:
                break
        if not progress:
            break
    for record in shuffled:
        if len(selected) >= route_samples:
            break
        if record["sample_id"] not in seen:
            selected.append(record)
            seen.add(record["sample_id"])
    return selected


def build_metadata(
    *,
    input_path: Path,
    selected: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    model: str,
) -> dict[str, Any]:
    return {
        "schema_version": AUGMENTATION_SCHEMA_VERSION,
        "prompt_version": AUGMENTATION_PROMPT_VERSION,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "llm_model": model,
        "target_types": list(dict.fromkeys(args.target_types)),
        "route_samples": args.route_samples,
        "route_seed": args.route_seed,
        "max_seq_length": args.max_seq_length,
        "max_words": args.max_words,
        "min_context_change_ratio": args.min_context_change_ratio,
        "max_context_change_ratio": args.max_context_change_ratio,
        "min_length_ratio": args.min_length_ratio,
        "max_length_ratio": args.max_length_ratio,
        "selection_uses_gold_training_labels": True,
        "official_validation_or_test_used": False,
        "selected": [
            {
                "sample_id": int(record["sample_id"]),
                "source_fingerprint": str(record["source_fingerprint"]),
                "entity_types": sorted(_entity_types(record)),
            }
            for record in selected
        ],
    }


def write_or_validate_metadata(
    path: Path, metadata: Mapping[str, Any], *, resume: bool
) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; use --resume")
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("existing augmentation metadata does not match this request")
        return
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _response_payload(cached: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: cached.get(field)
        for field in (
            "abstain",
            "context_segments",
            "used_image_ids",
            "change_summary",
            "abstain_reason",
        )
    }


def load_existing_records(
    path: Path,
    selected: Mapping[int, Mapping[str, Any]],
    *,
    model: str,
    args: argparse.Namespace,
) -> set[int]:
    if not path.exists():
        return set()
    existing: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            cached = json.loads(line)
            sample_id = cached.get("sample_id")
            if isinstance(sample_id, bool) or not isinstance(sample_id, int):
                raise ValueError(f"invalid sample_id at cache line {line_number}")
            source_record = selected.get(sample_id)
            if source_record is None or sample_id in existing:
                raise ValueError(f"unknown or duplicate sample_id={sample_id} in cache")
            if cached.get("schema_version") != AUGMENTATION_SCHEMA_VERSION:
                raise ValueError(f"cache schema mismatch at line {line_number}")
            if cached.get("prompt_version") != AUGMENTATION_PROMPT_VERSION:
                raise ValueError(f"prompt version mismatch at line {line_number}")
            if cached.get("llm_model") != model:
                raise ValueError(f"LLM model mismatch at line {line_number}")
            if cached.get("source_fingerprint") != source_record["source_fingerprint"]:
                raise ValueError(f"source fingerprint mismatch at line {line_number}")
            status = cached.get("status")
            if status == "rejected":
                if not str(cached.get("policy_rejection", "")).strip():
                    raise ValueError(
                        f"rejected cache record lacks policy_rejection at line {line_number}"
                    )
                if cached.get("labels") not in (None, []):
                    raise ValueError(
                        f"rejected cache record must not carry labels at line {line_number}"
                    )
                existing.add(sample_id)
                continue
            if status not in {"accepted", "abstained"}:
                raise ValueError(f"unknown cache status at line {line_number}")
            valid_ids = set(map(int, cached.get("valid_image_ids", [])))
            response = validate_augmentation_output(
                _response_payload(cached),
                source_record["words"],
                source_record["entities"],
                valid_ids,
                min_context_change_ratio=args.min_context_change_ratio,
                max_context_change_ratio=args.max_context_change_ratio,
                min_length_ratio=args.min_length_ratio,
                max_length_ratio=args.max_length_ratio,
                max_words=args.max_words,
            )
            if not response.abstain:
                expected_labels = labels_for_augmentation(
                    len(response.tokens), source_record["entities"], response.entity_spans
                )
                if cached.get("labels") != expected_labels:
                    raise ValueError(f"cached labels mismatch at line {line_number}")
                if status != "accepted":
                    raise ValueError(f"accepted status mismatch at line {line_number}")
            elif status != "abstained":
                raise ValueError(f"abstained status mismatch at line {line_number}")
            existing.add(sample_id)
    return existing


def main() -> None:
    args = parse_args()
    if args.route_samples < 1 or args.max_seq_length < 4 or args.max_words < 1:
        raise ValueError("route samples, sequence length, and max words must be positive")
    if args.max_words > args.max_seq_length - 2:
        raise ValueError("--max-words cannot exceed --max-seq-length - 2")
    if args.limit < 0 or args.request_delay < 0:
        raise ValueError("--limit and --request-delay must be non-negative")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be at least 1")
    input_path = Path(args.input)
    records = load_source_records(input_path)
    selected_records = select_source_records(
        records, args.target_types, args.route_samples, args.route_seed
    )
    if not selected_records:
        raise ValueError("no training records matched the augmentation route")

    client = LLMTeacherClient(
        args.base_url,
        args.api_key,
        args.model,
        args.timeout,
        args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; use --resume")
    metadata = build_metadata(
        input_path=input_path,
        selected=selected_records,
        args=args,
        model=client.model,
    )
    metadata_path = output.with_name(output.name + ".meta.json")
    write_or_validate_metadata(metadata_path, metadata, resume=args.resume)
    selected_by_id = {record["sample_id"]: record for record in selected_records}
    existing = (
        load_existing_records(
            output,
            selected_by_id,
            model=client.model,
            args=args,
        )
        if args.resume
        else set()
    )
    remaining = [
        record for record in selected_records if record["sample_id"] not in existing
    ]
    print(
        f"[resume] {len(existing)} resolved augmentation requests; "
        f"{len(remaining)} selected requests remain",
        flush=True,
    )

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    image_root = Path(args.image_root)
    twitter_root = Path(args.twitter_image_root)
    error_path = output.with_name(output.name + ".errors.jsonl")
    attempted = succeeded = accepted = abstained = rejected = failed = consecutive_failures = 0
    output_mode = "a" if args.resume else "x"
    with (
        output.open(output_mode, encoding="utf-8") as destination,
        error_path.open("a", encoding="utf-8") as error_file,
    ):
        for record in remaining:
            if args.limit and attempted >= args.limit:
                print(f"[limit] attempted {attempted} selected requests", flush=True)
                break
            attempted += 1
            sample_id = int(record["sample_id"])
            started = time.perf_counter()
            try:
                paths: list[Path | None] = [
                    image_path(name, image_root, twitter_root)
                    for name in record["images"][:4]
                ]
                paths.extend([None] * (4 - len(paths)))

                def validate_wordpieces(response: AugmentationResponse) -> None:
                    count = sum(
                        len(tokenizer.tokenize(token)) for token in response.tokens
                    )
                    if count > args.max_seq_length - 2:
                        raise ValueError(
                            f"augmentation has {count} WordPieces; maximum is "
                            f"{args.max_seq_length - 2}"
                        )

                response = client.complete_augmentation_json(
                    record["words"],
                    paths,
                    record["entities"],
                    min_context_change_ratio=args.min_context_change_ratio,
                    max_context_change_ratio=args.max_context_change_ratio,
                    min_length_ratio=args.min_length_ratio,
                    max_length_ratio=args.max_length_ratio,
                    max_words=args.max_words,
                    accepted_validator=validate_wordpieces,
                )
                latency = time.perf_counter() - started
                cached: dict[str, Any] = {
                    "schema_version": AUGMENTATION_SCHEMA_VERSION,
                    "prompt_version": AUGMENTATION_PROMPT_VERSION,
                    "sample_id": sample_id,
                    "source_fingerprint": record["source_fingerprint"],
                    "llm_model": client.model,
                    "source_entity_types": sorted(_entity_types(record)),
                    "images": record["images"],
                    "valid_image_ids": list(response.used_image_ids),
                    **response.to_dict(),
                    "request_latency_seconds": latency,
                    "request_attempts": client.last_attempt_count,
                    "usage": client.last_response_usage,
                }
                if response.abstain:
                    cached["status"] = "abstained"
                    cached["labels"] = []
                    abstained += 1
                else:
                    cached["status"] = "accepted"
                    cached["labels"] = labels_for_augmentation(
                        len(response.tokens), record["entities"], response.entity_spans
                    )
                    accepted += 1
                destination.write(json.dumps(cached, ensure_ascii=False) + "\n")
                destination.flush()
                succeeded += 1
                consecutive_failures = 0
                print(
                    f"[progress] total resolved {len(existing) + succeeded + rejected}/"
                    f"{len(selected_records)}; accepted={accepted}, abstained={abstained}, "
                    f"rejected={rejected} "
                    f"(sample_id={sample_id}, status={cached['status']})",
                    flush=True,
                )
                if args.request_delay:
                    time.sleep(args.request_delay)
            except ValueError as error:
                rejected += 1
                consecutive_failures = 0
                latency = time.perf_counter() - started
                cached = {
                    "schema_version": AUGMENTATION_SCHEMA_VERSION,
                    "prompt_version": AUGMENTATION_PROMPT_VERSION,
                    "sample_id": sample_id,
                    "source_fingerprint": record["source_fingerprint"],
                    "llm_model": client.model,
                    "source_entity_types": sorted(_entity_types(record)),
                    "images": record["images"],
                    "valid_image_ids": [],
                    "status": "rejected",
                    "policy_rejection": str(error),
                    "labels": [],
                    "request_latency_seconds": latency,
                    "request_attempts": client.last_attempt_count,
                    "usage": client.last_response_usage,
                }
                destination.write(json.dumps(cached, ensure_ascii=False) + "\n")
                destination.flush()
                print(
                    f"[REJECTED] total resolved "
                    f"{len(existing) + succeeded + rejected}/{len(selected_records)}; "
                    f"sample_id={sample_id}: {error}",
                    flush=True,
                )
                error_file.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "source_fingerprint": record["source_fingerprint"],
                            "status": "rejected",
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_file.flush()
                if args.request_delay:
                    time.sleep(args.request_delay)
            except Exception as error:
                failed += 1
                consecutive_failures += 1
                print(f"[ERROR] sample_id={sample_id}: {error}", flush=True)
                error_file.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "source_fingerprint": record["source_fingerprint"],
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_file.flush()
                if consecutive_failures >= args.max_consecutive_failures:
                    print(
                        f"[ABORT] {consecutive_failures} consecutive failures",
                        flush=True,
                    )
                    break

    print("\n=== LLM augmentation cache run complete ===")
    print(f"Selected source samples: {len(selected_records)}")
    print(f"Previously valid: {len(existing)}")
    print(f"Attempted: {attempted}")
    print(f"Succeeded responses: {succeeded}")
    print(f"Accepted augmentations this run: {accepted}")
    print(f"Valid abstentions this run: {abstained}")
    print(f"Policy-rejected responses this run: {rejected}")
    print(f"Unresolved request failures: {failed}")
    print(f"Total resolved cache records: {len(existing) + succeeded + rejected}")
    print(f"Cache: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()
