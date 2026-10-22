"""Query an LLM for independent reviews of fixed strict atomic CRF edits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.atomic_review_schema import (
    validate_atomic_baseline_entities,
    validate_atomic_edit_candidates,
    validate_atomic_edits_against_baseline,
    validate_atomic_review_output,
)
from llm.client import LLMTeacherClient
from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION
from scripts.export_atomic_edit_candidates import atomic_review_record_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--twitter-image-root", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scope", choices=("routed", "all"), default="routed")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def image_path(name: str, image_root: Path, twitter_root: Path) -> Path:
    if "twitter2017" in name:
        full_path = twitter_root / name
        if full_path.exists():
            return full_path
        parts = name.split("-", maxsplit=1)
        return twitter_root / (parts[1] if len(parts) == 2 else name)
    return image_root / name


def _validate_atomic_record(record: Mapping[str, Any], line_number: int) -> None:
    sample_id = record.get("sample_id")
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
        raise ValueError(f"Invalid sample_id in candidate line {line_number}")
    if record.get("prompt_version") != ATOMIC_REVIEW_PROMPT_VERSION:
        raise ValueError(
            f"Candidate sample_id={sample_id} uses an incompatible prompt version"
        )
    official_test = record.get("official_test")
    if not isinstance(official_test, bool):
        raise ValueError(f"Candidate sample_id={sample_id} is missing official_test")
    provenance = record.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("official_test") is not official_test
    ):
        raise ValueError(f"Candidate official_test provenance mismatch for sample_id={sample_id}")
    expected = atomic_review_record_fingerprint(record)
    if record.get("review_fingerprint") != expected:
        raise ValueError(f"Review fingerprint mismatch for sample_id={sample_id}")
    words = record.get("words")
    edits = record.get("edits")
    baseline = record.get("baseline_entities")
    if not isinstance(words, list) or not isinstance(edits, list):
        raise ValueError(f"Malformed atomic candidate sample_id={sample_id}")
    if not isinstance(baseline, list):
        raise ValueError(f"Missing baseline_entities for sample_id={sample_id}")
    typed_baseline = validate_atomic_baseline_entities(baseline, words)
    if edits:
        typed_edits = validate_atomic_edit_candidates(edits, words)
        validate_atomic_edits_against_baseline(typed_edits, typed_baseline)
    for edit in edits:
        if edit.get("baseline_entities") != baseline:
            raise ValueError(
                f"Edit baseline mismatch for sample_id={sample_id}, "
                f"edit_id={edit.get('edit_id')}"
            )
    if bool(record.get("review_route", {}).get("selected", False)) and not edits:
        raise ValueError(f"Empty atomic edit set cannot be routed for sample_id={sample_id}")


def load_atomic_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Candidate line {line_number} is not valid JSON"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Candidate line {line_number} must be an object")
            _validate_atomic_record(record, line_number)
            sample_id = int(record["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate candidate sample_id={sample_id}")
            sample_ids.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError("Atomic candidate cache is empty")

    immutable_fields = (
        "split",
        "checkpoint_sha256",
        "input_sha256",
        "source_ids_sha256",
        "prompt_version",
        "official_test",
    )
    first = records[0]
    for record in records[1:]:
        for field in immutable_fields:
            if record.get(field) != first.get(field):
                raise ValueError(f"Candidate cache mixes incompatible {field} values")
    return records


def load_existing_atomic_reviews(
    path: Path,
    candidates_by_id: Mapping[int, Mapping[str, Any]],
    model: str,
) -> set[int]:
    if not path.exists():
        return set()
    existing: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                cached = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Review line {line_number} is not valid JSON"
                ) from error
            if not isinstance(cached, dict):
                raise ValueError(f"Review line {line_number} must be an object")
            sample_id = cached.get("sample_id")
            if isinstance(sample_id, bool) or not isinstance(sample_id, int):
                raise ValueError(f"Invalid review sample_id at line {line_number}")
            if sample_id not in candidates_by_id:
                raise ValueError(f"Unknown review sample_id={sample_id}")
            if sample_id in existing:
                raise ValueError(f"Duplicate review sample_id={sample_id}")
            candidate = candidates_by_id[sample_id]
            if cached.get("review_fingerprint") != candidate.get("review_fingerprint"):
                raise ValueError(f"Stale atomic review cache for sample_id={sample_id}")
            if cached.get("prompt_version") != ATOMIC_REVIEW_PROMPT_VERSION:
                raise ValueError(f"Prompt version mismatch for sample_id={sample_id}")
            if cached.get("llm_model") != model:
                raise ValueError(f"LLM model mismatch for sample_id={sample_id}")
            for field in (
                "source_sample_id",
                "split",
                "checkpoint_sha256",
                "input_sha256",
                "source_ids_sha256",
                "official_test",
            ):
                if cached.get(field) != candidate.get(field):
                    raise ValueError(f"Review {field} mismatch for sample_id={sample_id}")
            typed_edits = validate_atomic_edit_candidates(
                candidate["edits"], candidate["words"]
            )
            validate_atomic_review_output(
                cached,
                typed_edits,
                set(map(int, candidate.get("valid_image_ids", []))),
                token_count=len(candidate["words"]),
            )
            existing.add(sample_id)
    return existing


def _selected_records(
    records: list[dict[str, Any]], scope: str
) -> list[dict[str, Any]]:
    if scope == "all":
        return sorted(
            [record for record in records if bool(record.get("edits"))],
            key=lambda record: int(record["sample_id"]),
        )
    selected = [
        record
        for record in records
        if bool(record.get("edits"))
        and bool(record.get("review_route", {}).get("selected", False))
    ]
    selected.sort(
        key=lambda record: (
            int(record.get("review_route", {}).get("route_order", 10**9)),
            int(record["sample_id"]),
        )
    )
    return selected


def validate_test_request_unlock(
    records: list[dict[str, Any]], *, allow_test: bool, scope: str
) -> None:
    """Enforce the immutable input-derived official-test lock."""
    official_test_values = {record.get("official_test") for record in records}
    if official_test_values - {False, True} or len(official_test_values) != 1:
        raise ValueError("Candidate cache mixes invalid official_test provenance")
    if True in official_test_values:
        if not allow_test:
            raise ValueError(
                "Official test atomic reviews are locked; pass --allow-test once frozen"
            )
        if scope != "all":
            raise ValueError("Official test atomic reviews require --scope all")


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be non-negative")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be at least 1")

    all_records = load_atomic_candidates(Path(args.candidates))
    validate_test_request_unlock(
        all_records, allow_test=args.allow_test, scope=args.scope
    )
    records = _selected_records(all_records, args.scope)
    if not records:
        raise ValueError(f"No atomic records selected for scope={args.scope}")

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
        raise FileExistsError(f"{output} exists; use --resume or choose a new output")
    candidates_by_id = {int(record["sample_id"]): record for record in all_records}
    existing_all = (
        load_existing_atomic_reviews(output, candidates_by_id, client.model)
        if args.resume
        else set()
    )
    selected_ids = {int(record["sample_id"]) for record in records}
    existing = existing_all & selected_ids
    print(
        f"[resume] {len(existing)} valid atomic reviews; "
        f"{len(records) - len(existing)} selected requests remain",
        flush=True,
    )

    image_root = Path(args.image_root)
    twitter_root = Path(args.twitter_image_root)
    error_path = output.with_name(output.name + ".errors.jsonl")
    output_mode = "a" if args.resume else "x"
    attempted = succeeded = failed = consecutive_failures = 0
    latencies: list[float] = []
    with (
        output.open(output_mode, encoding="utf-8") as destination,
        error_path.open("a", encoding="utf-8") as error_file,
    ):
        for record in records:
            sample_id = int(record["sample_id"])
            if sample_id in existing:
                continue
            if args.limit and attempted >= args.limit:
                print(f"[limit] attempted {attempted} selected samples", flush=True)
                break
            attempted += 1
            try:
                valid_image_ids = set(map(int, record.get("valid_image_ids", [])))
                paths: list[Path | None] = []
                for image_id, name in enumerate(record.get("image_names", [])[:4]):
                    paths.append(
                        image_path(str(name), image_root, twitter_root)
                        if image_id in valid_image_ids
                        else None
                    )
                paths.extend([None] * (4 - len(paths)))
                started = time.perf_counter()
                response = client.complete_atomic_review_json(
                    record["words"],
                    paths,
                    record["edits"],
                    valid_image_ids=valid_image_ids,
                    prompt_version=ATOMIC_REVIEW_PROMPT_VERSION,
                )
                latency = time.perf_counter() - started
                cached = {
                    "schema_version": "atomic-edit-reviews-v1",
                    "sample_id": sample_id,
                    "source_sample_id": record.get("source_sample_id"),
                    "split": record["split"],
                    "official_test": bool(record["official_test"]),
                    "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
                    "review_fingerprint": record["review_fingerprint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "input_sha256": record["input_sha256"],
                    "source_ids_sha256": record.get("source_ids_sha256"),
                    "llm_model": client.model,
                    "route_order": record.get("review_route", {}).get("route_order"),
                    **response.to_dict(),
                    "request_latency_seconds": latency,
                    "request_attempts": client.last_attempt_count,
                    "usage": client.last_response_usage,
                }
                destination.write(json.dumps(cached, ensure_ascii=False) + "\n")
                destination.flush()
                succeeded += 1
                latencies.append(latency)
                consecutive_failures = 0
                if succeeded % 10 == 0:
                    print(
                        f"[progress] succeeded {succeeded}; total valid "
                        f"{len(existing) + succeeded}/{len(records)} "
                        f"(sample_id={sample_id})",
                        flush=True,
                    )
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
                            "source_sample_id": record.get("source_sample_id"),
                            "review_fingerprint": record["review_fingerprint"],
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

    print("\n=== LLM atomic review cache run complete ===")
    print(f"Selected records: {len(records)}")
    print(f"Previously valid: {len(existing)}")
    print(f"Attempted: {attempted}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total valid: {len(existing) + succeeded}")
    if latencies:
        print(f"Mean successful request latency: {sum(latencies) / len(latencies):.2f}s")
    print(f"Review cache: {output}")
    print(f"Error log: {error_path}")


if __name__ == "__main__":
    main()
