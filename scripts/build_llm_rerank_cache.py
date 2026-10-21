"""Query an LLM to select among fixed CRF candidates without reading gold labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import LLMTeacherClient
from llm.prompts import RERANK_PROMPT_VERSION
from llm.rerank_schema import validate_rerank_output
from modules.reranker import (
    candidate_record_fingerprint,
    prepare_candidate_display,
    validate_rerank_policy,
)


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
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scope", choices=("pilot", "all"), default="pilot")
    parser.add_argument(
        "--policy",
        default=None,
        help="Frozen validation-selected policy required for test requests.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock test requests after the rerank policy is frozen.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_candidates(path: Path) -> list[dict[str, Any]]:
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
            sample_id = record.get("sample_id")
            if type(sample_id) is not int or sample_id < 0:
                raise ValueError(f"Invalid sample_id in candidate line {line_number}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate candidate sample_id={sample_id}")
            sample_ids.add(sample_id)
            if record.get("prompt_version") != RERANK_PROMPT_VERSION:
                raise ValueError(
                    f"Candidate sample_id={sample_id} uses an incompatible prompt version"
                )
            expected_fingerprint = candidate_record_fingerprint(record)
            if record.get("candidate_fingerprint") != expected_fingerprint:
                raise ValueError(
                    f"Candidate fingerprint mismatch for sample_id={sample_id}"
                )
            records.append(record)
    if not records:
        raise ValueError("Candidate cache is empty")
    return records


def image_path(name: str, image_root: Path, twitter_root: Path) -> Path:
    if "twitter2017" in name:
        full_path = twitter_root / name
        if full_path.exists():
            return full_path
        parts = name.split("-", maxsplit=1)
        return twitter_root / (parts[1] if len(parts) == 2 else name)
    return image_root / name


def candidate_ids(record: dict[str, Any]) -> set[str]:
    return {str(candidate["candidate_id"]) for candidate in record["candidates"]}


def load_existing(
    path: Path,
    candidates_by_id: dict[int, dict[str, Any]],
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
                decision = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Decision line {line_number} is not valid JSON"
                ) from error
            sample_id = decision.get("sample_id")
            if type(sample_id) is not int or sample_id not in candidates_by_id:
                raise ValueError(f"Decision line {line_number} has an unknown sample_id")
            if sample_id in existing:
                raise ValueError(f"Duplicate decision sample_id={sample_id}")
            candidate = candidates_by_id[sample_id]
            if decision.get("candidate_fingerprint") != candidate["candidate_fingerprint"]:
                raise ValueError(f"Stale decision cache for sample_id={sample_id}")
            if decision.get("prompt_version") != RERANK_PROMPT_VERSION:
                raise ValueError(f"Prompt version mismatch for sample_id={sample_id}")
            if decision.get("llm_model") != model:
                raise ValueError(f"LLM model mismatch for sample_id={sample_id}")
            for field in ("split", "checkpoint_sha256", "input_sha256"):
                if decision.get(field) != candidate.get(field):
                    raise ValueError(
                        f"Decision {field} mismatch for sample_id={sample_id}"
                    )
            prompt_candidates = prepare_candidate_display(
                candidate,
                RERANK_PROMPT_VERSION,
            )
            candidate_spans = {
                item["candidate_id"]: [
                    (int(entity["start"]), int(entity["end"]))
                    for entity in item["entities"]
                ]
                for item in prompt_candidates
            }
            validated = validate_rerank_output(
                decision,
                candidate_spans,
                set(map(int, candidate.get("valid_image_ids", []))),
                token_count=len(candidate.get("words", [])),
                candidate_spans=candidate_spans,
            )
            selected_candidate = next(
                item
                for item in candidate["candidates"]
                if item["candidate_id"] == validated.selected_candidate_id
            )
            if decision.get("selected_crf_rank") != int(
                selected_candidate["crf_rank"]
            ):
                raise ValueError(
                    f"Decision CRF rank mismatch for sample_id={sample_id}"
                )
            existing.add(sample_id)
    return existing


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be non-negative")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be at least 1")

    all_records = load_candidates(Path(args.candidates))
    splits = {str(record.get("split", "")) for record in all_records}
    if len(splits) != 1 or "" in splits:
        raise ValueError("Candidate cache must contain exactly one declared split")
    split = next(iter(splits))
    if split == "test":
        if not args.allow_test:
            raise ValueError(
                "Test requests are locked; freeze the prompt/route/threshold and pass "
                "--allow-test for the single final evaluation."
            )
        if args.scope != "all":
            raise ValueError("Test requests require --scope all")
    records = list(all_records)
    if args.scope == "pilot":
        records = [
            record
            for record in records
            if bool(record.get("pilot_route", {}).get("selected", False))
        ]
        records.sort(
            key=lambda record: (
                int(record["pilot_route"].get("route_order", 10**9)),
                int(record["sample_id"]),
            )
        )
    else:
        records.sort(key=lambda record: int(record["sample_id"]))
    if not records:
        raise ValueError(f"No candidate records selected for scope={args.scope}")

    client = LLMTeacherClient(
        args.base_url,
        args.api_key,
        args.model,
        args.timeout,
        args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )
    if args.policy:
        policy_path = Path(args.policy)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        validate_rerank_policy(policy, all_records, llm_model=client.model)
    elif split == "test":
        raise ValueError("Test requests require --policy from the validation pilot")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; use --resume or choose a new output")
    candidates_by_id = {int(record["sample_id"]): record for record in all_records}
    existing_all = (
        load_existing(output, candidates_by_id, client.model) if args.resume else set()
    )
    selected_ids = {int(record["sample_id"]) for record in records}
    existing = existing_all & selected_ids
    print(
        f"[resume] {len(existing)} valid decisions; "
        f"{len(records) - len(existing)} selected requests remain",
        flush=True,
    )

    image_root = Path(args.image_root)
    twitter_root = Path(args.twitter_image_root)
    error_path = output.with_name(output.name + ".errors.jsonl")
    output_mode = "a" if args.resume else "x"
    attempted = 0
    succeeded = 0
    failed = 0
    consecutive_failures = 0
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
                prompt_candidates = prepare_candidate_display(
                    record,
                    RERANK_PROMPT_VERSION,
                )
                if candidate_ids(record) != {
                    candidate["candidate_id"] for candidate in prompt_candidates
                }:
                    raise ValueError("Opaque candidate IDs do not match the export")
                valid_image_ids = set(map(int, record.get("valid_image_ids", [])))
                paths: list[Path | None] = []
                for image_id, name in enumerate(record.get("image_names", [])[:4]):
                    paths.append(
                        image_path(str(name), image_root, twitter_root)
                        if image_id in valid_image_ids
                        else None
                    )
                paths.extend([None] * (4 - len(paths)))
                request_started = time.perf_counter()
                decision = client.complete_rerank_json(
                    record["words"],
                    paths,
                    prompt_candidates,
                    valid_image_ids=valid_image_ids,
                    prompt_version=RERANK_PROMPT_VERSION,
                )
                request_latency = time.perf_counter() - request_started
                selected_candidate = next(
                    candidate
                    for candidate in record["candidates"]
                    if candidate["candidate_id"] == decision.selected_candidate_id
                )
                cached = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "split": record["split"],
                    "prompt_version": RERANK_PROMPT_VERSION,
                    "candidate_fingerprint": record["candidate_fingerprint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "input_sha256": record["input_sha256"],
                    "llm_model": client.model,
                    "route_arm": record.get("pilot_route", {}).get("arm"),
                    "selected_candidate_id": decision.selected_candidate_id,
                    "selected_crf_rank": int(selected_candidate["crf_rank"]),
                    "confidence": decision.confidence,
                    "uncertain": decision.uncertain,
                    "used_image_ids": list(decision.used_image_ids),
                    "decisive_spans": [
                        {
                            "start": span.start,
                            "end": span.end,
                            "evidence": span.evidence,
                        }
                        for span in decision.decisive_spans
                    ],
                    "brief_reason": decision.brief_reason,
                    "request_latency_seconds": request_latency,
                    "request_attempts": client.last_attempt_count,
                    "usage": client.last_response_usage,
                }
                destination.write(json.dumps(cached, ensure_ascii=False) + "\n")
                destination.flush()
                succeeded += 1
                latencies.append(request_latency)
                consecutive_failures = 0
                if succeeded % 10 == 0:
                    print(
                        f"[progress] succeeded {succeeded}; "
                        f"total valid {len(existing) + succeeded}/{len(records)} "
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
                            "candidate_fingerprint": record["candidate_fingerprint"],
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

    print("\n=== LLM reranker cache run complete ===")
    print(f"Selected records: {len(records)}")
    print(f"Previously valid: {len(existing)}")
    print(f"Attempted: {attempted}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total valid: {len(existing) + succeeded}")
    if latencies:
        print(f"Mean successful request latency: {sum(latencies) / len(latencies):.2f}s")
    print(f"Decision cache: {output}")
    print(f"Error log: {error_path}")


if __name__ == "__main__":
    main()
