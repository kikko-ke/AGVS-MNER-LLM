"""Build a resumable cache of role-separated reviews for fixed atomic edits.

Selection is deliberately gold-free: the split manifest supplies only immutable
sample roles, samples are ordered by their exported atomic margin, and edits are
ordered by their exported CRF score gap.  One cache row represents exactly one
``(sample_id, edit_id, reviewer_role)`` request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.atomic_multirole_schema import (
    ATOMIC_MULTIROLE_PROMPT_VERSION,
    ATOMIC_MULTIROLE_ROLES,
    ROLE_CRITIC,
    ROLE_TEXT,
    ROLE_VISUAL,
    validate_atomic_multirole_output,
)
from llm.atomic_review_schema import validate_atomic_edit_candidates
from llm.client import LLMTeacherClient
from scripts.build_llm_atomic_review_cache import (
    _selected_records,
    image_path,
    load_atomic_candidates,
    validate_test_request_unlock,
)


SCHEMA_VERSION = "atomic-edit-multirole-reviews-v1"
ALLOWED_META_ROLES = frozenset({"meta_train", "meta_calibration"})
DEFAULT_REVIEWER_ROLES = (ROLE_TEXT, ROLE_VISUAL, ROLE_CRITIC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--roles", nargs="+", default=["meta_train"], metavar="META_ROLE"
    )
    parser.add_argument(
        "--reviewer-roles",
        nargs="+",
        default=list(DEFAULT_REVIEWER_ROLES),
        metavar="REVIEWER_ROLE",
    )
    parser.add_argument("--route-samples", type=int, default=20)
    parser.add_argument("--max-edits-per-sample", type=int, default=1)
    parser.add_argument("--scope", choices=("routed", "all"), default="all")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--twitter-image-root", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_requested_meta_roles(values: Sequence[str]) -> list[str]:
    roles = list(dict.fromkeys(map(str, values)))
    if not roles or any(role not in ALLOWED_META_ROLES for role in roles):
        raise ValueError(
            "--roles may contain only meta_train or meta_calibration; "
            "meta_evaluation is permanently locked"
        )
    return roles


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_roles(path: Path) -> dict[int, tuple[int, str]]:
    """Load the manifest's local-id role mapping without consulting gold."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    if payload.get("gold_labels_used_for_assignment") is not False:
        raise ValueError("manifest must prove that gold labels were not used")
    if payload.get("official_validation_or_test_used") is not False:
        raise ValueError("manifest must not use official validation or test data")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("manifest must contain non-empty assignments")
    result: dict[int, tuple[int, str]] = {}
    seen_sources: set[int] = set()
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("manifest assignment must be an object")
        sample_id = item.get("sample_id")
        source_id = item.get("source_sample_id")
        role = item.get("role")
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or sample_id < 0
            or isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id < 0
        ):
            raise ValueError("manifest assignment IDs must be non-negative integers")
        if role not in {"meta_train", "meta_calibration", "meta_evaluation"}:
            raise ValueError(f"unsupported manifest role: {role!r}")
        if sample_id in result or source_id in seen_sources:
            raise ValueError("manifest assignments must have unique local and source IDs")
        result[sample_id] = (source_id, str(role))
        seen_sources.add(source_id)
    return result


def _finite_margin(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        margin = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(margin):
        raise ValueError(f"{field} must be finite")
    return margin


def _sample_margin(record: Mapping[str, Any]) -> float:
    route = record.get("route_features")
    if not isinstance(route, Mapping):
        raise ValueError(f"sample_id={record.get('sample_id')} lacks route_features")
    return _finite_margin(
        route.get("minimum_atomic_score_gap_per_word"),
        "minimum_atomic_score_gap_per_word",
    )


def select_atomic_edits(
    records: Sequence[dict[str, Any]],
    manifest_roles: Mapping[int, tuple[int, str]],
    *,
    roles: set[str],
    route_samples: int,
    max_edits_per_sample: int,
) -> list[dict[str, Any]]:
    """Select samples and edits using only exported, label-free margins."""
    eligible: list[tuple[float, int, dict[str, Any], str]] = []
    for record in records:
        sample_id = int(record["sample_id"])
        assignment = manifest_roles.get(sample_id)
        if assignment is None:
            raise ValueError(f"manifest has no assignment for sample_id={sample_id}")
        source_id, meta_role = assignment
        if record.get("source_sample_id") != source_id:
            raise ValueError(f"manifest source_sample_id mismatch for sample_id={sample_id}")
        if meta_role in roles and record.get("edits"):
            eligible.append((_sample_margin(record), sample_id, record, meta_role))
    eligible.sort(key=lambda item: (item[0], item[1]))
    if route_samples:
        eligible = eligible[:route_samples]

    selected: list[dict[str, Any]] = []
    for route_order, (_, sample_id, record, meta_role) in enumerate(eligible):
        typed = validate_atomic_edit_candidates(record["edits"], record["words"])
        typed_ids = {edit.edit_id for edit in typed}
        edits = sorted(
            record["edits"],
            key=lambda edit: (
                _finite_margin(edit.get("score_gap_per_word"), "score_gap_per_word"),
                str(edit.get("edit_id", "")),
            ),
        )[:max_edits_per_sample]
        for edit in edits:
            edit_id = str(edit.get("edit_id", ""))
            if edit_id not in typed_ids:
                raise ValueError(f"invalid selected edit_id={edit_id}")
            selected.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": record.get("source_sample_id"),
                    "edit_id": edit_id,
                    "meta_role": meta_role,
                    "route_order": route_order,
                    "score_gap_per_word": _finite_margin(
                        edit.get("score_gap_per_word"), "score_gap_per_word"
                    ),
                    "record": record,
                    "edit": edit,
                }
            )
    return selected


def _review_key(sample_id: int, edit_id: str, reviewer_role: str) -> tuple[int, str, str]:
    return (int(sample_id), str(edit_id), str(reviewer_role))


def expected_keys(
    selected: Sequence[Mapping[str, Any]], reviewer_roles: Sequence[str]
) -> list[tuple[int, str, str]]:
    return [
        _review_key(item["sample_id"], item["edit_id"], reviewer_role)
        for item in selected
        for reviewer_role in reviewer_roles
    ]


def load_existing_reviews(
    path: Path,
    selected_by_key: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    model: str,
    reviewer_roles: set[str],
) -> set[tuple[int, str, str]]:
    if not path.exists():
        return set()
    result: set[tuple[int, str, str]] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                cached = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Review line {line_number} is not valid JSON") from error
            if not isinstance(cached, Mapping):
                raise ValueError(f"Review line {line_number} must be an object")
            if cached.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Review schema mismatch at line {line_number}")
            sample_id = cached.get("sample_id")
            edit_id = cached.get("edit_id")
            reviewer_role = cached.get("role")
            if (
                isinstance(sample_id, bool)
                or not isinstance(sample_id, int)
                or not isinstance(edit_id, str)
                or not isinstance(reviewer_role, str)
            ):
                raise ValueError(f"Invalid review key at line {line_number}")
            key = _review_key(sample_id, edit_id, reviewer_role)
            if key in result:
                raise ValueError(f"Duplicate review key={key}")
            if reviewer_role not in reviewer_roles:
                raise ValueError(f"Unexpected reviewer role={reviewer_role}")
            selected = selected_by_key.get((sample_id, edit_id))
            if selected is None:
                raise ValueError(f"Unknown or unselected review key={key}")
            candidate = selected["record"]
            if cached.get("review_fingerprint") != candidate.get("review_fingerprint"):
                raise ValueError(f"Stale review fingerprint for key={key}")
            if cached.get("prompt_version") != ATOMIC_MULTIROLE_PROMPT_VERSION:
                raise ValueError(f"Prompt version mismatch for key={key}")
            if cached.get("llm_model") != model:
                raise ValueError(f"LLM model mismatch for key={key}")
            if cached.get("meta_role") != selected.get("meta_role"):
                raise ValueError(f"Meta role mismatch for key={key}")
            for field in (
                "source_sample_id",
                "split",
                "official_test",
                "checkpoint_sha256",
                "input_sha256",
                "source_ids_sha256",
            ):
                if cached.get(field) != candidate.get(field):
                    raise ValueError(f"Review {field} mismatch for key={key}")
            typed_edit = validate_atomic_edit_candidates(
                [selected["edit"]], candidate["words"]
            )[0]
            review_payload = {
                field: cached.get(field)
                for field in (
                    "edit_id",
                    "role",
                    "verdict",
                    "evidence_target",
                    "reason_code",
                    "used_image_ids",
                    "brief_evidence",
                )
            }
            validate_atomic_multirole_output(
                review_payload,
                typed_edit,
                reviewer_role,
                set(map(int, candidate.get("valid_image_ids", []))),
            )
            result.add(key)
    return result


def build_metadata(
    *,
    candidates_path: Path,
    manifest_path: Path,
    selected: Sequence[Mapping[str, Any]],
    reviewer_roles: Sequence[str],
    roles: Sequence[str],
    route_samples: int,
    max_edits_per_sample: int,
    scope: str,
) -> dict[str, Any]:
    selected_edits = [
        {
            "sample_id": int(item["sample_id"]),
            "source_sample_id": item.get("source_sample_id"),
            "edit_id": str(item["edit_id"]),
            "meta_role": str(item["meta_role"]),
            "route_order": int(item["route_order"]),
            "score_gap_per_word": float(item["score_gap_per_word"]),
        }
        for item in selected
    ]
    keys = expected_keys(selected, reviewer_roles)
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": ATOMIC_MULTIROLE_PROMPT_VERSION,
        "candidates": str(candidates_path),
        "candidates_sha256": sha256_file(candidates_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_rule": (
            "filter manifest meta role; order samples by ascending "
            "minimum_atomic_score_gap_per_word then sample_id; order edits by "
            "ascending score_gap_per_word then edit_id"
        ),
        "gold_entity_labels_used": False,
        "roles": list(roles),
        "reviewer_roles": list(reviewer_roles),
        "scope": scope,
        "route_samples": route_samples,
        "max_edits_per_sample": max_edits_per_sample,
        "selected_edits": selected_edits,
        "expected_review_keys": [
            {"sample_id": sample_id, "edit_id": edit_id, "role": reviewer_role}
            for sample_id, edit_id, reviewer_role in keys
        ],
        "expected_reviews": len(keys),
    }


def write_or_validate_metadata(path: Path, metadata: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; use --resume or choose a new output")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("Existing multirole metadata does not match this request")
        return
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.route_samples < 1:
        raise ValueError("--route-samples must be at least 1")
    if args.max_edits_per_sample < 1:
        raise ValueError("--max-edits-per-sample must be at least 1")
    if args.limit < 0 or args.request_delay < 0:
        raise ValueError("--limit and --request-delay must be non-negative")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be at least 1")
    roles = validate_requested_meta_roles(args.roles)
    reviewer_roles = list(dict.fromkeys(args.reviewer_roles))
    if (
        not reviewer_roles
        or len(reviewer_roles) != len(args.reviewer_roles)
        or any(role not in ATOMIC_MULTIROLE_ROLES for role in reviewer_roles)
    ):
        raise ValueError("--reviewer-roles must be unique supported reviewer roles")

    candidates_path = Path(args.candidates)
    manifest_path = Path(args.manifest)
    all_records = load_atomic_candidates(candidates_path)
    validate_test_request_unlock(
        all_records, allow_test=args.allow_test, scope=args.scope
    )
    scoped_records = _selected_records(all_records, args.scope)
    manifest_roles = load_manifest_roles(manifest_path)
    selected = select_atomic_edits(
        scoped_records,
        manifest_roles,
        roles=set(roles),
        route_samples=args.route_samples,
        max_edits_per_sample=args.max_edits_per_sample,
    )
    if not selected:
        raise ValueError("No atomic edits matched the requested manifest roles and scope")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; use --resume or choose a new output")
    metadata = build_metadata(
        candidates_path=candidates_path,
        manifest_path=manifest_path,
        selected=selected,
        reviewer_roles=reviewer_roles,
        roles=roles,
        route_samples=args.route_samples,
        max_edits_per_sample=args.max_edits_per_sample,
        scope=args.scope,
    )
    metadata_path = output.with_name(output.name + ".meta.json")
    write_or_validate_metadata(metadata_path, metadata, resume=args.resume)

    client = LLMTeacherClient(
        args.base_url,
        args.api_key,
        args.model,
        args.timeout,
        args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )
    selected_by_key = {
        (int(item["sample_id"]), str(item["edit_id"])): item for item in selected
    }
    existing = (
        load_existing_reviews(
            output,
            selected_by_key,
            model=client.model,
            reviewer_roles=set(reviewer_roles),
        )
        if args.resume
        else set()
    )
    tasks = [
        (item, reviewer_role)
        for item in selected
        for reviewer_role in reviewer_roles
    ]
    remaining = [
        (item, role)
        for item, role in tasks
        if _review_key(item["sample_id"], item["edit_id"], role) not in existing
    ]
    print(
        f"[resume] {len(existing)} valid multirole reviews; "
        f"{len(remaining)} selected requests remain",
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
        for item, reviewer_role in remaining:
            if args.limit and attempted >= args.limit:
                print(f"[limit] attempted {attempted} selected requests", flush=True)
                break
            attempted += 1
            record = item["record"]
            sample_id = int(item["sample_id"])
            edit_id = str(item["edit_id"])
            key = _review_key(sample_id, edit_id, reviewer_role)
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
                response = client.complete_atomic_multirole_json(
                    record["words"],
                    paths,
                    item["edit"],
                    reviewer_role,
                    valid_image_ids=valid_image_ids,
                    prompt_version=ATOMIC_MULTIROLE_PROMPT_VERSION,
                )
                latency = time.perf_counter() - started
                cached = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "source_sample_id": record.get("source_sample_id"),
                    "split": record["split"],
                    "official_test": bool(record["official_test"]),
                    "meta_role": item["meta_role"],
                    "route_order": int(item["route_order"]),
                    "prompt_version": ATOMIC_MULTIROLE_PROMPT_VERSION,
                    "review_fingerprint": record["review_fingerprint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "input_sha256": record["input_sha256"],
                    "source_ids_sha256": record.get("source_ids_sha256"),
                    "llm_model": client.model,
                    **response.to_dict(),
                    "request_latency_seconds": latency,
                    "request_attempts": client.last_attempt_count,
                    "usage": client.last_response_usage,
                }
                if _review_key(cached["sample_id"], cached["edit_id"], cached["role"]) != key:
                    raise RuntimeError("validated response key does not match request key")
                destination.write(json.dumps(cached, ensure_ascii=False) + "\n")
                destination.flush()
                succeeded += 1
                latencies.append(latency)
                consecutive_failures = 0
                if succeeded % 10 == 0:
                    print(
                        f"[progress] succeeded {succeeded}; total valid "
                        f"{len(existing) + succeeded}/{len(tasks)} (key={key})",
                        flush=True,
                    )
                if args.request_delay:
                    time.sleep(args.request_delay)
            except Exception as error:
                failed += 1
                consecutive_failures += 1
                print(f"[ERROR] key={key}: {error}", flush=True)
                error_file.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "source_sample_id": record.get("source_sample_id"),
                            "edit_id": edit_id,
                            "role": reviewer_role,
                            "meta_role": item["meta_role"],
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

    print("\n=== LLM atomic multirole cache run complete ===")
    print(f"Selected edits: {len(selected)}")
    print(f"Expected reviews: {len(tasks)}")
    print(f"Previously valid: {len(existing)}")
    print(f"Attempted: {attempted}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total valid: {len(existing) + succeeded}")
    if latencies:
        print(f"Mean successful request latency: {sum(latencies) / len(latencies):.2f}s")
    print(f"Review cache: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Error log: {error_path}")


if __name__ == "__main__":
    main()
