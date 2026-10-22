"""Build gold-aware OOF training rows for the atomic edit calibrator.

Candidate generation and LLM review must already be complete.  This is the
only V3 preprocessing stage that reads gold NER labels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.atomic_calibrator import (
    EXAMPLE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    build_feature_vector,
    count_delta,
    entity_counts,
    f1_from_counts,
    normalize_edit_class,
    outcome_from_gain,
    sha256_file,
)
from modules.crf_edit_decomposition import describe_atomic_entity_edit
from modules.llm_dataset import extract_bio_entities
from llm.atomic_review_schema import (
    validate_atomic_baseline_entities,
    validate_atomic_edit_candidates,
    validate_atomic_edits_against_baseline,
    validate_atomic_review_output,
)
from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION


META_ROLES = ("meta_train", "meta_calibration", "meta_evaluation")
CANDIDATE_SCHEMA_VERSION = "atomic-edit-candidates-v1"
REVIEW_SCHEMA_VERSION = "atomic-edit-reviews-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Gold training JSONL/text file")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help="Fraction assigned to policy calibration when no manifest assigns roles.",
    )
    parser.add_argument(
        "--evaluation-fraction",
        type=float,
        default=0.20,
        help="Independent go/no-go fraction when no manifest assigns roles.",
    )
    parser.add_argument(
        "--require-all-reviewed",
        action="store_true",
        help="Fail instead of safely skipping candidate samples with no review record.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(paths: list[Path], name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid {name} JSON in {path}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{name} record must be an object in {path}:{line_number}")
                sample_id = record.get("sample_id")
                if type(sample_id) is not int or sample_id < 0:
                    raise ValueError(f"Invalid {name} sample_id in {path}:{line_number}")
                if sample_id in seen:
                    raise ValueError(f"Duplicate {name} sample_id={sample_id}")
                seen.add(sample_id)
                records.append(record)
    return records


def load_gold(path: Path) -> list[list[str]]:
    result: list[list[str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = ast.literal_eval(line)
                labels = record["label"]
            except (KeyError, SyntaxError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid gold record at {path}:{line_number}") from error
            if not isinstance(labels, list):
                raise ValueError(f"Gold labels must be a list at {path}:{line_number}")
            result.append([str(label).replace("MISC", "MIS") for label in labels])
    return result


def _manifest_roles(path: Path | None) -> dict[str, dict[int, str]]:
    if path is None:
        return {"local": {}, "source": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    roles: dict[str, dict[int, str]] = {"local": {}, "source": {}}

    def add(sample_id: Any, role: Any, scope: str = "local") -> None:
        if isinstance(sample_id, bool) or not str(sample_id).isdigit():
            raise ValueError("manifest sample_id must be a non-negative integer")
        if scope not in roles:
            raise ValueError(f"unsupported manifest id scope: {scope}")
        normalized = str(role).strip().lower().replace("-", "_")
        aliases = {
            "train": "meta_train",
            "calibration": "meta_calibration",
            "calibrate": "meta_calibration",
            "cal": "meta_calibration",
            "evaluation": "meta_evaluation",
            "evaluate": "meta_evaluation",
            "eval": "meta_evaluation",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in META_ROLES:
            return
        key = int(sample_id)
        previous = roles[scope].get(key)
        if previous is not None and previous != normalized:
            raise ValueError(
                f"manifest assigns {scope} sample_id={key} to two meta roles"
            )
        roles[scope][key] = normalized

    if isinstance(payload, Mapping):
        for role in META_ROLES:
            values = payload.get(role)
            if isinstance(values, list):
                for sample_id in values:
                    add(sample_id, role)
        meta_roles = payload.get("meta_roles")
        if isinstance(meta_roles, Mapping):
            for role in META_ROLES:
                values = meta_roles.get(f"{role}_source_sample_ids")
                if isinstance(values, list):
                    for sample_id in values:
                        add(sample_id, role, "source")
        assignments = payload.get("assignments", payload.get("samples"))
        if isinstance(assignments, Mapping):
            for sample_id, value in assignments.items():
                role = value.get("role") if isinstance(value, Mapping) else value
                add(sample_id, role)
                if isinstance(value, Mapping) and value.get("source_sample_id") is not None:
                    add(value["source_sample_id"], role, "source")
        elif isinstance(assignments, list):
            for item in assignments:
                if isinstance(item, Mapping):
                    role = item.get("meta_role", item.get("role"))
                    if item.get("sample_id") is not None:
                        add(item["sample_id"], role, "local")
                    if item.get("source_sample_id") is not None:
                        add(item["source_sample_id"], role, "source")
    if not roles["local"] and not roles["source"]:
        raise ValueError("manifest does not contain meta role assignments")
    return roles


def stable_role(
    source_sample_id: int,
    seed: int,
    calibration_fraction: float,
    evaluation_fraction: float,
) -> str:
    """Assign 60/20/20 roles from the original source row, never gold."""
    digest = hashlib.sha256(f"{seed}:{source_sample_id}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < evaluation_fraction:
        return "meta_evaluation"
    if value < evaluation_fraction + calibration_fraction:
        return "meta_calibration"
    return "meta_train"


def candidate_by_id(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate record has no CRF candidates")
    result: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("CRF candidate must be an object")
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in result:
            raise ValueError("CRF candidate IDs must be non-empty and unique")
        bio = candidate.get("bio")
        if not isinstance(bio, list):
            raise ValueError("CRF candidate must contain a BIO sequence")
        result[candidate_id] = candidate
    return result


def atomic_edits(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("edits", record.get("atomic_edits"))
    if not isinstance(raw, list):
        raise ValueError("candidate record must contain an edits list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edit in raw:
        if not isinstance(edit, dict):
            raise ValueError("atomic edit must be an object")
        edit_id = str(edit.get("edit_id", ""))
        if not edit_id or edit_id in seen:
            raise ValueError("atomic edit IDs must be non-empty and unique")
        seen.add(edit_id)
        normalized = dict(edit)
        normalized["edit_id"] = edit_id
        normalized["candidate_id"] = str(edit.get("candidate_id", ""))
        normalized["edit_class"] = normalize_edit_class(edit.get("edit_class"))
        result.append(normalized)
    return result


def review_fingerprint(record: Mapping[str, Any]) -> str:
    value = record.get("review_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError("atomic record is missing review_fingerprint")
    return value


def validate_review_record(
    candidate: Mapping[str, Any],
    review_record: Mapping[str, Any],
    edits: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unsupported atomic candidate schema")
    if review_record.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported atomic review schema")
    if candidate.get("prompt_version") != ATOMIC_REVIEW_PROMPT_VERSION:
        raise ValueError("unsupported atomic candidate prompt version")
    if review_record.get("prompt_version") != ATOMIC_REVIEW_PROMPT_VERSION:
        raise ValueError("unsupported atomic review prompt version")
    if not isinstance(review_record.get("llm_model"), str) or not str(
        review_record["llm_model"]
    ).strip():
        raise ValueError("atomic review record is missing llm_model provenance")
    expected_fingerprint = review_fingerprint(candidate)
    actual_fingerprint = review_fingerprint(review_record)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"review fingerprint mismatch for sample_id={candidate['sample_id']}"
        )
    for field in (
        "source_sample_id",
        "split",
        "checkpoint_sha256",
        "input_sha256",
        "source_ids_sha256",
        "official_test",
    ):
        if candidate.get(field) != review_record.get(field):
            raise ValueError(
                f"review {field} mismatch for sample_id={candidate['sample_id']}"
            )
    raw_reviews = review_record.get("reviews")
    if not isinstance(raw_reviews, list):
        raise ValueError("review cache record must contain a reviews list")
    words = candidate.get("words")
    baseline = candidate.get("baseline_entities")
    if not isinstance(words, list) or not isinstance(baseline, list):
        raise ValueError("candidate record is missing words or baseline_entities")
    baseline_entities = validate_atomic_baseline_entities(baseline, words)
    validated_edits = validate_atomic_edit_candidates(edits, words)
    validate_atomic_edits_against_baseline(validated_edits, baseline_entities)
    response = validate_atomic_review_output(
        {"reviews": raw_reviews},
        validated_edits,
        [int(value) for value in candidate.get("valid_image_ids", [])],
        token_count=len(words),
    )
    return {review.edit_id: review.to_dict() for review in response.reviews}


def _project_gold(
    all_gold: list[list[str]],
    candidate: Mapping[str, Any],
) -> list[str]:
    sample_id = int(candidate["sample_id"])
    if sample_id >= len(all_gold):
        raise ValueError(f"candidate sample_id={sample_id} is outside the gold file")
    raw_indices = candidate.get("retained_word_indices")
    if not isinstance(raw_indices, list):
        raise ValueError("candidate is missing retained_word_indices")
    projected = [all_gold[sample_id][int(index)] for index in raw_indices]
    if len(projected) != len(candidate.get("words", [])):
        raise ValueError(f"gold/candidate length mismatch for sample_id={sample_id}")
    return projected


def _counts_dict(counts: tuple[int, int, int]) -> dict[str, int]:
    return {"tp": counts[0], "fp": counts[1], "fn": counts[2]}


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be strictly between 0 and 1")
    if not 0.0 < args.evaluation_fraction < 1.0:
        raise ValueError("--evaluation-fraction must be strictly between 0 and 1")
    if args.calibration_fraction + args.evaluation_fraction >= 1.0:
        raise ValueError(
            "--calibration-fraction and --evaluation-fraction must sum to less than 1"
        )
    output = Path(args.output)
    metadata_path = output.with_name(output.name + ".meta.json")
    for path in (output, metadata_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    candidate_paths = [Path(path) for path in args.candidates]
    review_paths = [Path(path) for path in args.reviews]
    candidates = load_jsonl(candidate_paths, "candidate")
    reviews = load_jsonl(review_paths, "review")
    review_models = {str(record.get("llm_model", "")).strip() for record in reviews}
    if "" in review_models or len(review_models) != 1:
        raise ValueError("review artifacts must use exactly one non-empty llm_model")
    reviews_by_id = {int(record["sample_id"]): record for record in reviews}
    gold = load_gold(input_path)
    roles_from_manifest = _manifest_roles(
        None if args.manifest is None else Path(args.manifest)
    )

    prepared: list[dict[str, Any]] = []
    role_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    role_samples: dict[str, int] = defaultdict(int)
    all_baseline_counts = [0, 0, 0]
    skipped_without_review = 0

    for candidate in candidates:
        sample_id = int(candidate["sample_id"])
        source_sample_id = int(candidate.get("source_sample_id", sample_id))
        role = roles_from_manifest["local"].get(sample_id)
        source_role = roles_from_manifest["source"].get(source_sample_id)
        if role is not None and source_role is not None and role != source_role:
            raise ValueError(
                f"manifest role mismatch for sample_id={sample_id}, "
                f"source_sample_id={source_sample_id}"
            )
        role = role if role is not None else source_role
        if role is None:
            role = str(candidate.get("meta_role", ""))
            if role not in META_ROLES:
                role = stable_role(
                    source_sample_id,
                    args.seed,
                    args.calibration_fraction,
                    args.evaluation_fraction,
                )
        candidate_map = candidate_by_id(candidate)
        top1_id = str(
            candidate.get(
                "top1_candidate_id", candidate.get("baseline_candidate_id", "")
            )
        )
        if top1_id not in candidate_map:
            raise ValueError(f"sample_id={sample_id} has an invalid top1_candidate_id")
        gold_bio = _project_gold(gold, candidate)
        gold_entities = extract_bio_entities(gold_bio)
        top1_entities = extract_bio_entities(candidate_map[top1_id]["bio"])
        baseline_counts = entity_counts(gold_entities, top1_entities)
        for index, value in enumerate(baseline_counts):
            all_baseline_counts[index] += value
            role_counts[role][index] += value
        role_samples[role] += 1
        edits = atomic_edits(candidate)
        review_record = reviews_by_id.get(sample_id)
        if review_record is None:
            skipped_without_review += 1
            if args.require_all_reviewed and edits:
                raise ValueError(f"missing review record for sample_id={sample_id}")
            continue
        review_map = validate_review_record(candidate, review_record, edits)
        prepared.append(
            {
                "candidate": candidate,
                "candidate_map": candidate_map,
                "edits": edits,
                "reviews": review_map,
                "gold_entities": gold_entities,
                "top1_entities": top1_entities,
                "baseline_counts": baseline_counts,
                "role": role,
            }
        )

    if not prepared:
        raise ValueError("no reviewed atomic candidate records are available")
    if role_samples.get("meta_train", 0) == 0:
        raise ValueError("no meta_train samples are available to estimate Dinkelbach q")
    # Freeze q from meta-train only.  Calibration gold must not influence the
    # labels or objective used to fit the calibrator.
    q = f1_from_counts(role_counts["meta_train"])
    examples: list[dict[str, Any]] = []
    reviewed_by_role: dict[str, set[int]] = defaultdict(set)
    for item in prepared:
        candidate = item["candidate"]
        sample_id = int(candidate["sample_id"])
        expected_fingerprint = review_fingerprint(candidate)
        actual_fingerprint = review_fingerprint(reviews_by_id[sample_id])
        for edit in item["edits"]:
            edit_id = str(edit["edit_id"])
            candidate_id = str(edit["candidate_id"])
            if candidate_id not in item["candidate_map"]:
                raise ValueError(
                    f"sample_id={sample_id} edit={edit_id} references an unknown candidate"
                )
            alternative = item["candidate_map"][candidate_id]
            alternative_entities = extract_bio_entities(alternative["bio"])
            descriptor = describe_atomic_entity_edit(
                item["top1_entities"], alternative_entities
            )
            if descriptor is None or normalize_edit_class(descriptor["edit_class"]) != edit[
                "edit_class"
            ]:
                raise ValueError(
                    f"sample_id={sample_id} edit={edit_id} is not the declared atomic edit"
                )
            edited_counts = entity_counts(item["gold_entities"], alternative_entities)
            delta = count_delta(item["baseline_counts"], edited_counts)
            gain = delta.dinkelbach_gain(q)
            review = item["reviews"][edit_id]
            features = build_feature_vector(candidate, edit, review)
            example = {
                "schema_version": EXAMPLE_SCHEMA_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "sample_id": sample_id,
                "source_sample_id": candidate.get("source_sample_id", sample_id),
                "split": candidate.get("split"),
                "fold_id": candidate.get("fold_id"),
                "meta_role": item["role"],
                "checkpoint_sha256": candidate.get("checkpoint_sha256"),
                "input_sha256": candidate.get("input_sha256"),
                "expected_review_fingerprint": expected_fingerprint,
                "review_fingerprint": actual_fingerprint,
                "llm_model": reviews_by_id[sample_id]["llm_model"],
                "edit_id": edit_id,
                "candidate_id": candidate_id,
                "edit_class": edit["edit_class"],
                "crf_rank": int(edit["crf_rank"]),
                "features": features,
                "review": review,
                "baseline_counts": _counts_dict(item["baseline_counts"]),
                "edited_counts": _counts_dict(edited_counts),
                "count_delta": {
                    "tp": delta.delta_tp,
                    "fp": delta.delta_fp,
                    "fn": delta.delta_fn,
                },
                "dinkelbach_q": q,
                "dinkelbach_gain": gain,
                "outcome": outcome_from_gain(gain),
                "baseline_exact": item["top1_entities"] == item["gold_entities"],
                "edited_exact": alternative_entities == item["gold_entities"],
            }
            examples.append(example)
            reviewed_by_role[item["role"]].add(sample_id)

    examples.sort(key=lambda row: (int(row["sample_id"]), int(row["crf_rank"])))
    mode = "w" if args.overwrite else "x"
    with output.open(mode, encoding="utf-8") as destination:
        for example in examples:
            destination.write(
                json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n"
            )

    roles: dict[str, Any] = {}
    for role in sorted(role_samples):
        role_examples = [row for row in examples if row["meta_role"] == role]
        roles[role] = {
            "samples": role_samples[role],
            "baseline_counts": role_counts[role],
            "baseline_f1": f1_from_counts(role_counts[role]),
            "reviewed_samples": len(reviewed_by_role[role]),
            "edit_examples": len(role_examples),
        }
    roles["all"] = {
        "samples": len(candidates),
        "baseline_counts": all_baseline_counts,
        "baseline_f1": f1_from_counts(all_baseline_counts),
        "reviewed_samples": len({int(row["sample_id"]) for row in examples}),
        "edit_examples": len(examples),
    }
    metadata = {
        "schema_version": "atomic-calibrator-data-v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "candidate_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in candidate_paths
        ],
        "review_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in review_paths
        ],
        "manifest": None
        if args.manifest is None
        else {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
        },
        "split_seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "dinkelbach_q": q,
        "dinkelbach_q_scope": "meta_train_top1_only",
        "llm_model": next(iter(review_models)),
        "skipped_candidate_samples_without_review": skipped_without_review,
        "outcomes": dict(
            sorted(
                defaultdict(int, {
                    key: sum(row["outcome"] == key for row in examples)
                    for key in ("beneficial", "neutral", "harmful")
                }).items()
            )
        ),
        "roles": roles,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Atomic calibrator data complete ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Examples: {output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
