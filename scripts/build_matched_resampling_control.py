"""Build a training-feature-matched resampling control without an LLM/API call.

The reference distribution comes from the existing LLM+validator-selected arm.
Candidate scoring uses ONLY original training text/labels/images. Overlap with
the reference is allowed, reported, and never rewarded or penalised in matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import BertTokenizer

from llm.augmentation_schema import AUGMENTATION_PROMPT_VERSION, AUGMENTATION_SCHEMA_VERSION
from modules.text_utils import normalize_words
from scripts.build_augmented_dataset import _load_source, _write_lines
from scripts.build_llm_augmentation_cache import image_path, load_source_records, sha256_file


ENTITY_TYPES = ("MIS", "ORG", "PER", "LOC")
SWAP_ATTEMPTS = 50000
MAX_STANDARDIZED_DIFFERENCE = 0.10
MAX_REFERENCE_OVERLAP = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--candidate-metadata", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--twitter-image-root", required=True)
    parser.add_argument("--bert-model", required=True)
    parser.add_argument("--sampling-seed", type=int, default=20261031)
    parser.add_argument("--dataset", default="UNI")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def image_info(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {"decodable": False, "sha256": None, "status": "missing_or_unreadable"}
    digest = hashlib.sha256(raw).hexdigest()
    for status, candidate in (("direct", raw), ("repaired", raw + b"\xff\xd9")):
        try:
            with Image.open(BytesIO(candidate)) as image:
                image.convert("RGB").load()
            return {"decodable": True, "sha256": digest, "status": status}
        except (OSError, UnidentifiedImageError, ValueError):
            continue
    return {"decodable": False, "sha256": digest, "status": "failed"}


def original_features(record: dict[str, Any], tokenizer: Any, valid_images: int):
    words = normalize_words(record["words"])
    counts = Counter(entity.label for entity in record["entities"])
    twitter = ["twitter2017" in name for name in record["images"][:4]]
    domain = "twitter2017" if twitter and all(twitter) else "mixed" if any(twitter) else "mi"
    type_signature = "+".join(label for label in ENTITY_TYPES if counts[label])
    stratum = (domain, type_signature, valid_images)
    word_count = len(words)
    wordpieces = sum(len(tokenizer.tokenize(word)) for word in words)
    entity_count = sum(counts.values())
    features = {
        "word_count": float(word_count), "wordpiece_count": float(wordpieces),
        "entity_count": float(entity_count),
        **{f"entities_{label}": float(counts[label]) for label in ENTITY_TYPES},
    }
    for prefix, value, boundaries in (
        ("words_bin", word_count, (10, 20, 30)),
        ("wordpieces_bin", wordpieces, (16, 32, 48)),
        ("entities_bin", entity_count, (1, 2, 3)),
    ):
        bucket = sum(value > boundary for boundary in boundaries)
        for index in range(len(boundaries) + 1):
            features[f"{prefix}_{index}"] = float(bucket == index)
    return stratum, features


def sample_matched(
    groups: dict[tuple, list[int]], quotas: Counter,
    vectors: dict[int, np.ndarray], target_sum: np.ndarray, scales: np.ndarray,
    *, seed: int, swap_attempts: int = SWAP_ATTEMPTS,
) -> tuple[dict[tuple, list[int]], dict[str, Any]]:
    """Exact categorical quotas plus fixed-budget within-stratum feature swaps.

    No reference IDs, acceptance flags, generated content, validation scores or
    test scores are arguments to this sampler. Only aggregate targets are used.
    """
    generator = random.Random(seed)
    selected, remaining = {}, {}
    for key in sorted(groups):
        population = sorted(groups[key])
        quota = quotas.get(key, 0)
        if not 0 <= quota <= len(population):
            raise ValueError(f"Infeasible categorical quota: {key}")
        selected[key] = generator.sample(population, quota)
        selected_set = set(selected[key])
        remaining[key] = [sample_id for sample_id in population if sample_id not in selected_set]
    if set(quotas) - set(groups):
        raise ValueError("Reference has a stratum absent from the candidate pool")
    count = sum(quotas.values())
    if count < 1:
        raise ValueError("Reference must contain at least one sample")
    selected_sum = sum((vectors[i] for ids in selected.values() for i in ids), np.zeros_like(target_sum))
    denominator = count * scales
    objective = lambda total: float(np.square((total - target_sum) / denominator).sum())
    initial_cost = current_cost = objective(selected_sum)
    exchangeable = [key for key in sorted(groups) if selected[key] and remaining[key]]
    swaps = 0
    for _ in range(swap_attempts if exchangeable else 0):
        key = generator.choice(exchangeable)
        first = generator.randrange(len(selected[key]))
        second = generator.randrange(len(remaining[key]))
        old, new = selected[key][first], remaining[key][second]
        proposed_sum = selected_sum - vectors[old] + vectors[new]
        proposed_cost = objective(proposed_sum)
        if proposed_cost < current_cost - 1e-15:
            selected[key][first], remaining[key][second] = new, old
            selected_sum, current_cost = proposed_sum, proposed_cost
            swaps += 1
    return selected, {"initial_objective": initial_cost, "final_objective": current_cost,
                      "swap_attempts": swap_attempts, "accepted_feature_swaps": swaps}


def main() -> None:
    cli = parse_args()
    source_dir, output_dir = Path(cli.input_dir), Path(cli.output_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; preserve it or choose a new path")
    train_name = f"MNER-{cli.dataset}_train.txt"
    train_path = source_dir / train_name
    metadata_path, reference_path = Path(cli.candidate_metadata), Path(cli.reference_manifest)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != AUGMENTATION_SCHEMA_VERSION
            or metadata.get("prompt_version") != AUGMENTATION_PROMPT_VERSION):
        raise ValueError("Candidate metadata schema/prompt mismatch")
    if reference.get("arm") != "duplicate_control":
        raise ValueError("Reference must be the existing accepted-source duplication arm")
    train_hash = sha256_file(train_path)
    if metadata.get("input_sha256") != train_hash or reference.get("source_train_sha256") != train_hash:
        raise ValueError("Training source SHA256 differs from candidate/reference metadata")
    raw_lines, raw_records = _load_source(train_path)
    records = {row["sample_id"]: row for row in load_source_records(train_path)}
    candidates = metadata.get("selected", [])
    ids = [row["sample_id"] for row in candidates]
    if (not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Candidate IDs must be non-empty, unique integers")
    for row in candidates:
        source = records.get(row["sample_id"])
        if source is None or source["source_fingerprint"] != row["source_fingerprint"]:
            raise ValueError(f"Candidate source mismatch: {row['sample_id']}")
    reference_ids = reference.get("source_sample_ids", [])
    if (not reference_ids or any(type(i) is not int for i in reference_ids)
            or len(set(reference_ids)) != len(reference_ids)
            or not set(reference_ids).issubset(ids)
            or len(reference_ids) != reference.get("accepted_augmentations")):
        raise ValueError("Reference IDs must be unique candidates matching the reference count")
    if reference.get("original_examples") != len(raw_lines):
        raise ValueError("Reference original example count mismatch")
    # Check the treatment dataset itself, not only its descriptive manifest.
    reference_train = reference_path.parent / train_name
    expected_reference = raw_lines + [raw_records[i]["raw_line"] for i in reference_ids]
    actual_reference = reference_train.read_text(encoding="utf-8").splitlines()
    if actual_reference != expected_reference:
        raise ValueError("Reference training file does not match its declared source IDs")

    tokenizer = BertTokenizer.from_pretrained(cli.bert_model, do_lower_case=True, local_files_only=True)
    image_root, twitter_root = Path(cli.image_root), Path(cli.twitter_image_root)
    physical_images: dict[str, dict[str, Any]] = {}
    strata, features, groups = {}, {}, defaultdict(list)
    for sample_id in tqdm(sorted(ids), desc="Audit original candidate features", unit="sample"):
        record = records[sample_id]
        valid = 0
        for name in record["images"][:4]:
            path = str(image_path(name, image_root, twitter_root))
            if path not in physical_images:
                physical_images[path] = image_info(Path(path))
            valid += int(physical_images[path]["decodable"])
        stratum, feature = original_features(record, tokenizer, valid)
        strata[sample_id], features[sample_id] = stratum, feature
        groups[stratum].append(sample_id)
    if any(strata[i][2] == 0 for i in reference_ids):
        raise ValueError("A reference accepted sample now has no decodable images; check image data")
    names = list(features[ids[0]])
    vectors = {i: np.array([features[i][name] for name in names], dtype=np.float64) for i in ids}
    matrix = np.stack([vectors[i] for i in sorted(ids)])
    pool_std = matrix.std(axis=0)
    scales = np.where(pool_std > 1e-12, pool_std, 1.0)
    target_sum = np.stack([vectors[i] for i in reference_ids]).sum(axis=0)
    quotas = Counter(strata[i] for i in reference_ids)
    sampled, optimisation = sample_matched(groups, quotas, vectors, target_sum, scales, seed=cli.sampling_seed)
    # Preserve the reference arm's categorical schedule in the appended block.
    positions = Counter()
    matched_ids = []
    for sample_id in reference_ids:
        key = strata[sample_id]
        matched_ids.append(sampled[key][positions[key]])
        positions[key] += 1
    count = len(reference_ids)
    selected_sum = np.stack([vectors[i] for i in matched_ids]).sum(axis=0)
    differences = (selected_sum - target_sum) / (count * scales)
    max_difference = float(np.abs(differences).max())
    overlap = len(set(matched_ids) & set(reference_ids))
    forced_overlap = sum(max(0, 2 * quota - len(groups[key])) for key, quota in quotas.items())
    balanced = max_difference <= MAX_STANDARDIZED_DIFFERENCE
    informative = overlap / count <= MAX_REFERENCE_OVERLAP
    for split in ("val", "test"):
        name = f"MNER-{cli.dataset}_{split}.txt"
        if sha256_file(source_dir / name) != sha256_file(reference_path.parent / name):
            raise ValueError(f"Reference {split} file differs from original")
    manifest = {
        "schema_version": "feature-matched-resampling-v1", "arm": "feature_matched_no_llm",
        "dataset": cli.dataset,
        "hypothesis_origin": "post-hoc after existing V6.1 results; validation-only exploration",
        "llm_api_requests": 0, "candidate_scoring_uses_llm_outputs": False,
        "reference_distribution_comes_from_llm_and_validator_selection": True,
        "selection_uses_validation_or_test_labels": False,
        "source_train": str(train_path), "source_train_sha256": train_hash,
        "candidate_metadata": str(metadata_path), "candidate_metadata_sha256": sha256_file(metadata_path),
        "reference_manifest": str(reference_path), "reference_manifest_sha256": sha256_file(reference_path),
        "reference_train_sha256": sha256_file(reference_train),
        "candidate_count": len(ids), "original_examples": len(raw_lines),
        "added_examples": count, "total_examples": len(raw_lines) + count,
        "sampling_seed": cli.sampling_seed, "source_sample_ids": matched_ids,
        "reference_source_sample_ids": reference_ids,
        "exact_matching": ["image_filename_domain", "entity_type_presence_signature", "actual_decodable_images_first4"],
        "strata": [{"domain": key[0], "entity_types": key[1], "valid_images": key[2],
                    "pool": len(groups[key]), "reference": quota, "matched": len(sampled[key])}
                   for key, quota in sorted(quotas.items())],
        "features": {name: {"reference_mean": float(target_sum[j] / count),
                            "matched_mean": float(selected_sum[j] / count),
                            "candidate_pool_std": float(pool_std[j]),
                            "difference_in_pool_sd_units": float(differences[j])}
                     for j, name in enumerate(names)},
        "optimisation": optimisation,
        "balance_check": {"max_abs_difference_in_pool_sd_units": max_difference,
                          "threshold": MAX_STANDARDIZED_DIFFERENCE, "passed": balanced},
        "overlap_check": {"count": overlap, "fraction": overlap / count,
                          "minimum_forced_by_exact_strata": forced_overlap,
                          "maximum_fraction_for_training": MAX_REFERENCE_OVERLAP, "passed": informative},
        "ready_for_training": balanced and informative,
        "bert_model": cli.bert_model,
        "tokenizer_files_sha256": {path.name: sha256_file(path) for path in sorted(Path(cli.bert_model).glob("*"))
                                  if path.is_file() and path.suffix in {".json", ".txt"}},
        "physical_candidate_images": physical_images,
        "note": "Acceptance is a generation/validator outcome, not verified semantic quality. "
                "Matching conditions on observed reference distributions; no causal benefit is assumed. "
                "Overlap is allowed and not used in the feature optimisation objective. "
                "Standardised differences use candidate-pool SD, not a significance test.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_lines(output_dir / train_name, raw_lines + [raw_records[i]["raw_line"] for i in matched_ids])
    for split in ("val", "test"):
        name = f"MNER-{cli.dataset}_{split}.txt"
        shutil.copyfile(source_dir / name, output_dir / name)
    manifest["output_files_sha256"] = {name: sha256_file(output_dir / name) for name in (
        train_name, f"MNER-{cli.dataset}_val.txt", f"MNER-{cli.dataset}_test.txt",
    )}
    manifest_path = output_dir / "matched_resampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n=== Matched resampling control build complete ===")
    print(f"Candidates: {len(ids)}; added original examples: {count}; total: {len(raw_lines) + count}")
    print(f"Exact strata matched: {len(quotas)}")
    print(f"Max standardised feature difference: {max_difference:.6f}; pass={balanced}")
    print(f"Overlap with reference: {overlap}/{count} ({overlap/count:.1%}); pass={informative}")
    print(f"Ready for training: {manifest['ready_for_training']}")
    print(f"Manifest: {manifest_path}")
    if not manifest["ready_for_training"]:
        raise RuntimeError("Matching/overlap gate failed; keep the report and inspect it before training")


if __name__ == "__main__":
    main()
