"""Match training-only cohorts for source-entity dropout consistency."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modules.context_consistency import ContextViewDataset, entity_alignment, verify_internal_manifest
from modules.text_utils import normalize_words
from scripts.build_llm_augmentation_cache import image_path, load_source_records, sha256_file
from scripts.build_matched_resampling_control import (
    MAX_REFERENCE_OVERLAP, MAX_STANDARDIZED_DIFFERENCE, image_info,
    original_features, sample_matched,
)


SCHEMA = "internal-dropout-selection-v1"
ARMS = ("llm_selected", "feature_matched")
FIXED_TRAINING = {"dataset": "MI", "seed": 1234, "num_epochs": 15, "batch_size": 4,
                  "num_workers": 4, "lr": 5e-5, "bert_lr": 1e-5, "vit_lr": 5e-6,
                  "warmup_ratio": 0.01, "weight_decay": 0.01, "max_grad_norm": 1.0,
                  "patience": 3, "max_seq_length": 64, "topk_img_patches": 12,
                  "image_dropout_prob": 0.2, "output_dropout": 0.2, "visual_fusion": "legacy",
                  "skip_test": True, "consistency_mode": "dropout", "consistency_weight": 0.5,
                  "consistency_temperature": 2.0, "consistency_warmup_epochs": 2}


def tokenizer_hashes(directory):
    return {p.name: sha256_file(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.suffix in {".json", ".txt"}}


def source_alignment(base, record):
    src, dst, weights = entity_alignment(record["words"], record["labels"], record["words"],
                                          record["labels"], base.tokenizer, base.max_sequence_length)
    active = weights.gt(0)
    return src, dst, weights, {"positions": src[active].tolist(), "weights": weights[active].tolist()}


def prepare_cohorts(base, internal_manifest, cache, dataset, image_root, twitter_root, bert_model, seed=20261102):
    manifest = verify_internal_manifest(internal_manifest, dataset)
    # Revalidate the existing selected cohort; held-out cache bodies are skipped.
    previous = ContextViewDataset(base, internal_manifest, cache, dataset)
    train_ids = manifest["partitions"]["train"]["source_ids"]
    records = {r["sample_id"]: r for r in load_source_records(Path(manifest["source"]))}
    metadata_path = Path(str(cache) + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference = sorted(train_ids[i] for i in previous.views)
    candidate_ids = sorted(r["sample_id"] for r in metadata["selected"] if r["sample_id"] in set(train_ids))
    alignments = {i: source_alignment(base, records[i])[3] for i in candidate_ids}
    candidates = [i for i in candidate_ids if alignments[i]["positions"]]
    if not reference or not set(reference).issubset(candidates):
        raise ValueError("Selected cohort must have usable source-only entity positions")
    groups, strata, features, physical_images = defaultdict(list), {}, {}, {}
    for i in candidates:
        record, valid = records[i], 0
        for name in record["images"][:4]:
            path = str(image_path(name, Path(image_root), Path(twitter_root)))
            if path not in physical_images:
                physical_images[path] = image_info(Path(path))
            valid += int(physical_images[path]["decodable"])
        stratum, feature = original_features(record, base.tokenizer, valid)
        encoded = base._encode_text(record["words"], record["labels"])
        labels = encoded[3]
        counts = Counter()
        for position in alignments[i]["positions"]:
            label = list(base.label_mapping)[int(labels[position])]
            if label.startswith("B-"):
                counts[label[2:]] += 1
        # Retained entity types are an exact stratum; retained counts/word budget
        # are numerical features. No generated text or acceptance score is used.
        stratum = (*stratum, "+".join(sorted(counts)))
        feature.update({"retained_entity_words": float(len(alignments[i]["positions"])),
                        "retained_entities": float(sum(counts.values())),
                        **{f"retained_{label}": float(counts[label]) for label in ("MIS", "ORG", "PER", "LOC")}})
        strata[i], features[i] = stratum, feature
        groups[stratum].append(i)
    if any(strata[i][2] == 0 for i in reference):
        raise ValueError("Selected source no longer has a decodable image; inspect dataset files")
    names = list(features[candidates[0]])
    vectors = {i: np.array([features[i][name] for name in names], dtype=np.float64) for i in candidates}
    std = np.stack([vectors[i] for i in candidates]).std(axis=0)
    scales = np.where(std > 1e-12, std, 1.0)
    target = np.stack([vectors[i] for i in reference]).sum(axis=0)
    quotas = Counter(strata[i] for i in reference)
    sampled, optimisation = sample_matched(groups, quotas, vectors, target, scales, seed=seed)
    matched = sorted(i for ids in sampled.values() for i in ids)
    count = len(reference)
    differences = (np.stack([vectors[i] for i in matched]).sum(axis=0) - target) / (count * scales)
    overlap = len(set(reference) & set(matched)) / count
    balanced = float(np.abs(differences).max()) <= MAX_STANDARDIZED_DIFFERENCE
    informative = overlap <= MAX_REFERENCE_OVERLAP

    def arm(ids):
        return {"source_ids": ids, "source_alignments": {str(i): alignments[i] for i in ids},
                "source_fingerprints": {str(i): records[i]["source_fingerprint"] for i in ids}}

    return {"schema_version": SCHEMA, "dataset": dataset, "sampling_seed": seed,
            "internal_manifest": str(internal_manifest), "internal_manifest_sha256": sha256_file(Path(internal_manifest)),
            "cache": str(cache), "cache_sha256": sha256_file(Path(cache)),
            "metadata": str(metadata_path), "metadata_sha256": sha256_file(metadata_path),
            "tokenizer_files_sha256": tokenizer_hashes(bert_model), "max_seq_length": base.max_sequence_length,
            "arms": {"llm_selected": arm(reference), "feature_matched": arm(matched)},
            "training_examples": len(base), "eligible_per_arm": count, "candidate_count": len(candidates),
            "exact_strata": [{"key": list(key), "pool": len(groups[key]), "per_arm": quota}
                             for key, quota in sorted(quotas.items())],
            "features": {name: {"reference_mean": float(target[j] / count),
                                  "difference_in_pool_sd_units": float(differences[j])} for j, name in enumerate(names)},
            "optimisation": optimisation, "physical_candidate_images": physical_images,
            "balance_check": {"maximum": float(np.abs(differences).max()),
                              "threshold": MAX_STANDARDIZED_DIFFERENCE, "passed": balanced},
            "overlap_check": {"fraction": overlap, "threshold": MAX_REFERENCE_OVERLAP, "passed": informative,
                              "minimum_forced_count": sum(max(0, 2*q-len(groups[k])) for k, q in quotas.items())},
            "ready_for_training": balanced and informative,
            "llm_api_requests": 0, "official_val_or_test_read": False, "internal_dev_responses_used": False,
            "control_scoring_uses_generated_content": False,
            "note": "Post-hoc pilot of joint LLM+validator selection, not semantic-quality proof. "
                    "Both arms use source-only entity masks and original text. Matching allows overlap "
                    "and uses reference aggregates; it is not an unconditional causal LLM comparison."}


class DropoutSelectionDataset(Dataset):
    def __init__(self, base, cohort_path, arm, internal_manifest, cache, dataset, bert_model):
        self.base = base
        manifest = verify_internal_manifest(internal_manifest, dataset)
        cohort = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
        if (cohort.get("schema_version") != SCHEMA or cohort.get("dataset") != dataset
                or cohort.get("ready_for_training") is not True or arm not in ARMS
                or cohort.get("official_val_or_test_read") is not False
                or cohort.get("internal_dev_responses_used") is not False):
            raise ValueError("Expected a ready training-only dropout cohort manifest")
        for path, key in ((internal_manifest, "internal_manifest"), (cache, "cache"),
                          (str(cache) + ".meta.json", "metadata")):
            if sha256_file(Path(path)) != cohort[key + "_sha256"]:
                raise ValueError(f"Cohort binding changed: {key}")
        if (base.max_sequence_length != cohort["max_seq_length"]
                or tokenizer_hashes(bert_model) != cohort["tokenizer_files_sha256"]):
            raise ValueError("Cohort tokenizer or maximum sequence length changed")
        if not cohort["balance_check"]["passed"] or not cohort["overlap_check"]["passed"]:
            raise ValueError("Cohort matching gate failed")
        for path, info in cohort["physical_candidate_images"].items():
            actual_hash = sha256_file(Path(path)) if Path(path).is_file() else None
            if actual_hash != info["sha256"]:
                raise ValueError("Candidate image changed after matching")
        ids = manifest["partitions"]["train"]["source_ids"]
        records = {r["sample_id"]: r for r in load_source_records(Path(manifest["source"]))}
        for value in cohort["arms"].values():
            chosen = value["source_ids"]
            if (any(type(i) is not int for i in chosen) or len(set(chosen)) != len(chosen)
                    or len(chosen) != cohort["eligible_per_arm"] or not set(chosen).issubset(ids)):
                raise ValueError("Cohort must contain unique internal-train source IDs")
        if len(base) != len(ids) or len(base) != cohort["training_examples"]:
            raise ValueError("Cohort must use each original internal-train row once")
        self.alignments = {}
        chosen = cohort["arms"][arm]
        for local, i in enumerate(ids):
            record = records[i]
            if (normalize_words(record["words"]) != base.examples["words"][local]
                    or record["labels"] != base.examples["labels"][local]
                    or record["images"] != base.examples["images"][local]):
                raise ValueError("Cohort source mapping differs from the loaded dataset")
            if i in chosen["source_ids"]:
                src, dst, weights, compact = source_alignment(base, record)
                if (chosen["source_fingerprints"][str(i)] != record["source_fingerprint"]
                        or compact != chosen["source_alignments"][str(i)] or not compact["positions"]):
                    raise ValueError("Cohort source-entity alignment changed")
                self.alignments[local] = (src, dst, weights)
        self.summary = {"arm": arm, "training_examples": len(base), "eligible_per_epoch": len(self.alignments),
                        "original_example_repetitions_per_epoch": 1, "source_only_entity_masks": True,
                        "internal_dev_views_used": 0, "cohort_sha256": sha256_file(Path(cohort_path))}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        main = self.base[index]
        length = len(main[0])
        alignment = self.alignments.get(index, (torch.zeros(length, dtype=torch.long),
                                              torch.zeros(length, dtype=torch.long), torch.zeros(length)))
        return (*main, main[0], main[1], main[2], *alignment)
