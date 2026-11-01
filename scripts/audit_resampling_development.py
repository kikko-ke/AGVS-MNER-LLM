"""Audit raw training cohorts and saved VALIDATION reports using the stdlib only.

This does not load models, read test reports/examples, query an LLM, or train.
Distribution changes are descriptive checks, not causal explanations of errors.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


TYPES = ("LOC", "MIS", "ORG", "PER")
METHODS = ("baseline", "legacy_selected", "legacy_matched", "token_selected")
SEEDS = (42, 1234, 2026)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str):
    if sha(path) != expected:
        raise ValueError(f"Artifact changed: {path}")


def fingerprint(words, labels, images):
    value = {"text": words, "label": labels, "images": images}
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode("ascii")).hexdigest()


def entities(words, labels):
    spans, current, start, repaired = [], None, None, 0
    for i, label in enumerate([*labels, "O"]):
        if label != "O" and (len(label) < 3 or label[:2] not in {"B-", "I-"} or label[2:] not in TYPES):
            raise ValueError(f"Unsupported raw BIO label: {label}")
        kind = None if label == "O" else label[2:]
        if current is not None and (label == "O" or label.startswith("B-") or kind != current):
            spans.append((current, " ".join(words[start:i]).casefold()))
            current, start = None, None
        if kind is not None and current is None:
            repaired += int(label.startswith("I-"))
            current, start = kind, i
    return spans, repaired


def load_training(path):
    records = {}
    with path.open(encoding="utf-8") as source:
        for sample_id, line in enumerate(source):
            if not line.strip():
                continue
            raw = ast.literal_eval(line)
            words = [str(x) for x in raw["text"]]
            labels = [str(x).upper().replace("MISC", "MIS") for x in raw["label"]]
            images = [str(x) for x in raw["images"]]
            if len(words) != len(labels):
                raise ValueError(f"Word/label length mismatch: {path}:{sample_id}")
            chunks, repaired = entities(words, labels)
            twitter = ["twitter2017" in name for name in images[:4]]
            domain = "none" if not twitter else "twitter2017" if all(twitter) else "mixed" if any(twitter) else "mi"
            records[sample_id] = {"words": len(words), "entities": chunks, "repaired_i": repaired,
                                  "listed_images": min(len(images), 4), "domain": domain,
                                  "fingerprint": fingerprint(words, labels, images), "raw_line": line.rstrip("\r\n")}
    return records


def profile(rows):
    counts, surfaces, domains, images = Counter(), Counter(), Counter(), Counter()
    lengths = []
    for row in rows:
        lengths.append(row["words"])
        domains[row["domain"]] += 1
        images[row["listed_images"]] += 1
        for kind, surface in row["entities"]:
            counts[kind] += 1
            surfaces[kind, surface] += 1
    total, n = sum(counts.values()), len(rows)
    return {"examples": n, "mean_raw_words": statistics.mean(lengths) if n else None,
            "median_raw_words": statistics.median(lengths) if n else None,
            "entity_mentions": {kind: counts[kind] for kind in TYPES},
            "entity_share": {kind: counts[kind] / total if total else 0.0 for kind in TYPES},
            "entity_mentions_per_example": total / n if n else 0.0,
            "unique_entity_surfaces": len(surfaces),
            "top10_entity_mention_share": sum(x for _, x in surfaces.most_common(10)) / total if total else 0.0,
            "domain_counts_from_listed_filenames": dict(domains),
            "domain_share_from_listed_filenames": {name: count / n for name, count in domains.items()} if n else {},
            "listed_image_count_first4": dict(sorted(images.items())),
            "distinct_source_fingerprints": len({row["fingerprint"] for row in rows}),
            "orphan_or_type_changed_I_starts": sum(row["repaired_i"] for row in rows)}


def parse_class_report(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in TYPES:
            continue
        if len(fields) != 5 or fields[0] in result:
            raise ValueError("Malformed or repeated class report row")
        result[fields[0]] = {"precision": float(fields[1]), "recall": float(fields[2]),
                             "f1": float(fields[3]), "support": int(fields[4])}
    if set(result) != set(TYPES):
        raise ValueError("Expected all four entity types in the saved validation report")
    return result


def validation_summary(entry):
    rows = entry["per_seed"]
    if len(rows) != 3 or {row["seed"] for row in rows} != set(SEEDS):
        raise ValueError("Validation report must include every fixed seed")
    classes, micro, supports = {}, {}, {}
    for method in METHODS:
        parsed = [parse_class_report(row["metrics"][method]["report"]) for row in rows]
        micro[method] = statistics.mean(row["metrics"][method]["f1"] for row in rows)
        classes[method] = {}
        for kind in TYPES:
            current = {value[kind]["support"] for value in parsed}
            if len(current) != 1:
                raise ValueError("Class support changed between seeds")
            support = next(iter(current))
            if kind in supports and supports[kind] != support:
                raise ValueError("Class support changed between methods")
            supports[kind] = support
            classes[method][kind] = {metric: statistics.mean(value[kind][metric] for value in parsed)
                                     for metric in ("precision", "recall", "f1")}
            classes[method][kind]["sample_std_f1"] = statistics.stdev(value[kind]["f1"] for value in parsed)
            classes[method][kind]["support_per_seed"] = support
    deltas = {kind: {metric: classes["legacy_selected"][kind][metric] - classes["baseline"][kind][metric]
                     for metric in ("precision", "recall", "f1")} for kind in TYPES}
    return {"mean_micro_f1": micro, "per_class": classes, "selection_minus_baseline": deltas}


def audit_dataset(dataset, frozen, validation):
    entry = frozen["datasets"][dataset]
    source = Path(entry["inputs"]["train"]["path"])
    require_hash(source, entry["inputs"]["train"]["sha256"])
    records = load_training(source)
    path = Path(entry["matched_manifest"])
    require_hash(path, entry["matched_manifest_sha256"])
    manifest = read_json(path)
    require_hash(source, manifest["source_train_sha256"])
    metadata_path, reference_path = Path(manifest["candidate_metadata"]), Path(manifest["reference_manifest"])
    require_hash(metadata_path, manifest["candidate_metadata_sha256"])
    require_hash(reference_path, manifest["reference_manifest_sha256"])
    metadata, reference = read_json(metadata_path), read_json(reference_path)
    if metadata["input_sha256"] != sha(source):
        raise ValueError("Candidate source hash mismatch")
    candidates = []
    for row in metadata["selected"]:
        sample_id = row["sample_id"]
        if type(sample_id) is not int or sample_id in candidates or sample_id not in records:
            raise ValueError("Invalid/duplicate candidate ID")
        if records[sample_id]["fingerprint"] != row["source_fingerprint"]:
            raise ValueError("Candidate fingerprint mismatch")
        candidates.append(sample_id)
    accepted, matched = manifest["reference_source_sample_ids"], manifest["source_sample_ids"]
    if (not accepted or len(accepted) != len(matched) or reference["source_sample_ids"] != accepted
            or len(records) != manifest["original_examples"] or len(accepted) != manifest["added_examples"]
            or len(candidates) != manifest["candidate_count"]):
        raise ValueError("Cohort size or reference IDs changed")
    for ids in (accepted, matched):
        if (any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)
                or not set(ids).issubset(candidates)):
            raise ValueError("Resampled IDs must be unique source candidates")
    originals = list(records.values())
    for directory, ids, digest in (
        (reference_path.parent, accepted, manifest["reference_train_sha256"]),
        (path.parent, matched, manifest["output_files_sha256"][source.name]),
    ):
        training = directory / source.name
        require_hash(training, digest)
        actual = training.read_text(encoding="utf-8").splitlines()
        expected = [row["raw_line"] for row in originals] + [records[i]["raw_line"] for i in ids]
        if actual != expected:
            raise ValueError("Training content differs from declared original-only resampling")
    cohorts = {"original": originals, "candidate_pool": [records[i] for i in candidates],
               "accepted_sources": [records[i] for i in accepted], "matched_sources": [records[i] for i in matched],
               "selected_training": originals + [records[i] for i in accepted],
               "matched_training": originals + [records[i] for i in matched]}
    profiles = {name: profile(rows) for name, rows in cohorts.items()}
    return {"raw_training_profiles": profiles,
            "accepted_fraction_of_candidates": len(accepted) / len(candidates),
            "added_fraction_of_training": len(accepted) / (len(originals) + len(accepted)),
            "validation": validation_summary(validation["datasets"][dataset]),
            "selection_entity_share_change_pp": {
                kind: 100 * (profiles["selected_training"]["entity_share"][kind] - profiles["original"]["entity_share"][kind])
                for kind in TYPES},
            "candidate_fingerprints": sorted({records[i]["fingerprint"] for i in candidates}),
            "accepted_fingerprints": sorted({records[i]["fingerprint"] for i in accepted})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    output, suite = Path(cli.output), Path(cli.suite_dir)
    if output.exists():
        raise FileExistsError("Diagnostic report already exists; choose a new output path")
    frozen_path, validation_path = suite / "frozen_manifest.json", suite / "validation_metrics.json"
    frozen, validation = read_json(frozen_path), read_json(validation_path)
    if (frozen.get("schema_version") != "frozen-component-suite-v1"
            or validation.get("schema_version") != "component-suite-metrics-v1" or validation.get("split") != "val"
            or validation.get("frozen_manifest_sha256") != sha(frozen_path)
            or set(frozen["datasets"]) != {"MI", "UNI"} or set(validation["datasets"]) != {"MI", "UNI"}):
        raise ValueError("Expected the bound two-dataset frozen/VALIDATION reports")
    datasets = {name: audit_dataset(name, frozen, validation) for name in ("MI", "UNI")}
    result = {"schema_version": "resampling-development-audit-v1", "test_report_read": False,
              "test_examples_read": False, "model_forward_passes": 0, "llm_api_requests": 0,
              "frozen_manifest_sha256": sha(frozen_path), "validation_report_sha256": sha(validation_path),
              "datasets": datasets,
              "cross_dataset_overlap": {
                  kind: len(set(datasets["MI"][kind]) & set(datasets["UNI"][kind]))
                  for kind in ("candidate_fingerprints", "accepted_fingerprints")},
              "note": "Training counts/surfaces use full raw labels BEFORE WordPiece truncation; validation "
                      "supports are after the evaluator's truncation/filtering and are not directly equal. "
                      "Image counts are listed filenames, not decoded validity. Distribution differences "
                      "and surface concentration are sample-size dependent; compare accepted/matched "
                      "cohorts of equal size before interpreting top10 concentration. These indicators "
                      "are descriptive and cannot establish the cause of any performance change. "
                      "The official test was already inspected in development history; this audit does not "
                      "restore an independent holdout or choose a new architecture/weight automatically."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for dataset, entry in datasets.items():
        print(f"\n=== {dataset}: raw training cohorts (before WordPiece truncation) ===")
        print("cohort                 N    words   LOC%   MIS%   ORG%   PER%   top10-entity-mention%")
        for name, values in entry["raw_training_profiles"].items():
            shares = " ".join(f"{100 * values['entity_share'][kind]:6.2f}" for kind in TYPES)
            print(f"{name:<20} {values['examples']:5d} {values['mean_raw_words']:7.2f} {shares} "
                  f"{100 * values['top10_entity_mention_share']:9.2f}")
        print("Listed filename domains, percentage of examples (original/candidate/accepted):")
        for name in ("original", "candidate_pool", "accepted_sources"):
            print(name, {key: round(100 * value, 2) for key, value in
                         entry["raw_training_profiles"][name]["domain_share_from_listed_filenames"].items()})
        print("\nValidation class F1, mean over 3 seeds; deltas in percentage points:")
        print("type  baseline selected matched token_selected   delta_P   delta_R   delta_F1")
        for kind in TYPES:
            metrics = entry["validation"]["per_class"]
            values = " ".join(f"{metrics[method][kind]['f1']:.4f}" for method in METHODS)
            delta = entry["validation"]["selection_minus_baseline"][kind]
            print(kind, values, " ".join(f"{100 * delta[key]:+.3f}" for key in ("precision", "recall", "f1")))
    print("\nCross-dataset overlap:", result["cross_dataset_overlap"])
    print("Raw training distribution changes are hypotheses, not causal findings.")
    print(f"Diagnostic report: {output}")


if __name__ == "__main__":
    main()
