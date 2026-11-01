"""Partition ONLY the original training file into a grouped internal train/dev.

Assignment is independent of augmentation acceptance and model scores. Identical
normalised text+image records stay together, even if their annotations differ.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.text_utils import normalize_words


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_split(source, output, dataset, seed=20261101, dev_ratio=0.2):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Internal split exists; preserve it rather than selecting a different split")
    if dataset not in {"MI", "UNI"} or not 0 < dev_ratio < 0.5:
        raise ValueError("Expected MI/UNI and dev_ratio in (0,0.5)")
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 10 or any(not line.strip() for line in lines):
        raise ValueError("Need at least ten non-blank training rows")
    groups, signatures = defaultdict(list), defaultdict(set)
    for i, line in enumerate(lines):
        raw = ast.literal_eval(line)
        words, images = [word.casefold() for word in normalize_words(raw["text"])], list(map(str, raw["images"]))
        if len(words) != len(raw["label"]):
            raise ValueError(f"Word/label length mismatch at {i}")
        key = hashlib.sha256(json.dumps({"text": words, "images": images}, ensure_ascii=True,
                                       sort_keys=True).encode("ascii")).hexdigest()
        groups[key].append(i)
        signatures[key].update(str(label).upper().replace("MISC", "MIS")[2:]
                               for label in raw["label"] if str(label).upper().startswith(("B-", "I-")))
    strata = defaultdict(list)
    for key, ids in groups.items():
        images = ast.literal_eval(lines[ids[0]])["images"][:4]
        twitter = ["twitter2017" in str(name) for name in images]
        domain = "none" if not twitter else "twitter2017" if all(twitter) else "mixed" if any(twitter) else "mi"
        strata[domain, tuple(sorted(signatures[key]))].append(key)
    generator, dev_groups = random.Random(seed), set()
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        generator.shuffle(keys)
        # Singleton strata remain in training; no record can cross the boundary.
        count = min(len(keys) - 1, max(1, round(len(keys) * dev_ratio))) if len(keys) > 1 else 0
        dev_groups.update(keys[:count])
    dev = sorted(i for key in dev_groups for i in groups[key])
    train = sorted(set(range(len(lines))) - set(dev))
    if not dev or not train:
        raise ValueError("Grouped split produced an empty partition")
    output.mkdir(parents=True, exist_ok=False)
    partitions = {}
    for role, ids, split in (("train", train, "train"), ("dev", dev, "val")):
        path = output / f"MNER-{dataset}_{split}.txt"
        path.write_text("\n".join(lines[i] for i in ids) + "\n", encoding="utf-8")
        partitions[role] = {"path": str(path), "sha256": sha(path), "count": len(ids), "source_ids": ids}
    manifest = {"schema_version": "context-internal-split-v1", "dataset": dataset,
                "source": str(source), "source_sha256": sha(source), "seed": seed,
                "target_dev_ratio": dev_ratio, "partitions": partitions,
                "grouping": "normalised casefolded text+ordered image names", "group_overlap": 0,
                "stratification": "training entity-type presence and listed image domain",
                "augmentation_acceptance_used": False, "official_val_or_test_read": False,
                "note": "A new training-internal development split; earlier public-test inspection is not undone."}
    path = output / "context_internal_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Internal {dataset}: train={len(train)}, dev={len(dev)}, grouped overlap=0")
    print(f"Manifest: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("MI", "UNI"), required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    create_split(args.input, args.output_dir, args.dataset, args.seed)


if __name__ == "__main__":
    main()
