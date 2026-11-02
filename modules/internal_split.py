"""Dependency-free verification of the fixed training-internal development split."""
import ast
import hashlib
import json
from pathlib import Path

from modules.text_utils import normalize_words


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_internal_manifest(path, dataset):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "context-internal-split-v1" or manifest["dataset"] != dataset:
        raise ValueError("Expected this dataset's training-internal split manifest")
    if manifest.get("official_val_or_test_read") is not False or manifest.get("augmentation_acceptance_used") is not False:
        raise ValueError("Split must not depend on acceptance or official held-out data")
    if sha256_file(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("Original training source changed")
    if set(manifest["partitions"]) != {"train", "dev"}:
        raise ValueError("Expected exactly internal train and dev partitions")
    for role, value in manifest["partitions"].items():
        split = "train" if role == "train" else "val"
        if Path(value["path"]).resolve() != (path.parent / f"MNER-{dataset}_{split}.txt").resolve():
            raise ValueError("Internal partition must use the declared split directory and filename")
        if sha256_file(value["path"]) != value["sha256"]:
            raise ValueError("Internal split file changed")
        if (any(type(i) is not int for i in value["source_ids"])
                or len(value["source_ids"]) != value["count"]
                or len(set(value["source_ids"])) != value["count"]):
            raise ValueError("Internal source IDs must be unique")
    train = set(manifest["partitions"]["train"]["source_ids"])
    dev = set(manifest["partitions"]["dev"]["source_ids"])
    originals = Path(manifest["source"]).read_text(encoding="utf-8").splitlines()
    if train & dev or train | dev != set(range(len(originals))):
        raise ValueError("Train/dev IDs must form a disjoint partition of the source")
    for partition in manifest["partitions"].values():
        actual = Path(partition["path"]).read_text(encoding="utf-8").splitlines()
        if actual != [originals[i] for i in partition["source_ids"]]:
            raise ValueError("Internal rows differ from declared original rows")
    keys = {}
    for i, line in enumerate(originals):
        raw = ast.literal_eval(line)
        keys[i] = (tuple(w.casefold() for w in normalize_words(raw["text"])), tuple(map(str, raw["images"])))
    if {keys[i] for i in train} & {keys[i] for i in dev}:
        raise ValueError("Normalised text/image groups overlap internal train/dev")
    return manifest
