"""Data, metrics and artifact utilities; importable without GPU packages."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import tempfile

BIO = ("O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
TAGS = (*BIO, "X")
LABELS = ("LOC", "MIS", "ORG", "PER")
SCHEMA = "tpm-mi-frozen-word-features-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".tpm-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def normalize_word(value):
    if not isinstance(value, str):
        raise ValueError("Original tokens must be strings")
    return value.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")


def read_rows(path, role, labels=True):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank source row at {path}:{line_number}")
        try:
            raw = json.loads(line)
        except ValueError:
            raw = ast.literal_eval(line)
        if (not isinstance(raw, dict) or not isinstance(raw.get("text"), list)
                or not raw["text"] or not isinstance(raw.get("images"), list)
                or len(raw["images"]) > 4):
            raise ValueError(f"Invalid MI record at {path}:{line_number}")
        words = [normalize_word(word) for word in raw["text"]]
        row = {"source_id": f"{role}:{line_number - 1}", "original_id": str(raw.get("id", "")),
               "tokens": words, "images": [str(name) for name in raw["images"]]}
        if labels:
            gold = raw.get("label")
            if not isinstance(gold, list) or len(gold) != len(words):
                raise ValueError(f"Word/gold mismatch at {path}:{line_number}")
            gold = [label.replace("MISC", "MIS") for label in gold]
            if any(label not in BIO for label in gold):
                raise ValueError(f"Unsupported BIO label at {path}:{line_number}")
            row["labels"] = gold
        rows.append(row)
    return rows


def save_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
                          encoding="utf-8")


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def spans(tags):
    """Match the existing evaluator: isolated I starts a span; type changes close it."""
    output, start, kind = set(), None, None
    for index, tag in enumerate([*tags, "O"]):
        prefix, current = tag.split("-", 1) if tag != "O" else ("O", None)
        if kind is not None and (current != kind or prefix != "I"):
            output.add((start, index, kind))
            start, kind = None, None
        if current is not None and kind is None:
            start, kind = index, current
    return output


def counts_score(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Prediction count differs from the evaluation rows")
    counts = {label: Counter() for label in LABELS}
    for row, prediction in zip(rows, predictions):
        if len(prediction) != len(row["tokens"]) or any(tag not in BIO for tag in prediction):
            raise ValueError("Prediction must cover every original word with a BIO label")
        gold, predicted = spans(row["labels"]), spans(prediction)
        for key, changes in (("tp", gold & predicted), ("fp", predicted - gold),
                             ("fn", gold - predicted)):
            for _, _, label in changes:
                counts[label][key] += 1
    by_type = {label: counts_score(*(counts[label][key] for key in ("tp", "fp", "fn")))
               for label in LABELS}
    return {**counts_score(*(sum(counts[label][key] for label in LABELS)
                             for key in ("tp", "fp", "fn"))),
            "samples": len(rows), "by_type": by_type,
            "scope": "strict entity spans and types over all original words"}


def project_path(path, positions, indices, word_count):
    if len(positions) != len(indices) or len(set(indices)) != len(indices):
        raise ValueError("Invalid first-WordPiece mapping")
    output = ["O"] * word_count
    for position, original_index in zip(positions, indices):
        if not 0 <= position < len(path) or not 0 <= original_index < word_count:
            raise ValueError("WordPiece mapping is out of range")
        tag = TAGS[path[position]]
        output[original_index] = "O" if tag == "X" else tag
    return output


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime():
    import torch
    import transformers
    return {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def average(values):
    return {"mean_f1": statistics.mean(values),
            "sample_std_f1": statistics.stdev(values) if len(values) > 1 else 0.0}


def filtered_indices(reference, candidates, image_hashes):
    """Exclude connected components linked to fitting OR development data."""
    rows = reference + candidates
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        keys = [("text", tuple(word.casefold() for word in row["tokens"]))]
        keys.extend(("image", image_hashes[name]) for name in row["images"]
                    if image_hashes.get(name) is not None)
        for key in keys:
            previous = owners.setdefault(key, index)
            parent[root(index)] = root(previous)
    excluded = {root(index) for index in range(len(reference))}
    return [index for index in range(len(candidates))
            if root(len(reference) + index) not in excluded]
