"""Gold-supervised VLM NER data and strict original-word-span evaluation."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

from llm.augmentation_schema import extract_protected_entities
from modules.internal_split import sha256_file, verify_internal_manifest
from modules.text_utils import normalize_words

SCHEMA = "gold-vlm-ner-v1"
LABELS = ("LOC", "MIS", "ORG", "PER")
SYSTEM = ('Extract named entities from the numbered WORDS and any supplied images. '
          'Use PER (person), ORG (organization), LOC (location), MIS (other named entity). '
          'Return only JSON: {"entities":[{"start":0,"end":1,"label":"PER"}]}. '
          'Indices refer to WORDS, start inclusive and end exclusive. '
          'Entities must not overlap. Return {"entities":[]} if there are none.')


def image_path(name, image_root, twitter_root):
    if "twitter2017" not in name:
        return Path(image_root) / name
    path = Path(twitter_root) / name
    return path if path.exists() else Path(twitter_root) / name.split("-", 1)[-1]


def open_image(raw):
    import math
    from PIL import Image
    for candidate in (raw, raw + b"\xff\xd9"):
        try:
            with Image.open(BytesIO(candidate)) as image:
                result = image.convert("RGBA").convert("RGB")
                result.load()
                width, height = result.size
                if max(width, height) / min(width, height) > 200:
                    raise ValueError("Image aspect ratio exceeds the Qwen processor limit")
                if min(width, height) < 28:
                    scale = 28 / min(width, height)
                    result = result.resize((math.ceil(width*scale), math.ceil(height*scale)), Image.Resampling.BICUBIC)
                return result
        except (OSError, ValueError):
            continue
    raise ValueError("Image cannot be decoded, including the existing JPEG repair rule")


def prepare_data(internal_manifest, output, image_root, twitter_root, panel_size=128):
    if panel_size < 1:
        raise ValueError("Positive panel_size required")
    manifest = verify_internal_manifest(internal_manifest, "MI")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Prepared data exists; preserve and validate it")
    raw_lines = Path(manifest["source"]).read_text(encoding="utf-8").splitlines()
    image_audit, rows, status = {}, {}, Counter()
    for role in ("train", "dev"):
        rows[role] = []
        for source_id in manifest["partitions"][role]["source_ids"]:
            raw = ast.literal_eval(raw_lines[source_id])
            words = normalize_words(raw["text"])
            entities = [{"start":e.start, "end":e.end, "label":e.label}
                        for e in extract_protected_entities(words, raw["label"])]
            images = []
            for name in raw["images"][:4]:
                path = str(image_path(str(name), image_root, twitter_root))
                if path not in image_audit:
                    try:
                        content = Path(path).read_bytes()
                        image = open_image(content)
                        image.close()
                        image_audit[path] = {"valid": True, "sha256": hashlib.sha256(content).hexdigest()}
                    except (OSError, ValueError):
                        image_audit[path] = {"valid": False, "sha256": sha256_file(path) if Path(path).is_file() else None}
                if image_audit[path]["valid"]:
                    images.append({"path": path, "sha256": image_audit[path]["sha256"]})
            status[f"{role}_valid_images_{len(images)}"] += 1
            rows[role].append({"source_id": source_id, "tokens": words, "images": images, "entities": entities})
    panel = sorted(rows["dev"], key=lambda r: hashlib.sha256(f"20261103:{r['source_id']}".encode()).hexdigest())[:panel_size]
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, values in (("train", rows["train"]), ("dev", rows["dev"]), ("dev_panel", panel)):
        path = output / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in values), encoding="utf-8")
        files[name] = {"path": str(path), "sha256": sha256_file(path), "count": len(values)}
    data = {"schema_version": SCHEMA, "dataset": "MI", "internal_manifest": str(internal_manifest),
            "internal_manifest_sha256": sha256_file(internal_manifest), "files": files,
            "panel_source_ids": [r["source_id"] for r in panel], "panel_selection_uses_labels": False,
            "image_audit": image_audit, "image_counts": dict(status),
            "gold_scope": "all original words, without BERT WordPiece truncation",
            "official_val_or_test_read": False, "llm_api_requests": 0,
            "note": "Exploratory internal dev panel; not directly comparable with old truncated scores."}
    path = output / "manifest.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def load_prepared(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("official_val_or_test_read") is not False:
        raise ValueError("Expected gold-supervised internal VLM data")
    if sha256_file(manifest["internal_manifest"]) != manifest["internal_manifest_sha256"]:
        raise ValueError("Internal manifest changed")
    split = verify_internal_manifest(manifest["internal_manifest"], "MI")
    rows = {}
    for role, info in manifest["files"].items():
        if sha256_file(info["path"]) != info["sha256"]:
            raise ValueError("Prepared data changed")
        rows[role] = [json.loads(line) for line in Path(info["path"]).read_text(encoding="utf-8").splitlines()]
        if len(rows[role]) != info["count"]:
            raise ValueError("Prepared count mismatch")
    for role in ("train", "dev"):
        if [r["source_id"] for r in rows[role]] != split["partitions"][role]["source_ids"]:
            raise ValueError("Prepared source IDs differ from the fixed split")
    if ([r["source_id"] for r in rows["dev_panel"]] != manifest["panel_source_ids"]
            or not set(manifest["panel_source_ids"]).issubset(split["partitions"]["dev"]["source_ids"])):
        raise ValueError("Invalid internal dev panel")
    originals = Path(split["source"]).read_text(encoding="utf-8").splitlines()
    for values in rows.values():
        for row in values:
            raw = ast.literal_eval(originals[row["source_id"]])
            expected = [{"start":e.start, "end":e.end, "label":e.label}
                        for e in extract_protected_entities(normalize_words(raw["text"]), raw["label"])]
            if row["tokens"] != normalize_words(raw["text"]) or row["entities"] != expected:
                raise ValueError("Prepared gold/tokens differ from original training source")
            parse_entities(json.dumps({"entities":row["entities"]}), len(row["tokens"]))
    return manifest, rows


def messages(row, mode):
    if mode not in {"text", "multi_image"}:
        raise ValueError("Expected text or multi_image")
    content = []
    if mode == "multi_image":
        content.extend({"type":"image", "image":item["path"]} for item in row["images"])
    numbered = json.dumps(list(enumerate(row["tokens"])), ensure_ascii=False).replace("<", "\\u003c")
    content.append({"type":"text", "text":"WORDS=" + numbered})
    return [{"role":"system", "content":SYSTEM}, {"role":"user", "content":content}]


def target(row):
    return json.dumps({"entities":row["entities"]}, ensure_ascii=True, separators=(",", ":"))


def parse_entities(text, token_count):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"entities"} or not isinstance(data["entities"], list):
        raise ValueError("Expected an entities array")
    entities = []
    for entity in data["entities"]:
        if not isinstance(entity, dict) or set(entity) != {"start", "end", "label"}:
            raise ValueError("Malformed entity object")
        a, b, label = entity["start"], entity["end"], entity["label"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= token_count or label not in LABELS:
            raise ValueError("Invalid span or entity type")
        entities.append((a, b, label))
    entities.sort()
    if any(a[1] > b[0] for a, b in zip(entities, entities[1:])):
        raise ValueError("Duplicate or overlapping entities")
    return entities


def span_metrics(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Every evaluated example must have a prediction")
    by_type = {}
    for label in LABELS:
        tp = fp = fn = 0
        for first, second in zip(gold, predicted):
            left, right = {e for e in first if e[2] == label}, {e for e in second if e[2] == label}
            tp += len(left & right); fp += len(right-left); fn += len(left-right)
        by_type[label] = {"tp":tp, "fp":fp, "fn":fn}
    def score(counts):
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        return {**counts, "precision":tp/(tp+fp) if tp+fp else 0., "recall":tp/(tp+fn) if tp+fn else 0.,
                "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {**score({k:sum(v[k] for v in by_type.values()) for k in ("tp", "fp", "fn")}),
            "by_type":{k:score(v) for k,v in by_type.items()}, "samples":len(gold)}
