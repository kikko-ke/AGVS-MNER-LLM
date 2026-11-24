"""A/B/C comparison that reads the original run without modifying its artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from modules.tpm_llm_v1 import pipeline as old
from modules.tpm_llm_v1.common import (
    atomic_json, average, digest, filtered_indices, fingerprint, metrics,
    read_json, read_rows, runtime, save_rows, spans,
)

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "tpm-qwen-image-conditioned-v1"
DEFAULT_ROOT = "outputs_vlm_gold/tpm_qwen_visual_v1"


def code_binding():
    files = [*Path(__file__).parent.glob("*.py"),
             PROJECT / "scripts/run_tpm_qwen_visual_v1.py",
             PROJECT / "scripts/run_tpm_qwen_visual_v1.sh"]
    # Git may check out shell scripts with CRLF; normalize only the shell binding.
    return {str(p.relative_to(PROJECT)): (
        fingerprint(p.read_text(encoding="utf-8")) if p.suffix == ".sh" else digest(p))
        for p in sorted(files)}


def source_run(protocol):
    source = Path(protocol["source_run"])
    original = old.load_protocol(source)
    if original["protocol_id"] != protocol["source_protocol_id"]:
        raise ValueError("The original text-only experiment protocol changed")
    if not old.completed(source / "calibration/completed.json", original):
        raise ValueError("The original development calibration is not complete")
    return source, original


def checked_native(source, original, seed, role):
    path = source / f"emissions/seed{seed}_{role}.pt"
    if not old.completed(path.with_suffix(".json"), original):
        raise ValueError(f"Missing verified TPM-MI emissions: {path}. Finish the original run first.")
    return path, old.load_torch(path)


def prepare(root, args):
    source = Path(args.source_run).resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Use a separate sibling output directory; do not nest or overwrite the original run")
    original = old.load_protocol(source)
    if not old.completed(source / "calibration/completed.json", original):
        raise ValueError("Complete the original text-only development calibration first")
    config = {**original["config"], "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
              "max_input_tokens": args.max_input_tokens, "shard_size": args.shard_size,
              "visual_feature_batch_size": 1, "architecture": "bigru"}
    if (config["min_pixels"] < 28 * 28 or config["min_pixels"] > config["max_pixels"]
            or config["max_input_tokens"] < 512 or config["shard_size"] < 1
            or config["projection_seed"] != 1234 or 1234 not in config["seeds"]):
        raise ValueError("Invalid image/token budget or unmatched source feature recipe")
    checked_native(source, original, 1234, "dev")
    old.checked_head(source, original, 1234, "qwen", "bigru")
    if (root / "protocol.json").is_file():
        saved = load_protocol(root)
        if saved["source_run"] != str(source) or saved["config"] != config:
            raise ValueError("Settings changed; choose a new --run directory")
        print("Existing visual augmentation protocol verified", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": SCHEMA, "source_run": str(source),
             "source_protocol_id": original["protocol_id"], "config": config,
             "code_sha256": code_binding(), "selection_seed": 1234,
             "systems": {"A": "frozen TPM-MI", "B": "TPM-MI + frozen Qwen text features",
                         "C": "TPM-MI + frozen Qwen text-and-image features"},
             "feature_format": "native image prefixes followed by unchanged raw post text; no chat prompt",
             "test_metrics_used_for_training_or_selection": False,
             "previously_used_official_test_in_project": True}
    value["protocol_id"] = fingerprint(value)
    atomic_json(root / "protocol.json", value)
    print({"run": str(root), "source_run": str(source), "protocol_id": value["protocol_id"],
           "seeds": config["seeds"], "max_pixels_per_image": config["max_pixels"]}, flush=True)


def load_protocol(root):
    value = read_json(root / "protocol.json")
    if (value.get("schema_version") != SCHEMA
            or fingerprint({k: v for k, v in value.items() if k != "protocol_id"}) != value["protocol_id"]
            or value["code_sha256"] != code_binding()):
        raise ValueError("Visual experiment code/protocol changed; choose a new --run directory")
    source_run(value)
    return value


def feature_path(root, role):
    return root / f"features/qwen_multimodal_{role}.pt"


def head_path(root, seed):
    return root / f"heads/seed{seed}/qwen_multimodal_bigru.pt"


def verify_ids(bundle, rows):
    if bundle["source_ids"] != [row["source_id"] for row in rows]:
        raise ValueError("Artifact source IDs/order do not match the original experiment")


def smoke(root, protocol):
    import torch
    from .features import FrozenVisualExtractor, usage_summary
    from modules.tpm_llm_v1.features import fit_head, infer_head
    done = root / "smoke/completed.json"
    if old.completed(done, protocol):
        print("Visual feature/head GPU smoke already passed", flush=True)
        return
    source, _ = source_run(protocol)
    audit = read_json(source / "data/image_audit.json")
    rows = old.data_rows(source, "train")
    count = lambda row: sum(audit[name]["valid"] for name in row["images"])
    ordered = sorted(rows, key=lambda row: (-count(row), -sum(map(len, row["tokens"])), row["source_id"]))
    chosen = [next(row for row in ordered if count(row) == number)
              for number in sorted({count(row) for row in ordered}, reverse=True)]
    ids = {row["source_id"] for row in chosen}
    chosen = (chosen + [row for row in ordered if row["source_id"] not in ids])[:8]
    if not chosen or count(chosen[0]) == 0:
        raise ValueError("No valid training images found for the visual smoke test")
    extractor = FrozenVisualExtractor(protocol["config"])
    try:
        features, usage = [], []
        for index, row in enumerate(chosen):
            values, _, record = extractor.row(row, audit, check_visual=index == 0)
            features.append(values)
            usage.append(record)
    finally:
        extractor.close()
    config = {**protocol["config"], "head_epochs": 1}
    head, history = fit_head(features, chosen, config, 1234)
    logits = infer_head(head, features, config["feature_batch_size"])
    if any(value.shape != (len(row["tokens"]), 9) for row, value in zip(chosen, logits)):
        raise ValueError("Visual smoke head lost original-word alignment")
    old.seal(done, protocol, [], extra={"passed": True, "runtime": runtime(), "history": history,
                                      "head_parameters": sum(p.numel() for p in head.parameters()),
                                      "usage": usage, "summary": usage_summary(usage),
                                      "test_scored": False})
    del head, logits
    torch.cuda.empty_cache()
    print({"visual_smoke_passed": True, "summary": usage_summary(usage), "report": str(done)}, flush=True)


def prepare_features(root, protocol, test=False):
    from .features import FrozenVisualExtractor, usage_summary
    if not old.completed(root / "smoke/completed.json", protocol):
        raise ValueError("Run --stage smoke before feature extraction")
    if test and not old.completed(root / "calibration/completed.json", protocol):
        raise ValueError("Freeze development coefficients before reading test inputs")
    source, _ = source_run(protocol)
    audit = read_json(source / "data/image_audit.json")
    extractor = None
    try:
        for role in (("test",) if test else ("train", "dev")):
            file = feature_path(root, role)
            if old.completed(file.with_suffix(".json"), protocol):
                checked_features(root, role, protocol)
                print(f"Reusing verified multimodal {role} features", flush=True)
                continue
            rows = (read_rows(Path(protocol["config"]["data_dir"]) / "MNER-MI_test.txt", "test", labels=False)
                    if test else old.data_rows(source, role))
            if test:
                if len(rows) != 860:
                    raise ValueError("Expected all 860 official test posts")
                audit = old.audit_images(rows, protocol["config"], audit)
            audit_path = root / f"features/{role}_image_audit.json"
            atomic_json(audit_path, audit)
            combined = {key: [] for key in ("features", "aligned", "source_ids", "image_usage")}
            shard_files = []
            size = protocol["config"]["shard_size"]
            for start in range(0, len(rows), size):
                shard = root / f"features/shards/{role}/{start:06d}.pt"
                selected = rows[start:start + size]
                if old.completed(shard.with_suffix(".json"), protocol):
                    bundle = old.load_torch(shard)
                else:
                    if extractor is None:
                        extractor = FrozenVisualExtractor(protocol["config"])
                    bundle = extractor.rows(selected, audit)
                    old.save_torch(shard, bundle)
                    old.seal(shard.with_suffix(".json"), protocol, [shard],
                             inputs=[audit_path], extra={"test_labels_read": False})
                verify_ids(bundle, selected)
                for key in combined:
                    combined[key].extend(bundle[key])
                shard_files.append(shard)
                print(f"{role}: {min(start + size, len(rows))}/{len(rows)} posts cached", flush=True)
            combined["encoder"] = "qwen_multimodal"
            old.save_torch(file, combined)
            summary_path = file.with_suffix(".summary.json")
            atomic_json(summary_path, usage_summary(combined["image_usage"]))
            old.seal(file.with_suffix(".json"), protocol, [file, summary_path, audit_path],
                     inputs=shard_files, extra={"test_labels_read": False, "runtime": runtime()})
    finally:
        if extractor is not None:
            extractor.close()


def checked_features(root, role, protocol):
    path = feature_path(root, role)
    if not old.completed(path.with_suffix(".json"), protocol):
        raise ValueError(f"Missing completed multimodal features: {path}")
    for info in read_json(root / f"features/{role}_image_audit.json").values():
        image = Path(info["path"])
        stat = image.stat() if image.is_file() else None
        if stat is None:
            if info["sha256"] is not None:
                raise ValueError(f"Cached feature image disappeared: {image}")
        elif stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
            if digest(image) != info["sha256"]:
                raise ValueError(f"Cached feature image changed: {image}")
    return path, old.load_torch(path)


def train_head(root, protocol, seed):
    import torch
    from modules.tpm_llm_v1.features import fit_head, infer_head
    file = head_path(root, seed)
    if old.completed(file.with_suffix(".json"), protocol):
        print(f"Reusing multimodal head seed={seed}", flush=True)
        return
    source, original = source_run(protocol)
    old.checked_head(source, original, seed, "qwen", "bigru")
    rows = old.data_rows(source, "train")
    train_path, train = checked_features(root, "train", protocol)
    dev_path, dev = checked_features(root, "dev", protocol)
    verify_ids(train, rows)
    verify_ids(dev, old.data_rows(source, "dev"))
    head, history = fit_head(train["features"], rows, protocol["config"], seed, "bigru")
    logits = infer_head(head, dev["features"], protocol["config"]["feature_batch_size"])
    old.save_torch(file, {"head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                          "architecture": "bigru", "encoder": "qwen_multimodal", "seed": seed,
                          "dev_logits": logits, "dev_aligned": dev["aligned"],
                          "dev_source_ids": dev["source_ids"], "history": history,
                          "runtime": runtime(), "train_samples": len(rows)})
    old.seal(file.with_suffix(".json"), protocol, [file], inputs=[train_path, dev_path])
    del head
    torch.cuda.empty_cache()


def checked_head(root, protocol, seed):
    file = head_path(root, seed)
    if not old.completed(file.with_suffix(".json"), protocol):
        raise ValueError(f"Fit the multimodal head first: {file}")
    return file, old.load_torch(file)


def calibrate(root, protocol):
    from modules.tpm_llm_v1.features import fuse
    done = root / "calibration/completed.json"
    if old.completed(done, protocol):
        print("Visual development coefficient already frozen", flush=True)
        return
    source, original = source_run(protocol)
    rows = old.data_rows(source, "dev")
    native_path, native = checked_native(source, original, 1234, "dev")
    verify_ids(native, rows)
    text_path, text = old.checked_head(source, original, 1234, "qwen", "bigru")
    visual_path, visual = checked_head(root, protocol, 1234)
    if text["dev_source_ids"] != native["source_ids"] or visual["dev_source_ids"] != native["source_ids"]:
        raise ValueError("Development head source IDs differ")
    original_report = source / "calibration/dev_report.json"
    text_weight = read_json(original_report)["coefficients"]["qwen_bigru"]
    def score(head, weight):
        values = fuse(native["emissions"], rows, native["mappings"], head["dev_logits"],
                      head["dev_aligned"], weight)
        return metrics(rows, old.decode(native, rows, values))
    trials = [{"coefficient": weight, "metrics": score(visual, weight)}
              for weight in protocol["config"]["coefficient_grid"]]
    selected = max(trials, key=lambda item: (item["metrics"]["f1"], -item["coefficient"]))
    baseline = metrics(rows, old.decode(native, rows))
    text_scores = score(text, text_weight)
    report = {"protocol_id": protocol["protocol_id"], "selection_seed": 1234,
              "coefficients": {"qwen_text": text_weight, "qwen_multimodal": selected["coefficient"]},
              "baseline": baseline, "qwen_text": text_scores, "qwen_multimodal": selected["metrics"],
              "trials": trials, "development_delta_visual_minus_text": selected["metrics"]["f1"] - text_scores["f1"],
              "test_metrics_used_for_selection": False,
              "selection_scope": "official dev, seed 1234, same coefficient grid; fixed across paired seeds"}
    file = root / "calibration/dev_report.json"
    atomic_json(file, report)
    old.seal(done, protocol, [file], inputs=[native_path, text_path, visual_path, original_report])
    print({"development": report, "report": str(file)}, flush=True)


def test_logits(root, protocol):
    import torch
    from modules.tpm_llm_v1.features import WordHead, infer_head
    if not old.completed(root / "calibration/completed.json", protocol):
        raise ValueError("Freeze development calibration first")
    path, features = checked_features(root, "test", protocol)
    for seed in protocol["config"]["seeds"]:
        file = root / f"test_logits/seed{seed}.pt"
        if old.completed(file.with_suffix(".json"), protocol):
            continue
        fitted, saved = checked_head(root, protocol, seed)
        head = WordHead("bigru").to("cuda:0")
        head.load_state_dict(saved["head"], strict=True)
        head.eval().requires_grad_(False)
        logits = infer_head(head, features["features"], protocol["config"]["feature_batch_size"])
        old.save_torch(file, {"logits": logits, "aligned": features["aligned"],
                              "source_ids": features["source_ids"]})
        old.seal(file.with_suffix(".json"), protocol, [file], inputs=[path, fitted])
        del head
        torch.cuda.empty_cache()


def summarize(scores, seeds):
    names = scores[str(seeds[0])]
    result = {name: average([scores[str(seed)][name]["f1"] for seed in seeds]) for name in names}
    comparisons = {"text_minus_tpm": ("tpm_qwen_text", "tpm"),
                   "multimodal_minus_text": ("tpm_qwen_multimodal", "tpm_qwen_text"),
                   "multimodal_minus_tpm": ("tpm_qwen_multimodal", "tpm"),
                   "multimodal_minus_text_shared1": ("tpm_qwen_multimodal_shared1", "tpm_qwen_text_shared1")}
    result["paired_deltas"] = {}
    for key, (candidate, reference) in comparisons.items():
        values = {str(seed): scores[str(seed)][candidate]["f1"] - scores[str(seed)][reference]["f1"]
                  for seed in seeds}
        result["paired_deltas"][key] = {"by_seed": values, "mean": sum(values.values()) / len(values),
                                        "mean_percentage_points": 100 * sum(values.values()) / len(values),
                                        "positive_seeds": sum(value > 0 for value in values.values())}
    return result


def evaluate(root, protocol):
    from modules.tpm_llm_v1.features import fuse
    done = root / "evaluation/completed.json"
    if old.completed(done, protocol):
        print("A/B/C official test comparison already complete", flush=True)
        return
    if not old.completed(root / "calibration/completed.json", protocol):
        raise ValueError("Freeze development calibration first")
    source, original = source_run(protocol)
    rows = read_rows(Path(protocol["config"]["data_dir"]) / "MNER-MI_test.txt", "test")
    if len(rows) != 860:
        raise ValueError("Expected all 860 official test posts")
    feature_file, features = checked_features(root, "test", protocol)
    verify_ids(features, rows)
    calibration_path = root / "calibration/dev_report.json"
    calibration = read_json(calibration_path)
    audit = read_json(root / "features/test_image_audit.json")
    hashes = {name: info["sha256"] for name, info in audit.items()}
    retained = filtered_indices(old.data_rows(source, "train") + old.data_rows(source, "dev"), rows, hashes)
    scores = {"official_test_full": {}, "official_test_filtered": {}}
    strata = {"0": [], "1": [], "2plus": []}
    for index, usage in enumerate(features["image_usage"]):
        strata[str(usage["valid_images"]) if usage["valid_images"] < 2 else "2plus"].append(index)
    by_count, inputs, outputs = {}, [calibration_path, feature_file], []
    for seed in protocol["config"]["seeds"]:
        native_path, native = checked_native(source, original, seed, "test")
        verify_ids(native, rows)
        text_path = source / f"test_logits/seed{seed}_qwen_bigru.pt"
        visual_path = root / f"test_logits/seed{seed}.pt"
        if not old.completed(text_path.with_suffix(".json"), original):
            raise ValueError("Complete the original run's test logits first")
        if not old.completed(visual_path.with_suffix(".json"), protocol):
            raise ValueError("Run --stage test-logits first")
        text, visual = old.load_torch(text_path), old.load_torch(visual_path)
        verify_ids(text, rows)
        verify_ids(visual, rows)
        inputs.extend([native_path, text_path, visual_path])
        predictions = {"tpm": old.decode(native, rows)}
        for name, head in (("qwen_text", text), ("qwen_multimodal", visual)):
            for suffix, weight in (("", calibration["coefficients"][name]), ("_shared1", 1.0)):
                values = fuse(native["emissions"], rows, native["mappings"], head["logits"], head["aligned"], weight)
                predictions[f"tpm_{name}{suffix}"] = old.decode(native, rows, values)
        def subset(positions):
            return {name: metrics([rows[i] for i in positions], [tags[i] for i in positions])
                    for name, tags in predictions.items()}
        scores["official_test_full"][str(seed)] = subset(range(len(rows)))
        scores["official_test_filtered"][str(seed)] = subset(retained)
        by_count[str(seed)] = {key: subset(indices) for key, indices in strata.items()}
        for name, tags in predictions.items():
            file = root / f"evaluation/seed{seed}_{name}_predictions.jsonl"
            file.parent.mkdir(parents=True, exist_ok=True)
            save_rows(file, [{"source_id": row["source_id"], "labels": value,
                              "predicted": sorted(spans(value))} for row, value in zip(rows, tags)])
            outputs.append(file)
    summary = {scope: summarize(values, protocol["config"]["seeds"]) for scope, values in scores.items()}
    report = {"schema_version": SCHEMA, "protocol_id": protocol["protocol_id"],
              "source_protocol_id": original["protocol_id"], "seeds": protocol["config"]["seeds"],
              "test_samples": len(rows), "filtered_test_samples": len(retained),
              "filtered_test_source_ids": [rows[i]["source_id"] for i in retained],
              "coefficients": calibration["coefficients"], "metrics": scores, "summary": summary,
              "valid_image_count_samples": {key: len(value) for key, value in strata.items()},
              "metrics_by_valid_image_count": by_count,
              "previously_used_official_test_in_project": True,
              "test_metrics_used_for_training_or_selection": False}
    file = root / "evaluation/comparison_report.json"
    atomic_json(file, report)
    old.seal(done, protocol, [file, *outputs], inputs=inputs)
    print({"summary": summary, "report": str(file)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "smoke", "features", "heads", "calibrate",
                                                         "test-features", "test-logits", "evaluate", "status"))
    parser.add_argument("--run", default=DEFAULT_ROOT)
    parser.add_argument("--source-run", default=old.DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.run).resolve()
    if args.stage == "status":
        print(read_json(root / "status.json") if (root / "status.json").exists() else "No visual run started")
        report = root / "calibration/dev_report.json"
        if report.exists():
            print(read_json(report))
        print({"test_complete": (root / "evaluation/completed.json").exists()})
        return
    if args.stage == "prepare":
        source = Path(args.source_run).resolve()
        if root == source or source in root.parents or root in source.parents:
            parser.error("--run must be separate from --source-run, without nesting")
    with old.stage_lock(root):
        status = {"stage": args.stage, "seed": args.seed, "state": "running", "started_unix": time.time()}
        atomic_json(root / "status.json", status)
        try:
            if args.stage == "prepare":
                prepare(root, args)
            else:
                protocol = load_protocol(root)
                if args.seed not in protocol["config"]["seeds"]:
                    raise ValueError("Seed must match the original paired experiment")
                old.require_cuda()
                stages = {"smoke": smoke, "features": prepare_features,
                          "heads": lambda r, p: train_head(r, p, args.seed), "calibrate": calibrate,
                          "test-features": lambda r, p: prepare_features(r, p, test=True),
                          "test-logits": test_logits, "evaluate": evaluate}
                stages[args.stage](root, protocol)
            atomic_json(root / "status.json", {**status, "state": "completed", "finished_unix": time.time()})
        except Exception as error:
            atomic_json(root / "status.json", {**status, "state": "failed", "error": str(error),
                                               "traceback": traceback.format_exc(), "finished_unix": time.time()})
            raise
