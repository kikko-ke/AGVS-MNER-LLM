"""Dev-selected cached fusion and paired 7B text/image studies; reuse TPM-MI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.tpm_llm_v1 import pipeline as old
from modules.tpm_llm_v1.common import atomic_json, average, digest, fingerprint, metrics, read_json, read_rows, runtime, save_rows
from modules.tpm_qwen_visual_v1 import pipeline as visual

SCHEMA = "tpm-qwen-fusion-scale-studies-v1"
DEFAULT_VISUAL = "outputs_vlm_gold/tpm_qwen_visual_min28_v1"
BRANCHES = ("text", "multimodal")


def code_binding():
    files = (Path(__file__), PROJECT / "scripts/run_tpm_qwen_studies_v1.sh",
             PROJECT / "scripts/evaluate_tpm_qwen_diagnostics_v1.py")
    return {p.name: fingerprint(p.read_text(encoding="utf-8")) for p in files}


def parent(root):
    p = visual.load_protocol(root)
    if not old.completed(root / "evaluation/completed.json", p):
        raise ValueError("Complete the existing 3B A/B/C evaluation first")
    source, original = visual.source_run(p)
    return p, source, original


def validate_7b(directory):
    from scripts.download_qwen7b_v1 import validate
    validate(directory)
    config = read_json(Path(directory) / "config.json")
    if config.get("model_type") != "qwen2_5_vl" or config.get("hidden_size") != 3584:
        raise ValueError("Expected Qwen2.5-VL-7B-Instruct (hidden_size=3584), not the 3B weights")


def prepare(root, args):
    source = Path(args.visual_run).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Use a separate sibling study output directory")
    p, base, _ = parent(source)
    if root == base or root in base.parents or base in root.parents:
        raise ValueError("Do not overwrite or nest inside the original TPM-MI run")
    config = dict(p["config"])
    if args.experiment == "7b":
        location = old.model_location(args.qwen7b_model, ("Qwen2.5-VL-7B-Instruct",))
        validate_7b(location)
        config["qwen_model"] = location
    value = {"schema_version": SCHEMA, "experiment": args.experiment,
             "visual_run": str(source), "visual_protocol_id": p["protocol_id"],
             "config": config, "code_sha256": code_binding(),
             "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
             "bootstrap_replicates": 10000, "bootstrap_seed": 1234,
             "selection_seed": 1234, "parameters_updated": "none for fusion; two matched BiGRU heads per seed for 7B",
             "post_hoc_study": True, "test_metrics_used_for_training_or_selection": False}
    if (root / "protocol.json").exists():
        saved = load_protocol(root)
        if any(saved[k] != v for k, v in value.items()):
            raise ValueError("Study settings changed; use a new output directory")
        print("Existing study protocol verified", flush=True)
        return
    value["model_files"] = old.model_binding(config["qwen_model"]) if args.experiment == "7b" else {}
    value["protocol_id"] = fingerprint(value)
    atomic_json(root / "protocol.json", value)
    print({"experiment": args.experiment, "run": str(root), "seeds": config["seeds"],
           "qwen_model": config["qwen_model"], "TPM_checkpoint_retrained": False}, flush=True)


def load_protocol(root):
    p = read_json(root / "protocol.json")
    if (p.get("schema_version") != SCHEMA or p["code_sha256"] != code_binding()
            or fingerprint({k: v for k, v in p.items() if k != "protocol_id"}) != p["protocol_id"]):
        raise ValueError("Study source/protocol changed; use a new output directory")
    original, _, _ = parent(Path(p["visual_run"]))
    if original["protocol_id"] != p["visual_protocol_id"]:
        raise ValueError("The completed 3B protocol changed")
    for name, info in p["model_files"].items():
        stat = Path(name).stat()
        if (stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]) and digest(name) != info["sha256"]:
            raise ValueError(f"7B model file changed: {name}")
    return p


def source_rows(p, role, labels=True):
    source = Path(p["visual_run"])
    _, base, _ = parent(source)
    if role != "test":
        return old.data_rows(base, role)
    rows = read_rows(Path(p["config"]["data_dir"]) / "MNER-MI_test.txt", "test", labels=labels)
    if len(rows) != 860:
        raise ValueError("Expected all 860 official test posts")
    return rows


def native(p, seed, role):
    _, base, original = parent(Path(p["visual_run"]))
    return visual.checked_native(base, original, seed, role)


def mix_emissions(text, multimodal, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Fusion alpha must be in [0,1]")
    if alpha == 0:
        return text
    if alpha == 1:
        return multimodal
    return (1 - alpha) * text + alpha * multimodal


def cached_3b(p, seed, role, rows, emissions):
    from modules.tpm_llm_v1.features import fuse
    source = Path(p["visual_run"])
    vp, base, original = parent(source)
    weights = read_json(source / "calibration/dev_report.json")["coefficients"]
    paths, results = [], {}
    for branch in BRANCHES:
        if role == "dev":
            file, saved = (old.checked_head(base, original, seed, "qwen", "bigru") if branch == "text"
                           else visual.checked_head(source, vp, seed))
            scores, valid, ids = saved["dev_logits"], saved["dev_aligned"], saved["dev_source_ids"]
        else:
            file = (base / f"test_logits/seed{seed}_qwen_bigru.pt" if branch == "text"
                    else source / f"test_logits/seed{seed}.pt")
            proto = original if branch == "text" else vp
            if not old.completed(file.with_suffix(".json"), proto):
                raise ValueError(f"Missing verified 3B logits: {file}")
            saved = old.load_torch(file)
            scores, valid, ids = saved["logits"], saved["aligned"], saved["source_ids"]
        visual.verify_ids({"source_ids": ids}, rows)
        results[branch] = fuse(emissions["emissions"], rows, emissions["mappings"], scores, valid,
                               weights["qwen_text" if branch == "text" else "qwen_multimodal"])
        paths.append(file)
    return results, paths


def calibrate_fusion(root, p):
    done = root / "calibration/completed.json"
    if old.completed(done, p):
        return
    rows = source_rows(p, "dev")
    native_file, bundle = native(p, 1234, "dev")
    visual.verify_ids(bundle, rows)
    branches, files = cached_3b(p, 1234, "dev", rows, bundle)
    trials = [{"alpha": a, "metrics": metrics(rows, old.decode(
        bundle, rows, mix_emissions(branches["text"], branches["multimodal"], a)))} for a in p["alpha_grid"]]
    selected = max(trials, key=lambda v: (v["metrics"]["f1"], -v["alpha"]))
    report = {"protocol_id": p["protocol_id"], "selected_alpha": selected["alpha"],
              "selected": selected, "trials": trials, "selection_seed": 1234,
              "test_metrics_used_for_selection": False, "scope": "official dev only; same alpha across fitted seeds"}
    file = root / "calibration/dev_report.json"
    atomic_json(file, report)
    old.seal(done, p, [file], inputs=[native_file, *files])
    print({"fusion_dev": report}, flush=True)


def feature_file(root, role):
    return root / f"features/paired7b_{role}.pt"


def fitted_file(root, seed, branch):
    return root / f"heads/seed{seed}/{branch}.pt"


def text_row(extractor, row, device="cuda:0"):
    import torch
    from modules.tpm_llm_v1.features import word_offsets, last_overlapping
    from modules.tpm_qwen_visual_v1.features import project_words
    text, bounds = word_offsets(row["tokens"])
    tokens = extractor.processor.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    if not 1 <= tokens["input_ids"].shape[1] <= 512:
        raise ValueError(f"7B text exceeds the original 512-subtoken recipe: {row['source_id']}")
    positions = last_overlapping(bounds, tokens["offset_mapping"][0].tolist())
    inputs = {k: tokens[k].to(device) for k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        hidden = extractor.model.get_decoder()(**inputs, use_cache=False, return_dict=True).last_hidden_state
        values, valid, extractor.projection = project_words(
            hidden, positions, extractor.config["projection_seed"], extractor.projection)
    return values, valid


def paired_rows(extractor, rows, audit):
    from tqdm.auto import tqdm
    result = {b: {"features": [], "aligned": [], "source_ids": []} for b in BRANCHES}
    result["multimodal"]["image_usage"] = []
    for row in tqdm(rows, desc="Frozen 7B text + multimodal", unit="post"):
        text, tv = text_row(extractor, row)
        mm, mv, usage = extractor.row(row, audit)
        if not tv.equal(mv):
            raise ValueError("7B text and multimodal word alignment differs")
        for branch, values, valid in (("text", text, tv), ("multimodal", mm, mv)):
            result[branch]["features"].append(values)
            result[branch]["aligned"].append(valid)
            result[branch]["source_ids"].append(row["source_id"])
        result["multimodal"]["image_usage"].append(usage)
    return result


def smoke_7b(root, p):
    import torch
    from modules.tpm_llm_v1.features import fit_head, infer_head
    from modules.tpm_qwen_visual_v1.features import FrozenVisualExtractor
    done = root / "smoke/completed.json"
    if old.completed(done, p):
        return
    source = Path(p["visual_run"])
    vp, base, _ = parent(source)
    audit = read_json(base / "data/image_audit.json")
    rows = sorted(source_rows(p, "train"), key=lambda r: (
        -sum(audit[n]["valid"] for n in r["images"]), -sum(map(len, r["tokens"])), r["source_id"]))[:4]
    torch.cuda.reset_peak_memory_stats()
    extractor = FrozenVisualExtractor(p["config"])
    try:
        # Explicitly check that image pixels affect aligned text states.
        extractor.row(rows[0], audit, check_visual=True)
        bundle = paired_rows(extractor, rows, audit)
        peak = torch.cuda.max_memory_allocated()
    finally:
        extractor.close()
    histories = {}
    for branch in BRANCHES:
        head, history = fit_head(bundle[branch]["features"], rows, {**p["config"], "head_epochs": 1}, 1234)
        values = infer_head(head, bundle[branch]["features"], p["config"]["feature_batch_size"])
        if any(v.shape != (len(r["tokens"]), 9) for r, v in zip(rows, values)):
            raise ValueError("7B smoke head dimensions differ")
        histories[branch] = history
        del head
        torch.cuda.empty_cache()
    old.seal(done, p, [], extra={"passed": True, "runtime": runtime(), "histories": histories,
                                "peak_allocated_gib": peak / 1024 ** 3, "test_scored": False})
    print({"paired_7b_smoke_passed": True, "peak_allocated_gib": peak / 1024 ** 3,
           "report": str(done)}, flush=True)


def features_7b(root, p, test=False):
    from modules.tpm_qwen_visual_v1.features import FrozenVisualExtractor, usage_summary
    if not old.completed(root / "smoke/completed.json", p):
        raise ValueError("Run the paired 7B GPU smoke first")
    if test and not old.completed(root / "calibration/completed.json", p):
        raise ValueError("Freeze 7B development coefficients before extracting test features")
    _, base, _ = parent(Path(p["visual_run"]))
    audit = read_json(base / "data/image_audit.json")
    extractor = None
    try:
        for role in (("test",) if test else ("train", "dev")):
            file = feature_file(root, role)
            if old.completed(file.with_suffix(".json"), p):
                print(f"Reusing verified paired 7B {role} features", flush=True)
                continue
            rows = source_rows(p, role, labels=False)
            if test:
                audit = old.audit_images(rows, p["config"], audit)
            audit_path = root / f"features/{role}_image_audit.json"
            atomic_json(audit_path, audit)
            combined = {b: {"features": [], "aligned": [], "source_ids": []} for b in BRANCHES}
            combined["multimodal"]["image_usage"] = []
            shards, size = [], p["config"]["shard_size"]
            for start in range(0, len(rows), size):
                shard = root / f"features/shards/{role}/{start:06d}.pt"
                selected = rows[start:start + size]
                if old.completed(shard.with_suffix(".json"), p):
                    bundle = old.load_torch(shard)
                else:
                    if extractor is None:
                        extractor = FrozenVisualExtractor(p["config"])
                    bundle = paired_rows(extractor, selected, audit)
                    old.save_torch(shard, bundle)
                    old.seal(shard.with_suffix(".json"), p, [shard], inputs=[audit_path])
                for branch in BRANCHES:
                    visual.verify_ids(bundle[branch], selected)
                    for key in combined[branch]:
                        combined[branch][key].extend(bundle[branch][key])
                shards.append(shard)
                print(f"paired 7B {role}: {start + len(selected)}/{len(rows)} cached", flush=True)
            old.save_torch(file, combined)
            summary_file = file.with_suffix(".summary.json")
            atomic_json(summary_file, usage_summary(combined["multimodal"]["image_usage"]))
            old.seal(file.with_suffix(".json"), p, [file, summary_file, audit_path], inputs=shards,
                     extra={"runtime": runtime(), "test_labels_read": False})
    finally:
        if extractor is not None:
            extractor.close()


def checked_features(root, role, p):
    file = feature_file(root, role)
    if not old.completed(file.with_suffix(".json"), p):
        raise ValueError(f"Missing completed 7B features: {file}")
    return file, old.load_torch(file)


def heads_7b(root, p, seed):
    import torch
    from modules.tpm_llm_v1.features import fit_head, infer_head
    rows, dev_rows = source_rows(p, "train"), source_rows(p, "dev")
    train_file, train = checked_features(root, "train", p)
    dev_file, dev = checked_features(root, "dev", p)
    for branch in BRANCHES:
        file = fitted_file(root, seed, branch)
        if old.completed(file.with_suffix(".json"), p):
            continue
        visual.verify_ids(train[branch], rows)
        visual.verify_ids(dev[branch], dev_rows)
        head, history = fit_head(train[branch]["features"], rows, p["config"], seed, "bigru")
        logits = infer_head(head, dev[branch]["features"], p["config"]["feature_batch_size"])
        old.save_torch(file, {"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                              "dev_logits": logits, "dev_aligned": dev[branch]["aligned"],
                              "dev_source_ids": dev[branch]["source_ids"], "history": history,
                              "branch": branch, "seed": seed, "runtime": runtime()})
        old.seal(file.with_suffix(".json"), p, [file], inputs=[train_file, dev_file])
        del head
        torch.cuda.empty_cache()


def checked_head(root, p, seed, branch):
    file = fitted_file(root, seed, branch)
    if not old.completed(file.with_suffix(".json"), p):
        raise ValueError(f"Fit the 7B head first: {file}")
    return file, old.load_torch(file)


def calibrate_7b(root, p):
    from modules.tpm_llm_v1.features import fuse
    done = root / "calibration/completed.json"
    if old.completed(done, p):
        return
    rows = source_rows(p, "dev")
    native_file, bundle = native(p, 1234, "dev")
    visual.verify_ids(bundle, rows)
    trials, selected, inputs = {}, {}, [native_file]
    for branch in BRANCHES:
        file, head = checked_head(root, p, 1234, branch)
        inputs.append(file)
        visual.verify_ids({"source_ids": head["dev_source_ids"]}, rows)
        trials[branch] = []
        for weight in p["config"]["coefficient_grid"]:
            values = fuse(bundle["emissions"], rows, bundle["mappings"], head["dev_logits"], head["dev_aligned"], weight)
            trials[branch].append({"coefficient": weight, "metrics": metrics(rows, old.decode(bundle, rows, values))})
        selected[branch] = max(trials[branch], key=lambda v: (v["metrics"]["f1"], -v["coefficient"]))
    report = {"protocol_id": p["protocol_id"], "selection_seed": 1234,
              "coefficients": {b: selected[b]["coefficient"] for b in BRANCHES},
              "selected": selected, "trials": trials, "test_metrics_used_for_selection": False}
    file = root / "calibration/dev_report.json"
    atomic_json(file, report)
    old.seal(done, p, [file], inputs=inputs)
    print({"paired_7b_dev": report}, flush=True)


def test_logits_7b(root, p):
    import torch
    from modules.tpm_llm_v1.features import WordHead, infer_head
    if not old.completed(root / "calibration/completed.json", p):
        raise ValueError("Freeze development coefficients first")
    feature_path, bundle = checked_features(root, "test", p)
    rows = source_rows(p, "test", labels=False)
    for seed in p["config"]["seeds"]:
        for branch in BRANCHES:
            file = root / f"test_logits/seed{seed}_{branch}.pt"
            if old.completed(file.with_suffix(".json"), p):
                continue
            visual.verify_ids(bundle[branch], rows)
            fitted, saved = checked_head(root, p, seed, branch)
            head = WordHead("bigru").to("cuda:0")
            head.load_state_dict(saved["head"], strict=True)
            head.eval().requires_grad_(False)
            values = infer_head(head, bundle[branch]["features"], p["config"]["feature_batch_size"])
            old.save_torch(file, {"source_ids": bundle[branch]["source_ids"], "aligned": bundle[branch]["aligned"],
                                  "logits": values})
            old.seal(file.with_suffix(".json"), p, [file], inputs=[fitted, feature_path])
            del head
            torch.cuda.empty_cache()


def evaluate(root, p):
    import numpy as np
    from modules.tpm_llm_v1.features import fuse
    from scripts.evaluate_tpm_qwen_diagnostics_v1 import component_ids, paired_bootstrap, post_counts, predictions
    done = root / "evaluation/completed.json"
    if old.completed(done, p):
        summary(root)
        return
    if not old.completed(root / "calibration/completed.json", p):
        raise ValueError("Freeze development choices first")
    rows = source_rows(p, "test")
    source = Path(p["visual_run"])
    original = read_json(source / "evaluation/comparison_report.json")
    calibration = read_json(root / "calibration/dev_report.json")
    retained = set(original["filtered_test_source_ids"])
    scopes = {"official_test_full": list(range(len(rows))),
              "official_test_filtered": [i for i, r in enumerate(rows) if r["source_id"] in retained]}
    groups = component_ids(rows, read_json(source / "features/test_image_audit.json"))
    scores = {scope: {} for scope in scopes}
    counts, inputs, outputs = {}, [root / "calibration/dev_report.json"], []
    for seed in p["config"]["seeds"]:
        native_file, bundle = native(p, seed, "test")
        visual.verify_ids(bundle, rows)
        inputs.append(native_file)
        pred = {}
        for name in ("tpm", "tpm_qwen_text", "tpm_qwen_multimodal"):
            file, tags = predictions(source, seed, name, rows)
            pred[name] = tags
            inputs.append(file)
        if p["experiment"] == "fusion":
            branches, files = cached_3b(p, seed, "test", rows, bundle)
            inputs.extend(files)
            for branch, name in (("text", "tpm_qwen_text"), ("multimodal", "tpm_qwen_multimodal")):
                if old.decode(bundle, rows, branches[branch]) != pred[name]:
                    raise ValueError("3B cached score fusion failed exact prediction replay")
            values = mix_emissions(branches["text"], branches["multimodal"], calibration["selected_alpha"])
            pred["tpm_qwen_fusion"] = old.decode(bundle, rows, values)
        else:
            for branch in BRANCHES:
                file = root / f"test_logits/seed{seed}_{branch}.pt"
                if not old.completed(file.with_suffix(".json"), p):
                    raise ValueError("Infer paired 7B test logits first")
                saved = old.load_torch(file)
                visual.verify_ids(saved, rows)
                values = fuse(bundle["emissions"], rows, bundle["mappings"], saved["logits"], saved["aligned"],
                              calibration["coefficients"][branch])
                pred[f"tpm_qwen7b_{branch}"] = old.decode(bundle, rows, values)
                # Same-coefficient control, fixed in advance rather than selected on test.
                values = fuse(bundle["emissions"], rows, bundle["mappings"], saved["logits"], saved["aligned"], 1.0)
                pred[f"tpm_qwen7b_{branch}_shared1"] = old.decode(bundle, rows, values)
                inputs.append(file)
        for scope, indices in scopes.items():
            scores[scope][str(seed)] = {name: metrics([rows[i] for i in indices], [tags[i] for i in indices])
                                       for name, tags in pred.items()}
        for name, tags in pred.items():
            counts.setdefault(name, []).append(post_counts(rows, tags))
            if name not in ("tpm", "tpm_qwen_text", "tpm_qwen_multimodal"):
                file = root / f"evaluation/seed{seed}_{name}_predictions.jsonl"
                file.parent.mkdir(parents=True, exist_ok=True)
                save_rows(file, [{"source_id": r["source_id"], "labels": t} for r, t in zip(rows, tags)])
                outputs.append(file)
    pairs = ({"fusion_minus_text3b": ("tpm_qwen_fusion", "tpm_qwen_text"),
              "fusion_minus_multimodal3b": ("tpm_qwen_fusion", "tpm_qwen_multimodal")}
             if p["experiment"] == "fusion" else {
                 "visual_gain_7b": ("tpm_qwen7b_multimodal", "tpm_qwen7b_text"),
                 "visual_gain_7b_shared1": ("tpm_qwen7b_multimodal_shared1", "tpm_qwen7b_text_shared1"),
                 "scale_text_7b_minus_3b": ("tpm_qwen7b_text", "tpm_qwen_text"),
                 "scale_multimodal_7b_minus_3b": ("tpm_qwen7b_multimodal", "tpm_qwen_multimodal")})
    counts = {name: np.stack(value) for name, value in counts.items()}
    result = {}
    for scope, indices in scopes.items():
        result[scope] = {"samples": len(indices), "systems": {name: average(
            [v[name]["f1"] for v in scores[scope].values()]) for name in counts}, "paired": {}}
        for name, (candidate, reference) in pairs.items():
            result[scope]["paired"][name] = paired_bootstrap(
                counts[candidate][:, indices], counts[reference][:, indices], [groups[i] for i in indices],
                p["bootstrap_replicates"], p["bootstrap_seed"])
    report = {"protocol_id": p["protocol_id"], "experiment": p["experiment"],
              "seeds": p["config"]["seeds"], "selection": calibration, "summary": result, "metrics": scores,
              "post_hoc_study": True, "test_metrics_used_for_training_or_selection": False}
    file = root / "evaluation/study_report.json"
    atomic_json(file, report)
    old.seal(done, p, [file, *outputs], inputs=inputs)
    summary(root)


def summary(root):
    dev = root / "calibration/dev_report.json"
    if dev.exists():
        print({"dev_selection": read_json(dev)}, flush=True)
    file = root / "evaluation/study_report.json"
    if file.exists():
        r = read_json(file)
        print({"experiment": r["experiment"], "summary": r["summary"], "report": str(file)}, flush=True)
    else:
        print({"evaluation_complete": False, "expected_report": str(file)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("fusion", "7b"), required=True)
    parser.add_argument("--stage", required=True, choices=("prepare", "fusion-calibrate", "smoke", "features", "heads",
                                                         "calibrate", "test-features", "test-logits", "evaluate", "summary"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--visual-run", default=DEFAULT_VISUAL)
    parser.add_argument("--qwen7b-model")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    root = Path(args.run).resolve()
    if args.stage == "summary":
        summary(root)
        return
    if (root / "protocol.json").exists() and read_json(root / "protocol.json").get("schema_version") != SCHEMA:
        parser.error("The output directory belongs to another experiment")
    source = Path(args.visual_run).resolve()
    if root == source or root in source.parents or source in root.parents:
        parser.error("Use a separate sibling study directory")
    # Before creating status/lock files, also protect the original TPM-MI output.
    if args.stage == "prepare":
        _, base, _ = parent(source)
        if root == base or root in base.parents or base in root.parents:
            parser.error("Do not write into the original TPM-MI run")
    with old.stage_lock(root):
        status = {"stage": args.stage, "seed": args.seed, "state": "running", "started_unix": time.time()}
        atomic_json(root / "status.json", status)
        try:
            if args.stage == "prepare":
                prepare(root, args)
            else:
                p = load_protocol(root)
                if p["experiment"] != args.experiment or args.seed not in p["config"]["seeds"]:
                    raise ValueError("Experiment/seed differs from the frozen study protocol")
                if args.stage in ("smoke", "features", "heads", "test-features", "test-logits"):
                    if args.experiment != "7b":
                        raise ValueError("Cached fusion requires no feature extraction or head fitting")
                    old.require_cuda()
                if args.stage == "fusion-calibrate" and args.experiment != "fusion":
                    raise ValueError("The alpha grid is only used by the cached fusion study")
                if args.stage == "calibrate" and args.experiment != "7b":
                    raise ValueError("Use fusion-calibrate for cached fusion")
                {"fusion-calibrate": calibrate_fusion, "smoke": smoke_7b, "features": features_7b,
                 "heads": lambda r, v: heads_7b(r, v, args.seed), "calibrate": calibrate_7b,
                 "test-features": lambda r, v: features_7b(r, v, test=True),
                 "test-logits": test_logits_7b, "evaluate": evaluate}[args.stage](root, p)
            atomic_json(root / "status.json", {**status, "state": "completed", "finished_unix": time.time()})
        except Exception as error:
            atomic_json(root / "status.json", {**status, "state": "failed", "error": str(error),
                                               "traceback": traceback.format_exc(), "finished_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
