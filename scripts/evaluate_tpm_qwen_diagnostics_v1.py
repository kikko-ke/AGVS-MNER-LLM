"""Post-hoc paired analysis and fixed-geometry Qwen image mismatch; no training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.tpm_llm_v1 import pipeline as old
from modules.tpm_llm_v1.common import (
    atomic_json, average, digest, fingerprint, load_rows, metrics,
    read_json, read_rows, runtime, save_rows, spans,
)
from modules.tpm_qwen_visual_v1 import pipeline as visual

SCHEMA = "tpm-qwen-posthoc-diagnostics-v1"
DEFAULT_VISUAL = "outputs_vlm_gold/tpm_qwen_visual_min28_v1"
DEFAULT_RUN = "outputs_vlm_gold/tpm_qwen_diagnostics_v1"
SYSTEMS = ("tpm", "tpm_qwen_text", "tpm_qwen_multimodal")


def code_binding():
    return {p.name: fingerprint(p.read_text(encoding="utf-8")) for p in
            (Path(__file__), PROJECT / "scripts/run_tpm_qwen_diagnostics_v1.sh")}


def component_ids(rows, audit):
    """Cluster exact text/shared image hashes, including transitive links."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        keys = [("text", tuple(w.casefold() for w in row["tokens"]))]
        keys += [("image", audit[n]["sha256"]) for n in row["images"] if audit[n]["sha256"]]
        for key in keys:
            previous = owners.setdefault(key, i)
            parent[find(i)] = find(previous)
    lookup = {}
    return [lookup.setdefault(find(i), len(lookup)) for i in range(len(rows))]


def mismatch_plan(rows, audit, seed):
    """Select donors using inputs only, within the same valid-image-count bucket."""
    groups = component_ids(rows, audit)
    slots = [[(slot, name) for slot, name in enumerate(r["images"]) if audit[name]["valid"]] for r in rows]
    buckets = defaultdict(list)
    for i, value in enumerate(slots):
        buckets[len(value)].append(i)
    rng, plan = random.Random(seed), []
    for i, row in enumerate(rows):
        candidates = [j for j in buckets[len(slots[i])] if groups[j] != groups[i]] if slots[i] else []
        record = {"source_id": row["source_id"], "valid_images": len(slots[i]),
                  "eligible": bool(candidates), "component": groups[i]}
        if candidates:
            j = rng.choice(candidates)
            record.update({"donor_source_id": rows[j]["source_id"],
                           "pairs": [{"slot": a, "original_name": n, "donor_slot": b, "donor_name": m}
                                     for (a, n), (b, m) in zip(slots[i], slots[j])]})
        else:
            record["reason"] = "no_valid_images" if not slots[i] else "no_disjoint_donor_in_count_bucket"
        plan.append(record)
    return plan


def source_context(protocol, labels=False):
    root = Path(protocol["visual_run"])
    p = visual.load_protocol(root)
    if p["protocol_id"] != protocol["visual_protocol_id"]:
        raise ValueError("The completed visual experiment protocol changed")
    if not old.completed(root / "evaluation/completed.json", p):
        raise ValueError("Complete the original A/B/C evaluation first")
    for name, expected in protocol["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Frozen diagnostic input changed: {name}")
    rows = read_rows(Path(p["config"]["data_dir"]) / "MNER-MI_test.txt", "test", labels=labels)
    if len(rows) != 860:
        raise ValueError("Expected all 860 official test posts")
    return root, p, rows, read_json(root / "features/test_image_audit.json")


def prepare(root, args):
    source = Path(args.visual_run).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Use a separate diagnostic output directory without nesting")
    p = visual.load_protocol(source)
    if not old.completed(source / "evaluation/completed.json", p):
        raise ValueError("The original A/B/C evaluation is not complete")
    if args.bootstrap_replicates < 100 or args.shard_size < 1:
        raise ValueError("Use at least 100 bootstrap draws and a positive shard size")
    rows = read_rows(Path(p["config"]["data_dir"]) / "MNER-MI_test.txt", "test", labels=False)
    if len(rows) != 860:
        raise ValueError("Expected 860 official test posts")
    _, features = visual.checked_features(source, "test", p)
    visual.verify_ids(features, rows)
    audit = read_json(source / "features/test_image_audit.json")
    plan = mismatch_plan(rows, audit, args.mismatch_seed)
    bindings = [source / "protocol.json", source / "evaluation/comparison_report.json",
                source / "features/test_image_audit.json", source / "calibration/dev_report.json"]
    bindings += [visual.head_path(source, seed) for seed in p["config"]["seeds"]]
    value = {"schema_version": SCHEMA, "visual_run": str(source), "visual_protocol_id": p["protocol_id"],
             "source_sha256": {str(f): digest(f) for f in bindings}, "code_sha256": code_binding(),
             "config": {"mismatch_seed": args.mismatch_seed, "bootstrap_seed": args.bootstrap_seed,
                        "bootstrap_replicates": args.bootstrap_replicates, "shard_size": args.shard_size,
                        "seeds": p["config"]["seeds"]}, "plan_fingerprint": fingerprint(plan),
             "intervention": "Qwen pixels only; same image count, token IDs, grids and text positions",
             "donor_sampling": "with replacement from a different exact-text/shared-image component in the same count bucket",
             "post_hoc_diagnostic": True, "parameters_updated": 0,
             "test_metrics_used_for_training_or_selection": False}
    value["protocol_id"] = fingerprint(value)
    if (root / "protocol.json").exists():
        previous = load_protocol(root)
        if previous != value:
            raise ValueError("Diagnostic recipe changed; choose a new --run directory")
        print("Existing diagnostic protocol verified", flush=True)
        return
    atomic_json(root / "mismatch_plan.json", plan)
    atomic_json(root / "protocol.json", value)
    print({"run": str(root), "source_visual_run": str(source),
           "eligible_posts": sum(x["eligible"] for x in plan),
           "unchanged_reasons": dict(Counter(x.get("reason") for x in plan if not x["eligible"])),
           "parameters_updated": 0}, flush=True)


def load_protocol(root):
    p = read_json(root / "protocol.json")
    if (p.get("schema_version") != SCHEMA or p["code_sha256"] != code_binding()
            or fingerprint({k: v for k, v in p.items() if k != "protocol_id"}) != p["protocol_id"]
            or fingerprint(read_json(root / "mismatch_plan.json")) != p["plan_fingerprint"]):
        raise ValueError("Diagnostic protocol, code or assignments changed; use a new output directory")
    source_context(p)
    return p


def post_counts(rows, predictions):
    import numpy as np
    if len(rows) != len(predictions):
        raise ValueError("Prediction row count differs")
    metrics(rows, predictions)  # Validate original-word coverage and BIO labels.
    values = []
    for row, tags in zip(rows, predictions):
        gold, predicted = spans(row["labels"]), spans(tags)
        values.append((len(gold & predicted), len(predicted - gold), len(gold - predicted)))
    return np.asarray(values, dtype=np.int64).reshape(len(rows), 3)


def f1(counts):
    import numpy as np
    denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(2 * counts[..., 0], denominator, out=np.zeros(denominator.shape), where=denominator != 0)


def paired_bootstrap(candidate, reference, groups, replicates=10000, seed=1234):
    """Resample the same text/image clusters for both systems and all fitted seeds."""
    import numpy as np
    if candidate.shape != reference.shape or candidate.shape[1:] != (len(groups), 3):
        raise ValueError("Paired count arrays must have shape (model_seeds, posts, 3)")
    lookup = {value: i for i, value in enumerate(sorted(set(groups)))}
    units = len(lookup)
    point = float((f1(candidate.sum(1)) - f1(reference.sum(1))).mean())
    result = {"samples": len(groups), "clusters": units, "delta_pp": point * 100,
              "replicates": replicates, "bootstrap_seed": seed, "ci95_pp": None,
              "scope": "sample uncertainty conditional on the existing fitted models; model seeds are not resampled"}
    if units < 2:
        return result
    left = np.zeros((candidate.shape[0], units, 3), dtype=np.int64)
    right = np.zeros_like(left)
    for i, group in enumerate(groups):
        left[:, lookup[group]] += candidate[:, i]
        right[:, lookup[group]] += reference[:, i]
    rng, differences = np.random.default_rng(seed), []
    for start in range(0, replicates, 256):
        weights = rng.multinomial(units, np.full(units, 1 / units), size=min(256, replicates - start))
        a = np.einsum("bc,scm->bsm", weights, left)
        b = np.einsum("bc,scm->bsm", weights, right)
        differences.extend((f1(a) - f1(b)).mean(1).tolist())
    result["ci95_pp"] = (np.quantile(differences, [0.025, 0.975]) * 100).tolist()
    return result


def predictions(root, seed, system, rows):
    file = root / f"evaluation/seed{seed}_{system}_predictions.jsonl"
    records = load_rows(file)
    visual.verify_ids({"source_ids": [r["source_id"] for r in records]}, rows)
    return file, [r["labels"] for r in records]


def analyze(root, protocol):
    import numpy as np
    done = root / "analysis/completed.json"
    if old.completed(done, protocol):
        print_summary(root)
        return
    source, _, rows, audit = source_context(protocol, labels=True)
    original = read_json(source / "evaluation/comparison_report.json")
    groups = component_ids(rows, audit)
    counts, inputs = {name: [] for name in SYSTEMS}, []
    for seed in protocol["config"]["seeds"]:
        for name in SYSTEMS:
            file, tags = predictions(source, seed, name, rows)
            scores = metrics(rows, tags)
            if abs(scores["f1"] - original["metrics"]["official_test_full"][str(seed)][name]["f1"]) > 1e-12:
                raise ValueError("Original prediction files do not reproduce the reported F1")
            counts[name].append(post_counts(rows, tags))
            inputs.append(file)
    counts = {name: np.stack(values) for name, values in counts.items()}
    filtered = set(original["filtered_test_source_ids"])
    scopes = {"official_test_full": list(range(len(rows))),
              "official_test_filtered": [i for i, row in enumerate(rows) if row["source_id"] in filtered]}
    analysis = {}
    for scope, indices in scopes.items():
        selected_groups = [groups[i] for i in indices]
        analysis[scope] = {"samples": len(indices), "systems": {
            name: average(f1(values[:, indices].sum(1)).tolist()) for name, values in counts.items()},
            "C_minus_B": paired_bootstrap(counts[SYSTEMS[2]][:, indices], counts[SYSTEMS[1]][:, indices],
                                          selected_groups, protocol["config"]["bootstrap_replicates"],
                                          protocol["config"]["bootstrap_seed"])}
    image_counts = {}
    for group, n in original["valid_image_count_samples"].items():
        image_counts[group] = {"samples": n, "mean_f1": {name: sum(
            original["metrics_by_valid_image_count"][str(seed)][group][name]["f1"]
            for seed in protocol["config"]["seeds"]) / len(protocol["config"]["seeds"]) for name in SYSTEMS}}
        image_counts[group]["C_minus_B_pp"] = 100 * (
            image_counts[group]["mean_f1"][SYSTEMS[2]] - image_counts[group]["mean_f1"][SYSTEMS[1]])
    report = {"protocol_id": protocol["protocol_id"], "seeds": protocol["config"]["seeds"],
              "analysis": analysis, "by_valid_image_count": image_counts,
              "bootstrap_method": "paired percentile cluster bootstrap of corpus entity micro-F1; identical resamples for every fitted seed",
              "clusters": "exact casefolded text and shared image SHA256 components in the original test set",
              "post_hoc_diagnostic": True, "test_metrics_used_for_training_or_selection": False}
    file = root / "analysis/analysis_report.json"
    atomic_json(file, report)
    old.seal(done, protocol, [file], inputs=inputs)
    print_summary(root)


def mismatch_inputs(row, assignment, processor, audit, maximum):
    """Replace pixel content at EXACT original grids; never change text/prefix IDs."""
    import torch
    from modules.tpm_qwen_visual_v1.features import open_qwen_image, prepare_input
    correct, positions, usage = prepare_input(row, processor, audit, maximum)
    if not assignment["eligible"] or assignment["source_id"] != row["source_id"]:
        raise ValueError("Mismatch input preparation requires this row's eligible assignment")
    if [p["slot"] for p in assignment["pairs"]] != usage["used_slots"]:
        raise ValueError("Assigned image slots differ from the original valid slots")
    patches = []
    for pair, grid in zip(assignment["pairs"], correct["image_grid_thw"]):
        if grid[0].item() != 1:
            raise ValueError("This diagnostic handles static images only")
        name = pair["donor_name"]
        if not audit[name]["valid"]:
            raise ValueError("Assigned donor image is invalid")
        size = (int(grid[2]) * processor.image_processor.patch_size,
                int(grid[1]) * processor.image_processor.patch_size)
        factor = processor.image_processor.patch_size * processor.image_processor.merge_size
        image, _ = open_qwen_image(Path(audit[name]["path"]), factor)
        try:
            resized = image.resize(size, processor.image_processor.resample)
            try:
                encoded = processor.image_processor(images=[resized], do_resize=False, return_tensors="pt")
            finally:
                resized.close()
        finally:
            image.close()
        if not torch.equal(encoded["image_grid_thw"][0], grid):
            raise ValueError("Donor preprocessing changed the original image grid")
        patches.append(encoded["pixel_values"])
    pixels = torch.cat(patches).to(correct["pixel_values"])
    if pixels.shape != correct["pixel_values"].shape:
        raise ValueError("Donor patches changed the original visual token budget")
    changed = {**correct, "pixel_values": pixels}
    usage.update({"eligible": True, "donor_source_id": assignment["donor_source_id"],
                  "donor_pairs": assignment["pairs"], "geometry_preserved": True,
                  "pixel_max_delta": (pixels - correct["pixel_values"]).abs().max().item()})
    return correct, changed, positions, usage


def extract_row(extractor, row, assignment, audit, matched, aligned, verify=False):
    import torch
    from modules.tpm_qwen_visual_v1.features import feature_forward, project_words
    if not assignment["eligible"]:
        return matched.clone(), aligned.clone(), {**assignment, "geometry_preserved": True}
    correct, changed, positions, usage = mismatch_inputs(row, assignment, extractor.processor, audit,
                                                       extractor.config["max_input_tokens"])
    def forward(inputs):
        hidden = feature_forward(extractor.model, {k: v.to("cuda:0") for k, v in inputs.items()})
        values, valid, extractor.projection = project_words(
            hidden, positions, extractor.config["projection_seed"], extractor.projection)
        return values, valid
    with torch.inference_mode():
        if verify:
            replay, valid = forward(correct)
            if not torch.equal(valid, aligned):
                raise ValueError("Original word alignment failed replay")
            torch.testing.assert_close(replay.float(), matched.float(), atol=0.01, rtol=0.01)
            usage["matched_replay_max_delta"] = (replay.float() - matched.float()).abs().max().item()
        values, valid = forward(changed)
    if not torch.equal(valid, aligned):
        raise ValueError("Mismatch changed original-word alignment")
    usage["word_feature_max_delta"] = (values.float() - matched.float()).abs().max().item()
    return values, valid, usage


def smoke(root, protocol):
    from modules.tpm_qwen_visual_v1.features import FrozenVisualExtractor
    done = root / "smoke/completed.json"
    if old.completed(done, protocol):
        print("Frozen mismatch GPU smoke already passed", flush=True)
        return
    source, p, rows, audit = source_context(protocol)
    _, matched = visual.checked_features(source, "test", p)
    plan = read_json(root / "mismatch_plan.json")
    eligible = sorted((i for i, a in enumerate(plan) if a["eligible"]),
                      key=lambda i: (-plan[i]["valid_images"], i))
    if not eligible:
        raise ValueError("No eligible disjoint donors; CPU analysis remains available")
    extractor, records = FrozenVisualExtractor(p["config"]), []
    try:
        for i in eligible[:3]:
            _, _, usage = extract_row(extractor, rows[i], plan[i], audit, matched["features"][i],
                                      matched["aligned"][i], verify=True)
            records.append(usage)
    finally:
        extractor.close()
    if not any(r["pixel_max_delta"] > 0 and r["word_feature_max_delta"] > 0 for r in records):
        raise ValueError("Donor pixels did not affect the extracted text features")
    old.seal(done, protocol, [], extra={"passed": True, "runtime": runtime(), "records": records,
                                      "parameters_updated": 0})
    print({"mismatch_gpu_smoke_passed": True, "posts": len(records), "report": str(done)}, flush=True)


def feature_path(root):
    return root / "features/mismatched_test.pt"


def extract_features(root, protocol):
    from tqdm.auto import tqdm
    from modules.tpm_qwen_visual_v1.features import FrozenVisualExtractor
    if not old.completed(root / "smoke/completed.json", protocol):
        raise ValueError("Run the fixed-geometry GPU smoke first")
    file = feature_path(root)
    if old.completed(file.with_suffix(".json"), protocol):
        return
    source, p, rows, audit = source_context(protocol)
    original_file, matched = visual.checked_features(source, "test", p)
    plan_path = root / "mismatch_plan.json"
    plan = read_json(plan_path)
    visual.verify_ids(matched, rows)
    combined = {k: [] for k in ("features", "aligned", "source_ids", "image_usage")}
    shards, extractor = [], None
    try:
        size = protocol["config"]["shard_size"]
        for start in range(0, len(rows), size):
            shard = root / f"features/shards/{start:06d}.pt"
            selected = rows[start:start + size]
            if old.completed(shard.with_suffix(".json"), protocol):
                bundle = old.load_torch(shard)
            else:
                bundle = {k: [] for k in combined}
                for i in tqdm(range(start, start + len(selected)), desc="Frozen Qwen mismatched test images", unit="post"):
                    if plan[i]["eligible"] and extractor is None:
                        extractor = FrozenVisualExtractor(p["config"])
                    values, valid, usage = extract_row(extractor, rows[i], plan[i], audit,
                                                       matched["features"][i], matched["aligned"][i])
                    bundle["features"].append(values)
                    bundle["aligned"].append(valid)
                    bundle["source_ids"].append(rows[i]["source_id"])
                    bundle["image_usage"].append(usage)
                old.save_torch(shard, bundle)
                old.seal(shard.with_suffix(".json"), protocol, [shard], inputs=[plan_path, original_file])
            visual.verify_ids(bundle, selected)
            for key in combined:
                combined[key].extend(bundle[key])
            shards.append(shard)
            print(f"mismatched test: {start + len(selected)}/{len(rows)} cached", flush=True)
        old.save_torch(file, combined)
        old.seal(file.with_suffix(".json"), protocol, [file], inputs=shards)
    finally:
        if extractor is not None:
            extractor.close()


def infer_logits(root, protocol):
    import torch
    from modules.tpm_llm_v1.features import WordHead, infer_head
    file = feature_path(root)
    if not old.completed(file.with_suffix(".json"), protocol):
        raise ValueError("Extract mismatched test features first")
    features = old.load_torch(file)
    source, p, rows, _ = source_context(protocol)
    visual.verify_ids(features, rows)
    for seed in protocol["config"]["seeds"]:
        output = root / f"logits/seed{seed}.pt"
        if old.completed(output.with_suffix(".json"), protocol):
            continue
        fitted, saved = visual.checked_head(source, p, seed)
        head = WordHead("bigru").to("cuda:0")
        head.load_state_dict(saved["head"], strict=True)
        head.eval().requires_grad_(False)
        values = infer_head(head, features["features"], p["config"]["feature_batch_size"])
        old.save_torch(output, {"source_ids": features["source_ids"], "logits": values,
                                "aligned": features["aligned"]})
        old.seal(output.with_suffix(".json"), protocol, [output], inputs=[file, fitted])
        del head
        torch.cuda.empty_cache()


def evaluate(root, protocol):
    from modules.tpm_llm_v1.features import fuse
    done = root / "evaluation/completed.json"
    if old.completed(done, protocol):
        print_summary(root)
        return
    source, p, rows, _ = source_context(protocol, labels=True)
    original_root, original_protocol = visual.source_run(p)
    original_report = read_json(source / "evaluation/comparison_report.json")
    plan = read_json(root / "mismatch_plan.json")
    eligible = [i for i, a in enumerate(plan) if a["eligible"]]
    retained = set(original_report["filtered_test_source_ids"])
    scopes = {"official_test_full": list(range(len(rows))),
              "eligible_mismatch": eligible,
              "official_test_filtered": [i for i, row in enumerate(rows) if row["source_id"] in retained],
              "eligible_filtered": [i for i in eligible if rows[i]["source_id"] in retained],
              "eligible_multiimage": [i for i in eligible if plan[i]["valid_images"] >= 2]}
    scores = {scope: {} for scope in scopes}
    inputs, outputs = [root / "mismatch_plan.json"], []
    coefficient = original_report["coefficients"]["qwen_multimodal"]
    for seed in protocol["config"]["seeds"]:
        native_file, native = visual.checked_native(original_root, original_protocol, seed, "test")
        visual.verify_ids(native, rows)
        original_logits = source / f"test_logits/seed{seed}.pt"
        if not old.completed(original_logits.with_suffix(".json"), p):
            raise ValueError("Original visual test logits are not verified")
        logits_file = root / f"logits/seed{seed}.pt"
        if not old.completed(logits_file.with_suffix(".json"), protocol):
            raise ValueError("Infer frozen-head mismatched logits first")
        heads = [old.load_torch(f) for f in (original_logits, logits_file)]
        predictions_by_system = {}
        for name in SYSTEMS:
            pred_file, values = predictions(source, seed, name, rows)
            predictions_by_system[name] = values
            inputs.append(pred_file)
        for head, name in zip(heads, ("replayed_C", "tpm_qwen_mismatched")):
            visual.verify_ids(head, rows)
            values = fuse(native["emissions"], rows, native["mappings"], head["logits"], head["aligned"], coefficient)
            predictions_by_system[name] = old.decode(native, rows, values)
        if predictions_by_system.pop("replayed_C") != predictions_by_system["tpm_qwen_multimodal"]:
            raise ValueError("Original matched-image C predictions failed exact replay")
        # Ineligible posts have no intervention; reuse their original C predictions exactly.
        for i, assignment in enumerate(plan):
            if not assignment["eligible"]:
                predictions_by_system["tpm_qwen_mismatched"][i] = predictions_by_system["tpm_qwen_multimodal"][i]
        for scope, indices in scopes.items():
            scores[scope][str(seed)] = {name: metrics([rows[i] for i in indices], [tags[i] for i in indices])
                                       for name, tags in predictions_by_system.items()}
        output = root / f"evaluation/seed{seed}_mismatched_predictions.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        save_rows(output, [{"source_id": row["source_id"], "labels": value}
                          for row, value in zip(rows, predictions_by_system["tpm_qwen_mismatched"])])
        outputs.append(output)
        inputs.extend([native_file, original_logits, logits_file])
    summary = {}
    for scope, values in scores.items():
        differences = {seed: item["tpm_qwen_multimodal"]["f1"] - item["tpm_qwen_mismatched"]["f1"]
                       for seed, item in values.items()}
        summary[scope] = {"samples": len(scopes[scope]), "systems": {
            name: average([v[name]["f1"] for v in values.values()])
            for name in (*SYSTEMS, "tpm_qwen_mismatched")},
            "C_minus_D_pp": {"by_seed": {seed: 100 * d for seed, d in differences.items()},
                              "mean": 100 * sum(differences.values()) / len(differences),
                              "positive_seeds": sum(d > 0 for d in differences.values())}}
    report = {"protocol_id": protocol["protocol_id"], "seeds": protocol["config"]["seeds"],
              "summary": summary, "metrics": scores, "frozen_C_coefficient": coefficient,
              "unchanged_posts": len(rows) - len(eligible), "geometry_preserved": True,
              "TPM_images": "original correct images, reused native emissions",
              "parameters_updated": 0, "post_hoc_diagnostic": True,
              "interpretation": "one fixed mismatched assignment; no claim of uncertainty over donor assignments",
              "test_metrics_used_for_training_or_selection": False}
    file = root / "evaluation/mismatch_report.json"
    atomic_json(file, report)
    old.seal(done, protocol, [file, *outputs], inputs=inputs)
    print_summary(root)


def print_summary(root):
    file = root / "analysis/analysis_report.json"
    if file.exists():
        r = read_json(file)
        for scope, value in r["analysis"].items():
            print({"scope": scope, "samples": value["samples"], "mean_F1": {
                k: round(v["mean_f1"] * 100, 4) for k, v in value["systems"].items()},
                   "C_minus_B": value["C_minus_B"]}, flush=True)
        print({"by_valid_image_count": r["by_valid_image_count"]}, flush=True)
    file = root / "evaluation/mismatch_report.json"
    if file.exists():
        r = read_json(file)
        for scope, value in r["summary"].items():
            print({"scope": scope, "samples": value["samples"], "mean_F1": {
                k: round(v["mean_f1"] * 100, 4) for k, v in value["systems"].items()},
                   "C_minus_D_pp": value["C_minus_D_pp"]}, flush=True)
    print({"analysis_report": str(root / "analysis/analysis_report.json"),
           "mismatch_report": str(root / "evaluation/mismatch_report.json")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "analyze", "smoke", "features", "logits",
                                                         "evaluate", "summary", "status"))
    parser.add_argument("--visual-run", default=DEFAULT_VISUAL)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--mismatch-seed", type=int, default=1234)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.run).resolve()
    if args.stage in ("status", "summary"):
        if args.stage == "status":
            print(read_json(root / "status.json") if (root / "status.json").exists() else "No diagnostic run started")
        print_summary(root)
        return
    if (root / "protocol.json").exists() and read_json(root / "protocol.json").get("schema_version") != SCHEMA:
        parser.error("The output directory belongs to another experiment; choose a new --run directory")
    source = Path(args.visual_run).resolve()
    if args.stage == "prepare" and (root == source or root in source.parents or source in root.parents):
        parser.error("Diagnostic and visual output directories must be separate, without nesting")
    with old.stage_lock(root):
        status = {"stage": args.stage, "state": "running", "started_unix": time.time()}
        atomic_json(root / "status.json", status)
        try:
            if args.stage == "prepare":
                prepare(root, args)
            else:
                protocol = load_protocol(root)
                if args.stage in ("smoke", "features", "logits"):
                    old.require_cuda()
                {"analyze": analyze, "smoke": smoke, "features": extract_features,
                 "logits": infer_logits, "evaluate": evaluate}[args.stage](root, protocol)
            atomic_json(root / "status.json", {**status, "state": "completed", "finished_unix": time.time()})
        except Exception as error:
            atomic_json(root / "status.json", {**status, "state": "failed", "error": str(error),
                                               "traceback": traceback.format_exc(), "finished_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
