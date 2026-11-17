"""Staged AutoDL workflow. Selection reads official dev, never test metrics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

from .common import (SCHEMA, TAGS, atomic_json, average, digest, filtered_indices,
                     fingerprint, load_rows, metrics, project_path, read_json,
                     read_rows, runtime, save_rows, seed_everything, spans)

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = "outputs_vlm_gold/tpm_mi_llm_v1"


def code_binding():
    paths = [*Path(__file__).parent.glob("*.py"), PROJECT / "scripts/run_tpm_llm_v1.py"]
    return {str(path.relative_to(PROJECT)): digest(path) for path in sorted(paths)}


def model_location(explicit, names):
    paths = [Path(explicit)] if explicit else [
        parent / name for parent in (PROJECT.parent / "pretrained_models", PROJECT / "pretrained_models")
        for name in names]
    for path in paths:
        if (path / "config.json").is_file():
            return str(path.resolve())
    raise FileNotFoundError(f"Local model not found. Checked: {[str(path) for path in paths]}")


def model_binding(directory):
    files = sorted(path for path in Path(directory).iterdir()
                   if path.is_file() and path.suffix in (".json", ".txt", ".model", ".safetensors", ".bin"))
    if not any(path.suffix in (".bin", ".safetensors") for path in files):
        raise FileNotFoundError(f"Model weights are missing: {directory}")
    print(f"Binding local model files: {directory}", flush=True)
    return {str(path.resolve()): {"sha256": digest(path), "size": path.stat().st_size,
                                  "mtime_ns": path.stat().st_mtime_ns} for path in files}


def audit_images(rows, config, prior=None):
    from .data import image_path, open_rgb
    result = dict(prior or {})
    names = sorted({name for row in rows for name in row["images"]} - set(result))
    from tqdm.auto import tqdm
    for name in tqdm(names, desc="Audit image files", unit="image"):
        path = image_path(name, config)
        valid, error = False, None
        try:
            with open_rgb(path):
                valid = True
        except (OSError, ValueError) as exc:
            error = str(exc)
        stat = path.stat() if path.is_file() else None
        result[name] = {"path": str(path.resolve()), "valid": valid, "error": error,
                        "sha256": digest(path) if stat else None,
                        "size": stat.st_size if stat else None,
                        "mtime_ns": stat.st_mtime_ns if stat else None}
    return result


def prepare(root, args):
    config = {"data_dir": str(Path(args.data_dir).resolve()),
              "image_dir": str(Path(args.image_dir).resolve()),
              "twitter_image_dir": str(Path(args.twitter_image_dir).resolve()),
              "bert_model": model_location(args.bert_model, ("bert-base-uncased",)),
              "vit_model": model_location(args.vit_model, ("ViTB-16", "vit-base-patch16-224")),
              "qwen_model": model_location(args.qwen_model, ("Qwen2.5-VL-3B-Instruct",)),
              "max_length": args.max_length, "batch_size": args.batch_size,
              "baseline_epochs": args.baseline_epochs, "patience": args.patience,
              "baseline_lr": args.baseline_lr, "warmup_ratio": 0.01,
              "head_epochs": args.head_epochs, "head_lr": 0.001,
              "feature_batch_size": args.feature_batch_size, "projection_seed": 1234,
              "seeds": args.seeds or [1234, 42, 2026],
              "coefficient_grid": [0.0, 0.25, 0.5, 1.0], "architectures": ["bigru", "linear"]}
    if (config["seeds"][0] != 1234 or len(set(config["seeds"])) != len(config["seeds"])
            or min(config["batch_size"], config["baseline_epochs"], config["head_epochs"],
                   config["feature_batch_size"]) < 1 or not 3 <= config["max_length"] <= 508
            or config["baseline_lr"] <= 0 or config["patience"] < 0):
        raise ValueError("Invalid recipe; seed 1234 must be the first calibration seed")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        old = load_protocol(root)
        if old["config"] != config:
            raise ValueError("Recipe differs from the saved protocol. Use a new --run directory.")
        print("Existing official-split protocol verified.", flush=True)
        return
    data = Path(config["data_dir"])
    sources = {role: data / f"MNER-MI_{name}.txt"
               for role, name in (("train", "train"), ("dev", "val"), ("test", "test"))}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"Expected original official split: {path}")
    rows = {role: read_rows(sources[role], role) for role in ("train", "dev")}
    if len(rows["train"]) != 6856 or len(rows["dev"]) != 860:
        raise ValueError(
            f"Expected official train/dev=6856/860; got {len(rows['train'])}/{len(rows['dev'])}. "
            "Do not use internal 5484/1372 or 4388 splits. Pass --data-dir with original official files.")
    prepared = root / "data"
    prepared.mkdir(exist_ok=True)
    for role in rows:
        save_rows(prepared / f"{role}.jsonl", rows[role])
    audit = audit_images(rows["train"] + rows["dev"], config)
    atomic_json(prepared / "image_audit.json", audit)
    bindings = {key: model_binding(config[f"{key}_model"]) for key in ("bert", "vit", "qwen")}
    source_hashes = {str(path.resolve()): digest(path) for path in sources.values()}
    value = {"schema_version": SCHEMA, "config": config, "code_sha256": code_binding(),
             "source_sha256": source_hashes, "model_files": bindings,
             "prepared_sha256": {path.name: digest(path) for path in prepared.iterdir() if path.is_file()},
             "samples": {role: len(values) for role, values in rows.items()},
             "upstream_model": "https://github.com/JinFish/MNER-MI/blob/main/models/BERTModels.py",
             "upstream_model_blob": "b7e3de6187808bf95b3d54778efbb36440d3a98b",
             "baseline_description": "TPM-MI audited reproduction; published score not assumed",
             "implementation_changes": [
                 "batch_first image Transformer prevents cross-post attention",
                 "tokenize original words once; do not re-tokenize WordPieces",
                 "mask missing and padded image slots in image attention and visual prefixes",
                 "select checkpoints using full-original-word dev entity F1"],
             "previously_used_official_test_in_project": True,
             "test_metrics_used_for_training_or_selection": False,
             "test_source_hashed_only_at_prepare": True,
             "image_summary": {"unique": len(audit),
                               "invalid": sum(not item["valid"] for item in audit.values())}}
    value["protocol_id"] = fingerprint(value)
    atomic_json(root / "protocol.json", value)
    print({"run": str(root), "samples": value["samples"], "image_summary": value["image_summary"],
           "protocol_id": value["protocol_id"]}, flush=True)


def load_protocol(root):
    value = read_json(root / "protocol.json")
    identity = {key: item for key, item in value.items() if key != "protocol_id"}
    if (value.get("schema_version") != SCHEMA or fingerprint(identity) != value["protocol_id"]
            or value["code_sha256"] != code_binding()):
        raise ValueError("Protocol or source code changed; use a new --run directory")
    for name, expected in value["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Official source changed: {name}")
    for name, expected in value["prepared_sha256"].items():
        if digest(root / "data" / name) != expected:
            raise ValueError(f"Prepared source changed: {name}")
    for info in read_json(root / "data/image_audit.json").values():
        path = Path(info["path"])
        stat = path.stat() if path.is_file() else None
        if stat is None and info["sha256"] is not None:
            raise ValueError(f"Image disappeared: {path}")
        if stat is not None and (stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]):
            if digest(path) != info["sha256"]:
                raise ValueError(f"Image changed: {path}")
    for binding in value["model_files"].values():
        for name, info in binding.items():
            stat = Path(name).stat()
            if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
                if digest(name) != info["sha256"]:
                    raise ValueError(f"Frozen pretrained model changed: {name}")
    return value


def seal(path, protocol, outputs, inputs=(), extra=None):
    atomic_json(path, {"protocol_id": protocol["protocol_id"],
                       "output_sha256": {str(Path(file).resolve()): digest(file) for file in outputs},
                       "input_sha256": {str(Path(file).resolve()): digest(file) for file in inputs},
                       **(extra or {})})


def completed(path, protocol):
    if not path.exists():
        return False
    report = read_json(path)
    if report["protocol_id"] != protocol["protocol_id"]:
        raise ValueError(f"Artifact protocol mismatch: {path}")
    for group in ("output_sha256", "input_sha256"):
        for name, expected in report[group].items():
            if digest(name) != expected:
                raise ValueError(f"Changed stage artifact: {name}")
    return report


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".tpm-", dir=path.parent)
    try:
        # File objects also support hidden temp names on Windows and Linux.
        with os.fdopen(handle, "wb") as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_torch(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def require_cuda():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One CUDA GPU is required. Run with CUDA_VISIBLE_DEVICES=0.")


def data_rows(root, role):
    return load_rows(root / "data" / f"{role}.jsonl")


def data_loader(root, protocol, role, seed=1234, shuffle=False, limit=None, test_audit=None):
    import torch
    from torch.utils.data import DataLoader
    from .data import WordImageDataset, collate
    rows = data_rows(root, role)
    if limit is not None:
        rows = rows[:limit]
    audit = test_audit or read_json(root / "data/image_audit.json")
    dataset = WordImageDataset(rows, protocol["config"], audit)
    loader = DataLoader(dataset, batch_size=protocol["config"]["batch_size"], shuffle=shuffle,
                        num_workers=0, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    return rows, loader


def collect(model, loader, rows):
    import torch
    from tqdm.auto import tqdm
    from .data import to_device
    values, masks, mappings = [], [], []
    observed = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="TPM-MI full-word inference", unit="batch"):
            observed.extend(batch["index"])
            values.append(model(to_device(batch, "cuda:0")).cpu())
            masks.append(batch["attention_mask"])
            mappings.extend({"positions": positions, "word_indices": indices}
                            for positions, indices in zip(batch["positions"], batch["word_indices"]))
    if observed != list(range(len(rows))):
        raise ValueError("Inference loader must preserve all original source rows")
    return {"emissions": torch.cat(values), "mask": torch.cat(masks), "mappings": mappings,
            "crf": {key: value.detach().cpu() for key, value in model.crf.state_dict().items()},
            "source_ids": [row["source_id"] for row in rows], "runtime": runtime()}


def decode(bundle, rows, emissions=None):
    import torch
    from torchcrf import CRF
    decoder = CRF(len(TAGS), batch_first=True)
    decoder.load_state_dict(bundle["crf"])
    values = bundle["emissions"] if emissions is None else emissions
    with torch.inference_mode():
        paths = decoder.decode(values, mask=bundle["mask"].bool())
    return [project_path(path, mapping["positions"], mapping["word_indices"], len(row["tokens"]))
            for row, path, mapping in zip(rows, paths, bundle["mappings"])]


def baseline_paths(root, seed):
    directory = root / f"baseline/seed{seed}"
    return directory, directory / "best.pt", directory / "completed.json"


def baseline(root, protocol, seed):
    import torch
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup
    from .data import to_device
    from .model import TPMMI
    directory, checkpoint, done = baseline_paths(root, seed)
    if completed(done, protocol):
        print(f"TPM-MI seed={seed}: completed checkpoint verified; skipping", flush=True)
        return
    config = protocol["config"]
    if not completed(root / "smoke/base.json", protocol):
        raise ValueError("Run --stage smoke-base first")
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free space is needed for atomic best-checkpoint replacement")
    if checkpoint.exists():
        print("Incomplete baseline stage: restarting from pretrained weights at epoch 1.", flush=True)
    seed_everything(seed)
    _, train_loader = data_loader(root, protocol, "train", seed, shuffle=True)
    dev_rows, dev_loader = data_loader(root, protocol, "dev")
    model = TPMMI.from_pretrained(config).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["baseline_lr"], weight_decay=0.01)
    steps = len(train_loader) * config["baseline_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config["warmup_ratio"]), steps)
    history, best, best_epoch, stale = [], -1.0, None, 0
    for epoch in range(1, config["baseline_epochs"] + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"TPM-MI seed={seed} epoch={epoch}/{config['baseline_epochs']}")
        for batch in progress:
            batch = to_device(batch, "cuda:0")
            loss = model.loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite baseline loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach().item())
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        scores = metrics(dev_rows, decode(collect(model, dev_loader, dev_rows), dev_rows))
        history.append({"epoch": epoch, "mean_loss": sum(losses)/len(losses), "dev": scores})
        atomic_json(directory / "history.json", history)
        print(f"TPM-MI seed={seed} epoch={epoch} dev_F1={scores['f1']:.6f}", flush=True)
        if scores["f1"] > best:
            best, best_epoch, stale = scores["f1"], epoch, 0
            save_torch(checkpoint, {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                    "epoch": epoch, "dev": scores, "seed": seed,
                                    "protocol_id": protocol["protocol_id"], "runtime": runtime()})
        else:
            stale += 1
        if config["patience"] and stale >= config["patience"]:
            print(f"Early stopping after {stale} dev epochs without improvement", flush=True)
            break
    seal(done, protocol, [checkpoint, directory / "history.json"],
         extra={"seed": seed, "best_epoch": best_epoch, "best_dev_f1": best,
                "runtime": runtime(), "test_scored": False})
    print(f"TPM-MI seed={seed} complete: best_epoch={best_epoch}, dev_F1={best:.6f}", flush=True)


def smoke_base(root, protocol):
    import torch
    from .model import TPMMI
    from .data import to_device
    path = root / "smoke/base.json"
    if completed(path, protocol):
        print("TPM-MI GPU smoke already passed", flush=True)
        return
    seed_everything(1234)
    rows, loader = data_loader(root, protocol, "train", limit=4)
    model = TPMMI.from_pretrained(protocol["config"]).to("cuda:0")
    batch = to_device(next(iter(loader)), "cuda:0")
    model.train()
    loss = model.loss(model(batch), batch)
    loss.backward()
    if not torch.isfinite(loss) or not any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Baseline smoke backward failed")
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        together = model(batch)
        reversed_batch = {key: value.flip(0) if torch.is_tensor(value) else value
                          for key, value in batch.items()}
        torch.testing.assert_close(together, model(reversed_batch).flip(0), atol=2e-4, rtol=2e-4)
        first_only = {key: value[:1] if torch.is_tensor(value) else value
                      for key, value in batch.items()}
        torch.testing.assert_close(together[:1], model(first_only), atol=2e-4, rtol=2e-4)
    predictions = decode(collect(model, loader, rows), rows)
    if any(len(prediction) != len(row["tokens"]) for row, prediction in zip(rows, predictions)):
        raise ValueError("Smoke predictions lost original words")
    seal(path, protocol, [], extra={"passed": True, "runtime": runtime(),
                                    "finite_loss": loss.detach().item(),
                                    "batch_isolation_checked": True, "test_scored": False})
    print(f"TPM-MI GPU smoke passed: loss={loss.detach().item():.6f}", flush=True)


def smoke_features(root, protocol):
    import torch
    from .features import extract, fit_head, infer_head
    path = root / "smoke/features.json"
    if completed(path, protocol):
        print("Frozen-feature GPU smoke already passed", flush=True)
        return
    rows = data_rows(root, "train")[:4]
    config = {**protocol["config"], "head_epochs": 1}
    result = {}
    for encoder in ("bert", "qwen"):
        bundle = extract(rows, config, encoder)
        head, history = fit_head(bundle["features"], rows, config, 1234)
        logits = infer_head(head, bundle["features"])
        if not all(values.shape == (len(row["tokens"]), 9) for row, values in zip(rows, logits)):
            raise ValueError("Frozen-feature smoke output shape mismatch")
        result[encoder] = {"history": history, "words": sum(len(row["tokens"]) for row in rows)}
        del head, bundle, logits
        torch.cuda.empty_cache()
    seal(path, protocol, [], extra={"passed": True, "runtime": runtime(), "encoders": result,
                                    "test_scored": False})
    print("BERT and Qwen feature/head GPU smoke passed", flush=True)


def feature_paths(root, encoder, role):
    file = root / f"features/{encoder}_{role}.pt"
    return file, file.with_suffix(".json")


def prepare_features(root, protocol, test=False):
    from .features import extract
    if not completed(root / "smoke/features.json", protocol):
        raise ValueError("Run --stage smoke-features first")
    if test and not completed(root / "calibration/completed.json", protocol):
        raise ValueError("Freeze development coefficients with --stage calibrate before reading test inputs")
    for encoder in ("bert", "qwen"):
        for role in (("test",) if test else ("train", "dev")):
            file, done = feature_paths(root, encoder, role)
            if completed(done, protocol):
                print(f"Reusing verified {encoder}/{role} features", flush=True)
                continue
            rows = (read_rows(Path(protocol["config"]["data_dir"]) / "MNER-MI_test.txt", "test", labels=False)
                    if test else data_rows(root, role))
            if test and len(rows) != 860:
                raise ValueError(f"Expected 860 official test rows, found {len(rows)}")
            value = extract(rows, protocol["config"], encoder)
            save_torch(file, value)
            seal(done, protocol, [file], extra={"samples": len(rows), "runtime": runtime(),
                                               "test_labels_read": False})


def head_paths(root, seed, encoder, architecture):
    file = root / f"heads/seed{seed}/{encoder}_{architecture}.pt"
    return file, file.with_suffix(".json")


def train_heads(root, protocol, seed):
    import torch
    from .features import fit_head, infer_head
    config = protocol["config"]
    rows = data_rows(root, "train")
    for encoder in ("bert", "qwen"):
        train_path, train_done = feature_paths(root, encoder, "train")
        dev_path, dev_done = feature_paths(root, encoder, "dev")
        if not completed(train_done, protocol) or not completed(dev_done, protocol):
            raise ValueError("Run --stage features first")
        train = load_torch(train_path)
        dev = load_torch(dev_path)
        if train["source_ids"] != [row["source_id"] for row in rows]:
            raise ValueError("Feature source IDs differ from official training rows")
        for architecture in config["architectures"]:
            file, done = head_paths(root, seed, encoder, architecture)
            if completed(done, protocol):
                print(f"Reusing head seed={seed} {encoder}/{architecture}", flush=True)
                continue
            head, history = fit_head(train["features"], rows, config, seed, architecture)
            logits = infer_head(head, dev["features"], config["feature_batch_size"])
            save_torch(file, {"head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                              "architecture": architecture, "seed": seed, "encoder": encoder,
                              "dev_logits": logits, "dev_aligned": dev["aligned"],
                              "dev_source_ids": dev["source_ids"], "history": history,
                              "train_samples": len(rows), "runtime": runtime()})
            seal(done, protocol, [file], inputs=[train_path, dev_path],
                 extra={"seed": seed, "encoder": encoder, "architecture": architecture,
                        "test_scored": False})
            del head, logits
            torch.cuda.empty_cache()


def base_cache(root, protocol, seed, role, test_audit=None):
    import torch
    from .model import TPMMI
    file = root / f"emissions/seed{seed}_{role}.pt"
    done = file.with_suffix(".json")
    _, checkpoint, baseline_done = baseline_paths(root, seed)
    if not completed(baseline_done, protocol):
        raise ValueError(f"Baseline seed={seed} has not completed")
    if completed(done, protocol):
        return load_torch(file)
    saved = load_torch(checkpoint)
    if saved["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Baseline checkpoint protocol differs")
    current = runtime()
    if any(saved["runtime"][key] != current[key] for key in ("torch", "transformers")):
        raise ValueError("Use the same base runtime for baseline training and scoring")
    model = TPMMI.from_pretrained(protocol["config"])
    model.load_state_dict(saved["model"], strict=True)
    del saved["model"]
    model.to("cuda:0").eval().requires_grad_(False)
    rows, loader = data_loader(root, protocol, role, test_audit=test_audit)
    result = collect(model, loader, rows)
    result["metrics"] = metrics(rows, decode(result, rows))
    if role == "dev" and abs(result["metrics"]["f1"] - saved["dev"]["f1"]) > 1e-12:
        raise ValueError("Best-checkpoint dev predictions failed exact metric reproduction")
    save_torch(file, result)
    seal(done, protocol, [file], inputs=[checkpoint], extra={"seed": seed, "role": role})
    del model
    torch.cuda.empty_cache()
    return result


def checked_head(root, protocol, seed, encoder, architecture):
    file, done = head_paths(root, seed, encoder, architecture)
    if not completed(done, protocol):
        raise ValueError(f"Missing fitted head: {file}")
    return file, load_torch(file)


def calibrate(root, protocol):
    from .features import fuse
    output = root / "calibration/completed.json"
    if completed(output, protocol):
        print("Development fusion coefficients already frozen", flush=True)
        return
    rows = data_rows(root, "dev")
    native = base_cache(root, protocol, 1234, "dev")
    selected, trials, inputs = {}, {}, [root / "emissions/seed1234_dev.pt"]
    for encoder in ("bert", "qwen"):
        for architecture in protocol["config"]["architectures"]:
            file, head = checked_head(root, protocol, 1234, encoder, architecture)
            inputs.append(file)
            if head["dev_source_ids"] != native["source_ids"]:
                raise ValueError("Head/base development source IDs differ")
            key = f"{encoder}_{architecture}"
            trials[key] = []
            for coefficient in protocol["config"]["coefficient_grid"]:
                values = fuse(native["emissions"], rows, native["mappings"], head["dev_logits"],
                              head["dev_aligned"], coefficient)
                scores = metrics(rows, decode(native, rows, values))
                trials[key].append({"coefficient": coefficient, "metrics": scores})
            selected[key] = max(trials[key], key=lambda item: (item["metrics"]["f1"], -item["coefficient"]))
    report = {"protocol_id": protocol["protocol_id"], "selection_seed": 1234,
              "scope": "official development, frozen choices shared across all seeds",
              "coefficients": {key: value["coefficient"] for key, value in selected.items()},
              "baseline": native["metrics"], "selected": selected, "trials": trials,
              "test_metrics_used_for_selection": False}
    qwen = selected["qwen_bigru"]["metrics"]["f1"]
    bert = selected["bert_bigru"]["metrics"]["f1"]
    report["development_deltas"] = {"qwen_minus_tpm": qwen - native["metrics"]["f1"],
                                     "qwen_minus_bert": qwen - bert}
    report["passes_development_gate"] = qwen >= max(bert, native["metrics"]["f1"]) + 0.003
    path = root / "calibration/dev_report.json"
    atomic_json(path, report)
    seal(output, protocol, [path], inputs=inputs)
    print({"dev_f1": {"tpm": native["metrics"]["f1"], "tpm_bert": bert, "tpm_qwen": qwen},
           "coefficients": report["coefficients"], "development_deltas": report["development_deltas"],
           "passes_development_gate": report["passes_development_gate"], "report": str(path)}, flush=True)


def test_logits(root, protocol):
    import torch
    from .features import WordHead, infer_head
    if not completed(root / "calibration/completed.json", protocol):
        raise ValueError("Run development calibration first")
    for encoder in ("bert", "qwen"):
        feature_path, feature_done = feature_paths(root, encoder, "test")
        if not completed(feature_done, protocol):
            raise ValueError("Run --stage test-features first")
        features = load_torch(feature_path)
        for seed in protocol["config"]["seeds"]:
            for architecture in protocol["config"]["architectures"]:
                source, saved = checked_head(root, protocol, seed, encoder, architecture)
                file = root / f"test_logits/seed{seed}_{encoder}_{architecture}.pt"
                done = file.with_suffix(".json")
                if completed(done, protocol):
                    continue
                head = WordHead(architecture).to("cuda:0")
                head.load_state_dict(saved["head"], strict=True)
                head.eval().requires_grad_(False)
                logits = infer_head(head, features["features"], protocol["config"]["feature_batch_size"])
                save_torch(file, {"logits": logits, "aligned": features["aligned"],
                                  "source_ids": features["source_ids"]})
                seal(done, protocol, [file], inputs=[source, feature_path])
                del head, logits
                torch.cuda.empty_cache()


def evaluate(root, protocol):
    from .features import fuse
    completed_path = root / "evaluation/completed.json"
    if completed(completed_path, protocol):
        print("Official test evaluation already complete", flush=True)
        return
    if not completed(root / "calibration/completed.json", protocol):
        raise ValueError("Development choices are not frozen")
    calibration_path = root / "calibration/dev_report.json"
    calibration = read_json(calibration_path)
    config = protocol["config"]
    test_path = Path(config["data_dir"]) / "MNER-MI_test.txt"
    rows = read_rows(test_path, "test")
    if len(rows) != 860:
        raise ValueError("Official test must contain 860 posts")
    save_rows(root / "data/test.jsonl", rows)
    prior = read_json(root / "data/image_audit.json")
    audit = audit_images(rows, config, prior)
    atomic_json(root / "evaluation/image_audit.json", audit)
    image_hashes = {key: info["sha256"] for key, info in audit.items()}
    retained = filtered_indices(data_rows(root, "train") + data_rows(root, "dev"), rows, image_hashes)
    scores = {"official_test_full": {}, "official_test_filtered": {}}
    inputs, outputs = [calibration_path], [root / "evaluation/image_audit.json"]
    for seed in config["seeds"]:
        native = base_cache(root, protocol, seed, "test", test_audit=audit)
        inputs.append(root / f"emissions/seed{seed}_test.pt")
        predictions = {"tpm": decode(native, rows)}
        for encoder in ("bert", "qwen"):
            for architecture in config["architectures"]:
                file = root / f"test_logits/seed{seed}_{encoder}_{architecture}.pt"
                if not completed(file.with_suffix(".json"), protocol):
                    raise ValueError("Run --stage test-logits in mner-vlm before official scoring")
                saved = load_torch(file)
                if saved["source_ids"] != native["source_ids"]:
                    raise ValueError("Test logits and baseline source IDs differ")
                inputs.append(file)
                coefficient = calibration["coefficients"][f"{encoder}_{architecture}"]
                variants = [("head" if architecture == "bigru" else "linear", coefficient, True)]
                if architecture == "bigru":
                    variants += [("no_clip", coefficient, False), ("shared1", 1.0, True)]
                for variant, weight, clip in variants:
                    values = fuse(native["emissions"], rows, native["mappings"], saved["logits"],
                                  saved["aligned"], weight, clip=clip)
                    predictions[f"tpm_{encoder}_{variant}"] = decode(native, rows, values)
        for scope, positions in (("official_test_full", list(range(len(rows)))),
                                 ("official_test_filtered", retained)):
            scores[scope][str(seed)] = {
                name: metrics([rows[index] for index in positions], [values[index] for index in positions])
                for name, values in predictions.items()}
        for name, values in predictions.items():
            file = root / f"evaluation/seed{seed}_{name}_predictions.jsonl"
            save_rows(file, [{"source_id": row["source_id"], "labels": prediction,
                              "predicted": sorted(spans(prediction))}
                             for row, prediction in zip(rows, values)])
            outputs.append(file)
    summary = {}
    for scope, values in scores.items():
        systems = values[str(config["seeds"][0])]
        summary[scope] = {name: average([entry[name]["f1"] for entry in values.values()]) for name in systems}
        deltas = {reference: [entry["tpm_qwen_head"]["f1"] - entry[reference]["f1"]
                              for entry in values.values()] for reference in ("tpm", "tpm_bert_head")}
        summary[scope]["paired_deltas"] = {
            reference: {"by_seed": dict(zip(map(str, config["seeds"]), differences)),
                        "mean": sum(differences)/len(differences),
                        "positive_seeds": sum(delta > 0 for delta in differences)}
            for reference, differences in deltas.items()}
    report = {"schema_version": SCHEMA, "protocol_id": protocol["protocol_id"],
              "baseline_description": protocol["baseline_description"],
              "train_samples": len(data_rows(root, "train")), "dev_samples": len(data_rows(root, "dev")),
              "test_samples": len(rows), "filtered_test_samples": len(retained),
              "filtered_test_source_ids": [rows[index]["source_id"] for index in retained],
              "filter_reference": "all official training and development posts; exact text/image components",
              "coefficients": calibration["coefficients"], "seeds": config["seeds"],
              "metrics": scores, "summary": summary,
              "previously_used_official_test_in_project": True,
              "test_metrics_used_for_training_or_selection": False}
    file = root / "evaluation/evaluation_report.json"
    atomic_json(file, report)
    seal(completed_path, protocol, [*outputs, file], inputs=inputs)
    print({"summary": summary, "report": str(file)}, flush=True)


@contextmanager
def stage_lock(root):
    # AutoDL is Linux; CPU unit tests need not import fcntl on Windows.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".stage.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another TPM-MI stage is running in this output directory") from exc
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = ("prepare", "smoke-base", "smoke-features", "baseline", "features", "heads",
              "calibrate", "test-features", "test-logits", "evaluate", "status")
    parser.add_argument("--stage", required=True, choices=stages)
    parser.add_argument("--run", default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-dir", default="dataset/text")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--twitter-image-dir", default="dataset/twitter2017_images")
    parser.add_argument("--bert-model")
    parser.add_argument("--vit-model")
    parser.add_argument("--qwen-model")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--baseline-lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    root = Path(args.run).resolve()
    if args.stage == "status":
        print(read_json(root / "status.json") if (root / "status.json").is_file() else "No TPM run started")
        for seed in (1234, 42, 2026):
            path = root / f"baseline/seed{seed}/history.json"
            history = read_json(path) if path.is_file() else []
            print({"seed": seed, "completed_baseline_epochs": [row["epoch"] for row in history],
                   "baseline_complete": (root / f"baseline/seed{seed}/completed.json").is_file()})
        report = root / "calibration/dev_report.json"
        if report.exists():
            print({"dev_report": str(report), "development_deltas": read_json(report)["development_deltas"]})
        print({"test_complete": (root / "evaluation/completed.json").is_file()})
        return
    with stage_lock(root):
        status = {"stage": args.stage, "seed": args.seed, "state": "running", "started_unix": time.time()}
        atomic_json(root / "status.json", status)
        try:
            if args.stage == "prepare":
                prepare(root, args)
            else:
                protocol = load_protocol(root)
                if args.seed not in protocol["config"]["seeds"]:
                    raise ValueError("Seed was not specified in the frozen protocol")
                require_cuda()
                functions = {"smoke-base": smoke_base, "smoke-features": smoke_features,
                             "baseline": lambda r, p: baseline(r, p, args.seed),
                             "features": prepare_features, "heads": lambda r, p: train_heads(r, p, args.seed),
                             "calibrate": calibrate, "test-features": lambda r, p: prepare_features(r, p, test=True),
                             "test-logits": test_logits, "evaluate": evaluate}
                functions[args.stage](root, protocol)
            atomic_json(root / "status.json", {**status, "state": "completed", "finished_unix": time.time()})
        except Exception as error:
            atomic_json(root / "status.json", {**status, "state": "failed", "error": str(error),
                                               "traceback": traceback.format_exc(), "finished_unix": time.time()})
            raise
