from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_component_suite as suite
from scripts.evaluate_crf_ensemble import decode_blend, sha256_file


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "data"
    source.mkdir()
    roots = {name: tmp_path / name for name in ("mi", "uni", "matched", "components")}
    manifests, artifacts = {}, {}
    for dataset in suite.DATASETS:
        selected_dir, matched_dir = tmp_path / f"{dataset}_selected", tmp_path / f"{dataset}_matched"
        for directory, role in ((source, "source"), (selected_dir, "selected"), (matched_dir, "matched")):
            directory.mkdir(exist_ok=True)
            for split in ("train", "val", "test"):
                text = f"{dataset}:{role if split == 'train' else split}"
                (directory / f"MNER-{dataset}_{split}.txt").write_text(text, encoding="utf-8")
        reference_path = selected_dir / "augmentation_manifest.json"
        write_json(reference_path, {})
        manifest_path = matched_dir / "matched_resampling_manifest.json"
        manifest = {"dataset": dataset, "reference_manifest": str(reference_path),
                    "source_train_sha256": sha256_file(source / f"MNER-{dataset}_train.txt"),
                    "reference_train_sha256": sha256_file(selected_dir / f"MNER-{dataset}_train.txt"),
                    "output_files_sha256": {p.name: sha256_file(p) for p in matched_dir.glob("*.txt")},
                    "candidate_count": 5, "added_examples": 1,
                    "balance_check": {"passed": True}, "overlap_check": {"fraction": 0.4}}
        write_json(manifest_path, manifest)
        manifests[dataset] = manifest_path
        for seed in suite.SEEDS:
            for method in suite.METHODS:
                if method == "token_selected":
                    root, arm = roots["components"], f"{dataset.lower()}_token_evidence_selected"
                elif method == "legacy_matched":
                    root, arm = roots["mi" if dataset == "MI" else "matched"], "matched_no_llm"
                else:
                    root = roots[dataset.lower()]
                    arm = "strict_baseline" if method == "baseline" else "llm_selected" if dataset == "MI" else "duplicate_control"
                run = root / f"{arm}_seed{seed}" / "run"
                run.mkdir(parents=True)
                checkpoint = run / "best_model.pt"
                checkpoint.write_bytes(f"{dataset}:{seed}:{method}".encode())
                data = source if method == "baseline" else matched_dir if method == "legacy_matched" else selected_dir
                config = {"dataset": dataset, "seed": seed, "data_path": str(data), "output_dir": str(root),
                          "skip_test": True, "max_seq_length": 4}
                if method == "token_selected":
                    config["visual_fusion"] = "token_evidence"
                write_json(run / "config.json", config)
                dev = {"baseline": 0.6667, "legacy_selected": 1.0, "legacy_matched": 0.0,
                       "token_selected": 0.6667 if dataset == "MI" else 1.0}[method]
                (run / "metrics.csv").write_text(f"epoch,dev_f1\n1,{dev}\n", encoding="utf-8")
                artifacts[dataset, seed, method] = checkpoint

    def audit(path, data, dataset):
        assert data == source and path == manifests[dataset]
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(suite, "audit_data", audit)
    label_map = {"O": 0, "B-PER": 1, "X": 2}

    class TinyDataset:
        def __init__(self, args, file_name):
            self.dataset = args.dataset
            self.split = "val" if file_name.endswith("_val.txt") else "test"
            self.processor = SimpleNamespace(get_label_mapping=lambda **kw: label_map)

        def __len__(self):
            return 2

    calls = []

    def collect(checkpoint, config, data, cli, labels, name):
        calls.append(name)
        method = name.split("/")[-1]
        choices = {"baseline": (1, 1), "legacy_selected": (1, 0), "legacy_matched": (0, 0),
                   "token_selected": (1, 1) if data.dataset == "MI" else (1, 0)}[method]
        if data.split == "test":
            if method == "legacy_matched":
                choices = (1, 1)
            elif method == "token_selected" and data.dataset == "UNI":
                choices = (0, 0)
        emissions = torch.zeros(2, 4, 3)
        emissions[:, (0, 2), 2] = 5
        for i, choice in enumerate(choices):
            emissions[i, 1, choice] = 4
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
        state = {"start_transitions": torch.zeros(3), "end_transitions": torch.zeros(3), "transitions": torch.zeros(3, 3)}
        return {"emissions": emissions, "mask": mask,
                "labels": torch.tensor([[2, 1, 2, 0], [2, 0, 2, 0]]), "crf": state,
                "native_paths": decode_blend(emissions, emissions, state, state, mask, 0)}

    monkeypatch.setattr(suite, "MultimodalNerDataset", TinyDataset)
    monkeypatch.setattr(suite, "_collect", collect)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["evaluate_component_suite.py", "--data-path", str(source),
                        "--mi-runs-root", str(roots["mi"]), "--uni-runs-root", str(roots["uni"]),
                        "--uni-matched-root", str(roots["matched"]), "--component-root", str(roots["components"]),
                        "--mi-matched-manifest", str(manifests["MI"]), "--uni-matched-manifest", str(manifests["UNI"]),
                        "--device", "cpu", "--output-dir", str(output)])
    return output, artifacts, calls


def test_freezes_all_24_models_and_reports_test_declines_without_search(tmp_path, monkeypatch):
    output, _, calls = fixture(tmp_path, monkeypatch)
    suite.main()
    assert len(calls) == 48
    assert all(name.startswith("val/") for name in calls[:24])
    assert all(name.startswith("test/") for name in calls[24:])
    frozen = json.loads((output / "frozen_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert frozen["parameter_search"] is False and frozen["models_chosen_using_test"] is False
    assert frozen["training_runs"] == 0 and frozen["llm_api_requests"] == 0
    assert report["frozen_manifest_sha256"] == sha256_file(output / "frozen_manifest.json")
    assert set(report["datasets"]) == {"MI", "UNI"}
    for dataset in suite.DATASETS:
        assert len(frozen["datasets"][dataset]["models"]) == 12
        rows = report["datasets"][dataset]["per_seed"]
        assert [row["seed"] for row in rows] == [42, 1234, 2026]
        assert all(set(row["metrics"]) == set(suite.METHODS) for row in rows)
        assert report["datasets"][dataset]["aggregate"]["comparisons"]["component_on_selected"]["mean_f1_difference"] < 0
    # UNI token test F1 differs from its recorded dev F1, which must be allowed.
    assert report["datasets"]["UNI"]["aggregate"]["methods"]["token_selected"]["mean_f1"] == 0


@pytest.mark.parametrize("problem", ["mode", "hyperparameter", "data", "missing_model"])
def test_preflight_errors_stop_before_any_inference(tmp_path, monkeypatch, problem):
    output, artifacts, calls = fixture(tmp_path, monkeypatch)
    checkpoint = artifacts["UNI", 2026, "token_selected"]
    path = checkpoint.parent / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if problem == "mode":
        config["visual_fusion"] = "legacy"
    elif problem == "hyperparameter":
        config["max_seq_length"] = 8
    elif problem == "data":
        test_path = Path(config["data_path"]) / "MNER-UNI_test.txt"
        test_path.write_bytes(b"changed evaluation split")
    else:
        checkpoint.unlink()
    write_json(path, config)
    with pytest.raises(ValueError):
        suite.main()
    assert not calls and not output.exists()


def test_failed_validation_replay_cannot_proceed_to_test(tmp_path, monkeypatch):
    output, artifacts, calls = fixture(tmp_path, monkeypatch)
    metrics = artifacts["MI", 42, "baseline"].parent / "metrics.csv"
    metrics.write_text("epoch,dev_f1\n1,0.1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reproduce recorded"):
        suite.main()
    assert calls == ["val/MI/seed42/baseline"]
    assert (output / "frozen_manifest.json").exists()
    assert not (output / "test_metrics.json").exists()


def test_changed_checkpoint_after_validation_is_rejected_before_test(tmp_path, monkeypatch):
    output, artifacts, calls = fixture(tmp_path, monkeypatch)
    original = suite.verify_plan

    def mutate(plan):
        artifacts["UNI", 2026, "token_selected"].write_bytes(b"changed after val")
        original(plan)

    monkeypatch.setattr(suite, "verify_plan", mutate)
    with pytest.raises(ValueError, match="SHA256 changed"):
        suite.main()
    assert len(calls) == 24 and all(name.startswith("val/") for name in calls)
    assert (output / "validation_metrics.json").exists()
    assert not (output / "test_metrics.json").exists()


def test_does_not_allow_seed_selection_or_report_overwrite(tmp_path, monkeypatch):
    output, _, calls = fixture(tmp_path, monkeypatch)
    args = list(sys.argv)
    monkeypatch.setattr(sys, "argv", [*args, "--seeds", "42"])
    with pytest.raises(SystemExit):
        suite.main()
    assert not calls and not output.exists()
    monkeypatch.setattr(sys, "argv", args)
    output.mkdir()
    with pytest.raises(FileExistsError):
        suite.main()
    assert not calls


def test_changed_evaluation_code_stops_before_test(tmp_path, monkeypatch):
    output, _, calls = fixture(tmp_path, monkeypatch)
    code = tmp_path / "evaluation_fixture.py"
    code.write_bytes(b"original evaluation code")
    prepare, verify = suite.prepare_plan, suite.verify_plan

    def bound(cli):
        plan = prepare(cli)
        plan["evaluation_code"][str(code)] = sha256_file(code)
        return plan

    def changed(plan):
        code.write_bytes(b"changed during validation")
        verify(plan)

    monkeypatch.setattr(suite, "prepare_plan", bound)
    monkeypatch.setattr(suite, "verify_plan", changed)
    with pytest.raises(ValueError, match="Evaluation code SHA256 changed"):
        suite.main()
    assert len(calls) == 24 and all(name.startswith("val/") for name in calls)
    assert not (output / "test_metrics.json").exists()
