from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_frozen_crf_test as final
from scripts.evaluate_crf_ensemble import decode_blend, sha256_file


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "data"
    source.mkdir()
    (source / "MNER-UNI_val.txt").write_text("validation", encoding="utf-8")
    (source / "MNER-UNI_test.txt").write_text("test", encoding="utf-8")
    root = tmp_path / "runs"
    artifacts = {}
    for seed in (42, 1234, 2026):
        artifacts[seed] = {}
        for name, arm in (("control", "duplicate_control"), ("baseline", "strict_baseline")):
            directory = root / f"{arm}_seed{seed}" / "uni_run"
            directory.mkdir(parents=True)
            checkpoint = directory / "best_model.pt"
            checkpoint.write_bytes(f"{seed}:{name}".encode())
            config = {"dataset": "UNI", "seed": seed, "max_seq_length": 4,
                      "data_path": str(source), "output_dir": str(directory.parent)}
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            # Deliberately different from test F1. Final scoring must not compare them.
            (directory / "metrics.csv").write_text("epoch,dev_f1\n3,0.1234\n", encoding="utf-8")
            artifacts[seed][name] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "config": config}
    val_hash = sha256_file(source / "MNER-UNI_val.txt")
    policy = {"schema_version": "crf-score-ensemble-policy-v1", "selection_split": "val",
              "weights": {"control": 0.7, "baseline": 0.3}, "validation_input_sha256": val_hash,
              "checkpoints": artifacts[1234]}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    confirmation = {
        "schema_version": "fixed-crf-ensemble-seed-confirmation-v1", "split": "val",
        "weights_searched": False, "selection_seed": 1234, "confirmation_seeds": [42, 2026],
        "source_policy_sha256": sha256_file(policy_path), "weights": policy["weights"],
        "input_sha256": val_hash,
        "per_seed": [{"seed": seed, "checkpoints": artifacts[seed]} for seed in (42, 2026)],
    }
    report_path = tmp_path / "confirmation.json"
    report_path.write_text(json.dumps(confirmation), encoding="utf-8")
    label_map = {"O": 0, "B-PER": 1, "X": 2}

    class TinyDataset:
        def __init__(self, args, file_name):
            assert file_name == "MNER-UNI_test.txt"
            self.processor = SimpleNamespace(get_label_mapping=lambda **kw: label_map)

        def __len__(self):
            return 2

    calls = []

    def collect(checkpoint, config, dataset, cli, labels, name):
        calls.append((config["seed"], name))
        emissions = torch.zeros(2, 4, 3)
        emissions[:, (0, 2), 2] = 4
        emissions[:, 1, 1] = 4
        if "duplicate_control" in str(checkpoint):
            emissions[1, 1, 1] = 0
            emissions[1, 1, 0] = 1
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
        state = {"start_transitions": torch.zeros(3), "end_transitions": torch.zeros(3),
                 "transitions": torch.zeros(3, 3)}
        return {"emissions": emissions, "mask": mask,
                "labels": torch.tensor([[2, 1, 2, 0], [2, 1, 2, 0]]), "crf": state,
                "native_paths": decode_blend(emissions, emissions, state, state, mask, 0)}

    monkeypatch.setattr(final, "MultimodalNerDataset", TinyDataset)
    monkeypatch.setattr(final, "_collect", collect)
    output = tmp_path / "test_metrics.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_frozen_crf_test.py", "--policy", str(policy_path),
                        "--confirmation-report", str(report_path), "--runs-root", str(root),
                        "--data-path", str(source), "--device", "cpu", "--output", str(output)])
    return output, policy_path, report_path, artifacts, calls


def test_final_test_reports_all_seeds_without_search_or_dev_f1_comparison(tmp_path, monkeypatch):
    output, _, _, _, calls = _fixture(tmp_path, monkeypatch)
    final.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert report["seeds"] == [42, 1234, 2026]
    assert report["weights"] == {"control": 0.7, "baseline": 0.3}
    assert report["weights_searched"] is False
    assert report["models_selected_using_test"] is False
    assert report["aggregate"]["seed_count"] == 3
    assert report["aggregate"]["ensemble"]["mean_f1"] == 1.0
    assert report["aggregate"]["ensemble"]["sample_std_f1"] == 0
    assert report["aggregate"]["positive_seeds_over_baseline"] == 0
    assert report["aggregate"]["positive_seeds_over_control"] == 3
    assert {seed for seed, _ in calls} == {42, 1234, 2026}
    assert len(calls) == 6


def test_does_not_allow_cherry_picking_test_seeds(tmp_path, monkeypatch):
    output, _, _, _, calls = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--seeds", "42", "2026"])
    with pytest.raises(ValueError, match="ALL confirmation seeds"):
        final.main()
    assert not calls and not output.exists()


def test_changed_policy_stops_before_test_inference(tmp_path, monkeypatch):
    output, policy_path, _, _, calls = _fixture(tmp_path, monkeypatch)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["weights"] = {"control": 0.8, "baseline": 0.2}
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="Policy SHA256"):
        final.main()
    assert not calls and not output.exists()


def test_changed_checkpoint_stops_before_test_inference(tmp_path, monkeypatch):
    output, _, _, artifacts, calls = _fixture(tmp_path, monkeypatch)
    checkpoint = tmp_path / "runs" / "strict_baseline_seed2026" / "uni_run" / "best_model.pt"
    assert str(checkpoint) == artifacts[2026]["baseline"]["path"]
    checkpoint.write_bytes(b"modified after confirmation")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        final.main()
    assert not calls and not output.exists()


def test_changed_test_data_in_one_arm_stops_before_inference(tmp_path, monkeypatch):
    output, _, _, artifacts, calls = _fixture(tmp_path, monkeypatch)
    copied = tmp_path / "changed_data"
    copied.mkdir()
    (copied / "MNER-UNI_val.txt").write_text("validation", encoding="utf-8")
    (copied / "MNER-UNI_test.txt").write_text("changed test", encoding="utf-8")
    config = artifacts[42]["control"]["config"]
    config = {**config, "data_path": str(copied)}
    path = tmp_path / "runs" / "duplicate_control_seed42" / "uni_run" / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Different MNER-UNI_test"):
        final.main()
    assert not calls and not output.exists()
