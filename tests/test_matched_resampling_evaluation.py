from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_matched_resampling as evaluation
from scripts.evaluate_crf_ensemble import decode_blend, sha256_file


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fixture(tmp_path, monkeypatch):
    source, selected, matched = (tmp_path / name for name in ("original", "selected", "matched"))
    lines = [repr({"text": [str(i)], "label": ["O"], "images": []}) for i in range(4)]
    for directory, extra in ((source, []), (selected, [lines[0]]), (matched, [lines[1]])):
        directory.mkdir()
        (directory / "MNER-UNI_train.txt").write_text("\n".join(lines + extra) + "\n", encoding="utf-8")
        for split in ("val", "test"):
            (directory / f"MNER-UNI_{split}.txt").write_bytes(split.encode())
    metadata = tmp_path / "candidate_metadata.json"
    _write_json(metadata, {"selected": [{"sample_id": i} for i in range(4)]})
    ref_manifest = selected / "augmentation_manifest.json"
    original_hash = sha256_file(source / "MNER-UNI_train.txt")
    _write_json(ref_manifest, {"arm": "duplicate_control", "source_sample_ids": [0],
                              "original_examples": 4, "accepted_augmentations": 1,
                              "source_train_sha256": original_hash})
    manifest_path = matched / "matched_resampling_manifest.json"
    _write_json(manifest_path, {
        "schema_version": "feature-matched-resampling-v1", "arm": "feature_matched_no_llm",
        "ready_for_training": True, "candidate_scoring_uses_llm_outputs": False,
        "selection_uses_validation_or_test_labels": False,
        "balance_check": {"passed": True, "max_abs_difference_in_pool_sd_units": 0.01},
        "overlap_check": {"passed": True, "fraction": 0, "count": 0},
        "source_sample_ids": [1], "reference_source_sample_ids": [0],
        "added_examples": 1, "original_examples": 4, "total_examples": 5, "candidate_count": 4,
        "sampling_seed": 20261031, "source_train_sha256": original_hash,
        "reference_train_sha256": sha256_file(selected / "MNER-UNI_train.txt"),
        "candidate_metadata": str(metadata), "candidate_metadata_sha256": sha256_file(metadata),
        "reference_manifest": str(ref_manifest), "reference_manifest_sha256": sha256_file(ref_manifest),
        "output_files_sha256": {p.name: sha256_file(p) for p in matched.glob("*.txt")},
    })
    old_root, new_root = tmp_path / "v61", tmp_path / "v62"
    artifacts = {}
    for seed in evaluation.SEEDS:
        artifacts[seed] = {}
        for name, root, directory, data in (
            ("baseline", old_root, "strict_baseline", source),
            ("control", old_root, "duplicate_control", selected),
            ("matched", new_root, "matched_no_llm", matched),
        ):
            run = root / f"{directory}_seed{seed}" / "uni_run"
            run.mkdir(parents=True)
            checkpoint = run / "best_model.pt"
            checkpoint.write_bytes(f"{seed}:{name}".encode())
            config = {"dataset": "UNI", "seed": seed, "data_path": str(data),
                      "output_dir": str(run), "max_seq_length": 4, "skip_test": True}
            _write_json(run / "config.json", config)
            # Test scores must NOT be compared with best validation scores.
            recorded_dev = "0.0000" if name == "matched" else "0.1234"
            (run / "metrics.csv").write_text(f"epoch,dev_f1\n3,{recorded_dev}\n", encoding="utf-8")
            artifacts[seed][name] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "config": config}
    policy_path, confirmation_path = tmp_path / "policy.json", tmp_path / "confirmation.json"
    weights = {"control": 0.7, "baseline": 0.3}
    val_hash = sha256_file(source / "MNER-UNI_val.txt")
    _write_json(policy_path, {
        "schema_version": "crf-score-ensemble-policy-v1", "selection_split": "val", "weights": weights,
        "validation_input_sha256": val_hash,
        "checkpoints": {name: artifacts[1234][name] for name in ("baseline", "control")},
    })
    _write_json(confirmation_path, {
        "schema_version": "fixed-crf-ensemble-seed-confirmation-v1", "split": "val", "weights_searched": False,
        "source_policy_sha256": sha256_file(policy_path), "weights": weights, "input_sha256": val_hash,
        "selection_seed": 1234, "confirmation_seeds": [42, 2026],
        "per_seed": [{"seed": seed, "checkpoints": {name: artifacts[seed][name] for name in ("baseline", "control")}}
                     for seed in (42, 2026)],
    })
    single = {"f1": 2 / 3, "precision": 0.5, "recall": 1.0}
    correct = {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    report_path = tmp_path / "reference_test.json"
    _write_json(report_path, {
        "schema_version": "frozen-crf-ensemble-final-test-v1", "split": "test", "seeds": list(evaluation.SEEDS),
        "weights_searched": False, "weights": weights, "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path), "confirmation_report": str(confirmation_path),
        "confirmation_report_sha256": sha256_file(confirmation_path), "validation_input_sha256": val_hash,
        "input_sha256": sha256_file(source / "MNER-UNI_test.txt"),
        "per_seed": [{"seed": seed, "checkpoints": {name: artifacts[seed][name] for name in ("baseline", "control")},
                      "metrics": {"baseline": single, "control": correct, "ensemble": correct}}
                     for seed in evaluation.SEEDS],
    })
    label_map = {"O": 0, "B-PER": 1, "X": 2}

    class TinyDataset:
        def __init__(self, args, file_name):
            assert file_name in {"MNER-UNI_val.txt", "MNER-UNI_test.txt"}
            self.processor = SimpleNamespace(get_label_mapping=lambda **kw: label_map)

        def __len__(self):
            return 2

    calls = []

    def collect(checkpoint, config, dataset, cli, labels, name):
        calls.append(name)
        emissions = torch.zeros(2, 4, 3)
        emissions[:, (0, 2), 2] = 4
        emissions[:, 1, 0] = 4
        if "duplicate_control" in str(checkpoint):
            emissions[0, 1, 0] = 0
            emissions[0, 1, 1] = 4
        elif "strict_baseline" in str(checkpoint):
            emissions[:, 1, 0] = 0
            emissions[:, 1, 1] = 4
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
        state = {"start_transitions": torch.zeros(3), "end_transitions": torch.zeros(3), "transitions": torch.zeros(3, 3)}
        return {"emissions": emissions, "mask": mask,
                "labels": torch.tensor([[2, 1, 2, 0], [2, 0, 2, 0]]), "crf": state,
                "native_paths": decode_blend(emissions, emissions, state, state, mask, 0)}

    monkeypatch.setattr(evaluation, "MultimodalNerDataset", TinyDataset)
    monkeypatch.setattr(evaluation, "_collect", collect)
    output = tmp_path / "output.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_matched_resampling.py", "--reference-report", str(report_path),
                        "--matched-manifest", str(manifest_path), "--matched-runs-root", str(new_root),
                        "--data-path", str(source), "--device", "cpu", "--output", str(output)])
    return output, report_path, manifest_path, artifacts, calls


def test_reports_all_seeds_primary_and_same_weight_secondary_without_search(tmp_path, monkeypatch):
    output, _, _, _, calls = _fixture(tmp_path, monkeypatch)
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 12
    assert report["seeds"] == [42, 1234, 2026]
    assert report["post_hoc_analysis"] is True and report["independent_holdout"] is False
    assert report["llm_api_requests"] == 0 and report["weights_searched"] is False
    assert report["weights"] == {"control": 0.7, "baseline": 0.3}
    assert report["aggregate"]["primary_llm_selected_minus_matched"]["mean_f1_difference"] == 1
    assert report["aggregate"]["primary_llm_selected_minus_matched"]["positive_seeds"] == 3
    assert all(set(row["metrics"]) == set(evaluation.METHODS) for row in report["per_seed"])
    assert len(report["matched_validation_replays"]) == 3


def test_cannot_omit_a_seed_from_reference(tmp_path, monkeypatch):
    output, reference, _, _, calls = _fixture(tmp_path, monkeypatch)
    data = json.loads(reference.read_text(encoding="utf-8"))
    data["seeds"] = [42, 1234]
    _write_json(reference, data)
    with pytest.raises(ValueError, match="all three fixed seeds"):
        evaluation.main()
    assert not calls and not output.exists()


def test_changed_reference_checkpoint_is_rejected_before_forward(tmp_path, monkeypatch):
    output, _, _, artifacts, calls = _fixture(tmp_path, monkeypatch)
    Path(artifacts[2026]["control"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        evaluation.main()
    assert not calls and not output.exists()


def test_config_mismatch_in_last_seed_stops_before_any_forward(tmp_path, monkeypatch):
    output, _, _, artifacts, calls = _fixture(tmp_path, monkeypatch)
    path = Path(artifacts[2026]["matched"]["path"]).parent / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["max_seq_length"] = 8
    _write_json(path, config)
    with pytest.raises(ValueError, match="training config differs"):
        evaluation.main()
    assert not calls and not output.exists()


@pytest.mark.parametrize("change", ["balance", "overlap", "matched_file", "test_file", "source_ids"])
def test_failed_or_modified_data_audit_stops_before_inference(tmp_path, monkeypatch, change):
    output, _, manifest_path, _, calls = _fixture(tmp_path, monkeypatch)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if change == "balance":
        data["balance_check"]["max_abs_difference_in_pool_sd_units"] = 0.11
    elif change == "overlap":
        data["overlap_check"]["fraction"] = 0.5
    elif change == "source_ids":
        data["source_sample_ids"] = [2]
    elif change == "matched_file":
        (manifest_path.parent / "MNER-UNI_train.txt").write_bytes(b"changed")
    else:
        (tmp_path / "original" / "MNER-UNI_test.txt").write_bytes(b"changed")
    _write_json(manifest_path, data)
    with pytest.raises(ValueError):
        evaluation.main()
    assert not calls and not output.exists()


def test_cannot_change_frozen_weights_via_reference_report(tmp_path, monkeypatch):
    output, reference, _, _, calls = _fixture(tmp_path, monkeypatch)
    data = json.loads(reference.read_text(encoding="utf-8"))
    data["weights"] = {"control": 0.6, "baseline": 0.4}
    _write_json(reference, data)
    with pytest.raises(ValueError, match="Frozen weights/seeds differ"):
        evaluation.main()
    assert not calls and not output.exists()


def test_prior_metrics_must_reproduce_and_output_is_preserved(tmp_path, monkeypatch):
    output, reference, _, _, calls = _fixture(tmp_path, monkeypatch)
    data = json.loads(reference.read_text(encoding="utf-8"))
    data["per_seed"][0]["metrics"]["baseline"]["f1"] = 0.1
    _write_json(reference, data)
    with pytest.raises(RuntimeError, match="metrics do not reproduce"):
        evaluation.main()
    assert calls and not output.exists()
    output.write_bytes(b"preserve")
    calls.clear()
    with pytest.raises(FileExistsError):
        evaluation.main()
    assert output.read_bytes() == b"preserve" and not calls


def test_matched_checkpoint_must_reproduce_recorded_dev_before_test(tmp_path, monkeypatch):
    output, _, _, artifacts, calls = _fixture(tmp_path, monkeypatch)
    metrics = Path(artifacts[42]["matched"]["path"]).parent / "metrics.csv"
    metrics.write_text("epoch,dev_f1\n3,0.5555\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reproduce recorded"):
        evaluation.main()
    assert calls == ["val/seed42/matched"] and not output.exists()
