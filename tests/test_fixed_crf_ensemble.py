from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import confirm_crf_ensemble as confirm
from scripts.evaluate_crf_ensemble import decode_blend, sha256_file


def _policy(data_path, input_hash):
    return {
        "schema_version": "crf-score-ensemble-policy-v1", "selection_split": "val",
        "weights": {"control": 0.7, "baseline": 0.3},
        "validation_input_sha256": input_hash,
        "checkpoints": {
            name: {"config": {"dataset": "UNI", "seed": 1234, "max_seq_length": 4,
                              "data_path": str(data_path), "output_dir": "selection"}}
            for name in ("control", "baseline")
        },
    }


@pytest.mark.parametrize("weights", [
    {"control": 0.7, "llm": 0.3},
    {"control": 0.7, "baseline": 0.2},
    {"control": True, "baseline": False},
    {"control": 0.7, "baseline": float("nan")},
])
def test_rejects_non_frozen_or_invalid_two_model_weights(tmp_path, weights):
    path = tmp_path / "policy.json"
    policy = _policy(tmp_path, "digest")
    policy["weights"] = weights
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        confirm.read_policy(path)


def test_checkpoint_discovery_refuses_to_select_between_repeated_runs(tmp_path):
    directory = tmp_path / "strict_baseline_seed42"
    for name in ("uni_first", "uni_second"):
        run = directory / name
        run.mkdir(parents=True)
        (run / "best_model.pt").write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="exactly one checkpoint"):
        confirm.find_checkpoint(tmp_path, "strict_baseline", 42)


def _confirmation_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    val_path = source / "MNER-UNI_val.txt"
    val_path.write_text("same validation", encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy(source, sha256_file(val_path))), encoding="utf-8")
    root = tmp_path / "runs"
    for seed in (42, 2026):
        for arm in ("strict_baseline", "duplicate_control"):
            run = root / f"{arm}_seed{seed}" / "uni_run"
            run.mkdir(parents=True)
            (run / "best_model.pt").write_bytes(f"{arm}_{seed}".encode())
            config = {"dataset": "UNI", "seed": seed, "max_seq_length": 4,
                      "data_path": str(source), "output_dir": str(run.parent)}
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            recorded = "1.0000" if arm == "strict_baseline" else "0.6667"
            (run / "metrics.csv").write_text(f"epoch,dev_f1\n3,{recorded}\n", encoding="utf-8")
    label_map = {"O": 0, "B-PER": 1, "X": 2}

    class TinyDataset:
        def __init__(self, args, file_name):
            assert file_name == "MNER-UNI_val.txt"
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
        targets = torch.tensor([[2, 1, 2, 0], [2, 1, 2, 0]])
        state = {"start_transitions": torch.zeros(3), "end_transitions": torch.zeros(3),
                 "transitions": torch.zeros(3, 3)}
        native = decode_blend(emissions, emissions, state, state, mask, 0)
        return {"emissions": emissions, "mask": mask, "labels": targets,
                "native_paths": native, "crf": state}

    monkeypatch.setattr(confirm, "MultimodalNerDataset", TinyDataset)
    monkeypatch.setattr(confirm, "_collect", collect)
    output = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["confirm_crf_ensemble.py", "--policy", str(policy_path),
                        "--runs-root", str(root), "--data-path", str(source), "--seeds", "42", "2026",
                        "--device", "cpu", "--output", str(output)])
    return output, policy_path, source, calls


def test_confirms_only_frozen_weights_and_compares_with_stronger_single_model(tmp_path, monkeypatch):
    output, _, _, calls = _confirmation_fixture(tmp_path, monkeypatch)
    confirm.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["weights"] == {"control": 0.7, "baseline": 0.3}
    assert result["weights_searched"] is False
    assert result["confirmation_seeds"] == [42, 2026]
    assert result["selection_seed"] == 1234
    assert len(calls) == 4
    assert result["aggregate"]["mean_ensemble_f1"] == 1.0
    assert result["aggregate"]["positive_seeds_over_best_single"] == 0
    assert all(row["gain_over_control"] > 0 for row in result["per_seed"])
    assert all(row["gain_over_best_single"] == 0 for row in result["per_seed"])
    assert result["llm_api_requests"] == 0


def test_changed_validation_hash_stops_before_inference(tmp_path, monkeypatch):
    output, _, source, calls = _confirmation_fixture(tmp_path, monkeypatch)
    (source / "MNER-UNI_val.txt").write_text("different validation", encoding="utf-8")
    with pytest.raises(ValueError, match="Validation SHA256"):
        confirm.main()
    assert not calls
    assert not output.exists()


def test_changed_training_protocol_stops_before_inference(tmp_path, monkeypatch):
    _, _, _, calls = _confirmation_fixture(tmp_path, monkeypatch)
    config_path = tmp_path / "runs" / "strict_baseline_seed2026" / "uni_run" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_seq_length"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="max_seq_length"):
        confirm.main()
    assert not calls


def test_selection_seed_is_not_counted_as_fresh_confirmation(tmp_path, monkeypatch):
    _, _, _, calls = _confirmation_fixture(tmp_path, monkeypatch)
    args = list(sys.argv)
    args[args.index("42")] = "1234"
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="weight-selection seed"):
        confirm.main()
    assert not calls
