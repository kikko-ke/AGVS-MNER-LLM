from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torchcrf import CRF

from scripts import evaluate_crf_ensemble as ensemble


def _state(crf):
    return {key: getattr(crf, key).detach().clone() for key in ensemble.CRF_KEYS}


def _sequence_score(emissions, state, path):
    result = state["start_transitions"][path[0]] + state["end_transitions"][path[-1]]
    for position, tag in enumerate(path):
        result = result + emissions[position, tag]
        if position:
            result = result + state["transitions"][path[position - 1], tag]
    return float(result)


def test_blend_matches_exhaustive_weighted_complete_crf_score():
    torch.manual_seed(17)
    left, right = CRF(3, batch_first=True), CRF(3, batch_first=True)
    with torch.no_grad():
        for crf in (left, right):
            for parameter in crf.parameters():
                parameter.uniform_(-3, 3)
    first, second = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    left_state, right_state = _state(left), _state(right)
    alpha = 0.3
    actual = ensemble.decode_blend(first, second, left_state, right_state, mask, alpha, batch_size=1)
    for index, length in enumerate(mask.sum(1).tolist()):
        possible = list(itertools.product(range(3), repeat=length))
        expected = max(possible, key=lambda path: (
            (1 - alpha) * _sequence_score(first[index], left_state, path)
            + alpha * _sequence_score(second[index], right_state, path)
        ))
        assert actual[index] == list(expected)


def test_endpoints_reproduce_native_paths_without_mutating_parameters():
    torch.manual_seed(5)
    first, second = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    left, right = CRF(3, batch_first=True), CRF(3, batch_first=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    left_state, right_state = _state(left), _state(right)
    assert ensemble.decode_blend(first, second, left_state, right_state, mask, 0) == left.decode(first, mask)
    assert ensemble.decode_blend(first, second, left_state, right_state, mask, 1) == right.decode(second, mask)
    for key in ensemble.CRF_KEYS:
        assert torch.equal(left_state[key], getattr(left, key))
        assert torch.equal(right_state[key], getattr(right, key))
    for alpha in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            ensemble.decode_blend(first, second, left_state, right_state, mask, alpha)


def test_llm_must_beat_the_best_non_llm_ensemble_not_just_control():
    rows = [{"name": "control", "member": "control", "metrics": {"f1": 0.80}},
            {"name": "ordinary", "member": "baseline", "metrics": {"f1": 0.83}},
            {"name": "llm", "member": "llm", "metrics": {"f1": 0.82}}]
    comparison = ensemble.compare_policies(rows)
    assert comparison["llm_gain_over_control"] > 0
    assert comparison["llm_gain_over_best_without_llm"] < 0
    assert not comparison["llm_increment_observed_on_validation"]


def test_config_check_rejects_protocol_drift():
    configs = {
        name: {"seed": 1234, "max_seq_length": 64, "data_path": name, "output_dir": name}
        for name in ("control", "baseline", "llm")
    }
    ensemble.validate_configs(configs)
    configs["llm"]["seed"] = 42
    with pytest.raises(ValueError, match="seed"):
        ensemble.validate_configs(configs)


def test_pipeline_only_reads_validation_and_writes_fixed_seven_policies(tmp_path, monkeypatch):
    checkpoints = {}
    for name in ("baseline", "control", "llm"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "MNER-UNI_val.txt").write_text("validation only", encoding="utf-8")
        config = {"dataset": "UNI", "seed": 1234, "max_seq_length": 4,
                  "data_path": str(directory), "output_dir": str(directory)}
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        checkpoint = directory / "best_model.pt"
        checkpoint.write_bytes(name.encode())
        checkpoints[name] = str(checkpoint)

    label_map = {"O": 0, "B-PER": 1, "X": 2}

    class TinyDataset:
        def __init__(self, args, file_name):
            assert file_name == "MNER-UNI_val.txt"
            self.processor = SimpleNamespace(get_label_mapping=lambda **kw: label_map)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return (torch.tensor([index, 10, 11, 0]), torch.zeros(4, dtype=torch.long),
                    torch.tensor([1, 1, 1, 0]), torch.zeros(4, 3, 1, 1),
                    torch.tensor([2, 1, 2, 0]), torch.tensor([1, 0, 0, 0]))

    class TinyModel(torch.nn.Module):
        def __init__(self, labels, args):
            super().__init__()
            self.crf = CRF(len(labels), batch_first=True)
            with torch.no_grad():
                for parameter in self.crf.parameters():
                    parameter.zero_()
            self.calls = 0

        def forward(self, input_ids, attention_mask, labels=None, **kwargs):
            assert labels is None
            self.calls += 1
            emissions = torch.zeros(input_ids.shape + (3,))
            emissions[:, 0, 2] = 3
            emissions[:, 1, 1] = 3
            emissions[:, 2, 2] = 3
            return (self.crf.decode(emissions, attention_mask.bool()), None,
                    {"token_emissions": emissions})

    monkeypatch.setattr(ensemble, "MultimodalNerDataset", TinyDataset)
    monkeypatch.setattr(ensemble, "AgvsMnerModel", TinyModel)
    monkeypatch.setattr(ensemble, "load_compatible_model_checkpoint", lambda *a, **kw: [])
    output = tmp_path / "results" / "val.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_crf_ensemble.py", "--baseline", checkpoints["baseline"],
                        "--control", checkpoints["control"], "--llm", checkpoints["llm"],
                        "--device", "cpu", "--batch-size", "1", "--output", str(output)])
    ensemble.main()
    data = json.loads(output.read_text(encoding="utf-8"))
    policy = json.loads(output.with_name("val.policy.json").read_text(encoding="utf-8"))
    assert len(data["policies"]) == 7
    assert all(row["metrics"]["f1"] == 1.0 for row in data["policies"])
    assert data["llm_api_requests"] == 0
    assert data["split"] == "val"
    assert data["complementarity"]["different_entity_sets"] == 0
    assert policy["weights"] == {"control": 1.0}
    assert policy["independent_confirmation_required"]


def test_script_imports_directly_and_rejects_test_grid():
    script = Path(ensemble.__file__)
    result = subprocess.run([sys.executable, "-I", str(script), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--baseline" in result.stdout
    with pytest.raises(SystemExit):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sys, "argv", [str(script), "--baseline", "b", "--control", "c",
                          "--llm", "l", "--split", "test", "--output", "unused.json"])
            ensemble.parse_args()
