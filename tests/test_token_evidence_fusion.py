from __future__ import annotations

from types import SimpleNamespace
import json
import re
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from models import agvs_mner
from models.token_evidence_fusion import TokenEvidenceFusion
from modules.trainer import load_compatible_model_checkpoint


def test_different_tokens_retrieve_different_patches():
    module = TokenEvidenceFusion(4, 4, topk=1, anchor_weight=0, dropout=0).eval()
    with torch.no_grad():
        module.query.weight.copy_(torch.eye(4))
        module.key.weight.copy_(torch.eye(4))
    text = torch.tensor([[[2., -2., 0., 0.], [-2., 2., 0., 0.]]])
    patches = torch.cat([text, torch.tensor([[[0., 0., 2., -2.]]])], dim=1)
    delta, diagnostics = module(text, patches, torch.ones(1, 3), torch.zeros(1, 3), torch.ones(1, 2))
    assert diagnostics["selected_indices"].tolist() == [[[0], [1]]]
    assert not torch.allclose(delta[:, 0], delta[:, 1])


def test_selected_scores_train_retrieval_query_and_key():
    torch.manual_seed(19)
    module = TokenEvidenceFusion(8, 4, topk=2, dropout=0)
    text, patches = torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 5, 8, requires_grad=True)
    delta, _ = module(text, patches, torch.ones(2, 5), torch.zeros(2, 5), torch.ones(2, 3))
    delta.square().sum().backward()
    for parameter in (module.query.weight, module.key.weight, module.value.weight, module.gate.weight,
                      module.null_score.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_invalid_pixels_and_padding_cannot_change_valid_evidence():
    torch.manual_seed(8)
    module = TokenEvidenceFusion(8, 4, topk=4, dropout=0).eval()
    text, patches = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
    text_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    anchor = torch.randn(2, 5)
    first, info = module(text, patches, mask, anchor, text_mask)
    changed = patches.masked_fill(~mask.bool().unsqueeze(-1), float("nan"))
    changed_anchor = anchor.masked_fill(~mask.bool(), float("inf"))
    second, other = module(text, changed, mask, changed_anchor, text_mask)
    assert torch.equal(first, second) and torch.isfinite(second).all()
    assert torch.equal(first[1], torch.zeros_like(first[1]))
    assert info["null_probability"][1].tolist() == [1, 1, 1]
    assert torch.equal(first[0, 2], torch.zeros(8))
    assert torch.equal(info["null_probability"], other["null_probability"])


class FakeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(200, 768)
        self.pooler = nn.Linear(768, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=states, pooler_output=torch.tanh(self.pooler(states[:, 0])))


class FakeVit(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)

    def forward(self, pixel_values):
        features = self.projection(pixel_values.mean(dim=(-1, -2)))
        positions = torch.arange(197, device=features.device).view(1, -1, 1)
        frequencies = torch.arange(1, 769, device=features.device).view(1, 1, -1) / 768
        states = features.unsqueeze(1) + 0.1 * torch.sin(positions * frequencies)
        return SimpleNamespace(last_hidden_state=states)


@pytest.fixture
def fake_encoders(monkeypatch):
    monkeypatch.setattr(agvs_mner.BertModel, "from_pretrained", lambda *a, **kw: FakeBert())
    monkeypatch.setattr(agvs_mner.ViTModel, "from_pretrained", lambda *a, **kw: FakeVit())
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def _model(mode=None):
    values = dict(bert_model="local-test", vit_model="local-test", image_dropout_prob=0,
                  output_dropout=0, topk_img_patches=3)
    if mode is not None:
        values["visual_fusion"] = mode
    labels = ["O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X"]
    return agvs_mner.AgvsMnerModel(labels, SimpleNamespace(**values)).eval()


def _inputs():
    return dict(input_ids=torch.tensor([[101, 3, 4, 102, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 1, 0]]), token_type_ids=torch.zeros(1, 5, dtype=torch.long),
                image_pixel_values=torch.randn(1, 4, 3, 2, 2), image_mask=torch.tensor([[1, 0, 0, 0]]),
                labels=torch.tensor([[9, 3, 0, 9, 0]]), return_features=True)


def test_default_legacy_checkpoint_and_explicit_legacy_are_identical(fake_encoders, tmp_path):
    torch.manual_seed(1234)
    default = _model()
    torch.manual_seed(1234)
    explicit = _model("legacy")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert all(torch.equal(value, explicit.state_dict()[key]) for key, value in default.state_dict().items())
    checkpoint = tmp_path / "legacy.pt"
    torch.save(default.state_dict(), checkpoint)
    assert load_compatible_model_checkpoint(explicit, checkpoint) == []
    inputs = _inputs()
    with torch.no_grad():
        left, right = default(**inputs), explicit(**inputs)
    assert left[0] == right[0] and torch.equal(left[1], right[1])
    assert torch.equal(left[2]["token_emissions"], right[2]["token_emissions"])


def test_new_component_preserves_shared_initialisation_and_sampler_rng(fake_encoders):
    torch.manual_seed(77)
    legacy = _model("legacy")
    expected = torch.rand(6)
    torch.manual_seed(77)
    updated = _model("token_evidence")
    actual = torch.rand(6)
    assert torch.equal(expected, actual)
    assert all(torch.equal(value, updated.state_dict()[key]) for key, value in legacy.state_dict().items())
    assert updated.token_evidence_fusion is not None


def test_full_model_has_finite_loss_and_exact_no_image_fallback(fake_encoders):
    torch.manual_seed(21)
    model = _model("token_evidence")
    inputs = _inputs()
    result = model(**inputs)
    assert torch.isfinite(result[1]) and len(result[0][0]) == 4
    result[1].backward()
    assert model.token_evidence_fusion.query.weight.grad.abs().sum() > 0
    assert "visual_null_probability" in model.last_loss_components
    inputs["image_mask"].zero_()
    with torch.no_grad():
        first = model(**inputs)
        inputs["image_pixel_values"] = torch.randn_like(inputs["image_pixel_values"]) * 100
        second = model(**inputs)
    assert torch.equal(first[2]["token_emissions"], second[2]["token_emissions"])
    assert model.last_loss_components["visual_null_probability"] == 1


def test_new_fusion_has_no_count_divided_by_four_scaling(fake_encoders):
    class ConstantFusion(nn.Module):
        def forward(self, text, patches, patch_mask, anchor_scores, text_mask):
            return torch.ones_like(text), {"null_probability": text.new_zeros(text.shape[:2]),
                                           "gate_mean": text.new_ones(text.shape[:2])}

    model = _model("token_evidence")
    model.token_evidence_fusion = ConstantFusion()
    inputs = _inputs()
    with torch.no_grad():
        one = model(**inputs)[2]["token_emissions"]
        inputs["image_mask"].fill_(1)
        four = model(**inputs)[2]["token_emissions"]
    assert torch.equal(one, four)


def test_legacy_weights_cannot_silently_initialise_an_untrained_new_component(fake_encoders, tmp_path):
    legacy = _model("legacy")
    path = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), path)
    new = _model("token_evidence")
    with pytest.raises(RuntimeError, match="token_evidence_fusion"):
        load_compatible_model_checkpoint(new, path)


@pytest.mark.parametrize("dataset", ["MI", "UNI"])
def test_training_cli_accepts_component_without_changing_the_default(monkeypatch, dataset):
    from run import parse_args
    monkeypatch.setattr(sys, "argv", ["run.py", "--dataset", dataset])
    assert parse_args().visual_fusion == "legacy"
    monkeypatch.setattr(sys, "argv", ["run.py", "--dataset", dataset, "--visual-fusion", "token_evidence", "--skip-test"])
    args = parse_args()
    assert args.visual_fusion == "token_evidence" and args.skip_test


def test_pilot_report_separates_component_and_selection_gains(tmp_path, monkeypatch):
    script = Path(agvs_mner.__file__).parents[1] / "scripts" / "train_token_evidence_pilot.sh"
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script.read_text(encoding="utf-8"), flags=re.DOTALL)
    for block in blocks:
        compile(block, "pilot embedded Python", "exec")
    original, selected = tmp_path / "original", tmp_path / "selected"
    for data in (original, selected):
        data.mkdir()
        (data / "MNER-MI_train.txt").write_text(data.name, encoding="utf-8")
    cases = [
        ("old/strict_baseline_seed1234", original, 0.75), ("old/llm_selected_seed1234", selected, 0.76),
        ("outputs_component_v1/mi_token_evidence_original_seed1234", original, 0.78),
        ("outputs_component_v1/mi_token_evidence_selected_seed1234", selected, 0.77),
    ]
    for name, data, value in cases:
        run = tmp_path / name / "mi_run"
        run.mkdir(parents=True)
        (run / "best_model.pt").write_bytes(b"test fixture, not a trained model")
        (run / "metrics.csv").write_text(f"epoch,dev_f1\n1,{value}\n", encoding="utf-8")
        (run / "config.json").write_text(json.dumps({"data_path": str(data)}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["-", "MI", "old", "llm_selected"])
    exec(blocks[-1], {"__name__": "__main__"})
    reports = list((tmp_path / "outputs_component_v1").glob("mi_pilot_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["component_gain_original"] == pytest.approx(0.03)
    assert report["component_gain_selected"] == pytest.approx(0.01)
    assert report["selected_minus_original_new_component"] == pytest.approx(-0.01)
    assert report["performance_confirmed"] is False and report["split"] == "val"
