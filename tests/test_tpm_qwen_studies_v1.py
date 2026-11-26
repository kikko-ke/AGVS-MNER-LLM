"""Cached-fusion endpoints, dev-only selection, paired 7B features and download checks."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import run_tpm_qwen_studies_v1 as study
from scripts import download_qwen7b_v1 as download
from modules.tpm_llm_v1 import pipeline as old
from modules.tpm_llm_v1.common import BIO, TAGS, atomic_json, metrics, save_rows
from modules.tpm_qwen_visual_v1.features import project_words
from test_tpm_qwen_visual_v1 import tiny_qwen


def native_fixture(tmp_path, monkeypatch, rows):
    from torchcrf import CRF
    decoder = CRF(10, batch_first=True)
    for parameter in decoder.parameters():
        parameter.data.zero_()
    emissions = torch.zeros(len(rows), 3, 10)
    emissions[:, [0, 2], 9] = 5
    emissions[:, 1, 0] = 2
    emissions[:, 1, TAGS.index("B-LOC")] = 1
    native = {"source_ids": [r["source_id"] for r in rows], "emissions": emissions,
              "mask": torch.ones(len(rows), 3, dtype=torch.bool), "crf": decoder.state_dict(),
              "mappings": [{"positions": [1], "word_indices": [0]} for _ in rows]}
    file = tmp_path / "native.pt"
    old.save_torch(file, native)
    monkeypatch.setattr(study, "native", lambda *args: (file, native))
    return native


def test_cached_fusion_has_exact_text_and_multimodal_endpoints():
    text, mm = torch.randn(2, 4, 10), torch.randn(2, 4, 10)
    assert study.mix_emissions(text, mm, 0) is text
    assert study.mix_emissions(text, mm, 1) is mm
    torch.testing.assert_close(study.mix_emissions(text, mm, 0.25), text * 0.75 + mm * 0.25)
    with pytest.raises(ValueError):
        study.mix_emissions(text, mm, -0.1)


def test_fusion_alpha_is_selected_on_dev_only_with_smallest_tie(tmp_path, monkeypatch):
    from modules.tpm_llm_v1.features import fuse
    rows = [{"source_id": "dev:0", "tokens": ["London"], "labels": ["B-LOC"]}]
    native = native_fixture(tmp_path, monkeypatch, rows)
    def source(p, role, labels=True):
        assert role == "dev"
        return rows
    monkeypatch.setattr(study, "source_rows", source)
    scores = torch.zeros(1, 9)
    scores[0, BIO.index("B-LOC")] = 9
    branches = {"text": native["emissions"], "multimodal": fuse(
        native["emissions"], rows, native["mappings"], [scores], [torch.tensor([True])], 1)}
    monkeypatch.setattr(study, "cached_3b", lambda *args: (branches, []))
    p = {"protocol_id": "fixture", "alpha_grid": [0, 0.25, 0.5, 0.75, 1]}
    study.calibrate_fusion(tmp_path, p)
    result = json.loads((tmp_path / "calibration/dev_report.json").read_text())
    assert result["selected_alpha"] == 0.25
    assert result["selected"]["metrics"]["f1"] == 1
    assert result["test_metrics_used_for_selection"] is False


def test_7b_text_path_uses_native_decoder_and_the_same_fixed_projection():
    model, _ = tiny_qwen()
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "New York"
            return {"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "offset_mapping": torch.tensor([[[0, 3], [4, 8]]])}
    extractor = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=Tokenizer()),
                                projection=None, config={"projection_seed": 1234})
    row = {"source_id": "train:0", "tokens": ["New", "York"]}
    actual, valid = study.text_row(extractor, row, device="cpu")
    with torch.inference_mode():
        hidden = model.get_decoder()(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.ones(1, 2),
                                     use_cache=False, return_dict=True).last_hidden_state
        expected, expected_valid, _ = project_words(hidden, [0, 1], 1234)
    assert torch.equal(actual, expected) and torch.equal(valid, expected_valid)
    assert actual.shape == (2, 768)


def test_7b_test_features_wait_for_dev_calibration(tmp_path):
    p = {"protocol_id": "fixture"}
    old.seal(tmp_path / "smoke/completed.json", p, [])
    with pytest.raises(ValueError, match="Freeze 7B development"):
        study.features_7b(tmp_path, p, test=True)


def test_paired_7b_feature_shards_resume_without_swapping_branches(tmp_path, monkeypatch):
    from modules.tpm_qwen_visual_v1 import features as vf
    source, output = tmp_path / "source", tmp_path / "study"
    atomic_json(source / "data/image_audit.json", {})
    rows = [{"source_id": f"train:{i}", "tokens": ["word"], "images": []} for i in range(3)]
    p = {"protocol_id": "fixture", "visual_run": str(source), "config": {"shard_size": 2}}
    old.seal(output / "smoke/completed.json", p, [])
    monkeypatch.setattr(study, "parent", lambda *a: ({}, source, {}))
    monkeypatch.setattr(study, "source_rows", lambda *a, **k: rows)
    monkeypatch.setattr(study, "runtime", lambda: {"torch": "fixture"})
    monkeypatch.setattr(study, "text_row", lambda *a: (torch.ones(1, 768).half(), torch.tensor([True])))
    seen, fail = [], [True]
    class Extractor:
        def __init__(self, config):
            pass
        def row(self, row, audit):
            if fail[0] and row["source_id"] == "train:2":
                raise RuntimeError("interrupted extraction")
            seen.append(row["source_id"])
            return torch.full((1, 768), 2.).half(), torch.tensor([True]), {
                "source_id": row["source_id"], "valid_images": 0, "missing": [], "sequence_tokens": 1}
        def close(self):
            pass
    monkeypatch.setattr(vf, "FrozenVisualExtractor", Extractor)
    with pytest.raises(RuntimeError, match="interrupted extraction"):
        study.features_7b(output, p)
    fail[0] = False
    study.features_7b(output, p)
    _, bundle = study.checked_features(output, "train", p)
    assert bundle["text"]["source_ids"] == bundle["multimodal"]["source_ids"] == [r["source_id"] for r in rows]
    assert all(t.eq(1).all() for t in bundle["text"]["features"])
    assert all(t.eq(2).all() for t in bundle["multimodal"]["features"])
    assert seen.count("train:0") == 2  # once for train and once for dev; completed train shard not replayed


def test_both_7b_weights_are_calibrated_on_dev_without_test_rows(tmp_path, monkeypatch):
    rows = [{"source_id": "dev:0", "tokens": ["London"], "labels": ["B-LOC"]}]
    native_fixture(tmp_path, monkeypatch, rows)
    def source(p, role, labels=True):
        assert role == "dev"
        return rows
    monkeypatch.setattr(study, "source_rows", source)
    def head(root, p, seed, branch):
        file = tmp_path / f"{branch}.pt"
        file.write_bytes(b"fixed fitted head")
        logits = torch.zeros(1, 9)
        if branch == "multimodal":
            logits[0, BIO.index("B-LOC")] = 9
        return file, {"dev_logits": [logits], "dev_aligned": [torch.tensor([True])],
                      "dev_source_ids": ["dev:0"]}
    monkeypatch.setattr(study, "checked_head", head)
    p = {"protocol_id": "fixture", "config": {"coefficient_grid": [0, 0.25, 0.5, 1]}}
    study.calibrate_7b(tmp_path, p)
    result = json.loads((tmp_path / "calibration/dev_report.json").read_text())
    assert result["coefficients"] == {"text": 0, "multimodal": 0.25}
    assert result["test_metrics_used_for_selection"] is False


@pytest.mark.parametrize("experiment", ["fusion", "7b"])
def test_study_reports_replay_existing_3b_results_and_preserve_source(tmp_path, monkeypatch, experiment):
    from modules.tpm_llm_v1.features import fuse
    source, output = tmp_path / "visual", tmp_path / "study"
    rows = [{"source_id": f"test:{i}", "tokens": [f"London{i}"], "labels": ["B-LOC"], "images": []} for i in range(2)]
    native = native_fixture(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(study, "source_rows", lambda *a, **k: rows)
    p = {"protocol_id": "fixture", "experiment": experiment, "visual_run": str(source),
         "config": {"seeds": [1234]}, "bootstrap_replicates": 100, "bootstrap_seed": 1234}
    atomic_json(source / "evaluation/comparison_report.json", {"filtered_test_source_ids": ["test:0", "test:1"]})
    atomic_json(source / "features/test_image_audit.json", {})
    for name, tag in (("tpm", "O"), ("tpm_qwen_text", "O"), ("tpm_qwen_multimodal", "B-LOC")):
        save_rows(source / f"evaluation/seed1234_{name}_predictions.jsonl",
                  [{"source_id": row["source_id"], "labels": [tag]} for row in rows])
    if experiment == "fusion":
        logits = [torch.zeros(1, 9) for _ in rows]
        for value in logits:
            value[0, BIO.index("B-LOC")] = 9
        branches = {"text": native["emissions"], "multimodal": fuse(native["emissions"], rows, native["mappings"],
                                                                     logits, [torch.tensor([True]) for _ in rows], 1)}
        monkeypatch.setattr(study, "cached_3b", lambda *a: (branches, []))
        selection = {"selected_alpha": 0}
    else:
        selection = {"coefficients": {"text": 0, "multimodal": 1}}
        for branch in study.BRANCHES:
            file = output / f"test_logits/seed1234_{branch}.pt"
            logits = [torch.zeros(1, 9) for _ in rows]
            if branch == "multimodal":
                for value in logits:
                    value[0, BIO.index("B-LOC")] = 9
            old.save_torch(file, {"source_ids": native["source_ids"], "logits": logits,
                                  "aligned": [torch.tensor([True]) for _ in rows]})
            old.seal(file.with_suffix(".json"), p, [file])
    calibration = output / "calibration/dev_report.json"
    atomic_json(calibration, selection)
    old.seal(output / "calibration/completed.json", p, [calibration])
    before = {str(f): old.digest(f) for f in source.rglob("*") if f.is_file()}
    study.evaluate(output, p)
    result = json.loads((output / "evaluation/study_report.json").read_text())
    full = result["summary"]["official_test_full"]
    if experiment == "fusion":
        assert full["paired"]["fusion_minus_text3b"]["delta_pp"] == 0
        assert full["paired"]["fusion_minus_multimodal3b"]["delta_pp"] == -100
    else:
        assert full["paired"]["visual_gain_7b"]["delta_pp"] == 100
        assert full["paired"]["scale_multimodal_7b_minus_3b"]["delta_pp"] == 0
    assert old.completed(output / "evaluation/completed.json", p)
    assert before == {str(f): old.digest(f) for f in source.rglob("*") if f.is_file()}


def local_model(tmp_path):
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    atomic_json(tmp_path / "config.json", {"model_type": "qwen2_5_vl", "hidden_size": 3584})
    atomic_json(tmp_path / "model.safetensors.index.json", {"metadata": {"total_size": 7},
                                                           "weight_map": {"weight": "model-00001.safetensors"}})
    (tmp_path / "model-00001.safetensors").write_bytes(b"fixture")


def test_7b_download_checks_reject_incomplete_shards_and_another_model(tmp_path, monkeypatch):
    local_model(tmp_path)
    download.validate(tmp_path)
    study.validate_7b(tmp_path)
    (tmp_path / "model-00001.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="weight shard"):
        download.validate(tmp_path)
    atomic_json(tmp_path / "config.json", {"model_type": "qwen2_5_vl", "hidden_size": 2048})
    monkeypatch.setattr("sys.argv", ["download", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="contains another model"):
        download.main()


def test_existing_7b_download_does_not_use_network(tmp_path, monkeypatch):
    local_model(tmp_path)
    monkeypatch.setattr("sys.argv", ["download", "--output", str(tmp_path)])
    # main returns before importing/calling the Hub for a complete local directory.
    download.main()


def test_download_retries_keep_one_resolved_model_revision(tmp_path, monkeypatch):
    calls, info_calls = [], []
    def model_info(repo, **kwargs):
        info_calls.append(repo)
        return SimpleNamespace(sha="a" * 40)
    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs["revision"]))
        if len(calls) == 1:
            raise TimeoutError("network interruption")
        local_model(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=model_info), snapshot_download=snapshot))
    monkeypatch.setattr(download.time, "sleep", lambda *a: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("sys.argv", ["download", "--output", str(tmp_path), "--attempts", "2"])
    download.main()
    assert info_calls == [download.MODEL] and calls == [(download.MODEL, "a" * 40)] * 2
    provenance = json.loads((tmp_path / "download_provenance.json").read_text())
    assert provenance["resolved_revision"] == "a" * 40


def test_7b_index_rejects_paths_outside_the_model_directory(tmp_path):
    local_model(tmp_path)
    atomic_json(tmp_path / "model.safetensors.index.json", {"weight_map": {"weight": "../other.safetensors"}})
    with pytest.raises(ValueError, match="Unsafe weight filename"):
        download.validate(tmp_path)


def test_study_cli_refuses_an_existing_experiment_directory(tmp_path, monkeypatch):
    atomic_json(tmp_path / "protocol.json", {"schema_version": "tpm-qwen-image-conditioned-v1"})
    monkeypatch.setattr("sys.argv", ["studies", "--experiment", "fusion", "--stage", "evaluate", "--run", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not (tmp_path / "status.json").exists()
