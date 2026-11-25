"""Input-only donor selection, native geometry, paired statistics and frozen scoring."""
import json

import numpy as np
import pytest
import torch

from scripts import evaluate_tpm_qwen_diagnostics_v1 as diag
from modules.tpm_llm_v1.common import BIO, TAGS, atomic_json, metrics, save_rows
from modules.tpm_llm_v1 import pipeline as old
from modules.tpm_qwen_visual_v1 import features
from test_tpm_qwen_visual_v1 import native_processor


def test_donors_exclude_transitive_text_image_components_and_use_no_gold():
    audit = {name: {"valid": True, "sha256": name} for name in ("a", "b", "c", "d")}
    rows = [{"source_id": f"test:{i}", "tokens": [word], "images": images}
            for i, (word, images) in enumerate([
                ("same", ["a"]), ("SAME", ["b"]), ("linked", ["b"]),
                ("other", ["c"]), ("fourth", ["d"]), ("two", ["a", "d"]), ("empty", [])])]
    ids = diag.component_ids(rows, audit)
    assert ids[0] == ids[1] == ids[2] == ids[4] == ids[5]
    plan = diag.mismatch_plan(rows, audit, 1234)
    assert plan == diag.mismatch_plan([{**r, "labels": ["unused"]} for r in rows], audit, 1234)
    lookup = {r["source_id"]: i for i, r in enumerate(rows)}
    for i, p in enumerate(plan):
        if p["eligible"]:
            j = lookup[p["donor_source_id"]]
            assert ids[i] != ids[j] and len(p["pairs"]) == p["valid_images"]
            assert len(rows[i]["images"]) == len(rows[j]["images"])
    assert plan[5]["reason"] == "no_disjoint_donor_in_count_bucket"
    assert plan[6]["reason"] == "no_valid_images"


def image_fixture(tmp_path):
    from PIL import Image
    audit = {}
    for name, size, color in (("tiny.png", (190, 21), "red"), ("tall.png", (56, 112), "blue"),
                              ("donor1.png", (100, 200), "green"), ("donor2.png", (200, 50), "yellow")):
        path = tmp_path / name
        Image.new("RGB", size, color).save(path)
        audit[name] = {"valid": True, "path": str(path), "sha256": old.digest(path)}
    audit["missing.png"] = {"valid": False, "path": str(tmp_path / "missing.png"), "sha256": None}
    row = {"source_id": "test:0", "tokens": ["New", "York"],
           "images": ["tiny.png", "missing.png", "tall.png"]}
    pairs = [{"slot": 0, "original_name": "tiny.png", "donor_slot": 0, "donor_name": "donor1.png"},
             {"slot": 2, "original_name": "tall.png", "donor_slot": 1, "donor_name": "donor2.png"}]
    return row, audit, {"source_id": "test:0", "eligible": True, "donor_source_id": "test:1", "pairs": pairs}


def test_native_mismatch_changes_only_pixels_preserving_grid_tokens_and_alignment(tmp_path, native_processor):
    row, audit, assignment = image_fixture(tmp_path)
    correct, changed, positions, usage = diag.mismatch_inputs(row, assignment, native_processor, audit, 2048)
    expected, expected_positions, _ = features.prepare_input(row, native_processor, audit, 2048)
    for key in correct:
        assert torch.equal(correct[key], expected[key])
        if key != "pixel_values":
            assert torch.equal(correct[key], changed[key])
    assert changed["pixel_values"].shape == correct["pixel_values"].shape
    assert not torch.equal(changed["pixel_values"], correct["pixel_values"])
    assert positions == expected_positions
    assert usage["used_slots"] == [0, 2] and usage["geometry_preserved"]
    assert usage["pixel_max_delta"] > 0


def test_native_same_image_geometry_replay_is_bitwise_equal_including_tiny_image(tmp_path, native_processor):
    row, audit, assignment = image_fixture(tmp_path)
    for pair in assignment["pairs"]:
        pair["donor_name"] = pair["original_name"]
    correct, changed, _, _ = diag.mismatch_inputs(row, assignment, native_processor, audit, 2048)
    assert torch.equal(correct["pixel_values"], changed["pixel_values"])


def test_ineligible_rows_reuse_cached_features_without_a_model():
    values, valid = torch.randn(2, 768).half(), torch.tensor([True, False])
    a, b, usage = diag.extract_row(None, {}, {"eligible": False, "reason": "no_valid_images"}, {}, values, valid)
    assert torch.equal(a, values) and torch.equal(b, valid) and usage["geometry_preserved"]


def test_bootstrap_recomputes_corpus_f1_instead_of_averaging_post_f1():
    candidate = np.array([[[1, 0, 0], [0, 0, 9]]])
    reference = np.array([[[0, 0, 1], [0, 0, 9]]])
    result = diag.paired_bootstrap(candidate, reference, [0, 1], replicates=500, seed=1234)
    assert result["delta_pp"] == pytest.approx(100 * 2 / 11)
    assert result == diag.paired_bootstrap(candidate, reference, [0, 1], replicates=500, seed=1234)
    identical = diag.paired_bootstrap(candidate, candidate, [0, 1], replicates=100)
    assert identical["ci95_pp"] == [0, 0]


def test_bootstrap_uses_cluster_units_and_preserves_paired_model_seeds():
    candidate = np.array([[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]])
    reference = candidate.copy()
    reference[0] = [[0, 0, 1], [0, 0, 1]]
    result = diag.paired_bootstrap(candidate, reference, [0, 1], replicates=100)
    assert result["ci95_pp"] == [50, 50]
    assert diag.paired_bootstrap(candidate, reference, [0, 0], replicates=100)["ci95_pp"] is None


def test_post_counts_validate_bio_coverage_and_entity_boundaries():
    rows = [{"tokens": ["New", "York"], "labels": ["B-LOC", "I-LOC"]}]
    assert diag.post_counts(rows, [["B-LOC", "O"]]).tolist() == [[0, 1, 1]]
    with pytest.raises(ValueError, match="cover every original word"):
        diag.post_counts(rows, [["O"]])


def test_cli_refuses_to_write_into_an_existing_visual_run(tmp_path, monkeypatch):
    atomic_json(tmp_path / "protocol.json", {"schema_version": "tpm-qwen-image-conditioned-v1"})
    before = (tmp_path / "protocol.json").read_bytes()
    monkeypatch.setattr("sys.argv", ["diagnostics", "--stage", "analyze", "--run", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        diag.main()
    assert error.value.code == 2
    assert (tmp_path / "protocol.json").read_bytes() == before
    assert not (tmp_path / "status.json").exists()


def test_analysis_replays_source_predictions_reports_filtered_groups_and_keeps_source_unchanged(tmp_path, monkeypatch):
    source, output = tmp_path / "visual", tmp_path / "diagnostic"
    rows = [{"source_id": f"test:{i}", "tokens": [f"place{i}"], "labels": ["B-LOC"], "images": []}
            for i in range(4)]
    p = {"protocol_id": "diagnostic", "config": {"seeds": [1234, 42],
                                                  "bootstrap_replicates": 100, "bootstrap_seed": 1234}}
    report = {"metrics": {"official_test_full": {}}, "filtered_test_source_ids": ["test:0", "test:1"],
              "valid_image_count_samples": {"0": 4}, "metrics_by_valid_image_count": {}}
    for seed in p["config"]["seeds"]:
        report["metrics"]["official_test_full"][str(seed)] = {}
        report["metrics_by_valid_image_count"][str(seed)] = {"0": {}}
        for name in diag.SYSTEMS:
            tags = [["O"] if name != "tpm_qwen_multimodal" else ["B-LOC"] for _ in rows]
            file = source / f"evaluation/seed{seed}_{name}_predictions.jsonl"
            file.parent.mkdir(parents=True, exist_ok=True)
            save_rows(file, [{"source_id": row["source_id"], "labels": value} for row, value in zip(rows, tags)])
            report["metrics"]["official_test_full"][str(seed)][name] = metrics(rows, tags)
            report["metrics_by_valid_image_count"][str(seed)]["0"][name] = metrics(rows, tags)
    atomic_json(source / "evaluation/comparison_report.json", report)
    monkeypatch.setattr(diag, "source_context", lambda *a, **k: (source, {}, rows, {}))
    before = {str(f): old.digest(f) for f in source.rglob("*") if f.is_file()}
    diag.analyze(output, p)
    result = json.loads((output / "analysis/analysis_report.json").read_text())
    assert result["analysis"]["official_test_full"]["C_minus_B"]["delta_pp"] == 100
    assert result["analysis"]["official_test_filtered"]["samples"] == 2
    assert result["by_valid_image_count"]["0"]["C_minus_B_pp"] == 100
    assert old.completed(output / "analysis/completed.json", p)
    assert before == {str(f): old.digest(f) for f in source.rglob("*") if f.is_file()}


def test_frozen_scoring_keeps_tpm_and_original_predictions_and_reports_eligible_scope(tmp_path, monkeypatch):
    from torchcrf import CRF
    source, output = tmp_path / "visual", tmp_path / "diagnostic"
    p, vp = {"protocol_id": "diagnostic", "config": {"seeds": [1234]}}, {"protocol_id": "visual"}
    rows = [{"source_id": f"test:{i}", "tokens": ["London"], "labels": ["B-LOC"], "images": []} for i in range(2)]
    report = {"coefficients": {"qwen_multimodal": 0.5}, "filtered_test_source_ids": ["test:0", "test:1"]}
    atomic_json(source / "evaluation/comparison_report.json", report)
    atomic_json(output / "mismatch_plan.json", [{"eligible": True, "valid_images": 2}, {"eligible": False}])
    monkeypatch.setattr(diag, "source_context", lambda *a, **k: (source, vp, rows, {}))
    monkeypatch.setattr(diag.visual, "source_run", lambda *a: (source, {}))
    decoder = CRF(10, batch_first=True)
    for parameter in decoder.parameters():
        parameter.data.zero_()
    emissions = torch.zeros(2, 3, 10)
    emissions[:, [0, 2], 9] = 5
    emissions[:, 1, 0] = 2
    emissions[:, 1, TAGS.index("B-LOC")] = 1
    native = {"source_ids": [r["source_id"] for r in rows], "emissions": emissions,
              "mask": torch.ones(2, 3, dtype=torch.bool), "crf": decoder.state_dict(),
              "mappings": [{"positions": [1], "word_indices": [0]} for _ in rows]}
    native_file = source / "native.pt"
    old.save_torch(native_file, native)
    monkeypatch.setattr(diag.visual, "checked_native", lambda *a: (native_file, native))
    for name in diag.SYSTEMS:
        file = source / f"evaluation/seed1234_{name}_predictions.jsonl"
        save_rows(file, [{"source_id": row["source_id"], "labels": ["B-LOC"]} for row in rows])
    for file, protocol, high in ((source / "test_logits/seed1234.pt", vp, True),
                                 (output / "logits/seed1234.pt", p, False)):
        logits = [torch.zeros(1, 9) for _ in rows]
        if high:
            for score in logits:
                score[0, BIO.index("B-LOC")] = 9
        old.save_torch(file, {"source_ids": native["source_ids"], "logits": logits,
                              "aligned": [torch.tensor([True]) for _ in rows]})
        old.seal(file.with_suffix(".json"), protocol, [file])
    before = {str(f): old.digest(f) for f in source.rglob("*") if f.is_file()}
    diag.evaluate(output, p)
    result = json.loads((output / "evaluation/mismatch_report.json").read_text())
    assert result["parameters_updated"] == 0
    assert result["summary"]["eligible_mismatch"]["C_minus_D_pp"]["mean"] == 100
    assert result["summary"]["official_test_full"]["C_minus_D_pp"]["mean"] == pytest.approx(100 / 3)
    assert before == {str(f): old.digest(f) for f in source.rglob("*") if f.is_file()}
    assert old.completed(output / "evaluation/completed.json", p)
