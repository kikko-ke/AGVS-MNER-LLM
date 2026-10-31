from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION, AUGMENTATION_SCHEMA_VERSION,
    labels_for_augmentation, validate_augmentation_output,
)
from scripts import build_matched_resampling_control as matched
from scripts import project_augmentation_selection as projection
from scripts.build_llm_augmentation_cache import load_source_records, sha256_file
from scripts.evaluate_matched_resampling import audit_data


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "source_train.txt"
    records = [{"text": ["watching", "Acme", "Corp", "at", "arena", str(i)],
                "label": ["O", "B-ORG", "I-ORG", "O", "O", "O"], "images": ["image.jpg"]}
               for i in range(4)]
    source.write_text("\n".join(map(repr, records)) + "\n", encoding="utf-8")
    normalised = load_source_records(source)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    train = target_dir / "MNER-MI_train.txt"
    train.write_text("\n".join(repr(records[i]) for i in (2, 1, 0)) + "\n", encoding="utf-8")
    # Held-out labels deliberately cannot pass BIO parsing: projection must
    # never read them. Only unlabeled text/images are used for overlap checks.
    for split in ("val", "test"):
        (target_dir / f"MNER-MI_{split}.txt").write_text(
            repr({"text": [split], "images": [split + ".jpg"], "label": ["INVALID"]}) + "\n",
            encoding="utf-8",
        )
    cache = tmp_path / "source_cache.jsonl"
    metadata = {
        "schema_version": AUGMENTATION_SCHEMA_VERSION, "prompt_version": AUGMENTATION_PROMPT_VERSION,
        "input": str(source), "input_sha256": sha256_file(source), "llm_model": "test-model",
        "min_context_change_ratio": 0.0, "max_context_change_ratio": 1.0,
        "min_length_ratio": 0.01, "max_length_ratio": 100, "max_words": 62,
        "selected": [{"sample_id": r["sample_id"], "source_fingerprint": r["source_fingerprint"]}
                     for r in normalised],
    }
    _write_json(Path(str(cache) + ".meta.json"), metadata)
    rows = []
    for i, record in enumerate(normalised):
        row = {"schema_version": AUGMENTATION_SCHEMA_VERSION, "prompt_version": AUGMENTATION_PROMPT_VERSION,
               "sample_id": i, "source_fingerprint": record["source_fingerprint"], "llm_model": "test-model",
               "images": record["images"], "valid_image_ids": [0]}
        if i == 1:
            row.update(status="rejected", policy_rejection="context policy rejection", labels=[])
        else:
            payload = {"abstain": i == 3, "used_image_ids": [] if i == 3 else [0],
                       "context_segments": [] if i == 3 else [["watching"], ["beside", "the", "bright", "arena", str(i)]],
                       "change_summary": "visible setting" if i != 3 else "",
                       "abstain_reason": "no safe rewrite" if i == 3 else ""}
            response = validate_augmentation_output(
                payload, record["words"], record["entities"], {0}, min_context_change_ratio=0.0,
                max_context_change_ratio=1.0, min_length_ratio=0.01, max_length_ratio=100,
            )
            row.update(response.to_dict())
            row["status"] = "abstained" if response.abstain else "accepted"
            row["labels"] = [] if response.abstain else labels_for_augmentation(
                len(response.tokens), record["entities"], response.entity_spans,
            )
        rows.append(row)
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_cache, output = tmp_path / "projected.jsonl", tmp_path / "selected"
    monkeypatch.setattr(sys, "argv", ["project_augmentation_selection.py", "--source-input", str(source),
                        "--source-cache", str(cache), "--target-input-dir", str(target_dir),
                        "--target-dataset", "MI", "--output-cache", str(output_cache), "--output-dir", str(output)])
    return source, target_dir, cache, output_cache, output


def test_remaps_ids_by_full_content_preserves_rejections_and_duplicates_only_originals(tmp_path, monkeypatch):
    _, target, cache, projected, output = _fixture(tmp_path, monkeypatch)
    projection.main()
    rows = [json.loads(line) for line in projected.read_text(encoding="utf-8").splitlines()]
    assert [(r["origin_source_sample_id"], r["sample_id"], r["status"]) for r in rows] == [
        (0, 2, "accepted"), (1, 1, "rejected"), (2, 0, "accepted"),
    ]
    originals = (target / "MNER-MI_train.txt").read_text(encoding="utf-8").splitlines()
    actual = (output / "MNER-MI_train.txt").read_text(encoding="utf-8").splitlines()
    assert actual == originals + [originals[2], originals[0]]
    metadata = json.loads(Path(str(projected) + ".meta.json").read_text(encoding="utf-8"))
    assert metadata["input_sha256"] == sha256_file(target / "MNER-MI_train.txt")
    assert metadata["projection"]["source_cache_sha256"] == sha256_file(cache)
    assert metadata["projection"]["llm_api_requests"] == 0
    assert metadata["projection"]["heldout_labels_read"] is False
    assert metadata["projection"]["projected_candidate_count"] == 3
    manifest = json.loads((output / "augmentation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "MI" and manifest["accepted_augmentations"] == 2
    for split in ("val", "test"):
        name = f"MNER-MI_{split}.txt"
        assert (target / name).read_bytes() == (output / name).read_bytes()
    with pytest.raises(FileExistsError):
        projection.main()


@pytest.mark.parametrize("field", ["text", "label", "images"])
def test_each_contract_field_must_match_not_just_the_row_number(tmp_path, monkeypatch, field):
    _, target, _, projected, _ = _fixture(tmp_path, monkeypatch)
    path = target / "MNER-MI_train.txt"
    records = [ast.literal_eval(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if field == "text":
        records[2]["text"][0] = "looking"
    elif field == "label":
        records[2]["label"][1:3] = ["B-PER", "I-PER"]
    else:
        records[2]["images"] = ["different.jpg"]
    path.write_text("\n".join(map(repr, records)) + "\n", encoding="utf-8")
    projection.main()
    rows = [json.loads(line) for line in projected.read_text(encoding="utf-8").splitlines()]
    assert {row["origin_source_sample_id"] for row in rows} == {1, 2}


def test_incomplete_source_cache_is_not_silently_used(tmp_path, monkeypatch):
    _, _, cache, projected, output = _fixture(tmp_path, monkeypatch)
    lines = cache.read_text(encoding="utf-8").splitlines()
    cache.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resolve every selected"):
        projection.main()
    assert not projected.exists() and not output.exists()


def test_heldout_overlap_is_rejected_without_creating_outputs(tmp_path, monkeypatch):
    _, target, _, projected, output = _fixture(tmp_path, monkeypatch)
    example = (target / "MNER-MI_train.txt").read_text(encoding="utf-8").splitlines()[0]
    (target / "MNER-MI_test.txt").write_text(example + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps held-out"):
        projection.main()
    assert not projected.exists() and not output.exists()


def test_ambiguous_duplicate_target_is_rejected(tmp_path, monkeypatch):
    _, target, _, projected, output = _fixture(tmp_path, monkeypatch)
    path = target / "MNER-MI_train.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Ambiguous"):
        projection.main()
    assert not projected.exists() and not output.exists()


def test_source_queries_must_also_respect_target_heldout_boundary(tmp_path, monkeypatch):
    source, target, _, projected, output = _fixture(tmp_path, monkeypatch)
    example = source.read_text(encoding="utf-8").splitlines()[3]
    (target / "MNER-MI_val.txt").write_text(example + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source cached candidates overlap target held-out"):
        projection.main()
    assert not projected.exists() and not output.exists()


def test_projected_mi_data_supports_existing_matching_and_generic_audit(tmp_path, monkeypatch):
    _, target, _, projected, selected = _fixture(tmp_path, monkeypatch)
    projection.main()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), "white").save(images / "image.jpg")
    monkeypatch.setattr(matched.BertTokenizer, "from_pretrained", lambda *a, **kw: type(
        "Tokenizer", (), {"tokenize": lambda self, word: [word]},
    )())
    output = tmp_path / "matched"
    monkeypatch.setattr(sys, "argv", ["build_matched_resampling_control.py", "--dataset", "MI",
                        "--input-dir", str(target), "--candidate-metadata", str(projected) + ".meta.json",
                        "--reference-manifest", str(selected / "augmentation_manifest.json"),
                        "--image-root", str(images), "--twitter-image-root", str(images),
                        "--bert-model", str(tmp_path / "tokenizer"), "--sampling-seed", "0", "--output-dir", str(output)])
    matched.main()
    manifest_path = output / "matched_resampling_manifest.json"
    manifest = audit_data(manifest_path, target, dataset="MI")
    assert manifest["ready_for_training"] is True and manifest["dataset"] == "MI"
    with pytest.raises(ValueError, match="Expected a passed"):
        audit_data(manifest_path, target, dataset="UNI")


def test_runner_completion_markers_preserve_and_verify_checkpoint_data_and_code(tmp_path, monkeypatch):
    script = Path(matched.__file__).with_name("train_mi_resampling.sh").read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    for block in blocks:
        compile(block, "train_mi_resampling.sh embedded Python", "exec")
    run = tmp_path / "outputs" / "mi_run"
    run.mkdir(parents=True)
    checkpoint = run / "best_model.pt"
    checkpoint.write_bytes(b"trained checkpoint")
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "metrics.csv").write_text("epoch,dev_f1\n1,0.8\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        (data / f"MNER-MI_{split}.txt").write_text(split, encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_mi_resampling.sh").write_text(script, encoding="utf-8")
    (tmp_path / "run.py").write_text("training entry", encoding="utf-8")
    log = tmp_path / "train.log"
    log.write_text("Skipping test evaluation", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "run.py\n")
    monkeypatch.setattr(sys, "argv", ["-", str(run.parent), str(data), str(log)])
    exec(blocks[-1], {"__name__": "__main__"})
    marker_path = run.parent / "completed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["checkpoint"] == str(checkpoint)
    assert marker["artifacts"]["run.py"] == sha256_file(tmp_path / "run.py")
    monkeypatch.setattr(sys, "argv", ["-", str(marker_path), str(data)])
    exec(blocks[-2], {"__name__": "__main__"})
    checkpoint.write_bytes(b"modified after completion")
    with pytest.raises(SystemExit, match="Completed artifact changed"):
        exec(blocks[-2], {"__name__": "__main__"})
