from __future__ import annotations

import json
import random
import sys

import numpy as np
import pytest
from PIL import Image

from scripts import build_matched_resampling_control as matched


def test_sampling_is_reproducible_unique_and_preserves_exact_quotas():
    groups = {("mi", "MIS", 1): list(range(6)), ("twitter", "ORG", 2): list(range(6, 12))}
    quotas = matched.Counter({key: 2 for key in groups})
    vectors = {i: np.array([float(i), float(i % 3)]) for i in range(12)}
    target = sum((vectors[i] for i in (1, 2, 9, 10)), np.zeros(2))
    first, audit = matched.sample_matched(groups, quotas, vectors, target, np.ones(2), seed=7, swap_attempts=1000)
    second, _ = matched.sample_matched(groups, quotas, vectors, target, np.ones(2), seed=7, swap_attempts=1000)
    assert first == second
    assert all(len(first[key]) == quotas[key] for key in groups)
    assert len({i for ids in first.values() for i in ids}) == 4
    assert audit["final_objective"] <= audit["initial_objective"]


def test_identical_feature_candidates_are_sampled_without_identity_preferences():
    key = ("mi", "MIS", 1)
    groups = {key: list(range(10))}
    vectors = {i: np.zeros(2) for i in range(10)}
    result, audit = matched.sample_matched(groups, matched.Counter({key: 4}), vectors,
                                         np.zeros(2), np.ones(2), seed=0, swap_attempts=20)
    assert result[key] == random.Random(0).sample(list(range(10)), 4)
    assert audit["accepted_feature_swaps"] == 0


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), "white").save(images / "a.png")
    records = [
        {"text": ["entity", "here"], "label": [f"B-{'MIS' if i < 4 else 'ORG'}", "O"],
         "images": ["a.png"]} for i in range(8)
    ]
    lines = [repr(record) for record in records]
    train = source / "MNER-UNI_train.txt"
    train.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        (source / f"MNER-UNI_{split}.txt").write_bytes(f"unchanged {split}\r\n".encode())
    rows = matched.load_source_records(train)
    metadata = {
        "schema_version": matched.AUGMENTATION_SCHEMA_VERSION,
        "prompt_version": matched.AUGMENTATION_PROMPT_VERSION,
        "input_sha256": matched.sha256_file(train),
        "selected": [{"sample_id": row["sample_id"], "source_fingerprint": row["source_fingerprint"]}
                     for row in rows],
    }
    metadata_path = tmp_path / "candidate_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / train.name).write_text("\n".join(lines + [lines[0], lines[4]]) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        name = f"MNER-UNI_{split}.txt"
        (reference_dir / name).write_bytes((source / name).read_bytes())
    reference_path = reference_dir / "augmentation_manifest.json"
    reference_path.write_text(json.dumps({
        "arm": "duplicate_control", "source_train_sha256": matched.sha256_file(train),
        "original_examples": 8, "accepted_augmentations": 2, "source_sample_ids": [0, 4],
    }), encoding="utf-8")

    class Tokenizer:
        def tokenize(self, word):
            return [word]

    monkeypatch.setattr(matched.BertTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    output = tmp_path / "matched"
    monkeypatch.setattr(sys, "argv", ["build_matched_resampling_control.py", "--input-dir", str(source),
                        "--candidate-metadata", str(metadata_path), "--reference-manifest", str(reference_path),
                        "--image-root", str(images), "--twitter-image-root", str(images),
                        "--bert-model", str(tmp_path / "tokenizer"), "--sampling-seed", "1",
                        "--output-dir", str(output)])
    return source, output, metadata_path, reference_path, lines


def test_builds_original_only_control_and_copies_evaluation_files_exactly(tmp_path, monkeypatch):
    source, output, _, _, lines = _fixture(tmp_path, monkeypatch)
    matched.main()
    manifest = json.loads((output / "matched_resampling_manifest.json").read_text(encoding="utf-8"))
    actual = (output / "MNER-UNI_train.txt").read_text(encoding="utf-8").splitlines()
    assert actual[:8] == lines
    assert actual[8:] == [lines[i] for i in manifest["source_sample_ids"]]
    assert len(actual) == 10
    assert manifest["added_examples"] == 2
    assert manifest["llm_api_requests"] == 0
    assert manifest["candidate_scoring_uses_llm_outputs"] is False
    assert manifest["ready_for_training"] is True
    assert manifest["balance_check"]["max_abs_difference_in_pool_sd_units"] == 0
    assert manifest["overlap_check"]["fraction"] == 0.5
    assert all(row["matched"] == row["reference"] for row in manifest["strata"])
    for split in ("val", "test"):
        name = f"MNER-UNI_{split}.txt"
        assert (output / name).read_bytes() == (source / name).read_bytes()
    with pytest.raises(FileExistsError):
        matched.main()


def test_source_hash_mismatch_is_rejected_before_output(tmp_path, monkeypatch):
    source, output, _, _, _ = _fixture(tmp_path, monkeypatch)
    with (source / "MNER-UNI_train.txt").open("a", encoding="utf-8") as target:
        target.write("\n")
    with pytest.raises(ValueError, match="Training source SHA256"):
        matched.main()
    assert not output.exists()


def test_incorrect_reference_duplication_is_rejected(tmp_path, monkeypatch):
    _, output, _, reference, lines = _fixture(tmp_path, monkeypatch)
    (reference.parent / "MNER-UNI_train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Reference training file"):
        matched.main()
    assert not output.exists()


def test_uninformative_exact_overlap_is_reported_and_blocks_training(tmp_path, monkeypatch):
    _, output, metadata_path, _, _ = _fixture(tmp_path, monkeypatch)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["selected"] = [row for row in metadata["selected"] if row["sample_id"] in (0, 4)]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Matching/overlap gate failed"):
        matched.main()
    manifest = json.loads((output / "matched_resampling_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ready_for_training"] is False
    assert manifest["overlap_check"]["minimum_forced_by_exact_strata"] == 2
    assert manifest["overlap_check"]["fraction"] == 1.0


def test_physical_image_check_does_not_trust_candidate_metadata(tmp_path):
    assert matched.image_info(tmp_path / "missing.png")["decodable"] is False
    damaged = tmp_path / "damaged.png"
    damaged.write_bytes(b"not an image")
    info = matched.image_info(damaged)
    assert info["status"] == "failed" and info["sha256"] is not None
    real = tmp_path / "real.png"
    Image.new("RGB", (8, 8), "white").save(real)
    assert matched.image_info(real)["status"] == "direct"
