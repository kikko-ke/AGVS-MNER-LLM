from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import audit_resampling_development as audit
from scripts.build_llm_augmentation_cache import load_source_records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def class_report(method, offset=0, changed_support=False):
    values = {"baseline": [0.8] * 4, "legacy_selected": [0.75, 0.9, 0.85, 0.78],
              "legacy_matched": [0.78] * 4, "token_selected": [0.79] * 4}[method]
    text = "precision recall f1-score support\n"
    for kind, value in zip(audit.TYPES, values):
        score = value + offset
        support = 6 if changed_support and kind == "PER" else 5
        text += f" {kind} {score:.6f} {score:.6f} {score:.6f} {support}\n"
    text += "micro avg 0.8 0.8 0.8 20\n"
    return text


def fixture(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    # A canary: this cannot be parsed, and the audit must never attempt to do so.
    (suite / "test_metrics.json").write_bytes(b"TEST REPORT MUST NOT BE READ")
    frozen = {"schema_version": "frozen-component-suite-v1", "datasets": {}}
    validation = {"schema_version": "component-suite-metrics-v1", "split": "val", "datasets": {}}
    records = [
        {"text": ["at", "Paris"], "label": ["O", "B-LOC"], "images": ["a.jpg"]},
        {"text": ["Acme", "here"], "label": ["B-ORG", "O"], "images": ["b.jpg"]},
        {"text": ["today", "Alice"], "label": ["O", "B-PER"], "images": ["twitter2017-c.jpg"]},
        {"text": ["music", "Festival"], "label": ["O", "B-MISC"], "images": ["d.jpg"]},
    ]
    for dataset in ("MI", "UNI"):
        root = tmp_path / dataset
        source, selected, matched = root / "source", root / "selected", root / "matched"
        for directory, ids in ((source, []), (selected, [1]), (matched, [3])):
            directory.mkdir(parents=True)
            (directory / f"MNER-{dataset}_train.txt").write_text(
                "\n".join(map(repr, records + [records[i] for i in ids])) + "\n", encoding="utf-8",
            )
        training = source / f"MNER-{dataset}_train.txt"
        rows = load_source_records(training)  # Cross-check the existing generator's fingerprint convention.
        metadata_path, reference_path = root / "metadata.json", selected / "augmentation_manifest.json"
        write_json(metadata_path, {"input_sha256": audit.sha(training), "selected": [
            {"sample_id": i, "source_fingerprint": rows[i]["source_fingerprint"]} for i in (1, 2, 3)
        ]})
        write_json(reference_path, {"source_sample_ids": [1]})
        manifest_path = matched / "matched_resampling_manifest.json"
        write_json(manifest_path, {
            "source_train_sha256": audit.sha(training), "reference_manifest": str(reference_path),
            "reference_manifest_sha256": audit.sha(reference_path), "candidate_metadata": str(metadata_path),
            "candidate_metadata_sha256": audit.sha(metadata_path), "reference_source_sample_ids": [1],
            "source_sample_ids": [3], "original_examples": 4, "added_examples": 1, "candidate_count": 3,
            "reference_train_sha256": audit.sha(selected / training.name),
            "output_files_sha256": {training.name: audit.sha(matched / training.name)},
        })
        frozen["datasets"][dataset] = {
            "inputs": {"train": {"path": str(training), "sha256": audit.sha(training)},
                       "test": {"path": "DO_NOT_OPEN_TEST_EXAMPLES", "sha256": "not-used"}},
            "matched_manifest": str(manifest_path), "matched_manifest_sha256": audit.sha(manifest_path),
        }
        validation["datasets"][dataset] = {"per_seed": [
            {"seed": seed, "metrics": {method: {"f1": 0.8 + offset, "report": class_report(method, offset)}
                                        for method in audit.METHODS}}
            for seed, offset in ((42, 0), (1234, 0.01), (2026, -0.01))
        ]}
    frozen_path = suite / "frozen_manifest.json"
    write_json(frozen_path, frozen)
    validation["frozen_manifest_sha256"] = audit.sha(frozen_path)
    write_json(suite / "validation_metrics.json", validation)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit_resampling_development.py", "--suite-dir", str(suite), "--output", str(output)])
    return suite, output


def test_reports_raw_class_reweighting_and_validation_deltas_without_test_or_models(tmp_path, monkeypatch):
    suite, output = fixture(tmp_path, monkeypatch)
    audit.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["test_report_read"] is False and report["test_examples_read"] is False
    assert report["model_forward_passes"] == 0 and report["llm_api_requests"] == 0
    for dataset in ("MI", "UNI"):
        entry = report["datasets"][dataset]
        assert entry["raw_training_profiles"]["original"]["entity_share"]["ORG"] == 0.25
        assert entry["raw_training_profiles"]["selected_training"]["entity_share"]["ORG"] == 0.4
        assert entry["raw_training_profiles"]["matched_training"]["entity_share"]["MIS"] == 0.4
        assert entry["selection_entity_share_change_pp"]["ORG"] == pytest.approx(15)
        assert entry["validation"]["selection_minus_baseline"]["ORG"]["f1"] == pytest.approx(0.05)
        assert entry["validation"]["selection_minus_baseline"]["LOC"]["recall"] == pytest.approx(-0.05)
        assert entry["added_fraction_of_training"] == 0.2
        assert entry["accepted_fraction_of_candidates"] == pytest.approx(1 / 3)
        assert entry["raw_training_profiles"]["original"]["domain_share_from_listed_filenames"]["twitter2017"] == 0.25
        assert entry["raw_training_profiles"]["selected_training"]["domain_share_from_listed_filenames"]["twitter2017"] == 0.2
    assert report["cross_dataset_overlap"] == {"candidate_fingerprints": 3, "accepted_fingerprints": 1}
    assert (suite / "test_metrics.json").read_bytes() == b"TEST REPORT MUST NOT BE READ"
    with pytest.raises(FileExistsError):
        audit.main()


def test_train_hash_change_is_rejected(tmp_path, monkeypatch):
    suite, output = fixture(tmp_path, monkeypatch)
    frozen = audit.read_json(suite / "frozen_manifest.json")
    Path(frozen["datasets"]["MI"]["inputs"]["train"]["path"]).write_bytes(b"changed training data")
    with pytest.raises(ValueError, match="Artifact changed"):
        audit.main()
    assert not output.exists()


def test_test_report_cannot_be_substituted_for_validation(tmp_path, monkeypatch):
    suite, output = fixture(tmp_path, monkeypatch)
    path = suite / "validation_metrics.json"
    data = audit.read_json(path)
    data["split"] = "test"
    write_json(path, data)
    with pytest.raises(ValueError, match="VALIDATION"):
        audit.main()
    assert not output.exists()


def test_inconsistent_class_support_is_not_silently_averaged(tmp_path, monkeypatch):
    suite, output = fixture(tmp_path, monkeypatch)
    path = suite / "validation_metrics.json"
    data = audit.read_json(path)
    data["datasets"]["MI"]["per_seed"][0]["metrics"]["legacy_selected"]["report"] = class_report(
        "legacy_selected", changed_support=True,
    )
    write_json(path, data)
    with pytest.raises(ValueError, match="support changed"):
        audit.main()
    assert not output.exists()


def test_entity_counting_keeps_mentions_and_surfaces_distinct():
    chunks, repairs = audit.entities(["Alice", "Smith", "Paris", "Paris"], ["I-PER", "I-PER", "B-LOC", "B-LOC"])
    assert chunks == [("PER", "alice smith"), ("LOC", "paris"), ("LOC", "paris")]
    assert repairs == 1
    counts = audit.profile([{"words": 4, "domain": "mi", "listed_images": 1,
                             "entities": chunks, "repaired_i": repairs, "fingerprint": "a"}])
    assert counts["entity_mentions"]["LOC"] == 2 and counts["unique_entity_surfaces"] == 2
    assert counts["entity_share"]["LOC"] == pytest.approx(2 / 3)


def test_malformed_report_is_rejected():
    with pytest.raises(ValueError, match="all four"):
        audit.parse_class_report("LOC 0.8 0.8 0.8 5\n")
    with pytest.raises(ValueError, match="Unsupported raw BIO"):
        audit.entities(["bad"], ["B-UNKNOWN"])
