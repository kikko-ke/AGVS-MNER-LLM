from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_atomic_preflight_split import create_split, partition_source_ids


def _write_records(path: Path, count: int, *, label: str = "O") -> None:
    with path.open("w", encoding="utf-8") as destination:
        for index in range(count):
            destination.write(
                repr(
                    {
                        "text": [f"word-{index}"],
                        "images": [f"{index}.jpg"],
                        "label": [label],
                    }
                )
                + "\n"
            )


def test_partition_is_deterministic_disjoint_and_72_8_20() -> None:
    first = partition_source_ids(100, seed=17)
    second = partition_source_ids(100, seed=17)

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "base_train": 72,
        "base_dev": 8,
        "meta": 20,
    }
    assert set(first["base_train"]).isdisjoint(first["base_dev"])
    assert set(first["base_train"]).isdisjoint(first["meta"])
    assert set(first["base_dev"]).isdisjoint(first["meta"])
    assert set().union(*map(set, first.values())) == set(range(100))


def test_split_manifest_tracks_source_ids_without_using_labels(tmp_path: Path) -> None:
    source_a = tmp_path / "train-a.txt"
    source_b = tmp_path / "train-b.txt"
    _write_records(source_a, 50, label="O")
    _write_records(source_b, 50, label="B-PER")

    manifest_a = create_split(
        source_a,
        tmp_path / "first",
        dataset="UNI",
        seed=123,
        base_dev_ratio=0.08,
        meta_ratio=0.20,
    )
    manifest_b = create_split(
        source_b,
        tmp_path / "second",
        dataset="UNI",
        seed=123,
        base_dev_ratio=0.08,
        meta_ratio=0.20,
    )

    first = json.loads(manifest_a.read_text(encoding="utf-8"))
    second = json.loads(manifest_b.read_text(encoding="utf-8"))
    assert first["gold_labels_used_for_assignment"] is False
    assert first["official_validation_or_test_used"] is False
    assert first["source_sha256"] != second["source_sha256"]
    assert set(first["meta_train"]).isdisjoint(first["meta_calibration"])
    assert set(first["meta_train"]).isdisjoint(first["meta_evaluation"])
    assert set(first["meta_calibration"]).isdisjoint(first["meta_evaluation"])
    assert (
        set(first["meta_train"])
        | set(first["meta_calibration"])
        | set(first["meta_evaluation"])
        == set(range(first["partitions"]["meta"]["count"]))
    )
    assert len(first["assignments"]) == first["partitions"]["meta"]["count"]
    assert len(first["meta_train"]) == 6
    assert len(first["meta_calibration"]) == 2
    assert len(first["meta_evaluation"]) == 2
    assert all(
        assignment["source_sample_id"]
        in first["meta_roles"][
            f"{assignment['role']}_source_sample_ids"
        ]
        for assignment in first["assignments"]
    )

    for partition in ("base_train", "base_dev", "meta"):
        first_sidecar = json.loads(
            Path(first["partitions"][partition]["source_ids_path"]).read_text(
                encoding="utf-8"
            )
        )
        second_sidecar = json.loads(
            Path(second["partitions"][partition]["source_ids_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert (
            first_sidecar["source_sample_ids"]
            == second_sidecar["source_sample_ids"]
        )


def test_split_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "train.txt"
    _write_records(source, 20)
    kwargs = {
        "dataset": "UNI",
        "seed": 1,
        "base_dev_ratio": 0.1,
        "meta_ratio": 0.2,
    }
    create_split(source, tmp_path / "split", **kwargs)
    with pytest.raises(FileExistsError, match="--overwrite"):
        create_split(source, tmp_path / "split", **kwargs)
