"""Create a leakage-resistant base-train/base-dev/meta preflight split.

The split assignment depends only on the source row id and a user supplied
seed.  Gold labels remain in the emitted files because the base model and the
later calibrator need them, but they are never parsed while assigning rows.
The official validation and test files are not read or copied by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.reranker import sha256_file


SCHEMA_VERSION = "atomic-preflight-split-v1"
PARTITION_NAMES = ("base_train", "base_dev", "meta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original MNER train file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="UNI")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument(
        "--base-dev-ratio",
        type=float,
        default=0.08,
        help="Fraction of the original train set reserved for base-model selection.",
    )
    parser.add_argument(
        "--meta-ratio",
        type=float,
        default=0.20,
        help="Fraction reserved for OOF atomic-edit reviews/calibration.",
    )
    parser.add_argument(
        "--meta-calibration-ratio",
        type=float,
        default=0.20,
        help="Fraction of the meta pool reserved for calibrator policy selection.",
    )
    parser.add_argument(
        "--meta-evaluation-ratio",
        type=float,
        default=0.20,
        help="Fraction of the meta pool held out for the preflight go/no-go audit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_ratios(base_dev_ratio: float, meta_ratio: float) -> None:
    values = (base_dev_ratio, meta_ratio)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("--base-dev-ratio and --meta-ratio must be positive finite values")
    if sum(values) >= 1.0:
        raise ValueError("base-dev and meta ratios must sum to less than 1")


def _assignment_key(source_sample_id: int, seed: int) -> str:
    payload = f"atomic-preflight-v1:{seed}:{source_sample_id}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def partition_source_ids(
    sample_count: int,
    *,
    seed: int,
    base_dev_ratio: float = 0.08,
    meta_ratio: float = 0.20,
) -> dict[str, list[int]]:
    """Return deterministic, disjoint source-row ids for a 72/8/20 split."""
    _validate_ratios(base_dev_ratio, meta_ratio)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be an integer")
    if sample_count < 3:
        raise ValueError("At least three source samples are required")

    meta_count = max(1, int(sample_count * meta_ratio + 0.5))
    base_dev_count = max(1, int(sample_count * base_dev_ratio + 0.5))
    if meta_count + base_dev_count >= sample_count:
        raise ValueError("Ratios leave no examples for base training")

    shuffled = sorted(
        range(sample_count),
        key=lambda sample_id: (_assignment_key(sample_id, seed), sample_id),
    )
    meta = set(shuffled[:meta_count])
    base_dev = set(shuffled[meta_count : meta_count + base_dev_count])
    base_train = set(shuffled[meta_count + base_dev_count :])
    return {
        "base_train": sorted(base_train),
        "base_dev": sorted(base_dev),
        "meta": sorted(meta),
    }


def _write_source_ids(
    path: Path,
    *,
    partition: str,
    source_sha256: str,
    source_ids: list[int],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "partition": partition,
        "source_sha256": source_sha256,
        "source_sample_ids": source_ids,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_split(
    input_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    seed: int,
    base_dev_ratio: float,
    meta_ratio: float,
    meta_calibration_ratio: float = 0.20,
    meta_evaluation_ratio: float = 0.20,
    overwrite: bool = False,
) -> Path:
    """Write split files, source-id sidecars, and an auditable manifest."""
    _validate_ratios(base_dev_ratio, meta_ratio)
    meta_role_ratios = (meta_calibration_ratio, meta_evaluation_ratio)
    if any(not math.isfinite(value) or value <= 0.0 for value in meta_role_ratios):
        raise ValueError("meta calibration/evaluation ratios must be positive and finite")
    if sum(meta_role_ratios) >= 1.0:
        raise ValueError("meta calibration/evaluation ratios must sum to less than 1")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not dataset or any(character in dataset for character in "/\\"):
        raise ValueError("dataset must be a non-empty filename-safe label")

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if any(not line.strip() for line in lines):
        raise ValueError("The source train file contains blank records")
    partitions = partition_source_ids(
        len(lines),
        seed=seed,
        base_dev_ratio=base_dev_ratio,
        meta_ratio=meta_ratio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "atomic_preflight_manifest.json"
    file_names = {
        "base_train": f"MNER-{dataset}_train.txt",
        "base_dev": f"MNER-{dataset}_val.txt",
        "meta": f"MNER-{dataset}_meta.txt",
    }
    paths: list[Path] = [manifest_path]
    for partition, file_name in file_names.items():
        data_path = output_dir / file_name
        paths.extend(
            [data_path, data_path.with_name(data_path.name + ".source_ids.json")]
        )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {existing[0]}; pass --overwrite to recreate the split"
        )

    source_sha256 = sha256_file(input_path)
    manifest_partitions: dict[str, Any] = {}
    for partition in PARTITION_NAMES:
        source_ids = partitions[partition]
        data_path = output_dir / file_names[partition]
        with data_path.open("w", encoding="utf-8", newline="") as destination:
            for source_id in source_ids:
                line = lines[source_id]
                destination.write(line if line.endswith(("\n", "\r")) else line + "\n")
        source_ids_path = data_path.with_name(data_path.name + ".source_ids.json")
        _write_source_ids(
            source_ids_path,
            partition=partition,
            source_sha256=source_sha256,
            source_ids=source_ids,
        )
        manifest_partitions[partition] = {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "count": len(source_ids),
            "source_ids_path": str(source_ids_path),
            "source_ids_sha256": sha256_file(source_ids_path),
        }

    union = set().union(*(set(values) for values in partitions.values()))
    if len(union) != len(lines) or sum(map(len, partitions.values())) != len(lines):
        raise AssertionError("Generated partitions are not a disjoint source cover")

    meta_source_ids = partitions["meta"]
    meta_calibration_count = max(
        1, int(len(meta_source_ids) * meta_calibration_ratio + 0.5)
    )
    meta_evaluation_count = max(
        1, int(len(meta_source_ids) * meta_evaluation_ratio + 0.5)
    )
    if meta_calibration_count + meta_evaluation_count >= len(meta_source_ids):
        raise ValueError("meta pool is too small to form train/calibration/evaluation roles")
    ranked_meta_local_ids = sorted(
        range(len(meta_source_ids)),
        key=lambda local_id: (
            _assignment_key(meta_source_ids[local_id], seed + 1),
            local_id,
        ),
    )
    meta_calibration = sorted(ranked_meta_local_ids[:meta_calibration_count])
    meta_evaluation = sorted(
        ranked_meta_local_ids[
            meta_calibration_count : meta_calibration_count + meta_evaluation_count
        ]
    )
    meta_train = sorted(
        ranked_meta_local_ids[meta_calibration_count + meta_evaluation_count :]
    )
    roles_by_local_id = {
        **{sample_id: "meta_train" for sample_id in meta_train},
        **{sample_id: "meta_calibration" for sample_id in meta_calibration},
        **{sample_id: "meta_evaluation" for sample_id in meta_evaluation},
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(input_path),
        "source_sha256": source_sha256,
        "source_samples": len(lines),
        "seed": seed,
        "ratios": {
            "base_train": 1.0 - base_dev_ratio - meta_ratio,
            "base_dev": base_dev_ratio,
            "meta": meta_ratio,
        },
        "assignment": "sha256(seed, source_sample_id)",
        "meta_role_assignment": "sha256(seed+1, source_sample_id)",
        "gold_labels_used_for_assignment": False,
        "official_validation_or_test_used": False,
        # These are local row ids in MNER-<dataset>_meta.txt, matching the
        # sample_id written by the atomic candidate exporter/calibrator input.
        "meta_train": meta_train,
        "meta_calibration": meta_calibration,
        "meta_evaluation": meta_evaluation,
        "assignments": [
            {
                "sample_id": local_id,
                "source_sample_id": meta_source_ids[local_id],
                "role": roles_by_local_id[local_id],
            }
            for local_id in range(len(meta_source_ids))
        ],
        "meta_roles": {
            "calibration_ratio": meta_calibration_ratio,
            "evaluation_ratio": meta_evaluation_ratio,
            "train_ratio": 1.0 - meta_calibration_ratio - meta_evaluation_ratio,
            "meta_train_source_sample_ids": [
                meta_source_ids[local_id] for local_id in meta_train
            ],
            "meta_calibration_source_sample_ids": [
                meta_source_ids[local_id] for local_id in meta_calibration
            ],
            "meta_evaluation_source_sample_ids": [
                meta_source_ids[local_id] for local_id in meta_evaluation
            ],
        },
        "partitions": manifest_partitions,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    manifest_path = create_split(
        Path(args.input),
        Path(args.output_dir),
        dataset=args.dataset,
        seed=args.seed,
        base_dev_ratio=args.base_dev_ratio,
        meta_ratio=args.meta_ratio,
        meta_calibration_ratio=args.meta_calibration_ratio,
        meta_evaluation_ratio=args.meta_evaluation_ratio,
        overwrite=args.overwrite,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("=== Atomic preflight split complete ===")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
