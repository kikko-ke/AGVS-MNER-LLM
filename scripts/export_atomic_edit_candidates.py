"""Export gold-free strict atomic CRF edits for LLM review/calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llm.prompts import ATOMIC_REVIEW_PROMPT_VERSION
from models.agvs_mner import AgvsMnerModel
from modules.crf_candidates import (
    build_allowed_tag_mask,
    deduplicate_topk_paths,
    serialize_entities,
)
from modules.crf_edit_decomposition import describe_atomic_entity_edit
from modules.crf_topk import topk_viterbi_decode
from modules.dataset import MultimodalInferenceDataset
from modules.reranker import (
    build_candidate_provenance,
    opaque_candidate_ids,
    sha256_file,
    sha256_inference_projection,
)
from modules.trainer import load_compatible_model_checkpoint
from run_llm import set_seed


SCHEMA_VERSION = "atomic-edit-candidates-v1"
_FINGERPRINT_FIELDS = {"review_fingerprint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--source-ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-label", default="meta")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Unlock an official test artifact only after all policy choices are frozen.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--oversample-factor", type=int, default=20)
    parser.add_argument(
        "--route-queries",
        type=int,
        default=1000,
        help="Number of lowest atomic-edit margins to mark for review; 0 selects all.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _stable_json_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Atomic candidate record is not canonical JSON") from error
    return serialized.encode("utf-8")


def atomic_review_record_fingerprint(record: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in record.items() if key not in _FINGERPRINT_FIELDS
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def load_source_ids(path: Path | None, sample_count: int) -> tuple[list[int], str | None]:
    """Load the local-row to original-source-row mapping from a split sidecar."""
    if path is None:
        return list(range(sample_count)), None
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid source-id sidecar: {path}") from error
    values = payload.get("source_sample_ids") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError("Source-id sidecar must contain source_sample_ids")
    if len(values) != sample_count:
        raise ValueError(
            f"Source-id count {len(values)} does not match input rows {sample_count}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Source sample ids must be non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("Source-id sidecar contains duplicates")
    return list(values), sha256_file(path)


def _entity_tuples(entities: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, str]]:
    return [
        (int(entity["start"]), int(entity["end"]), str(entity["label"]))
        for entity in entities
    ]


def _with_surface(entity: Mapping[str, Any] | None, words: Sequence[str]) -> dict | None:
    if entity is None:
        return None
    result = {
        "start": int(entity["start"]),
        "end": int(entity["end"]),
        "label": str(entity["label"]),
    }
    result["surface"] = " ".join(str(word) for word in words[result["start"] : result["end"]])
    return result


def _edit_order_key(sample_id: int, edit_id: str) -> str:
    return hashlib.sha256(
        f"{ATOMIC_REVIEW_PROMPT_VERSION}:{sample_id}:{edit_id}".encode("ascii")
    ).hexdigest()


def select_low_margin_atomic_routes(
    records: Sequence[Mapping[str, Any]],
    route_queries: int,
) -> list[dict[str, Any]]:
    """Select deterministic lowest-margin eligible records without reading gold."""
    if route_queries < 0:
        raise ValueError("route_queries must be non-negative")
    seen: set[int] = set()
    ranked: list[tuple[float, int, Mapping[str, Any]]] = []
    for record in records:
        sample_id = record.get("sample_id")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise ValueError("Every atomic record needs an integer sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id={sample_id}")
        seen.add(sample_id)
        edits = record.get("edits")
        if not isinstance(edits, list) or not edits:
            continue
        margin = record.get("route_features", {}).get(
            "minimum_atomic_score_gap_per_word"
        )
        if isinstance(margin, bool):
            raise ValueError("Atomic route margin must be finite")
        try:
            margin = float(margin)
        except (TypeError, ValueError) as error:
            raise ValueError("Atomic route margin must be finite") from error
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("Atomic route margin must be finite and non-negative")
        ranked.append((margin, sample_id, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = ranked if route_queries == 0 else ranked[:route_queries]
    return [
        {
            "sample_id": sample_id,
            "route_order": index,
            "minimum_atomic_score_gap_per_word": margin,
        }
        for index, (margin, sample_id, _) in enumerate(selected)
    ]


def _is_official_test(input_path: Path, split_label: str) -> bool:
    return split_label.lower() == "test" or input_path.name.lower().endswith("_test.txt")


def main() -> None:
    cli = parse_args()
    if cli.k < 2:
        raise ValueError("--k must be at least 2")
    if cli.oversample_factor <= 0:
        raise ValueError("--oversample-factor must be positive")
    if cli.route_queries < 0:
        raise ValueError("--route-queries must be non-negative")
    if not cli.split_label:
        raise ValueError("--split-label must be non-empty")

    checkpoint = Path(cli.checkpoint)
    input_path = Path(cli.input_file)
    output = Path(cli.output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.name.lower().endswith("_test.txt") and cli.split_label != "test":
        raise ValueError("An official *_test.txt input must use --split-label test")
    official_test = _is_official_test(input_path, cli.split_label)
    if official_test and not cli.allow_test:
        raise ValueError("Official test export is locked; pass --allow-test only after freezing")
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    if output.exists() and not cli.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["device"] = cli.device
    config["data_path"] = str(input_path.parent)
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.num_workers is not None:
        config["num_workers"] = cli.num_workers
    args = SimpleNamespace(**config)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    set_seed(int(getattr(args, "seed", 1234)))

    dataset = MultimodalInferenceDataset(args, input_path.name)
    default_sidecar = input_path.with_name(input_path.name + ".source_ids.json")
    sidecar_path = Path(cli.source_ids) if cli.source_ids else (
        default_sidecar if default_sidecar.is_file() else None
    )
    source_ids, source_ids_sha256 = load_source_ids(sidecar_path, len(dataset))
    if cli.split_label == "meta" and sidecar_path is None:
        raise ValueError("A meta export requires --source-ids (or the default sidecar)")

    input_sha256 = sha256_inference_projection(input_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    decode_k = cli.k * cli.oversample_factor
    provenance = build_candidate_provenance(
        checkpoint_sha256=checkpoint_sha256,
        input_sha256=input_sha256,
        split=cli.split_label,
        candidate_count=cli.k,
        decode_count=decode_k,
        generator_version="strict-atomic-crf-topk-v1",
    )
    provenance.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_ids_sha256": source_ids_sha256,
            "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
            "official_test": official_test,
        }
    )

    label_mapping = dataset.processor.get_label_mapping(include_subword_label=True)
    index_to_label = {index: label for label, index in label_mapping.items()}
    outside_label_id = label_mapping["O"]
    subword_label_id = label_mapping["X"]
    model = AgvsMnerModel(list(label_mapping), args)
    missing = load_compatible_model_checkpoint(model, checkpoint, map_location="cpu")
    if missing:
        print(f"Optional checkpoint keys initialized locally: {missing}")
    model.to(args.device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(getattr(args, "batch_size", 4)),
        num_workers=int(getattr(args, "num_workers", 0)),
        shuffle=False,
    )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Export atomic CRF edits", unit="batch"):
            (
                input_ids,
                token_type_ids,
                attention_mask,
                image_pixel_values,
                image_mask,
                sample_ids,
            ) = batch
            input_ids = input_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            image_pixel_values = image_pixel_values.to(args.device)
            image_mask_device = image_mask.to(args.device)
            _, _, features = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                image_pixel_values=image_pixel_values,
                image_mask=image_mask_device,
                labels=None,
                return_features=True,
            )
            emissions = features["token_emissions"]
            structures = torch.stack(
                [dataset.token_structure(int(sample_id))[0] for sample_id in sample_ids]
            ).to(args.device)
            allowed_tags = build_allowed_tag_mask(
                emissions,
                attention_mask,
                structures,
                subword_label_id,
            )
            topk_paths, topk_scores = topk_viterbi_decode(
                model.crf,
                emissions,
                attention_mask.bool(),
                decode_k,
                allowed_tags=allowed_tags,
            )

            for batch_index, sample_id_tensor in enumerate(sample_ids):
                local_sample_id = int(sample_id_tensor)
                active_length = int(attention_mask[batch_index].sum())
                structural = structures[batch_index].cpu()
                _, retained_words, retained_word_indices = dataset.token_structure(
                    local_sample_id
                )
                candidates = deduplicate_topk_paths(
                    topk_paths[batch_index],
                    topk_scores[batch_index],
                    structural_mask=structural,
                    active_length=active_length,
                    index_to_label=index_to_label,
                    outside_label_id=outside_label_id,
                    subword_label_id=subword_label_id,
                    maximum_candidates=cli.k,
                )
                opaque_seed = f"{cli.split_label}:{source_ids[local_sample_id]}"
                candidate_ids = opaque_candidate_ids(
                    opaque_seed,
                    len(candidates),
                    ATOMIC_REVIEW_PROMPT_VERSION,
                )
                for rank_index, candidate in enumerate(candidates):
                    candidate.update(
                        {
                            "candidate_id": candidate_ids[rank_index],
                            "crf_rank": rank_index + 1,
                            "entities": serialize_entities(
                                candidate["bio"], retained_words
                            ),
                        }
                    )
                if not candidates:
                    continue
                baseline_entities = list(candidates[0]["entities"])
                baseline_tuples = _entity_tuples(baseline_entities)
                top1_score = float(candidates[0]["crf_score"])
                word_count = max(len(retained_words), 1)
                edits: list[dict[str, Any]] = []
                for candidate in candidates[1:]:
                    description = describe_atomic_entity_edit(
                        baseline_tuples,
                        _entity_tuples(candidate["entities"]),
                    )
                    if description is None:
                        continue
                    candidate_id = str(candidate["candidate_id"])
                    edit = {
                        "edit_id": "E_" + candidate_id.removeprefix("C_"),
                        "candidate_id": candidate_id,
                        "edit_class": description["edit_class"],
                        "removed_entity": _with_surface(
                            description["removed_entity"], retained_words
                        ),
                        "added_entity": _with_surface(
                            description["added_entity"], retained_words
                        ),
                        "baseline_entities": baseline_entities,
                        "crf_rank": int(candidate["crf_rank"]),
                        "score_gap_per_word": max(
                            top1_score - float(candidate["crf_score"]), 0.0
                        )
                        / word_count,
                    }
                    edits.append(edit)
                edits.sort(
                    key=lambda edit: _edit_order_key(
                        local_sample_id, str(edit["edit_id"])
                    )
                )
                for display_position, edit in enumerate(edits):
                    edit["display_position"] = display_position

                image_names = list(dataset.examples["images"][local_sample_id][:4])
                valid_image_ids = [
                    image_id
                    for image_id, valid in enumerate(image_mask[batch_index].tolist())
                    if valid
                ]
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
                    "provenance": provenance,
                    "sample_id": local_sample_id,
                    "source_sample_id": source_ids[local_sample_id],
                    "split": cli.split_label,
                    "official_test": official_test,
                    "checkpoint_sha256": checkpoint_sha256,
                    "input_sha256": input_sha256,
                    "source_ids_sha256": source_ids_sha256,
                    "words": retained_words,
                    "retained_word_indices": retained_word_indices,
                    "image_names": image_names,
                    "valid_image_ids": valid_image_ids,
                    "baseline_candidate_id": str(candidates[0]["candidate_id"]),
                    "baseline_entities": baseline_entities,
                    "candidates": candidates,
                    "edits": edits,
                    "route_features": {
                        "minimum_atomic_score_gap_per_word": (
                            min(float(edit["score_gap_per_word"]) for edit in edits)
                            if edits
                            else None
                        )
                    },
                }
                records.append(record)

    routes = select_low_margin_atomic_routes(records, cli.route_queries)
    routes_by_id = {int(route["sample_id"]): route for route in routes}
    for record in records:
        route = routes_by_id.get(int(record["sample_id"]))
        record["review_route"] = {
            "selected": route is not None,
            "strategy": "lowest_atomic_score_gap" if route is not None else None,
            "route_order": None if route is None else int(route["route_order"]),
        }
        record["review_fingerprint"] = atomic_review_record_fingerprint(record)

    mode = "w" if cli.overwrite else "x"
    with output.open(mode, encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": ATOMIC_REVIEW_PROMPT_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "input": str(input_path),
        "input_sha256": input_sha256,
        "input_hash_scope": "ordered text+images projection",
        "source_ids": None if sidecar_path is None else str(sidecar_path),
        "source_ids_sha256": source_ids_sha256,
        "split": cli.split_label,
        "official_test": official_test,
        "k": cli.k,
        "decode_k": decode_k,
        "input_samples": len(dataset),
        "exported_samples": len(records),
        "eligible_samples": sum(bool(record["edits"]) for record in records),
        "selected_review_requests": len(routes),
        "routing": "lowest minimum atomic score gap",
        "gold_entity_labels_used": False,
    }
    metadata_path = output.with_name(output.name + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Atomic candidate export complete ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Candidates: {output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
