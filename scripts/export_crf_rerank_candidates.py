"""Export gold-free CRF top-k candidates for constrained LLM reranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.agvs_mner import AgvsMnerModel
from llm.prompts import RERANK_PROMPT_VERSION
from modules.crf_candidates import (
    build_allowed_tag_mask,
    deduplicate_topk_paths,
    serialize_entities,
)
from modules.crf_topk import topk_viterbi_decode
from modules.dataset import MultimodalInferenceDataset
from modules.reranker import (
    build_candidate_provenance,
    candidate_display_order,
    candidate_record_fingerprint,
    normalized_top1_top2_margin,
    opaque_candidate_ids,
    select_pilot_routes,
    sha256_file,
    sha256_inference_projection,
)
from modules.trainer import load_compatible_model_checkpoint
from run_llm import set_seed


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "meta", "test"),
        default="val",
        help="Dataset suffix used when --input-file is omitted.",
    )
    parser.add_argument(
        "--input-file",
        help="Custom inference file, such as the held-out atomic preflight meta pool.",
    )
    parser.add_argument(
        "--split-label",
        help="Provenance label for --input-file (for example meta or fold-0-meta).",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock test export only after the rerank policy is frozen.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=20,
        help="Decode 100 raw paths for K=5 before canonical BIO deduplication.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--uncertainty-queries", type=int, default=100)
    parser.add_argument("--random-control-queries", type=int, default=100)
    parser.add_argument("--route-seed", type=int, default=20261020)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    split_label = cli.split_label or cli.split
    custom_input = Path(cli.input_file) if cli.input_file else None
    custom_is_test = (
        custom_input is not None and custom_input.name.lower().endswith("_test.txt")
    )
    if custom_is_test and split_label != "test":
        raise ValueError("An official *_test.txt input must use --split-label test")
    official_test = split_label == "test" or custom_is_test
    if official_test and not cli.allow_test:
        raise ValueError(
            "Test export is locked during development; freeze the prompt/route/policy "
            "and pass --allow-test for the single final evaluation."
        )
    if cli.k < 2:
        raise ValueError("--k must be at least 2 for reranking")
    if cli.oversample_factor <= 0:
        raise ValueError("--oversample-factor must be positive")
    checkpoint = Path(cli.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    output = Path(cli.output)
    if output.exists() and not cli.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["device"] = cli.device
    if custom_input is not None:
        if not custom_input.is_file():
            raise FileNotFoundError(custom_input)
        config["data_path"] = str(custom_input.parent)
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.num_workers is not None:
        config["num_workers"] = cli.num_workers
    args = SimpleNamespace(**config)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    set_seed(int(getattr(args, "seed", 1234)))

    file_name = (
        custom_input.name
        if custom_input is not None
        else f"MNER-{args.dataset}_{cli.split}.txt"
    )
    input_path = Path(args.data_path) / file_name
    input_sha256 = sha256_inference_projection(input_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    provenance = build_candidate_provenance(
        checkpoint_sha256=checkpoint_sha256,
        input_sha256=input_sha256,
        split=split_label,
        candidate_count=cli.k,
        decode_count=cli.k * cli.oversample_factor,
    )
    dataset = MultimodalInferenceDataset(args, file_name)
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
    decode_k = cli.k * cli.oversample_factor
    with torch.no_grad():
        for batch in tqdm(loader, desc="Export CRF candidates", unit="batch"):
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
                sample_id = int(sample_id_tensor)
                active_length = int(attention_mask[batch_index].sum())
                structural = structures[batch_index].cpu()
                _, retained_words, retained_word_indices = dataset.token_structure(sample_id)
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
                candidate_ids = opaque_candidate_ids(
                    sample_id,
                    len(candidates),
                    RERANK_PROMPT_VERSION,
                )
                permutation = candidate_display_order(
                    sample_id,
                    len(candidates),
                    RERANK_PROMPT_VERSION,
                )
                display_positions = {
                    candidate_index: position
                    for position, candidate_index in enumerate(permutation)
                }
                for rank_index, candidate in enumerate(candidates):
                    candidate_id = candidate_ids[rank_index]
                    candidate.update(
                        {
                            "candidate_id": candidate_id,
                            "crf_rank": rank_index + 1,
                            "display_position": display_positions[rank_index],
                            "entities": serialize_entities(candidate["bio"], retained_words),
                        }
                    )
                image_names = list(dataset.examples["images"][sample_id][:4])
                valid_image_ids = [
                    image_id
                    for image_id, valid in enumerate(image_mask[batch_index].tolist())
                    if valid
                ]
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "prompt_version": RERANK_PROMPT_VERSION,
                    "provenance": provenance,
                    "sample_id": sample_id,
                    "split": split_label,
                    "checkpoint_sha256": checkpoint_sha256,
                    "input_sha256": input_sha256,
                    "k": cli.k,
                    "words": retained_words,
                    "retained_word_indices": retained_word_indices,
                    "image_names": image_names,
                    "valid_image_ids": valid_image_ids,
                    "top1_candidate_id": candidates[0]["candidate_id"],
                    "candidates": candidates,
                }
                record["route_features"] = {
                    "top1_top2_gap_per_word": (
                        normalized_top1_top2_margin(record)
                        if len(candidates) >= 2
                        else None
                    )
                }
                records.append(record)

    routes = select_pilot_routes(
        records,
        uncertainty_count=cli.uncertainty_queries,
        random_count=cli.random_control_queries,
        seed=cli.route_seed,
    )
    routes_by_id = {int(route["sample_id"]): route for route in routes}
    for record in records:
        route = routes_by_id.get(int(record["sample_id"]))
        record["pilot_route"] = {
            "selected": route is not None,
            "arm": None if route is None else route["route_group"],
            "route_order": None if route is None else route["route_order"],
            "route_seed": cli.route_seed,
        }
        record["candidate_fingerprint"] = candidate_record_fingerprint(record)
    mode = "w" if cli.overwrite else "x"
    with output.open(mode, encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    selected = [record for record in records if record["pilot_route"]["selected"]]
    arms: dict[str, int] = {}
    for record in selected:
        arm = record["pilot_route"]["arm"]
        arms[arm] = arms.get(arm, 0) + 1
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": RERANK_PROMPT_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "input": str(input_path),
        "input_sha256": input_sha256,
        "input_hash_scope": "ordered text+images projection",
        "split": split_label,
        "k": cli.k,
        "decode_k": decode_k,
        "samples": len(records),
        "pilot_queries": len(selected),
        "pilot_arms": arms,
        "route_seed": cli.route_seed,
        "gold_entity_labels_used": False,
    }
    metadata_path = output.with_name(output.name + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Candidate export complete ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Candidates: {output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
