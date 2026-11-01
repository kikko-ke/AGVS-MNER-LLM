"""Use cached LLM contexts as sparse entity-consistency views, not extra gold rows."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION, AUGMENTATION_SCHEMA_VERSION,
    extract_protected_entities, labels_for_augmentation, validate_augmentation_output,
)
from modules.text_utils import normalize_words
from modules.trainer import NerTrainer
from scripts.build_llm_augmentation_cache import load_source_records, sha256_file


def verify_internal_manifest(path, dataset):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "context-internal-split-v1" or manifest["dataset"] != dataset:
        raise ValueError("Expected this dataset's training-internal split manifest")
    if manifest.get("official_val_or_test_read") is not False or manifest.get("augmentation_acceptance_used") is not False:
        raise ValueError("Split must not depend on acceptance or official held-out data")
    if sha256_file(Path(manifest["source"])) != manifest["source_sha256"]:
        raise ValueError("Original training source changed")
    if set(manifest["partitions"]) != {"train", "dev"}:
        raise ValueError("Expected exactly internal train and dev partitions")
    for role, value in manifest["partitions"].items():
        split = "train" if role == "train" else "val"
        if Path(value["path"]).resolve() != (path.parent / f"MNER-{dataset}_{split}.txt").resolve():
            raise ValueError("Internal partition must use the declared split directory and filename")
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise ValueError("Internal split file changed")
        if (any(type(i) is not int for i in value["source_ids"])
                or len(value["source_ids"]) != value["count"]
                or len(set(value["source_ids"])) != value["count"]):
            raise ValueError("Internal source IDs must be unique")
    train = set(manifest["partitions"]["train"]["source_ids"])
    dev = set(manifest["partitions"]["dev"]["source_ids"])
    originals = Path(manifest["source"]).read_text(encoding="utf-8").splitlines()
    if train & dev or train | dev != set(range(len(originals))):
        raise ValueError("Train/dev IDs must form a disjoint partition of the source")
    for partition in manifest["partitions"].values():
        actual = Path(partition["path"]).read_text(encoding="utf-8").splitlines()
        if actual != [originals[i] for i in partition["source_ids"]]:
            raise ValueError("Internal rows differ from declared original rows")
    keys = {}
    for i, line in enumerate(originals):
        raw = ast.literal_eval(line)
        keys[i] = (tuple(word.casefold() for word in normalize_words(raw["text"])), tuple(map(str, raw["images"])))
    if {keys[i] for i in train} & {keys[i] for i in dev}:
        raise ValueError("Normalised text/image groups overlap internal train/dev")
    return manifest


def wordpiece_locations(words, tokenizer):
    positions, offset = [], 1
    for word in normalize_words(words):
        length = len(tokenizer.tokenize(word))
        positions.append((offset, offset + length) if length else None)
        offset += length
    return positions


def entity_alignment(words, labels, rewritten_words, rewritten_labels, tokenizer, max_length):
    original_entities = extract_protected_entities(words, labels)
    rewritten_entities = extract_protected_entities(rewritten_words, rewritten_labels)
    if len(original_entities) != len(rewritten_entities):
        raise ValueError("Entity counts differ between source and rewrite")
    source_locations = wordpiece_locations(words, tokenizer)
    destination_locations = wordpiece_locations(rewritten_words, tokenizer)
    aligned = []
    for first, second in zip(original_entities, rewritten_entities):
        if first.label != second.label or normalize_words(first.tokens) != normalize_words(second.tokens):
            raise ValueError("Entity contract differs between source and rewrite")
        left = source_locations[first.start:first.end]
        right = destination_locations[second.start:second.end]
        if any(location is None or location[1] > max_length - 1 for location in left + right):
            continue  # Skip the whole entity if any part does not fit before SEP.
        aligned.append([(a[0], b[0]) for a, b in zip(left, right)])
    source, target, weights = torch.zeros(max_length, dtype=torch.long), torch.zeros(max_length, dtype=torch.long), torch.zeros(max_length)
    index = 0
    for entity in aligned:
        for first, second in entity:
            source[index], target[index] = first, second
            weights[index] = 1.0 / (len(aligned) * len(entity))
            index += 1
    return source, target, weights


class ContextViewDataset(Dataset):
    def __init__(self, base, manifest_path, cache_path, dataset):
        self.base = base
        manifest = verify_internal_manifest(manifest_path, dataset)
        ids = manifest["partitions"]["train"]["source_ids"]
        if len(ids) != len(base):
            raise ValueError("Base dataset must be the complete internal training partition")
        source = {row["sample_id"]: row for row in load_source_records(Path(manifest["source"]))}
        metadata = json.loads(Path(str(cache_path) + ".meta.json").read_text(encoding="utf-8"))
        if (metadata.get("schema_version") != AUGMENTATION_SCHEMA_VERSION
                or metadata.get("prompt_version") != AUGMENTATION_PROMPT_VERSION
                or metadata.get("input_sha256") != manifest["source_sha256"]):
            raise ValueError("Existing augmentation cache must belong to the original outer training file")
        selected = {}
        for row in metadata["selected"]:
            i = row["sample_id"]
            if type(i) is not int or i in selected or i not in source or row["source_fingerprint"] != source[i]["source_fingerprint"]:
                raise ValueError("Invalid cache candidate/fingerprint")
            selected[i] = row
        local_ids = {source_id: index for index, source_id in enumerate(ids)}
        for i, index in local_ids.items():
            row = source[i]
            if (normalize_words(row["words"]) != base.examples["words"][index]
                    or row["labels"] != base.examples["labels"][index]
                    or row["images"] != base.examples["images"][index]):
                raise ValueError("Internal source mapping differs from the loaded dataset")
        self.views, seen = {}, set()
        with Path(cache_path).open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                i = row["sample_id"]
                if type(i) is not int or i in seen or i not in selected or row.get("source_fingerprint") != source[i]["source_fingerprint"]:
                    raise ValueError("Invalid/duplicate cache row")
                seen.add(i)
                if i not in local_ids:
                    continue  # Never use internal-dev augmentation outcomes.
                if (row.get("schema_version") != AUGMENTATION_SCHEMA_VERSION or row.get("prompt_version") != AUGMENTATION_PROMPT_VERSION
                        or row.get("llm_model") != metadata["llm_model"]):
                    raise ValueError("Cache schema/prompt/model changed")
                if row.get("status") not in {"accepted", "abstained", "rejected"}:
                    raise ValueError("Unknown cache resolution status")
                if row["status"] != "accepted":
                    continue
                original = source[i]
                response = validate_augmentation_output(
                    {key: row.get(key) for key in ("abstain", "context_segments", "used_image_ids", "change_summary", "abstain_reason")},
                    original["words"], original["entities"], set(row["valid_image_ids"]),
                    **{key: metadata[key] for key in ("min_context_change_ratio", "max_context_change_ratio", "min_length_ratio", "max_length_ratio", "max_words")},
                )
                rebuilt_labels = labels_for_augmentation(len(response.tokens), original["entities"], response.entity_spans)
                if response.abstain or row["tokens"] != list(response.tokens) or row["labels"] != rebuilt_labels:
                    raise ValueError("Accepted cache content violates its reconstructed entity contract")
                alignment = entity_alignment(original["words"], original["labels"], response.tokens,
                                             rebuilt_labels, base.tokenizer, base.max_sequence_length)
                if alignment[2].sum() > 0:
                    encoded = base._encode_text(list(response.tokens), rebuilt_labels)
                    self.views[local_ids[i]] = (*encoded[:3], *alignment)
        if seen != set(selected):
            raise ValueError("Resolve the existing cache before this controlled experiment")
        if not self.views:
            raise ValueError("No usable, fully aligned LLM views remain in internal training")
        self.summary = {"training_examples": len(base), "usable_training_views": len(self.views),
                        "original_example_repetitions_per_epoch": 1, "internal_dev_views_used": 0,
                        "cache_sha256": sha256_file(Path(cache_path)), "manifest_sha256": sha256_file(Path(manifest_path))}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        main = self.base[index]
        if index in self.views:
            view = self.views[index]
        else:
            length = len(main[0])
            view = (main[0], main[1], main[2], torch.zeros(length, dtype=torch.long),
                    torch.zeros(length, dtype=torch.long), torch.zeros(length))
        return (*main, *view)


def entity_consistency_kl(teacher, student, source_positions, target_positions, weights, temperature, batch_size):
    if temperature <= 0 or batch_size < 1:
        raise ValueError("Positive temperature and original batch_size required")
    size = teacher.shape[-1]
    left = teacher.gather(1, source_positions.unsqueeze(-1).expand(-1, -1, size)).detach()
    right = student.gather(1, target_positions.unsqueeze(-1).expand(-1, -1, size))
    token_kl = F.kl_div(F.log_softmax(right / temperature, dim=-1),
                      F.softmax(left / temperature, dim=-1), reduction="none").sum(-1)
    return (token_kl * weights).sum() * temperature ** 2 / batch_size


class ContextConsistencyTrainer(NerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_epoch, self.context_step = 0, 0

    def _run_epoch(self, data_loader, training, progress_bar=None):
        if training:
            self.context_epoch += 1
        return super()._run_epoch(data_loader, training, progress_bar)

    def _step(self, batch):
        if len(batch) == 6:
            return super()._step(batch)
        ids, types, mask, pixels, labels, image_mask, alt_ids, alt_types, alt_mask, src, dst, weights = batch
        decoded, loss, features = self.model(input_ids=ids, token_type_ids=types, attention_mask=mask,
                                              image_pixel_values=pixels, image_mask=image_mask, labels=labels, return_features=True)
        self.context_step += 1
        mode = self.args.consistency_mode
        eligible = weights.sum(-1).gt(0)
        active = self.model.training and mode != "none" and self.context_epoch > self.args.consistency_warmup_epochs and eligible.any()
        if active:
            indices = eligible.nonzero(as_tuple=True)[0]
            if mode == "dropout":
                view_ids, view_types, view_mask, target = ids, types, mask, src
            else:
                view_ids, view_types, view_mask, target = alt_ids, alt_types, alt_mask, dst
            original_components = dict(self.model.last_loss_components)
            # Auxiliary stochasticity must not shift later main-view dropout or
            # the original DataLoader RNG stream. Restore all visible RNGs.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed((self.args.seed * 1000003 + self.context_step) % (2 ** 63 - 1))
                _, _, alternate = self.model(input_ids=view_ids[indices], token_type_ids=view_types[indices],
                                              attention_mask=view_mask[indices], image_pixel_values=pixels[indices],
                                              image_mask=image_mask[indices], labels=None, return_features=True)
            consistency = entity_consistency_kl(features["token_emissions"][indices], alternate["token_emissions"],
                                                 src[indices], target[indices], weights[indices],
                                                 self.args.consistency_temperature, len(ids))
            loss = loss + self.args.consistency_weight * consistency
            original_components["entity_context_consistency"] = consistency.detach()
            original_components["view_fraction"] = eligible.float().mean()
            self.model.last_loss_components = original_components
        return mask, labels, decoded, loss
