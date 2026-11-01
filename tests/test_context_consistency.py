from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from llm.augmentation_schema import (
    AUGMENTATION_PROMPT_VERSION, AUGMENTATION_SCHEMA_VERSION,
    validate_augmentation_output, labels_for_augmentation,
)
from modules.context_consistency import (
    ContextConsistencyTrainer, ContextViewDataset, entity_alignment,
    entity_consistency_kl, verify_internal_manifest,
)
from modules.dataset import MultimodalNerDataset, MnerProcessor
from modules.text_utils import normalize_words
from scripts.build_llm_augmentation_cache import load_source_records, sha256_file
from scripts.create_context_internal_split import create_split


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    def tokenize(self, word):
        if word == "longerword":
            return ["longer", "##word"]
        return [word.lower()] if word else []

    def convert_tokens_to_ids(self, words):
        return [sum(map(ord, word)) % 80 + 3 for word in words]


def test_alignment_uses_first_wordpieces_and_skips_whole_truncated_entities():
    words, labels = ["A", "longerword", "B", "tail"], ["B-ORG", "I-ORG", "O", "B-PER"]
    rewritten = ["x", "x", *words]
    rewritten_labels = ["O", "O", *labels]
    src, dst, weights = entity_alignment(words, labels, rewritten, rewritten_labels, Tokenizer(), 9)
    assert src[:3].tolist() == [1, 2, 5] and dst[:3].tolist() == [3, 4, 7]
    assert weights[:3].tolist() == [0.25, 0.25, 0.5] and weights.sum() == 1
    _, _, partial = entity_alignment(words, labels, rewritten, rewritten_labels, Tokenizer(), 6)
    assert partial.sum() == 0
    with pytest.raises(ValueError, match="Entity contract"):
        entity_alignment(words, labels, ["x", "x", "different", *words[1:]], rewritten_labels, Tokenizer(), 9)


def test_kl_stops_teacher_gradient_and_normalises_by_original_batch():
    torch.manual_seed(8)
    teacher = torch.randn(1, 4, 5, requires_grad=True)
    student = torch.randn(1, 4, 5, requires_grad=True)
    source, target, weights = torch.tensor([[1, 0]]), torch.tensor([[2, 0]]), torch.tensor([[1., 0.]])
    full = entity_consistency_kl(teacher, student, source, target, weights, 2, 1)
    sparse = entity_consistency_kl(teacher, student, source, target, weights, 2, 4)
    assert float(sparse.detach()) == pytest.approx(float(full.detach()) / 4)
    sparse.backward()
    assert teacher.grad is None and student.grad is not None
    assert student.grad[:, 2].abs().sum() > 0 and student.grad[:, 0].abs().sum() == 0


class FakeBase:
    _encode_text = MultimodalNerDataset._encode_text

    def __init__(self, path):
        raw = [ast.literal_eval(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.examples = {"words": [normalize_words(r["text"]) for r in raw],
                         "labels": [r["label"] for r in raw], "images": [r["images"] for r in raw]}
        self.max_sequence_length, self.tokenizer = 16, Tokenizer()
        self.processor = MnerProcessor
        self.label_mapping = MnerProcessor.get_label_mapping(include_subword_label=True)

    def __len__(self):
        return len(self.examples["words"])

    def __getitem__(self, i):
        ids, types, mask, labels = self._encode_text(self.examples["words"][i], self.examples["labels"][i])
        return ids, types, mask, torch.zeros(4, 3, 2, 2), labels, torch.tensor([1, 0, 0, 0])


def fixture(tmp_path):
    records = [{"text": ["today", "Acme", "Corp", "near", f"scene{i}"],
                "label": ["O", "B-ORG", "I-ORG", "O", "O"], "images": ["a.jpg"]} for i in range(24)]
    records.append({**records[0], "text": ["TODAY", *records[0]["text"][1:]]})
    source = tmp_path / "MNER-MI_train.txt"
    source.write_text("\n".join(map(repr, records)) + "\n", encoding="utf-8")
    path = create_split(source, tmp_path / "split", "MI")
    manifest = verify_internal_manifest(path, "MI")
    metadata = {"schema_version": AUGMENTATION_SCHEMA_VERSION, "prompt_version": AUGMENTATION_PROMPT_VERSION,
                "input_sha256": sha256_file(source), "llm_model": "test", "min_context_change_ratio": 0.0,
                "max_context_change_ratio": 1.0, "min_length_ratio": 0.01, "max_length_ratio": 100., "max_words": 62}
    cached, selected = [], []
    for row in load_source_records(source):
        selected.append({"sample_id": row["sample_id"], "source_fingerprint": row["source_fingerprint"]})
        payload = {"abstain": False, "used_image_ids": [0], "context_segments": [["today"], ["around", "the", "bright", row["words"][-1]]],
                   "change_summary": "visible context", "abstain_reason": ""}
        response = validate_augmentation_output(payload, row["words"], row["entities"], {0},
                                                min_context_change_ratio=0, max_context_change_ratio=1,
                                                min_length_ratio=0.01, max_length_ratio=100)
        cached.append({"schema_version": AUGMENTATION_SCHEMA_VERSION, "prompt_version": AUGMENTATION_PROMPT_VERSION,
                       "sample_id": row["sample_id"], "source_fingerprint": row["source_fingerprint"], "llm_model": "test",
                       "status": "accepted", "valid_image_ids": [0], **response.to_dict(),
                       "labels": labels_for_augmentation(len(response.tokens), row["entities"], response.entity_spans)})
    metadata["selected"] = selected
    cache = tmp_path / "cache.jsonl"
    cache.write_text("".join(json.dumps(row) + "\n" for row in cached), encoding="utf-8")
    Path(str(cache) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    base = FakeBase(Path(manifest["partitions"]["train"]["path"]))
    return source, path, manifest, cache, cached, base


def test_grouped_split_is_reproducible_and_uses_only_original_train(tmp_path):
    source, path, manifest, _, _, _ = fixture(tmp_path)
    second_path = create_split(source, tmp_path / "second", "MI")
    second = verify_internal_manifest(second_path, "MI")
    for name in ("train", "dev"):
        assert manifest["partitions"][name]["source_ids"] == second["partitions"][name]["source_ids"]
    train, dev = (set(manifest["partitions"][name]["source_ids"]) for name in ("train", "dev"))
    assert not train & dev and train | dev == set(range(25))
    assert (0 in train) == (24 in train)
    assert manifest["official_val_or_test_read"] is False and manifest["augmentation_acceptance_used"] is False
    with pytest.raises(FileExistsError):
        create_split(source, path.parent, "MI")


def test_view_dataset_preserves_each_main_example_and_ignores_dev_cache_bodies(tmp_path):
    _, path, manifest, cache, cached, base = fixture(tmp_path)
    dev = set(manifest["partitions"]["dev"]["source_ids"])
    for row in cached:
        if row["sample_id"] in dev:
            row["tokens"] = ["invalid unused internal-dev outcome"]
    cache.write_text("".join(json.dumps(row) + "\n" for row in cached), encoding="utf-8")
    dataset = ContextViewDataset(base, path, cache, "MI")
    assert len(dataset) == len(base) and dataset.summary["usable_training_views"] == len(base)
    assert dataset.summary["internal_dev_views_used"] == 0
    for i in range(len(base)):
        assert all(torch.equal(a, b) for a, b in zip(dataset[i][:6], base[i]))
        assert dataset[i][-1].sum() == 1


def test_changed_train_cache_contract_and_split_file_fail(tmp_path):
    _, path, manifest, cache, cached, base = fixture(tmp_path)
    train_id = manifest["partitions"]["train"]["source_ids"][0]
    cached[train_id]["tokens"] = ["corrupt source view"]
    cache.write_text("".join(json.dumps(row) + "\n" for row in cached), encoding="utf-8")
    with pytest.raises(ValueError, match="entity contract"):
        ContextViewDataset(base, path, cache, "MI")
    Path(manifest["partitions"]["dev"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="split file changed"):
        verify_internal_manifest(path, "MI")


def test_manifest_cannot_redirect_the_development_loader(tmp_path):
    _, path, manifest, _, _, _ = fixture(tmp_path)
    manifest["partitions"]["dev"]["path"] = str(tmp_path / "other_val.txt")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split directory and filename"):
        verify_internal_manifest(path, "MI")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(200, 8)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(8, 10)
        self.calls = 0

    def forward(self, input_ids, attention_mask, token_type_ids, image_pixel_values, image_mask, labels=None, return_features=False):
        self.calls += 1
        emissions = self.classifier(self.dropout(self.embedding(input_ids)))
        decoded = [row[:int(mask.sum())].tolist() for row, mask in zip(emissions.argmax(-1), attention_mask)]
        loss = None if labels is None else F.cross_entropy(emissions[attention_mask.bool()], labels[attention_mask.bool()])
        if loss is not None:
            self.last_loss_components = {"ner": loss.detach()}
        if return_features:
            return decoded, loss, {"token_emissions": emissions}
        return decoded, loss


def trainer(model, mode, epoch=3):
    value = object.__new__(ContextConsistencyTrainer)
    value.model = model
    value.args = SimpleNamespace(seed=1234, consistency_mode=mode, consistency_weight=0.5,
                                 consistency_temperature=2., consistency_warmup_epochs=2)
    value.context_epoch, value.context_step = epoch, 0
    return value


def batch():
    ids = torch.arange(12).view(2, 6)
    mask = torch.ones_like(ids)
    positions = torch.zeros_like(ids)
    positions[0, 0] = 2
    weights = torch.zeros_like(ids, dtype=torch.float)
    weights[0, 0] = 1
    return (ids, ids * 0, mask, torch.zeros(2, 4, 3, 2, 2), ids * 0, torch.ones(2, 4),
            ids + 1, ids * 0, mask, positions, positions + 1, weights)


@pytest.mark.parametrize("mode", ["dropout", "llm_context"])
def test_auxiliary_rng_does_not_shift_main_dropout_stream_and_loss_is_finite(mode):
    torch.manual_seed(19)
    model = TinyModel().train()
    plain, augmented = trainer(copy.deepcopy(model), "none"), trainer(copy.deepcopy(model), mode)
    data = batch()
    torch.manual_seed(55)
    first = plain._step(data)
    expected_rng = torch.rand(8)
    torch.manual_seed(55)
    second = augmented._step(data)
    actual_rng = torch.rand(8)
    assert torch.equal(expected_rng, actual_rng) and first[2] == second[2]
    assert plain.model.calls == 1 and augmented.model.calls == 2
    assert torch.isfinite(second[3]) and second[3] >= first[3] - 1e-6
    second[3].backward()
    assert augmented.model.embedding.weight.grad.abs().sum() > 0
    assert "entity_context_consistency" in augmented.model.last_loss_components


def test_warmup_no_view_and_validation_do_not_add_forward_passes():
    model = TinyModel().train()
    value = trainer(model, "llm_context", epoch=2)
    data = batch()
    value._step(data)
    assert model.calls == 1
    value.context_epoch = 3
    zeros = (*data[:-1], torch.zeros_like(data[-1]))
    value._step(zeros)
    assert model.calls == 2
    model.eval()
    value._step(data[:6])
    assert model.calls == 3


def test_entrypoint_trains_and_saves_without_loading_any_test_file(tmp_path, monkeypatch):
    import run_context_consistency as entry

    _, path, manifest, cache, _, _ = fixture(tmp_path)
    loaded = []

    def dataset(args, filename):
        loaded.append(filename)
        assert filename in {"MNER-MI_train.txt", "MNER-MI_val.txt"}
        return FakeBase(Path(args.data_path) / filename)

    monkeypatch.setattr(entry, "MultimodalNerDataset", dataset)
    monkeypatch.setattr(entry, "AgvsMnerModel", lambda labels, args: TinyModel())
    monkeypatch.setattr(entry, "configure_logging", lambda output: None)
    monkeypatch.setattr("sys.argv", [
        "run_context_consistency.py", "--dataset", "MI", "--device", "cpu",
        "--data-path", str(path.parent), "--internal-manifest", str(path),
        "--augmentation-cache", str(cache), "--consistency-mode", "llm_context",
        "--consistency-warmup-epochs", "0", "--num-epochs", "1", "--batch-size", "32",
        "--output-dir", str(tmp_path / "outputs"),
    ])
    entry.main()
    checkpoints = list((tmp_path / "outputs").glob("*/best_model.pt"))
    assert len(checkpoints) == 1
    config = json.loads(checkpoints[0].with_name("config.json").read_text(encoding="utf-8"))
    assert config["skip_test"] is True and config["consistency_mode"] == "llm_context"
    assert loaded == ["MNER-MI_train.txt", "MNER-MI_val.txt"]
    assert checkpoints[0].with_name("metrics.csv").is_file()


def selection_fixture(tmp_path):
    from PIL import Image
    from modules.dropout_selection import prepare_cohorts

    _, path, manifest, cache, cached, base = fixture(tmp_path)
    selected = set(manifest["partitions"]["train"]["source_ids"][:4])
    for row in cached:
        if row["sample_id"] not in selected:
            row["status"] = "rejected"
    cache.write_text("".join(json.dumps(row) + "\n" for row in cached), encoding="utf-8")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.jpg")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.txt").write_text("fixture vocabulary", encoding="utf-8")
    cohort = prepare_cohorts(base, path, cache, "MI", tmp_path, tmp_path, tokenizer)
    cohort_path = tmp_path / "cohorts.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    return path, manifest, cache, base, tokenizer, cohort_path, cohort


def test_dropout_cohorts_match_training_features_and_use_only_source_entities(tmp_path):
    from modules.dropout_selection import ARMS, DropoutSelectionDataset

    path, manifest, cache, base, tokenizer, cohort_path, cohort = selection_fixture(tmp_path)
    assert cohort["ready_for_training"] is True
    assert cohort["balance_check"]["maximum"] == 0
    assert cohort["eligible_per_arm"] == 4 and cohort["official_val_or_test_read"] is False
    dev_ids = set(manifest["partitions"]["dev"]["source_ids"])
    for arm in ARMS:
        assert len(set(cohort["arms"][arm]["source_ids"])) == 4
        assert not set(cohort["arms"][arm]["source_ids"]) & dev_ids
        dataset = DropoutSelectionDataset(base, cohort_path, arm, path, cache, "MI", tokenizer)
        assert len(dataset) == len(base)
        for i in range(len(dataset)):
            row = dataset[i]
            assert all(torch.equal(a, b) for a, b in zip(row[:6], base[i]))
            assert torch.equal(row[0], row[6]) and torch.equal(row[9], row[10])
        assert sum(float(dataset[i][-1].sum()) for i in range(len(dataset))) == 4


@pytest.mark.parametrize("changed", ["cache", "tokenizer", "image", "dev_id", "positions"])
def test_dropout_cohort_bindings_reject_changed_inputs_or_heldout_ids(tmp_path, changed):
    from modules.dropout_selection import DropoutSelectionDataset

    path, manifest, cache, base, tokenizer, cohort_path, cohort = selection_fixture(tmp_path)
    if changed == "cache":
        cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif changed == "tokenizer":
        (tokenizer / "vocab.txt").write_text("changed", encoding="utf-8")
    elif changed == "image":
        (tmp_path / "a.jpg").write_bytes(b"changed")
    elif changed == "dev_id":
        cohort["arms"]["llm_selected"]["source_ids"][0] = manifest["partitions"]["dev"]["source_ids"][0]
    else:
        first = cohort["arms"]["llm_selected"]["source_ids"][0]
        cohort["arms"]["llm_selected"]["source_alignments"][str(first)]["positions"] = [0]
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    with pytest.raises(ValueError):
        DropoutSelectionDataset(base, cohort_path, "llm_selected", path, cache, "MI", tokenizer)


def test_dropout_cohort_full_overlap_blocks_training(tmp_path):
    from modules.dropout_selection import prepare_cohorts

    _, path, _, cache, _, base = fixture(tmp_path)
    from PIL import Image
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.jpg")
    cohort = prepare_cohorts(base, path, cache, "MI", tmp_path, tmp_path, tmp_path)
    assert cohort["overlap_check"]["fraction"] == 1
    assert cohort["ready_for_training"] is False


def test_dropout_selection_entrypoint_saves_completed_binding(tmp_path, monkeypatch):
    import run_context_consistency as entry
    from scripts.evaluate_dropout_selection import completed_run
    from modules.dropout_selection import FIXED_TRAINING

    path, _, cache, _, tokenizer, cohort_path, _ = selection_fixture(tmp_path)
    monkeypatch.setattr(entry, "MultimodalNerDataset", lambda args, filename: FakeBase(Path(args.data_path) / filename))
    monkeypatch.setattr(entry, "AgvsMnerModel", lambda labels, args: TinyModel())
    monkeypatch.setattr(entry, "configure_logging", lambda output: None)
    root = tmp_path / "outputs"
    monkeypatch.setattr("sys.argv", ["run_context_consistency.py", "--dataset", "MI", "--device", "cpu",
        "--data-path", str(path.parent), "--internal-manifest", str(path), "--augmentation-cache", str(cache),
        "--consistency-mode", "dropout", "--dropout-selection-manifest", str(cohort_path),
        "--dropout-selection-arm", "llm_selected", "--bert-model", str(tokenizer),
        "--num-epochs", "1", "--batch-size", "32", "--output-dir", str(root / "llm_selected_seed1234")])
    entry.main()
    checkpoint = list(root.glob("*/*/best_model.pt"))[0]
    completed = json.loads(checkpoint.with_name("completed.json").read_text(encoding="utf-8"))
    assert completed["checkpoint_sha256"] == sha256_file(checkpoint)
    assert completed["test_read"] is False
    # The production reader must reject our deliberately non-protocol smoke run.
    with pytest.raises(ValueError, match="fixed dropout selection protocol"):
        completed_run(root, "llm_selected", cohort_path)
    config_path = checkpoint.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(FIXED_TRAINING)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed["config_sha256"] = sha256_file(config_path)
    checkpoint.with_name("completed.json").write_text(json.dumps(completed), encoding="utf-8")
    assert completed_run(root, "llm_selected", cohort_path)[0] == checkpoint
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="binding changed"):
        completed_run(root, "llm_selected", cohort_path)
