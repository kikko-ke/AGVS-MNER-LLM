"""CPU checks for the new baseline's most consequential implementation choices."""
import json

import pytest

from modules.tpm_llm_v1.common import (
    BIO, TAGS, filtered_indices, metrics, project_path, read_rows, spans)


def test_bio_metrics_keep_original_boundaries():
    row = {"tokens": ["New", "York", "today"], "labels": ["B-LOC", "I-LOC", "O"]}
    assert metrics([row], [row["labels"]])["f1"] == 1
    result = metrics([row], [["B-LOC", "O", "O"]])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert spans(["I-PER", "I-PER", "B-PER", "O"]) == {(0, 2, "PER"), (2, 3, "PER")}


def test_source_reader_normalizes_misc_and_rejects_misalignment(tmp_path):
    path = tmp_path / "source.txt"
    row = {"text": ["name"], "images": ["first.jpg", "second.jpg"], "label": ["B-MISC"]}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = read_rows(path, "train")[0]
    assert result["labels"] == ["B-MIS"]
    assert result["images"] == row["images"]
    assert "labels" not in read_rows(path, "test", labels=False)[0]
    row["label"] = []
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Word/gold mismatch"):
        read_rows(path, "train")


def test_overlap_filter_handles_transitive_image_links():
    reference = [{"tokens": ["known"], "images": ["a"]}]
    candidates = [{"tokens": ["first"], "images": ["a", "b"]},
                  {"tokens": ["second"], "images": ["b"]},
                  {"tokens": ["unlinked"], "images": ["c"]},
                  {"tokens": ["KNOWN"], "images": []}]
    assert filtered_indices(reference, candidates, {"a": "ha", "b": "hb", "c": "hc"}) == [2]


def test_completed_stage_refuses_changed_inputs_and_outputs(tmp_path):
    from modules.tpm_llm_v1.pipeline import completed, seal
    source, result, done = (tmp_path / name for name in ("source", "result", "done.json"))
    source.write_text("original input", encoding="utf-8")
    result.write_text("original output", encoding="utf-8")
    protocol = {"protocol_id": "fixture"}
    seal(done, protocol, [result], inputs=[source])
    assert completed(done, protocol)
    source.write_text("different input", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed stage artifact"):
        completed(done, protocol)
    source.write_text("original input", encoding="utf-8")
    result.write_text("different output", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed stage artifact"):
        completed(done, protocol)


def test_subword_alignment_is_once_and_first_piece_projection():
    pytest.importorskip("torch")
    from modules.tpm_llm_v1.data import encode_words

    class Tokenizer:
        cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

        def tokenize(self, word):
            return ["mu", "##lti"] if word == "multi" else [word]

        def convert_tokens_to_ids(self, pieces):
            return [{"mu": 11, "##lti": 12, "London": 13}[piece] for piece in pieces]

        def encode_plus(self, *args, **kwargs):
            raise AssertionError("Already-tokenized WordPieces must not be encoded again")

    row = {"tokens": ["multi", "London"], "labels": ["B-ORG", "B-LOC"]}
    value = encode_words(row, Tokenizer(), 4)
    assert value["input_ids"].tolist() == [101, 11, 12, 102]
    assert value["positions"] == [1] and value["word_indices"] == [0]
    assert value["labels"].tolist() == [9, TAGS.index("B-ORG"), 9, 9]
    assert project_path([9, TAGS.index("B-ORG"), 9, 9], [1], [0], 2) == ["B-ORG", "O"]


def test_last_subtoken_uses_character_overlap_and_ignores_specials():
    pytest.importorskip("torch")
    from modules.tpm_llm_v1.features import last_overlapping, word_offsets
    text, bounds = word_offsets(["New", "York", ""])
    assert text == "New York "
    assert last_overlapping(bounds, [(0, 0), (0, 2), (2, 3), (3, 8), (0, 0)]) == [2, 3, -1]


def test_fusion_centers_nine_labels_masks_alignment_and_keeps_x():
    torch = pytest.importorskip("torch")
    from modules.tpm_llm_v1.features import fuse
    emissions = torch.zeros(1, 5, 10)
    rows = [{"tokens": ["first", "second"]}]
    mapping = [{"positions": [1, 3], "word_indices": [0, 1]}]
    logits = [torch.tensor([[5., 2., 1., 1., 0., 0., 0., 0., 0.], [100.] * 9])]
    valid = [torch.tensor([True, False])]
    result = fuse(emissions, rows, mapping, logits, valid, 1)
    assert result[0, 1, 0] == 4 and result[0, 1, 9] == 4
    assert result[0, [0, 2, 3, 4]].abs().sum() == 0
    assert torch.equal(fuse(emissions, rows, mapping, logits, valid, 0), emissions)
    logits[0][0, 0] = 1000
    clipped = fuse(emissions, rows, mapping, logits, valid, 2)
    assert clipped.abs().max() == 16


def test_centering_without_clip_preserves_crf_paths_including_x():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcrf")
    from torchcrf import CRF
    from modules.tpm_llm_v1.features import fuse
    torch.manual_seed(3)
    emissions = torch.randn(1, 4, 10)
    rows = [{"tokens": ["one", "two"]}]
    mapping = [{"positions": [1, 2], "word_indices": [0, 1]}]
    logits = [torch.randn(2, 9)]
    valid = [torch.tensor([True, True])]
    raw = emissions.clone()
    raw[0, [1, 2]] += logits[0][:, [*range(9), 0]]
    centered = fuse(emissions, rows, mapping, logits, valid, 1, clip=False)
    decoder = CRF(10, batch_first=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert decoder.decode(raw, mask=mask) == decoder.decode(centered, mask=mask)


def test_dev_calibration_uses_smallest_tied_weight_and_needs_no_test_rows(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcrf")
    from modules.tpm_llm_v1 import pipeline
    from modules.tpm_llm_v1.common import load_rows, save_rows
    from torchcrf import CRF
    row = {"source_id": "dev:0", "tokens": ["London"], "labels": ["B-LOC"]}
    (tmp_path / "data").mkdir()
    save_rows(tmp_path / "data/dev.jsonl", [row])
    decoder = CRF(10, batch_first=True)
    for parameter in decoder.parameters():
        parameter.data.zero_()
    emissions = torch.zeros(1, 3, 10)
    emissions[0, [0, 2], 9] = 5
    emissions[0, 1, 0] = 2
    emissions[0, 1, TAGS.index("B-LOC")] = 1
    native = {"emissions": emissions, "mask": torch.ones(1, 3, dtype=torch.bool),
              "mappings": [{"positions": [1], "word_indices": [0]}],
              "crf": decoder.state_dict(), "source_ids": ["dev:0"]}
    native["metrics"] = metrics([row], pipeline.decode(native, [row]))
    (tmp_path / "emissions").mkdir()
    (tmp_path / "emissions/seed1234_dev.pt").write_bytes(b"fixed development artifact")
    monkeypatch.setattr(pipeline, "base_cache", lambda *args: native)

    def head(root, protocol, seed, encoder, architecture):
        file = tmp_path / f"{encoder}_{architecture}.pt"
        file.write_bytes(b"fixed fitted head")
        logits = torch.zeros(1, 9)
        if encoder == "qwen" and architecture == "bigru":
            logits[0, BIO.index("B-LOC")] = 9
        return file, {"dev_logits": [logits], "dev_aligned": [torch.tensor([True])],
                      "dev_source_ids": ["dev:0"]}

    monkeypatch.setattr(pipeline, "checked_head", head)
    protocol = {"protocol_id": "fixture", "config": {"architectures": ["bigru", "linear"],
                                                       "coefficient_grid": [0., 0.25, 0.5, 1.]}}
    pipeline.calibrate(tmp_path, protocol)
    report = pipeline.read_json(tmp_path / "calibration/dev_report.json")
    assert report["coefficients"]["qwen_bigru"] == 0.25
    assert report["coefficients"]["bert_bigru"] == 0
    assert report["selected"]["qwen_bigru"]["metrics"]["f1"] == 1
    assert report["test_metrics_used_for_selection"] is False
    assert load_rows(tmp_path / "data/dev.jsonl") == [row]


def test_bigru_does_not_read_padding_features():
    torch = pytest.importorskip("torch")
    from modules.tpm_llm_v1.features import WordHead
    torch.manual_seed(11)
    head = WordHead().eval()
    features = torch.randn(2, 4, 768)
    changed = features.clone()
    changed[0, 2:] = 1000
    with torch.no_grad():
        original = head(features, [2, 4])
        modified = head(changed, [2, 4])
    torch.testing.assert_close(original[0, :2], modified[0, :2])
    assert sum(parameter.numel() for parameter in head.parameters()) == 693513


def test_head_fits_gold_rows_and_serializes_safely(tmp_path):
    torch = pytest.importorskip("torch")
    from modules.tpm_llm_v1.features import fit_head, infer_head
    from modules.tpm_llm_v1.pipeline import load_torch, save_torch
    torch.manual_seed(10)
    features = [torch.randn(2, 768), torch.randn(1, 768)]
    rows = [{"tokens": ["New", "York"], "labels": ["B-LOC", "I-LOC"]},
            {"tokens": ["London"], "labels": ["B-LOC"]}]
    config = {"head_lr": 1e-3, "feature_batch_size": 2, "head_epochs": 1}
    head, history = fit_head(features, rows, config, 1234, device="cpu")
    values = infer_head(head, features, device="cpu")
    assert [list(value.shape) for value in values] == [[2, 9], [1, 9]]
    assert history[0]["mean_loss"] > 0
    path = tmp_path / "head.pt"
    save_torch(path, {"head": head.state_dict(), "history": history, "torch": str(torch.__version__)})
    restored = load_torch(path)
    for key, value in head.state_dict().items():
        assert torch.equal(value, restored["head"][key])


def tiny_baseline():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcrf")
    transformers = pytest.importorskip("transformers")
    from modules.tpm_llm_v1.model import TPMMI
    torch.manual_seed(5)
    bert_config = transformers.BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=4, intermediate_size=64)
    vit_config = transformers.ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                                       intermediate_size=64, image_size=8, patch_size=4)
    bert_config._attn_implementation = "eager"
    vit_config._attn_implementation = "eager"
    model = TPMMI(transformers.BertModel(bert_config), transformers.ViTModel(vit_config),
                  prompt_width=48, image_heads=4, image_ffn=64)
    batch = {"input_ids": torch.tensor([[1, 3, 4, 2, 0], [1, 6, 7, 8, 2]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]).bool(),
             "token_type_ids": torch.zeros(2, 5, dtype=torch.long),
             "pixels": torch.randn(2, 4, 3, 8, 8),
             "image_mask": torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]).bool(),
             "labels": torch.tensor([[9, 3, 4, 9, 0], [9, 0, 0, 0, 9]])}
    return torch, model, batch


def test_tpm_has_no_cross_post_attention_and_handles_all_missing_images():
    torch, model, batch = tiny_baseline()
    model.eval()
    with torch.no_grad():
        together = model(batch)
        alone = model({key: value[:1] for key, value in batch.items()})
        permuted = model({key: value.flip(0) for key, value in batch.items()}).flip(0)
        changed = {**batch, "pixels": batch["pixels"].clone()}
        changed["pixels"][1] = 1000
        changed["pixels"][0, 2:] = -1000
        masked = model(changed)
    torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(together, permuted, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(together, masked, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(together).all()


def test_tpm_prefix_and_crf_backward_reaches_visual_prompt():
    torch, model, batch = tiny_baseline()
    loss = model.loss(model(batch), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.img_prompt_encoder[0].weight.grad.abs().sum() > 0
    assert model.temporalEmbedding.weight.grad.abs().sum() > 0
    assert model.crf.transitions.grad.abs().sum() > 0
