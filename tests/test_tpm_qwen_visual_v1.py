"""CPU verification of native VL states, word alignment, resume and comparison."""
import json

import pytest
import torch

from modules.tpm_qwen_visual_v1 import features, pipeline
from modules.tpm_llm_v1.common import BIO, TAGS, atomic_json, save_rows
from modules.tpm_llm_v1 import pipeline as old


@pytest.fixture
def native_processor():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special = ["<|endoftext|>", features.VISION_START, features.IMAGE_PAD, features.VISION_END]
    tokenizer.train_from_iterator(["New York office. London a b c"], trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=special))
    fast = Qwen2TokenizerFast(tokenizer_object=tokenizer, eos_token=special[0], pad_token=special[0])
    processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=3136, max_pixels=200704), tokenizer=fast)
    return processor


def make_images(tmp_path):
    from PIL import Image
    audit = {}
    for name, color in (("first.png", "red"), ("third.png", "blue")):
        path = tmp_path / name
        Image.new("RGB", (56, 56), color).save(path)
        audit[name] = {"path": str(path), "valid": True}
    audit["missing.png"] = {"path": str(tmp_path / "missing.png"), "valid": False}
    return audit


def test_native_processor_expansion_keeps_word_suffix_and_all_valid_images(tmp_path, native_processor):
    row = {"source_id": "train:0", "tokens": ["New", "York", "office."],
           "images": ["first.png", "missing.png", "third.png"]}
    inputs, positions, usage = features.prepare_input(row, native_processor, make_images(tmp_path), 2048)
    raw = native_processor.tokenizer("New York office.", return_offsets_mapping=True)
    bounds = features.word_offsets(row["tokens"])[1]
    relative = features.last_overlapping(bounds, raw["offset_mapping"])
    assert positions == [index + usage["prefix_tokens"] for index in relative]
    assert usage["used_slots"] == [0, 2] and usage["valid_images"] == 2
    assert usage["missing"] == [{"slot": 1, "name": "missing.png"}]
    assert inputs["image_grid_thw"].shape[0] == 2
    assert inputs["input_ids"][0, -len(raw["input_ids"]):].tolist() == raw["input_ids"]
    assert set(inputs) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    assert inputs["input_ids"][0, :min(positions)].eq(
        native_processor.tokenizer.convert_tokens_to_ids(features.IMAGE_PAD)).any()


@pytest.mark.parametrize("size,expected", [((190, 21), (254, 28)),
                                        ((21, 190), (28, 254)),
                                        ((7, 7), (28, 28))])
def test_tiny_images_reproduce_native_error_then_are_retained(tmp_path, native_processor, size, expected):
    from PIL import Image
    path = tmp_path / "tiny.png"
    Image.new("RGB", size, "green").save(path)
    original_bytes = path.read_bytes()
    # Reproduce the exact AutoDL smart_resize failure before testing the repair.
    with Image.open(path) as image:
        with pytest.raises(ValueError, match="larger than factor:28"):
            native_processor.image_processor(images=image, return_tensors="pt")
    audit = {"tiny.png": {"path": str(path), "valid": True}}
    row = {"source_id": "train:739", "tokens": ["New", "York"], "images": ["tiny.png"]}
    inputs, positions, usage = features.prepare_input(row, native_processor, audit, 2048)
    assert len(inputs["image_grid_thw"]) == usage["valid_images"] == 1
    assert torch.isfinite(inputs["pixel_values"]).all() and min(positions) > 0
    assert usage["used_slots"] == [0] and usage["missing"] == []
    assert usage["resized_images"] == [{"slot": 0, "name": "tiny.png",
                                        "original_size": list(size), "input_size": list(expected)}]
    summary = features.usage_summary([usage])
    assert summary["resized_image_slots"] == summary["samples_with_resized_images"] == 1
    assert path.read_bytes() == original_bytes


def test_regular_images_produce_exactly_the_original_processor_inputs(tmp_path, native_processor):
    audit = make_images(tmp_path)
    row = {"source_id": "train:0", "tokens": ["New", "York"], "images": ["first.png"]}
    image, adjustment = features.open_qwen_image(tmp_path / "first.png", 28)
    try:
        assert image.size == (56, 56) and adjustment is None
        expected = native_processor(text=[features.VISION_START + features.IMAGE_PAD +
                                          features.VISION_END + "New York"],
                                    images=[image], return_tensors="pt", padding=False, truncation=False)
    finally:
        image.close()
    observed, _, usage = features.prepare_input(row, native_processor, audit, 2048)
    for key, values in observed.items():
        torch.testing.assert_close(values, expected[key], atol=0, rtol=0)
    assert usage["resized_images"] == []


def test_missing_images_have_explicit_text_fallback_and_budget_is_never_truncated(tmp_path, native_processor):
    row = {"source_id": "train:1", "tokens": ["New", "York"], "images": ["missing.png"]}
    inputs, positions, usage = features.prepare_input(row, native_processor, make_images(tmp_path), 2048)
    raw = native_processor.tokenizer("New York", return_offsets_mapping=True)
    assert inputs["input_ids"][0].tolist() == raw["input_ids"]
    assert usage["valid_images"] == 0 and usage["prefix_tokens"] == 0
    assert "pixel_values" not in inputs and len(positions) == 2
    row["images"] = ["first.png", "third.png"]
    with pytest.raises(ValueError, match="No input was truncated"):
        features.prepare_input(row, native_processor, make_images(tmp_path), 3)


def test_alignment_rejects_a_processor_that_changes_the_post_suffix(native_processor):
    class Altered:
        tokenizer = native_processor.tokenizer

        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[999, 998]]), "attention_mask": torch.ones(1, 2)}

    with pytest.raises(ValueError, match="suffix differs"):
        features.prepare_input({"source_id": "x", "tokens": ["New"], "images": []}, Altered(), {}, 2048)


def tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(19)
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        image_token_id=4, video_token_id=7, vision_start_token_id=5, vision_end_token_id=6,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "out_hidden_size": 32, "num_heads": 4, "patch_size": 14,
                       "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "window_size": 28, "fullatt_block_indexes": [0]})
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
    inputs = {"input_ids": torch.tensor([[5, 4, 6, 5, 4, 6, 10, 11]]),
              "attention_mask": torch.ones(1, 8, dtype=torch.long),
              "pixel_values": torch.randn(8, 3 * 2 * 14 * 14),
              "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
    return model, inputs


def test_native_qwen_identity_readout_is_last_hidden_state_and_uses_both_images():
    # Real Transformers 4.51.3, random tiny model: no weights or network needed.
    model, inputs = tiny_qwen()
    with torch.inference_mode():
        expected = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states[-1]
        model.set_output_embeddings(torch.nn.Identity())
        observed = features.feature_forward(model, inputs)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        for start in (0, 4):
            changed = {**inputs, "pixel_values": inputs["pixel_values"].clone()}
            changed["pixel_values"][start:start + 4] = 0
            altered = features.feature_forward(model, changed)
            assert (observed[:, -2:] - altered[:, -2:]).abs().max() > 1e-7
    assert all(not p.requires_grad for p in model.parameters())


def test_projection_matches_original_fixed_random_matrix_and_masks_unaligned_words():
    hidden = torch.randn(1, 8, 32)
    output, valid, matrix = features.project_words(hidden, [6, -1, 7], 1234)
    generator = torch.Generator().manual_seed(1234)
    expected_matrix = torch.randn(32, 768, generator=generator) / 768 ** 0.5
    torch.testing.assert_close(matrix, expected_matrix)
    torch.testing.assert_close(output[[0, 2]], (hidden[0, [6, 7]] @ expected_matrix).half())
    assert valid.tolist() == [True, False, True] and output[1].abs().sum() == 0


def fixture_source(tmp_path, monkeypatch, rows):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    for role in ("train", "dev"):
        save_rows(source / f"data/{role}.jsonl", rows)
    atomic_json(source / "data/image_audit.json", {})
    protocol = {"protocol_id": "visual-fixture", "config": {"shard_size": 2}}
    monkeypatch.setattr(pipeline, "source_run", lambda p: (source, {"protocol_id": "source-fixture"}))
    monkeypatch.setattr(pipeline, "runtime", lambda: {"torch": "fixture"})
    root = tmp_path / "visual"
    root.mkdir()
    old.seal(root / "smoke/completed.json", protocol, [])
    return root, source, protocol


def test_feature_shards_resume_after_interruption_without_reextracting_completed_rows(tmp_path, monkeypatch):
    rows = [{"source_id": f"train:{i}", "tokens": ["word"], "images": []} for i in range(3)]
    root, _, protocol = fixture_source(tmp_path, monkeypatch, rows)
    seen, fail = [], [True]

    class Extractor:
        def __init__(self, config):
            pass

        def rows(self, selected, audit):
            if fail[0] and selected[0]["source_id"] == "train:2":
                raise RuntimeError("simulated GPU interruption")
            seen.append([r["source_id"] for r in selected])
            return {"source_ids": [r["source_id"] for r in selected],
                    "features": [torch.zeros(1, 768) for _ in selected],
                    "aligned": [torch.tensor([True]) for _ in selected],
                    "image_usage": [{"source_id": r["source_id"], "valid_images": 0,
                                     "missing": [], "sequence_tokens": 1} for r in selected]}

        def close(self):
            pass

    monkeypatch.setattr(features, "FrozenVisualExtractor", Extractor)
    with pytest.raises(RuntimeError, match="interruption"):
        pipeline.prepare_features(root, protocol)
    assert seen == [["train:0", "train:1"]]
    fail[0] = False
    pipeline.prepare_features(root, protocol)
    assert seen.count(["train:0", "train:1"]) == 2  # once for train and once for dev, never replayed
    _, bundle = pipeline.checked_features(root, "train", protocol)
    assert bundle["source_ids"] == [r["source_id"] for r in rows]
    assert len(bundle["features"]) == 3


def test_test_feature_extraction_is_blocked_until_dev_calibration(tmp_path):
    protocol = {"protocol_id": "fixture"}
    old.seal(tmp_path / "smoke/completed.json", protocol, [])
    with pytest.raises(ValueError, match="Freeze development"):
        pipeline.prepare_features(tmp_path, protocol, test=True)


def test_visual_calibration_compares_existing_text_head_without_reading_test(tmp_path, monkeypatch):
    from torchcrf import CRF
    row = {"source_id": "dev:0", "tokens": ["London"], "labels": ["B-LOC"], "images": []}
    root, source, protocol = fixture_source(tmp_path, monkeypatch, [row])
    protocol["config"]["coefficient_grid"] = [0, 0.25, 0.5, 1]
    decoder = CRF(10, batch_first=True)
    for p in decoder.parameters():
        p.data.zero_()
    emissions = torch.zeros(1, 3, 10)
    emissions[0, [0, 2], 9] = 5
    emissions[0, 1, 0] = 2
    emissions[0, 1, TAGS.index("B-LOC")] = 1
    native = {"source_ids": ["dev:0"], "emissions": emissions, "crf": decoder.state_dict(),
              "mask": torch.ones(1, 3, dtype=torch.bool),
              "mappings": [{"positions": [1], "word_indices": [0]}]}
    native_path = source / "native.pt"
    native_path.write_bytes(b"fixed native emissions")
    monkeypatch.setattr(pipeline, "checked_native", lambda *args: (native_path, native))
    text_path, visual_path = source / "text.pt", root / "head.pt"
    text_path.write_bytes(b"fixed existing text head")
    visual_path.write_bytes(b"new visual head")
    text = {"dev_logits": [torch.zeros(1, 9)], "dev_aligned": [torch.tensor([True])],
            "dev_source_ids": ["dev:0"]}
    visual = {**text, "dev_logits": [torch.zeros(1, 9)]}
    visual["dev_logits"][0][0, BIO.index("B-LOC")] = 9
    monkeypatch.setattr(old, "checked_head", lambda *args: (text_path, text))
    monkeypatch.setattr(pipeline, "checked_head", lambda *args: (visual_path, visual))
    atomic_json(source / "calibration/dev_report.json", {"coefficients": {"qwen_bigru": 0.5}})
    pipeline.calibrate(root, protocol)
    report = json.loads((root / "calibration/dev_report.json").read_text())
    assert report["coefficients"] == {"qwen_text": 0.5, "qwen_multimodal": 0.25}
    assert report["qwen_multimodal"]["f1"] == 1 and report["qwen_text"]["f1"] == 0
    assert report["test_metrics_used_for_selection"] is False
    assert not (source / "data/test.jsonl").exists()


def test_summary_reports_paired_visual_increment_and_percentage_points():
    scores = {str(seed): {"tpm": {"f1": 0.7}, "tpm_qwen_text": {"f1": 0.75},
                         "tpm_qwen_multimodal": {"f1": visual},
                         "tpm_qwen_text_shared1": {"f1": 0.74},
                         "tpm_qwen_multimodal_shared1": {"f1": visual - 0.01}}
              for seed, visual in ((1234, 0.77), (42, 0.74), (2026, 0.78))}
    result = pipeline.summarize(scores, [1234, 42, 2026])
    delta = result["paired_deltas"]["multimodal_minus_text"]
    assert delta["mean"] == pytest.approx(0.04 / 3)
    assert delta["mean_percentage_points"] == pytest.approx(4 / 3)
    assert delta["positive_seeds"] == 2 and len(delta["by_seed"]) == 3


def test_full_report_decodes_abc_preserves_negative_results_and_does_not_write_source(tmp_path, monkeypatch):
    from torchcrf import CRF
    rows = [{"source_id": "train:0", "tokens": ["training"], "labels": ["O"], "images": []}]
    root, source, protocol = fixture_source(tmp_path, monkeypatch, rows)
    source_protocol = {"protocol_id": "source-fixture"}
    data = tmp_path / "original_data"
    data.mkdir()
    test_file = data / "MNER-MI_test.txt"
    test_file.write_text("".join(json.dumps({"text": ["London", str(i)], "label": ["B-LOC", "O"],
                                            "images": []}) + "\n" for i in range(860)), encoding="utf-8")
    protocol["config"].update({"data_dir": str(data), "seeds": [1234]})
    ids = [f"test:{i}" for i in range(860)]
    decoder = CRF(10, batch_first=True)
    for p in decoder.parameters():
        p.data.zero_()
    emissions = torch.zeros(860, 3, 10)
    emissions[:, [0, 2], 9] = 5
    emissions[:, 1, 0] = 2
    emissions[:, 1, TAGS.index("B-LOC")] = 1
    native = {"source_ids": ids, "emissions": emissions, "crf": decoder.state_dict(),
              "mask": torch.ones(860, 3, dtype=torch.bool),
              "mappings": [{"positions": [1], "word_indices": [0]} for _ in ids]}
    native_path = source / "emissions/seed1234_test.pt"
    old.save_torch(native_path, native)
    old.seal(native_path.with_suffix(".json"), source_protocol, [native_path])
    text_scores = [torch.zeros(2, 9) for _ in ids]
    for score in text_scores:
        score[0, BIO.index("B-LOC")] = 9
    for file, p, scores in (
            (source / "test_logits/seed1234_qwen_bigru.pt", source_protocol, text_scores),
            (root / "test_logits/seed1234.pt", protocol, [torch.zeros(2, 9) for _ in ids])):
        old.save_torch(file, {"source_ids": ids, "logits": scores,
                              "aligned": [torch.tensor([True, True]) for _ in ids]})
        old.seal(file.with_suffix(".json"), p, [file])
    feature_file = pipeline.feature_path(root, "test")
    old.save_torch(feature_file, {"source_ids": ids, "image_usage": [{"valid_images": 0} for _ in ids]})
    audit_path = root / "features/test_image_audit.json"
    atomic_json(audit_path, {})
    old.seal(feature_file.with_suffix(".json"), protocol, [feature_file, audit_path])
    calibration_path = root / "calibration/dev_report.json"
    atomic_json(calibration_path, {"coefficients": {"qwen_text": 0.5, "qwen_multimodal": 0}})
    old.seal(root / "calibration/completed.json", protocol, [calibration_path])
    before = {str(p.relative_to(source)): old.digest(p) for p in source.rglob("*") if p.is_file()}
    pipeline.evaluate(root, protocol)
    report = json.loads((root / "evaluation/comparison_report.json").read_text())
    delta = report["summary"]["official_test_full"]["paired_deltas"]["multimodal_minus_text"]
    assert delta["mean"] == -1 and delta["positive_seeds"] == 0
    assert report["test_samples"] == report["filtered_test_samples"] == 860
    assert report["metrics"]["official_test_full"]["1234"]["tpm_qwen_text"]["f1"] == 1
    assert report["valid_image_count_samples"] == {"0": 860, "1": 0, "2plus": 0}
    after = {str(p.relative_to(source)): old.digest(p) for p in source.rglob("*") if p.is_file()}
    assert after == before
    assert old.completed(root / "evaluation/completed.json", protocol)
