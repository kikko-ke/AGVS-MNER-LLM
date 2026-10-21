from __future__ import annotations

from pathlib import Path

from modules.dataset import MnerProcessor, MultimodalInferenceDataset
from modules.reranker import sha256_inference_projection


class _Tokenizer:
    pieces = {
        "alpha": ["alpha"],
        "longword": ["long", "##word", "##tail"],
        "omega": ["omega"],
    }

    def tokenize(self, word: str) -> list[str]:
        return self.pieces[word]


def test_unlabeled_loader_does_not_require_label_field(tmp_path: Path) -> None:
    data = tmp_path / "sample.txt"
    data.write_text(
        repr({"text": ["Alice"], "images": ["a.jpg"]}) + "\n",
        encoding="utf-8",
    )
    processor = MnerProcessor.__new__(MnerProcessor)
    processor.data_path = tmp_path

    examples = processor.load_unlabeled_examples(data.name)

    assert examples == {"words": [["Alice"]], "images": [["a.jpg"]]}


def test_inference_projection_hash_is_independent_of_gold_labels(tmp_path: Path) -> None:
    data = tmp_path / "sample.txt"
    first = {"text": ["Alice"], "images": ["a.jpg"], "label": ["B-PER"]}
    second = {**first, "label": ["B-ORG"]}
    data.write_text(repr(first) + "\n", encoding="utf-8")
    first_hash = sha256_inference_projection(data)
    data.write_text(repr(second) + "\n", encoding="utf-8")

    assert sha256_inference_projection(data) == first_hash


def test_inference_projection_hash_accepts_escaped_surrogates(tmp_path: Path) -> None:
    data = tmp_path / "sample.txt"
    # Some MNER records store emoji as escaped UTF-16 surrogate pairs.  After
    # literal_eval these remain surrogate code units, which cannot be encoded
    # directly as UTF-8 but must still produce a stable inference-only hash.
    data.write_text(
        "{'text': ['before', '\\ud83d\\ude02', 'after'], "
        "'images': ['a.jpg'], 'label': ['O', 'O', 'O']}\n",
        encoding="utf-8",
    )

    first_hash = sha256_inference_projection(data)
    assert len(first_hash) == 64

    data.write_text(
        "{'text': ['before', '\\ud83d\\ude02', 'after'], "
        "'images': ['a.jpg'], 'label': ['B-PER', 'I-PER', 'O']}\n",
        encoding="utf-8",
    )
    assert sha256_inference_projection(data) == first_hash


def test_inference_structure_matches_partial_wordpiece_truncation() -> None:
    dataset = MultimodalInferenceDataset.__new__(MultimodalInferenceDataset)
    dataset.max_sequence_length = 5  # CLS + three pieces + SEP
    dataset.examples = {"words": [["alpha", "longword", "omega"]], "images": [[]]}
    dataset.tokenizer = _Tokenizer()

    structure, words, word_indices = dataset.token_structure(0)

    # alpha plus the first two pieces of longword fit. The partially retained
    # word still has a word-level CRF position, exactly as in _encode_text.
    assert words == ["alpha", "longword"]
    assert word_indices == [0, 1]
    assert structure.tolist() == [True, False, False, True, True]
