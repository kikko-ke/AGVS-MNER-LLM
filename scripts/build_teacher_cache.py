"""Generate a JSONL cache of structured LLM teacher signals."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import BertTokenizer

from llm.client import LLMTeacherClient
from llm.schema import teacher_to_soft_labels, validate_teacher_output
from modules.dataset import MnerProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--twitter-image-root", required=True)
    parser.add_argument("--bert-model", required=True)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def image_path(name: str, image_root: Path, twitter_root: Path) -> Path:
    if "twitter2017" in name:
        return twitter_root / name.split("-", maxsplit=1)[1]
    return image_root / name


def main() -> None:
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    label_mapping = MnerProcessor.get_label_mapping(include_subword_label=True)
    client = LLMTeacherClient(args.base_url, args.api_key, args.model, args.timeout)
    image_root = Path(args.image_root)
    twitter_root = Path(args.twitter_image_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for sample_id, line in enumerate(input_file):
            if not line.strip():
                continue
            record = ast.literal_eval(line)
            words = record["text"]
            image_names = record.get("images", [])[:4]
            paths = [image_path(name, image_root, twitter_root) for name in image_names]
            paths.extend([None] * (4 - len(paths)))
            raw_payload = client.complete_json(words, paths)
            teacher = validate_teacher_output(
                raw_payload,
                token_count=len(words),
                num_images=4,
                raw_text=json.dumps(raw_payload, ensure_ascii=False),
            )
            token_probs, image_scores = teacher_to_soft_labels(
                teacher,
                words,
                tokenizer,
                args.max_seq_length,
                label_mapping,
            )
            output_file.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "token_probs": token_probs,
                        "image_scores": image_scores,
                        "entities": [
                            {
                                "start": entity.start,
                                "end": entity.end,
                                "label": entity.label,
                                "image_ids": list(entity.image_ids),
                                "confidence": entity.confidence,
                                "evidence": entity.evidence,
                            }
                            for entity in teacher.entities
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()


