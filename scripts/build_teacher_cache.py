"""Generate a JSONL cache of structured LLM teacher signals."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time

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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="只处理N个待处理样本，0表示全部")
    parser.add_argument("--resume", action="store_true", help="断点续传，跳过已处理的样本")
    return parser.parse_args()


def image_path(name: str, image_root: Path, twitter_root: Path) -> Path:
    if "twitter2017" in name:
        return twitter_root / name.split("-", maxsplit=1)[1]
    return image_root / name


def load_valid_sample_ids(
    path: Path,
    max_seq_length: int,
    num_labels: int,
) -> set[int]:
    """Return complete cache IDs and reject mixed or malformed cache files."""
    if not path.exists():
        return set()

    sample_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"缓存第 {line_number} 行不是合法 JSON: {error}"
                ) from error

            sample_id = record.get("sample_id")
            probs = record.get("token_probs")
            scores = record.get("image_scores")
            valid = (
                type(sample_id) is int
                and sample_id >= 0
                and isinstance(probs, list)
                and len(probs) == max_seq_length
                and all(
                    isinstance(row, list) and len(row) == num_labels
                    for row in probs
                )
                and isinstance(scores, list)
                and len(scores) == 4
            )
            if not valid:
                raise ValueError(
                    f"缓存第 {line_number} 行是不完整的失败占位记录；"
                    "请先备份并提取有效记录，再使用 --resume"
                )
                continue
            if sample_id in sample_ids:
                raise ValueError(f"缓存中存在重复 sample_id={sample_id}")
            sample_ids.add(sample_id)

    return sample_ids


def main() -> None:
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    label_mapping = MnerProcessor.get_label_mapping(include_subword_label=True)
    client = LLMTeacherClient(
        args.base_url,
        args.api_key,
        args.model,
        args.timeout,
        args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )
    image_root = Path(args.image_root)
    twitter_root = Path(args.twitter_image_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.limit < 0:
        raise ValueError("--limit 不能为负数")
    if args.request_delay < 0:
        raise ValueError("--request-delay 不能为负数")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures 必须至少为 1")
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"{output_path} 已存在；请使用 --resume 或指定新输出文件"
        )

    existing_ids = (
        load_valid_sample_ids(output_path, args.max_seq_length, len(label_mapping))
        if args.resume
        else set()
    )
    print(
        f"[resume] 已有 {len(existing_ids)} 条有效缓存，只请求缺失的 sample_id",
        flush=True,
    )

    error_path = output_path.with_name(output_path.name + ".errors.jsonl")
    processed = 0
    failed = 0
    attempted = 0
    consecutive_failures = 0
    output_mode = "a" if args.resume else "x"

    with (
        Path(args.input).open("r", encoding="utf-8") as input_file,
        output_path.open(output_mode, encoding="utf-8") as output_file,
        error_path.open("a", encoding="utf-8") as error_file,
    ):
        for sample_id, line in enumerate(input_file):
            if not line.strip() or sample_id in existing_ids:
                continue

            if args.limit > 0 and attempted >= args.limit:
                print(f"[limit] 已尝试 {attempted} 个待处理样本")
                break

            attempted += 1
            try:
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
                output_file.flush()
                processed += 1
                consecutive_failures = 0
                if processed % 10 == 0:
                    print(
                        f"[progress] 本次成功 {processed} 条，"
                        f"累计有效 {len(existing_ids) + processed} 条 "
                        f"(sample_id={sample_id})",
                        flush=True,
                    )
                if args.request_delay > 0:
                    time.sleep(args.request_delay)
            except Exception as exc:
                failed += 1
                consecutive_failures += 1
                print(f"[ERROR] sample_id={sample_id} 失败: {exc}", flush=True)
                error_file.write(
                    json.dumps(
                        {"sample_id": sample_id, "error": str(exc)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_file.flush()
                if consecutive_failures >= args.max_consecutive_failures:
                    print(
                        f"[ABORT] 连续 {consecutive_failures} 个样本失败，"
                        "为避免持续无效请求，任务自动停止",
                        flush=True,
                    )
                    break

    print("\n=== 本次运行结束 ===")
    print(f"此前有效: {len(existing_ids)} 条")
    print(f"本次尝试: {attempted} 条")
    print(f"本次成功: {processed} 条")
    print(f"本次失败: {failed} 条")
    print(f"累计有效: {len(existing_ids) + processed} 条")
    print(f"缓存输出: {output_path}")
    print(f"错误日志: {error_path}")


if __name__ == "__main__":
    main()
