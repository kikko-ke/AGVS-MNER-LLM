"""Audit usable gold-consistent LLM entity-to-image supervision."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, UnidentifiedImageError

from modules.llm_dataset import extract_bio_entities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="MNER training text file")
    parser.add_argument("--cache", required=True, help="Teacher JSONL cache")
    parser.add_argument(
        "--teacher-task", choices=("mner", "grounding"), default="grounding"
    )
    parser.add_argument("--image-root")
    parser.add_argument("--twitter-image-root")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.5, 0.7, 0.8, 0.9),
        help="Teacher confidence thresholds to compare",
    )
    return parser.parse_args()


def normalise_type(value: object) -> str:
    label = str(value).upper().replace("MISC", "MIS")
    return label[2:] if label.startswith(("B-", "I-")) else label


def resolve_image(name: str, image_root: Path, twitter_root: Path) -> Path:
    if "twitter2017" not in name:
        return image_root / name
    full_path = twitter_root / name
    if full_path.exists():
        return full_path
    return twitter_root / name.split("-", 1)[1]


def is_decodable(path: Path) -> bool:
    if not path.exists():
        return False
    raw_bytes = path.read_bytes()
    for candidate in (raw_bytes, raw_bytes + b"\xff\xd9"):
        try:
            with Image.open(BytesIO(candidate)) as image:
                image.convert("RGB").load()
            return True
        except (OSError, UnidentifiedImageError):
            continue
    return False


def main() -> None:
    args = parse_args()
    examples = [
        ast.literal_eval(line)
        for line in Path(args.input).open(encoding="utf-8")
        if line.strip()
    ]
    cache: dict[int, dict[str, object]] = {}
    cache_tasks = Counter()
    with Path(args.cache).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source):
            if line.strip():
                item = json.loads(line)
                cache_tasks[str(item.get("teacher_task", "mner"))] += 1
                cache[int(item.get("sample_id", line_number))] = item

    print(f"训练样本: {len(examples)}")
    print(f"缓存覆盖: {len(cache)} / {len(examples)}")
    print("缓存任务:", dict(sorted(cache_tasks.items())))
    unexpected_tasks = set(cache_tasks) - {args.teacher_task}
    if unexpected_tasks:
        raise ValueError(
            f"缓存任务与 --teacher-task={args.teacher_task} 不一致: "
            f"{sorted(unexpected_tasks)}"
        )

    check_physical_images = bool(args.image_root and args.twitter_image_root)
    image_root = Path(args.image_root) if args.image_root else None
    twitter_root = Path(args.twitter_image_root) if args.twitter_image_root else None
    valid_image_ids_by_sample: dict[int, set[int]] = {}
    if check_physical_images:
        assert image_root is not None and twitter_root is not None
        for sample_id, example in enumerate(examples):
            valid_image_ids_by_sample[sample_id] = {
                image_id
                for image_id, image_name in enumerate(example.get("images", [])[:4])
                if is_decodable(resolve_image(image_name, image_root, twitter_root))
            }
        print("图片有效性: 已按实际可解码文件检查")
    else:
        print("图片有效性: 仅检查数据中列出的槽位（可传入两个 image root 做严格检查）")

    for threshold in sorted(set(args.thresholds)):
        accepted = Counter()
        grounding_modes = Counter()
        accepted_samples: set[int] = set()
        rejected = Counter()
        grounded_words = 0

        for sample_id, item in cache.items():
            if not 0 <= sample_id < len(examples):
                rejected["sample_id_out_of_range"] += 1
                continue
            example = examples[sample_id]
            gold = extract_bio_entities(
                [str(label).replace("MISC", "MIS") for label in example["label"]]
            )
            image_count = min(len(example.get("images", [])), 4)
            valid_image_ids = valid_image_ids_by_sample.get(
                sample_id, set(range(image_count))
            )

            for entity in item.get("entities", []):
                try:
                    start = int(entity["start"])
                    end = int(entity["end"])
                    entity_type = normalise_type(entity["label"])
                    confidence = float(entity.get("confidence", 1.0))
                except (KeyError, TypeError, ValueError):
                    rejected["malformed"] += 1
                    continue

                if (start, end, entity_type) not in gold:
                    rejected["span_or_type_mismatch"] += 1
                    continue
                if confidence < threshold:
                    rejected["below_confidence"] += 1
                    continue
                raw_image_ids = entity.get("image_ids", [])
                if not isinstance(raw_image_ids, (list, tuple)):
                    rejected["malformed_image_ids"] += 1
                    continue
                selected_image_ids = {
                    int(image_id)
                    for image_id in raw_image_ids
                    if str(image_id).lstrip("-").isdigit()
                    and int(image_id) in valid_image_ids
                }
                if raw_image_ids and not selected_image_ids:
                    rejected["selected_images_invalid"] += 1
                    continue

                accepted[entity_type] += 1
                accepted_samples.add(sample_id)
                grounded_words += end - start
                if not raw_image_ids:
                    grounding_modes["NONE"] += 1
                elif len(valid_image_ids) < 2:
                    grounding_modes["single_valid_image"] += 1
                elif selected_image_ids == valid_image_ids:
                    grounding_modes["all_valid_images"] += 1
                else:
                    grounding_modes["discriminative_subset"] += 1

        print(f"\n置信度阈值 >= {threshold:.2f}")
        print(f"可用实体: {sum(accepted.values())}，样本: {len(accepted_samples)}")
        print(f"受监督实体词数: {grounded_words}")
        print("类别分布:", dict(sorted(accepted.items())))
        print("Grounding 类型:", dict(sorted(grounding_modes.items())))
        print("过滤原因:", dict(sorted(rejected.items())))


if __name__ == "__main__":
    main()
