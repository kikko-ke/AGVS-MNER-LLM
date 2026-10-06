"""验证教师缓存文件的完整性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify_cache(cache_path: Path, expected_count: int) -> tuple[int, int, int]:
    valid = 0
    invalid = 0
    if not cache_path.exists():
        print(f"[MISSING] {cache_path} 不存在")
        return 0, 0, expected_count
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                token_probs = record.get("token_probs", [])
                image_scores = record.get("image_scores", [])
                if len(token_probs) > 0 and len(image_scores) == 4:
                    valid += 1
                else:
                    invalid += 1
            except Exception:
                invalid += 1
    missing = expected_count - valid - invalid
    return valid, invalid, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--dataset", default="UNI", choices=["MI", "UNI"])
    args = parser.parse_args()

    cache_dir = Path(args.teacher_cache)
    expected = {"MI": (6856, 860, 860), "UNI": (10229, 1583, 1583)}
    train_n, val_n, test_n = expected[args.dataset]

    print(f"=== 验证教师缓存 ({args.dataset}) ===")
    print(f"缓存目录: {cache_dir}")
    print()

    all_valid = True
    for split, expected_n in [("train", train_n), ("val", val_n), ("test", test_n)]:
        cache_file = cache_dir / f"MNER-{args.dataset}_{split}.jsonl"
        valid, invalid, missing = verify_cache(cache_file, expected_n)
        status = "OK" if valid == expected_n else "FAIL"
        if valid != expected_n:
            all_valid = False
        print(f"[{status}] {split}: 有效={valid}/{expected_n}, 无效={invalid}, 缺失={missing}")

    print()
    if all_valid:
        print("=== 全部验证通过 ===")
    else:
        print("=== 存在问题，请检查 ===")


if __name__ == "__main__":
    main()
