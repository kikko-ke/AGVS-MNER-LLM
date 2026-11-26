"""Download the official 7B checkpoint to its own directory, with resume/retries."""
import argparse
import json
import os
from pathlib import Path
import time

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT = "/root/autodl-tmp/pretrained_models/Qwen2.5-VL-7B-Instruct"


def validate(directory):
    root = Path(directory)
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                "vocab.json", "merges.txt", "model.safetensors.index.json")
    for name in required:
        if not (root / name).is_file() or not (root / name).stat().st_size:
            raise FileNotFoundError(f"Missing or empty 7B model file: {root / name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2_5_vl" or config.get("hidden_size") != 3584:
        raise ValueError("This is not the official 7B architecture; refusing to reuse/overwrite another model")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("7B weight index has no weight_map")
    total = 0
    for name in set(mapping.values()):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Unsafe weight filename in the 7B index")
        file = root / name
        if not file.is_file() or not file.stat().st_size:
            raise FileNotFoundError(f"Missing or empty 7B weight shard: {file}")
        total += file.stat().st_size
    if index.get("metadata", {}).get("total_size", 0) > total:
        raise ValueError("7B shard sizes are smaller than the indexed tensor bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    root = Path(args.output)
    if (root / "config.json").is_file():
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("hidden_size") != 3584 or config.get("model_type") != "qwen2_5_vl":
            raise ValueError("The output directory contains another model; choose a separate 7B directory")
    try:
        validate(root)
        print(f"Complete local 7B files found; skipping network download: {root}", flush=True)
        return
    except FileNotFoundError:
        pass
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, snapshot_download
    revision = None
    for attempt in range(1, args.attempts + 1):
        try:
            if revision is None:
                revision = HfApi().model_info(MODEL, revision=args.revision).sha
            print(f"Downloading {MODEL}@{revision}, attempt {attempt}/{args.attempts}", flush=True)
            snapshot_download(MODEL, revision=revision, local_dir=str(root), max_workers=2,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                              etag_timeout=60)
            validate(root)
        except Exception as error:
            if attempt == args.attempts:
                raise
            print(f"{type(error).__name__}: download incomplete; keeping cached files and retrying", flush=True)
            time.sleep(min(5 * 2 ** (attempt - 1), 30))
        else:
            (root / "download_provenance.json").write_text(json.dumps(
                {"repo_id": MODEL, "resolved_revision": revision}, indent=2) + "\n", encoding="utf-8")
            print(f"7B download complete: {root}", flush=True)
            return


if __name__ == "__main__":
    main()
