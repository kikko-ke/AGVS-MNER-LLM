"""Prepare gold-only VLM training data without API requests or new teacher caches."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.vlm_ner import prepare_data, load_prepared

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-manifest", default="dataset/context_internal_mi_seed20261101/context_internal_manifest.json")
    parser.add_argument("--output-dir", default="dataset/vlm_gold_mi_v1")
    parser.add_argument("--image-root", default="dataset/images")
    parser.add_argument("--twitter-image-root", default="dataset/twitter2017_images")
    args = parser.parse_args()
    path = Path(args.output_dir) / "manifest.json"
    if not path.exists():
        path = prepare_data(args.internal_manifest, args.output_dir, args.image_root, args.twitter_image_root)
    manifest, _ = load_prepared(path)
    print("Prepared data verified:", path)
    print(json.dumps({k:v["count"] for k,v in manifest["files"].items()}, sort_keys=True))
    print("Image counts:", manifest["image_counts"])
    print("Gold scope:", manifest["gold_scope"])

if __name__ == "__main__":
    main()
