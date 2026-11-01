"""Freeze two matched dropout cohorts inside the existing MI training split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset import MultimodalNerDataset
from modules.dropout_selection import ARMS, DropoutSelectionDataset, prepare_cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-manifest", default="dataset/context_internal_mi_seed20261101/context_internal_manifest.json")
    parser.add_argument("--data-path", default="dataset/context_internal_mi_seed20261101")
    parser.add_argument("--cache", default="augmentation_cache/mi_from_v61/MNER-MI_train.jsonl")
    parser.add_argument("--image-path", default="dataset/images")
    parser.add_argument("--twitter2017-image-path", default="dataset/twitter2017_images")
    parser.add_argument("--bert-model", default="/root/autodl-tmp/pretrained_models/bert-base-uncased")
    parser.add_argument("--vit-model", default="/root/autodl-tmp/pretrained_models/ViTB-16")
    parser.add_argument("--output", default="dataset/context_internal_mi_seed20261101/dropout_cohorts_seed20261102.json")
    args = parser.parse_args()
    args.dataset, args.max_seq_length = "MI", 64
    base = MultimodalNerDataset(args, "MNER-MI_train.txt")
    output = Path(args.output)
    if not output.exists():
        cohort = prepare_cohorts(base, args.internal_manifest, args.cache, "MI", args.image_path,
                                 args.twitter2017_image_path, args.bert_model)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(cohort, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    else:
        cohort = json.loads(output.read_text(encoding="utf-8"))
        print("Preserving existing cohort assignment:", output, flush=True)
    print("Candidate pool:", cohort["candidate_count"])
    print("Eligible sources per arm:", cohort["eligible_per_arm"])
    print("Balance gate:", cohort["balance_check"])
    print("Overlap gate:", cohort["overlap_check"])
    for arm in ARMS:
        data = DropoutSelectionDataset(base, output, arm, args.internal_manifest, args.cache, "MI", args.bert_model)
        print(json.dumps(data.summary, sort_keys=True))
    print("Cohort preparation passed:", output)


if __name__ == "__main__":
    main()
