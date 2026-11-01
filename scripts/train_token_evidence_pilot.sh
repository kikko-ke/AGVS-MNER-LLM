#!/usr/bin/env bash
# Fixed two-run component pilot; no API calls, weight searches or test scoring.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dataset="${1:-MI}"
case "$dataset" in
  MI) selected=dataset/v63_mi_llm_selected; old_root=outputs_mi_v63; old_arm=llm_selected ;;
  UNI) selected=dataset/v61_duplicate_control_seed20261027; old_root=outputs_v61; old_arm=duplicate_control ;;
  *) echo "Usage: bash scripts/train_token_evidence_pilot.sh MI|UNI"; exit 2 ;;
esac
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs

python - "$dataset" "$selected" "$old_root" "$old_arm" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path
import torch
dataset, selected, old_root, old_arm = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: use AutoDL GPU mode for the component pilot")
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
reference = json.loads((Path(selected) / "augmentation_manifest.json").read_text(encoding="utf-8"))
if reference["arm"] != "duplicate_control" or reference["source_train_sha256"] != sha(f"dataset/text/MNER-{dataset}_train.txt"):
    raise SystemExit("Selected-original data provenance differs")
original = Path(f"dataset/text/MNER-{dataset}_train.txt").read_text(encoding="utf-8").splitlines()
ids = reference["source_sample_ids"]
actual = (Path(selected) / f"MNER-{dataset}_train.txt").read_text(encoding="utf-8").splitlines()
if actual != original + [original[i] for i in ids]:
    raise SystemExit("Selected data is not the declared original-only resampling")
for split in ("val", "test"):
    name = f"MNER-{dataset}_{split}.txt"
    if sha(Path(selected) / name) != sha(Path("dataset/text") / name):
        raise SystemExit(f"Different evaluation data: {name}")
expected = {"dataset": dataset, "seed": 1234, "num_epochs": 15, "batch_size": 4,
            "lr": 5e-5, "bert_lr": 1e-5, "vit_lr": 5e-6, "warmup_ratio": 0.01,
            "weight_decay": 0.01, "max_grad_norm": 1.0, "patience": 3,
            "max_seq_length": 64, "topk_img_patches": 12, "image_dropout_prob": 0.2,
            "output_dropout": 0.2, "skip_test": True}
for arm, data in (("strict_baseline", "dataset/text"), (old_arm, selected)):
    checkpoints = list((Path(old_root) / f"{arm}_seed1234").glob("*/best_model.pt"))
    if len(checkpoints) != 1:
        raise SystemExit(f"Expected one existing reference checkpoint for {arm}, found {len(checkpoints)}")
    config = json.loads((checkpoints[0].parent / "config.json").read_text(encoding="utf-8"))
    wrong = [key for key, value in expected.items() if config.get(key) != value]
    if wrong or config.get("visual_fusion", "legacy") != "legacy" or Path(config["data_path"]).resolve() != Path(data).resolve():
        raise SystemExit(f"Reference protocol differs: {arm}, fields={wrong}")
print(f"Component pilot preflight passed: {dataset}, seed=1234, selected originals={len(ids)}")
PY

common=(
  --dataset "$dataset"
  --image-path dataset/images --twitter2017-image-path dataset/twitter2017_images
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16
  --device cuda --num-epochs 15 --batch-size 4 --num-workers 4
  --lr 5e-5 --bert-lr 1e-5 --vit-lr 5e-6
  --warmup-ratio 0.01 --weight-decay 0.01 --max-grad-norm 1.0
  --patience 3 --seed 1234 --max-seq-length 64 --topk-img-patches 12
  --image-dropout-prob 0.2 --output-dropout 0.2
  --visual-fusion token_evidence --skip-test
)
for arm in original selected; do
  data=dataset/text
  if [[ "$arm" == selected ]]; then data="$selected"; fi
  out="outputs_component_v1/${dataset,,}_token_evidence_${arm}_seed1234"
  if [[ -d "$out" ]] && [[ -n "$(ls -A "$out")" ]]; then
    echo "Existing pilot preserved: $out. Inspect its completion before retrying."
    exit 1
  fi
  log="logs/component_v1_${dataset,,}_${arm}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== Token evidence pilot: $dataset $arm; log=$log ==="
  python -u run.py "${common[@]}" --data-path "$data" --output-dir "$out" 2>&1 | tee "$log"
done

python - "$dataset" "$old_root" "$old_arm" <<'PY'
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
dataset, old_root, old_arm = sys.argv[1:]
def best(root):
    checkpoints = list(Path(root).glob("*/best_model.pt"))
    if len(checkpoints) != 1:
        raise SystemExit(f"Expected one checkpoint: {root}")
    path = checkpoints[0]
    with (path.parent / "metrics.csv").open(encoding="utf-8", newline="") as source:
        values = [float(row["dev_f1"]) for row in csv.DictReader(source) if row.get("dev_f1")]
    if not values or not all(math.isfinite(value) for value in values):
        raise SystemExit(f"Missing/nonfinite validation scores: {root}")
    return {"dev_f1_rounded": max(values), "checkpoint": str(path),
            "config": json.loads((path.parent / "config.json").read_text(encoding="utf-8"))}
rows = {
    "legacy_original": best(f"{old_root}/strict_baseline_seed1234"),
    "legacy_selected": best(f"{old_root}/{old_arm}_seed1234"),
    "new_original": best(f"outputs_component_v1/{dataset.lower()}_token_evidence_original_seed1234"),
    "new_selected": best(f"outputs_component_v1/{dataset.lower()}_token_evidence_selected_seed1234"),
}
for row in rows.values():
    row["data_train_sha256"] = hashlib.sha256(
        (Path(row["config"]["data_path"]) / f"MNER-{dataset}_train.txt").read_bytes()
    ).hexdigest()
for name in ("original", "selected"):
    if rows[f"legacy_{name}"]["data_train_sha256"] != rows[f"new_{name}"]["data_train_sha256"]:
        raise SystemExit("Legacy/new training data differ")
component_original = rows["new_original"]["dev_f1_rounded"] - rows["legacy_original"]["dev_f1_rounded"]
component_selected = rows["new_selected"]["dev_f1_rounded"] - rows["legacy_selected"]["dev_f1_rounded"]
selection_new = rows["new_selected"]["dev_f1_rounded"] - rows["new_original"]["dev_f1_rounded"]
report = {"schema_version": "token-evidence-component-pilot-v1", "dataset": dataset, "split": "val",
          "seed": 1234, "models": rows, "component_gain_original": component_original,
          "component_gain_selected": component_selected, "selected_minus_original_new_component": selection_new,
          "llm_api_requests": 0, "performance_confirmed": False,
          "note": "Four-decimal CSV development scores; not native replay, independent confirmation, "
                  "test performance, or proof that component gains are caused by LLM selection."}
output = Path("outputs_component_v1") / f"{dataset.lower()}_pilot_{datetime.now():%Y%m%d_%H%M%S}.json"
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("\n=== Fixed component pilot (validation only) ===")
for name, row in rows.items():
    print(f"{name:<20} F1={row['dev_f1_rounded']:.4f}")
print(f"Component gain, original data: {component_original:+.4f}")
print(f"Component gain, selected data: {component_selected:+.4f}")
print(f"Selection gain within new component: {selection_new:+.4f}")
print(f"Report: {output}")
PY
