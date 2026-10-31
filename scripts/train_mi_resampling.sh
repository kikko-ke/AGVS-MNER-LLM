#!/usr/bin/env bash
# Foreground MI experiments. Preparation reuses exact existing cache matches.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:-prepare}"
if [[ "$mode" != prepare && "$mode" != train ]]; then
  echo "Usage: bash scripts/train_mi_resampling.sh prepare|train"
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs

cache="augmentation_cache/mi_from_v61/MNER-MI_train.jsonl"
selected="dataset/v63_mi_llm_selected"
matched="dataset/v63_mi_feature_matched_seed20261031"

if [[ "$mode" == prepare ]]; then
  if [[ ! -f "$selected/augmentation_manifest.json" ]]; then
    python -u scripts/project_augmentation_selection.py \
      --source-input dataset/text/MNER-UNI_train.txt \
      --source-cache augmentation_cache/qwen3vl32b_v61/MNER-UNI_train.jsonl \
      --target-input-dir dataset/text \
      --target-dataset MI \
      --output-cache "$cache" \
      --output-dir "$selected" \
      2>&1 | tee "logs/mi_project_cache_$(date +%Y%m%d_%H%M%S).log"
  fi
  if [[ ! -f "$matched/matched_resampling_manifest.json" ]]; then
    python -u scripts/build_matched_resampling_control.py \
      --dataset MI \
      --input-dir dataset/text \
      --candidate-metadata "${cache}.meta.json" \
      --reference-manifest "$selected/augmentation_manifest.json" \
      --image-root dataset/images \
      --twitter-image-root dataset/twitter2017_images \
      --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased \
      --sampling-seed 20261031 \
      --output-dir "$matched" \
      2>&1 | tee "logs/mi_build_matched_$(date +%Y%m%d_%H%M%S).log"
  fi
fi

# Verify even when reusing prepared directories. Never silently accept partial
# output or a failed matching gate, and bind projection to the original cache.
python -u - <<'PY'
import json
from pathlib import Path
from scripts.evaluate_matched_resampling import audit_data, require_hash

root = Path("dataset/v63_mi_feature_matched_seed20261031")
manifest = audit_data(root / "matched_resampling_manifest.json", Path("dataset/text"), dataset="MI")
metadata_path = Path(manifest["candidate_metadata"])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
projection = metadata["projection"]
reference = json.loads(Path(manifest["reference_manifest"]).read_text(encoding="utf-8"))
require_hash(Path(reference["augmentation_cache"]), reference["augmentation_cache_sha256"], "Projected cache")
require_hash(metadata_path, reference["projection_metadata_sha256"], "Projection metadata")
for name in ("source_input", "source_cache", "source_metadata"):
    require_hash(Path(projection[name]), projection[name + "_sha256"], "Projection origin " + name)
print("MI data audit passed")
print("Original MI training examples:", manifest["original_examples"])
print("MI candidate pool:", manifest["candidate_count"])
print("Added original examples per resampling arm:", manifest["added_examples"])
print("Maximum standardised difference:", manifest["balance_check"]["max_abs_difference_in_pool_sd_units"])
print("Overlap fraction:", manifest["overlap_check"]["fraction"])
PY

if [[ "$mode" == prepare ]]; then
  echo "MI preparation complete. Next: bash scripts/train_mi_resampling.sh train"
  exit 0
fi

python - <<'PY'
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Start AutoDL with a GPU before training.")
for name in ("bert-base-uncased", "ViTB-16"):
    path = Path("/root/autodl-tmp/pretrained_models") / name
    if not path.is_dir():
        raise SystemExit(f"Pretrained model directory missing: {path}")
PY

common=(
  --dataset MI
  --image-path dataset/images
  --twitter2017-image-path dataset/twitter2017_images
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16
  --device cuda --num-epochs 15 --batch-size 4 --num-workers 4
  --lr 5e-5 --bert-lr 1e-5 --vit-lr 5e-6
  --warmup-ratio 0.01 --weight-decay 0.01 --max-grad-norm 1.0
  --patience 3 --max-seq-length 64 --topk-img-patches 12
  --image-dropout-prob 0.2 --output-dropout 0.2 --skip-test
)

index=0
for seed in 1234 42 2026; do
  for arm in strict_baseline llm_selected matched_no_llm; do
    index=$((index + 1))
    case "$arm" in
      strict_baseline) data=dataset/text ;;
      llm_selected) data="$selected" ;;
      matched_no_llm) data="$matched" ;;
    esac
    out="outputs_mi_v63/${arm}_seed${seed}"
    done_file="$out/completed.json"
    if [[ -f "$done_file" ]]; then
      python - "$done_file" "$data" <<'PY'
import json
import sys
from pathlib import Path
from scripts.build_llm_augmentation_cache import sha256_file
marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name, digest in marker["artifacts"].items():
    if sha256_file(Path(name)) != digest:
        raise SystemExit(f"Completed artifact changed: {name}")
if marker["data_path"] != sys.argv[2]:
    raise SystemExit("Completed run data directory changed")
print("Verified completed run:", marker["checkpoint"])
PY
      echo "=== $index/9 $arm seed=$seed already complete ==="
      continue
    fi
    if [[ -d "$out" ]] && [[ -n "$(ls -A "$out")" ]]; then
      echo "Incomplete run exists: $out. Inspect it before retrying; existing results are preserved."
      exit 1
    fi
    log="logs/mi_${arm}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== $index/9 $arm seed=$seed; log=$log ==="
    if python -u run.py "${common[@]}" --data-path "$data" --output-dir "$out" --seed "$seed" \
      2>&1 | tee "$log"; then
      python - "$out" "$data" "$log" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
from scripts.build_llm_augmentation_cache import sha256_file
root, data, log = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
checkpoints = list(root.glob("*/best_model.pt"))
if len(checkpoints) != 1 or "Skipping test evaluation" not in log.read_text(encoding="utf-8", errors="replace"):
    raise SystemExit("Training did not produce one completed validation-only checkpoint")
checkpoint = checkpoints[0]
config = checkpoint.parent / "config.json"
metrics = checkpoint.parent / "metrics.csv"
paths = [checkpoint, config, metrics, Path("scripts/train_mi_resampling.sh")]
paths += [Path(name) for name in subprocess.check_output(
    ["git", "ls-files", "--", "run.py", "models", "modules"], text=True,
).splitlines()]
paths += [data / f"MNER-MI_{split}.txt" for split in ("train", "val", "test")]
marker = {"schema_version": "mi-resampling-completed-v1", "data_path": str(data),
          "checkpoint": str(checkpoint), "log": str(log),
          "artifacts": {str(path): sha256_file(path) for path in paths}}
(root / "completed.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
print("Completed:", checkpoint)
PY
    else
      status=$?
      echo "Training or logging failed: $arm seed=$seed, status=$status; log=$log"
      exit "$status"
    fi
  done
done
echo "All 9 MI validation-only training runs completed. Results: outputs_mi_v63"
