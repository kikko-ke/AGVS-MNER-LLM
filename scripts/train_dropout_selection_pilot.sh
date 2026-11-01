#!/usr/bin/env bash
# Two fixed source-only dropout arms; foreground, no API calls or held-out input.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
manifest=dataset/context_internal_mi_seed20261101/context_internal_manifest.json
cohort=dataset/context_internal_mi_seed20261101/dropout_cohorts_seed20261102.json
test -f "$manifest" || { echo "Existing internal split missing; preserve the original assignment."; exit 1; }
python -u scripts/prepare_dropout_selection.py 2>&1 | tee "logs/dropout_selection_prepare_$stamp.log"

common=(
  --dataset MI --data-path dataset/context_internal_mi_seed20261101 --internal-manifest "$manifest"
  --augmentation-cache augmentation_cache/mi_from_v61/MNER-MI_train.jsonl
  --dropout-selection-manifest "$cohort" --consistency-mode dropout
  --image-path dataset/images --twitter2017-image-path dataset/twitter2017_images
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16
  --device cuda --num-epochs 15 --batch-size 4 --num-workers 4
  --lr 5e-5 --bert-lr 1e-5 --vit-lr 5e-6 --warmup-ratio 0.01
  --weight-decay 0.01 --max-grad-norm 1 --patience 3 --seed 1234
  --max-seq-length 64 --topk-img-patches 12 --image-dropout-prob 0.2 --output-dropout 0.2
  --visual-fusion legacy --consistency-weight 0.5 --consistency-temperature 2
  --consistency-warmup-epochs 2 --skip-test
)
for arm in llm_selected feature_matched; do
  out="outputs_dropout_selection_internal/${arm}_seed1234"
  if [[ -d "$out" && -n "$(ls -A "$out")" ]]; then
    # Only fully completed, hash-verified runs can be skipped. Partial runs stop.
    python -u scripts/evaluate_dropout_selection.py --check-only --arm "$arm"
    continue
  fi
  log="logs/dropout_selection_${arm}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== Source-only dropout: $arm; log=$log ==="
  python -u run_context_consistency.py "${common[@]}" --dropout-selection-arm "$arm" --output-dir "$out" \
    2>&1 | tee "$log"
done
python -u scripts/evaluate_dropout_selection.py 2>&1 | tee "logs/dropout_selection_evaluate_$stamp.log"
