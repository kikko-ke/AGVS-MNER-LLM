#!/usr/bin/env bash
# Explicit stages: smoke never automatically starts long training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
stage="${1:-smoke}"
if [[ "$stage" != smoke && "$stage" != train ]]; then
  echo "Usage: bash scripts/vlm_ner_pilot.sh smoke|train"
  exit 2
fi
python - <<'PY'
from modules.vlm_training import check_runtime
print('Environment:', check_runtime())
PY
mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
python -u scripts/prepare_vlm_ner.py 2>&1 | tee "logs/vlm_prepare_$stamp.log"
if [[ "$stage" == smoke ]]; then
  python -u scripts/download_vlm_model.py 2>&1 | tee "logs/vlm_download_$stamp.log"
  python -u scripts/run_vlm_ner.py --stage smoke \
    --output outputs_vlm_gold/mi_multi_image_smoke \
    2>&1 | tee "logs/vlm_smoke_$stamp.log"
  echo "Smoke complete. Inspect outputs_vlm_gold/mi_multi_image_smoke/smoke_report.json before training."
else
  python -u scripts/run_vlm_ner.py --stage train \
    --smoke-report outputs_vlm_gold/mi_multi_image_smoke/smoke_report.json \
    --output outputs_vlm_gold/mi_multi_image_seed1234 \
    2>&1 | tee "logs/vlm_train_$stamp.log"
fi
