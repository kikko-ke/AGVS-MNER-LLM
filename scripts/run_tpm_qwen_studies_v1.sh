#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
conda_bin="${TPM_CONDA_BIN:-/root/miniconda3/bin/conda}"
vlm_env="${TPM_VLM_ENV:-/root/autodl-tmp/envs/mner-vlm}"
visual_run="${TPM_VISUAL_RUN_DIR:-outputs_vlm_gold/tpm_qwen_visual_min28_v1}"
qwen7b_model="${TPM_QWEN7B_MODEL:-/root/autodl-tmp/pretrained_models/Qwen2.5-VL-7B-Instruct}"
mode="${1:-fusion}"
case "$mode" in download7b|fusion|smoke7b|pilot7b|all7b|summary-fusion|summary7b) ;; *)
    printf '%s\n' 'Usage: bash scripts/run_tpm_qwen_studies_v1.sh download7b|fusion|smoke7b|pilot7b|all7b|summary-fusion|summary7b' >&2
    exit 2 ;;
esac
if [[ "$mode" == download7b ]]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
    "$conda_bin" run --no-capture-output -p "$vlm_env" python -u scripts/download_qwen7b_v1.py --output "$qwen7b_model"
    exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
experiment=7b
run_dir="${TPM_7B_RUN_DIR:-outputs_vlm_gold/tpm_qwen_7b_v1}"
if [[ "$mode" == fusion || "$mode" == summary-fusion ]]; then
    experiment=fusion
    run_dir="${TPM_FUSION_RUN_DIR:-outputs_vlm_gold/tpm_qwen_fusion_v1}"
fi
base() { "$conda_bin" run --no-capture-output -n base python -u scripts/run_tpm_qwen_studies_v1.py --experiment "$experiment" --run "$run_dir" "$@"; }
vlm() { "$conda_bin" run --no-capture-output -p "$vlm_env" python -u scripts/run_tpm_qwen_studies_v1.py --experiment "$experiment" --run "$run_dir" "$@"; }
if [[ "$mode" == summary-fusion || "$mode" == summary7b ]]; then
    base --stage summary
    exit 0
fi
mkdir -p "$run_dir"
exec 9>"$run_dir/.pipeline.lock"
flock -n 9 || { printf '%s\n' 'This study is already running.' >&2; exit 1; }
base --stage prepare --visual-run "$visual_run" --qwen7b-model "$qwen7b_model"
if [[ "$experiment" == fusion ]]; then
    base --stage fusion-calibrate
    base --stage evaluate
    exit 0
fi
vlm --stage smoke
if [[ "$mode" == smoke7b ]]; then
    exit 0
fi
vlm --stage features
mapfile -t seeds < <("$conda_bin" run --no-capture-output -n base python -c \
    'import json,sys; print("\n".join(map(str,json.load(open(sys.argv[1]))["config"]["seeds"])))' "$run_dir/protocol.json")
[[ ${#seeds[@]} -gt 0 ]] || { printf '%s\n' 'No paired seeds in the protocol.' >&2; exit 1; }
[[ "$mode" != pilot7b ]] || seeds=(1234)
for seed in "${seeds[@]}"; do
    vlm --stage heads --seed "$seed"
    [[ "$seed" != 1234 ]] || base --stage calibrate
done
if [[ "$mode" == pilot7b ]]; then
    printf '%s\n' "Development-only 7B pilot complete: $run_dir/calibration/dev_report.json"
    exit 0
fi
vlm --stage test-features
vlm --stage test-logits
base --stage evaluate
