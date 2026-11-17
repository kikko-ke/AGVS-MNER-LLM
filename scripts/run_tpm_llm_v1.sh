#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
conda_bin="${TPM_CONDA_BIN:-/root/miniconda3/bin/conda}"
vlm_env="${TPM_VLM_ENV:-/root/autodl-tmp/envs/mner-vlm}"
run_dir="${TPM_RUN_DIR:-outputs_vlm_gold/tpm_mi_llm_v1}"
mode="${1:-pilot}"
if [[ "$mode" != smoke && "$mode" != pilot && "$mode" != all && "$mode" != evaluate ]]; then
    printf '%s\n' 'Usage: bash scripts/run_tpm_llm_v1.sh smoke|pilot|all|evaluate' >&2
    exit 2
fi
mkdir -p "$run_dir"
exec 9>"$run_dir/.pipeline.lock"
flock -n 9 || { printf '%s\n' 'This TPM-MI pipeline is already running.' >&2; exit 1; }
base() { "$conda_bin" run --no-capture-output -n base python -u scripts/run_tpm_llm_v1.py --run "$run_dir" "$@"; }
vlm() { "$conda_bin" run --no-capture-output -p "$vlm_env" python -u scripts/run_tpm_llm_v1.py --run "$run_dir" "$@"; }
if [[ "$mode" != evaluate ]]; then
    prepare_args=(--stage prepare --batch-size "${TPM_BATCH_SIZE:-8}")
    [[ -z "${TPM_DATA_DIR:-}" ]] || prepare_args+=(--data-dir "$TPM_DATA_DIR")
    [[ -z "${TPM_IMAGE_DIR:-}" ]] || prepare_args+=(--image-dir "$TPM_IMAGE_DIR")
    [[ -z "${TPM_BERT_MODEL:-}" ]] || prepare_args+=(--bert-model "$TPM_BERT_MODEL")
    [[ -z "${TPM_VIT_MODEL:-}" ]] || prepare_args+=(--vit-model "$TPM_VIT_MODEL")
    [[ -z "${TPM_QWEN_MODEL:-}" ]] || prepare_args+=(--qwen-model "$TPM_QWEN_MODEL")
    base "${prepare_args[@]}"
    base --stage smoke-base
    vlm --stage smoke-features
fi
if [[ "$mode" == smoke ]]; then
    printf '%s\n' "Smoke completed: $run_dir/smoke"
    exit 0
fi
if [[ "$mode" != evaluate ]]; then
    vlm --stage features
    seeds=(1234)
    [[ "$mode" != all ]] || seeds=(1234 42 2026)
    for seed in "${seeds[@]}"; do
        base --stage baseline --seed "$seed"
        vlm --stage heads --seed "$seed"
        [[ "$seed" != 1234 ]] || base --stage calibrate
    done
fi
if [[ "$mode" == pilot ]]; then
    printf '%s\n' "Development-only pilot completed: $run_dir/calibration/dev_report.json"
    printf '%s\n' 'No official test metrics were computed by the pilot.'
    exit 0
fi
vlm --stage test-features
vlm --stage test-logits
base --stage evaluate
printf '%s\n' "Completed: $run_dir/evaluation/evaluation_report.json"
