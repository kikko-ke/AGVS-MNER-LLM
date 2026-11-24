#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
conda_bin="${TPM_CONDA_BIN:-/root/miniconda3/bin/conda}"
vlm_env="${TPM_VLM_ENV:-/root/autodl-tmp/envs/mner-vlm}"
source_run="${TPM_SOURCE_RUN:-outputs_vlm_gold/tpm_mi_llm_v1}"
run_dir="${TPM_VISUAL_RUN_DIR:-outputs_vlm_gold/tpm_qwen_visual_v1}"
mode="${1:-pilot}"
if [[ "$mode" != smoke && "$mode" != pilot && "$mode" != all && "$mode" != evaluate ]]; then
    printf '%s\n' 'Usage: bash scripts/run_tpm_qwen_visual_v1.sh smoke|pilot|all|evaluate' >&2
    exit 2
fi
mkdir -p "$run_dir"
exec 9>"$run_dir/.pipeline.lock"
flock -n 9 || { printf '%s\n' 'This visual pipeline is already running.' >&2; exit 1; }
base() { "$conda_bin" run --no-capture-output -n base python -u scripts/run_tpm_qwen_visual_v1.py --run "$run_dir" "$@"; }
vlm() { "$conda_bin" run --no-capture-output -p "$vlm_env" python -u scripts/run_tpm_qwen_visual_v1.py --run "$run_dir" "$@"; }
if [[ "$mode" != evaluate ]]; then
    base --stage prepare --source-run "$source_run" \
        --min-pixels "${TPM_VISUAL_MIN_PIXELS:-3136}" \
        --max-pixels "${TPM_VISUAL_MAX_PIXELS:-200704}" \
        --max-input-tokens "${TPM_VISUAL_MAX_INPUT_TOKENS:-2048}" \
        --shard-size "${TPM_VISUAL_SHARD_SIZE:-64}"
    vlm --stage smoke
fi
if [[ "$mode" == smoke ]]; then
    printf '%s\n' "Visual GPU smoke completed: $run_dir/smoke/completed.json"
    exit 0
fi
if [[ "$mode" != evaluate ]]; then
    vlm --stage features
    # The original frozen seed list is authoritative, including custom source runs.
    mapfile -t seeds < <("$conda_bin" run --no-capture-output -n base python -c \
        'import json,sys; print("\n".join(map(str,json.load(open(sys.argv[1]))["config"]["seeds"])))' \
        "$run_dir/protocol.json")
    [[ ${#seeds[@]} -gt 0 ]] || { printf '%s\n' 'No paired seeds in the saved protocol.' >&2; exit 1; }
    [[ "$mode" != pilot ]] || seeds=(1234)
    for seed in "${seeds[@]}"; do
        vlm --stage heads --seed "$seed"
        [[ "$seed" != 1234 ]] || base --stage calibrate
    done
fi
if [[ "$mode" == pilot ]]; then
    printf '%s\n' "Development-only visual pilot completed: $run_dir/calibration/dev_report.json"
    printf '%s\n' 'No official test metrics were computed by this pilot.'
    exit 0
fi
vlm --stage test-features
vlm --stage test-logits
base --stage evaluate
printf '%s\n' "A/B/C comparison completed: $run_dir/evaluation/comparison_report.json"
