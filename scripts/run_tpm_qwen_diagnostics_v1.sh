#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
conda_bin="${TPM_CONDA_BIN:-/root/miniconda3/bin/conda}"
vlm_env="${TPM_VLM_ENV:-/root/autodl-tmp/envs/mner-vlm}"
visual_run="${TPM_VISUAL_RUN_DIR:-outputs_vlm_gold/tpm_qwen_visual_min28_v1}"
run_dir="${TPM_DIAG_RUN_DIR:-outputs_vlm_gold/tpm_qwen_diagnostics_v1}"
mode="${1:-analyze}"
case "$mode" in analyze|smoke|all|evaluate|summary) ;; *)
    printf '%s\n' 'Usage: bash scripts/run_tpm_qwen_diagnostics_v1.sh analyze|smoke|all|evaluate|summary' >&2
    exit 2 ;;
esac
base() { "$conda_bin" run --no-capture-output -n base python -u scripts/evaluate_tpm_qwen_diagnostics_v1.py --run "$run_dir" "$@"; }
vlm() { "$conda_bin" run --no-capture-output -p "$vlm_env" python -u scripts/evaluate_tpm_qwen_diagnostics_v1.py --run "$run_dir" "$@"; }
if [[ "$mode" == summary ]]; then
    base --stage summary
    exit 0
fi
mkdir -p "$run_dir"
exec 9>"$run_dir/.pipeline.lock"
flock -n 9 || { printf '%s\n' 'This diagnostic pipeline is already running.' >&2; exit 1; }
if [[ "$mode" != evaluate ]]; then
    base --stage prepare --visual-run "$visual_run" \
        --mismatch-seed "${TPM_DIAG_MISMATCH_SEED:-1234}" \
        --bootstrap-seed "${TPM_DIAG_BOOTSTRAP_SEED:-1234}" \
        --bootstrap-replicates "${TPM_DIAG_BOOTSTRAP_REPLICATES:-10000}" \
        --shard-size "${TPM_DIAG_SHARD_SIZE:-64}"
fi
if [[ "$mode" == analyze || "$mode" == all ]]; then
    base --stage analyze
fi
if [[ "$mode" == analyze ]]; then
    printf '%s\n' "CPU analysis completed: $run_dir/analysis/analysis_report.json"
    exit 0
fi
if [[ "$mode" == smoke || "$mode" == all ]]; then
    vlm --stage smoke
fi
if [[ "$mode" == smoke ]]; then
    exit 0
fi
if [[ "$mode" == all ]]; then
    vlm --stage features
    vlm --stage logits
fi
base --stage evaluate
printf '%s\n' "Frozen image-mismatch evaluation completed: $run_dir/evaluation/mismatch_report.json"
