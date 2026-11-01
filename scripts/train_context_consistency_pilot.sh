#!/usr/bin/env bash
# Fixed three-arm MI pilot on a new internal development split, foreground only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs

split=dataset/context_internal_mi_seed20261101
manifest="$split/context_internal_manifest.json"
cache=augmentation_cache/mi_from_v61/MNER-MI_train.jsonl
if [[ ! -f "$manifest" ]]; then
  python -u scripts/create_context_internal_split.py \
    --input dataset/text/MNER-MI_train.txt --dataset MI --seed 20261101 --output-dir "$split" \
    2>&1 | tee "logs/context_internal_split_$(date +%Y%m%d_%H%M%S).log"
fi

common=(
  --dataset MI --data-path "$split" --internal-manifest "$manifest" --augmentation-cache "$cache"
  --image-path dataset/images --twitter2017-image-path dataset/twitter2017_images
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16
  --device cuda --num-epochs 15 --batch-size 4 --num-workers 4
  --lr 5e-5 --bert-lr 1e-5 --vit-lr 5e-6 --warmup-ratio 0.01
  --weight-decay 0.01 --max-grad-norm 1.0 --patience 3 --seed 1234
  --max-seq-length 64 --topk-img-patches 12 --image-dropout-prob 0.2 --output-dropout 0.2
  --visual-fusion legacy --consistency-weight 0.5 --consistency-temperature 2
  --consistency-warmup-epochs 2 --skip-test
)
for mode in none dropout llm_context; do
  out="outputs_context_internal/${mode}_seed1234"
  if [[ -d "$out" ]] && [[ -n "$(ls -A "$out")" ]]; then
    echo "Existing run preserved: $out. Inspect it before retrying."
    exit 1
  fi
  log="logs/context_internal_${mode}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== Internal MI context pilot: $mode; log=$log ==="
  python -u run_context_consistency.py "${common[@]}" --consistency-mode "$mode" --output-dir "$out" \
    2>&1 | tee "$log"
done

python - <<'PY'
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from modules.dataset import MultimodalNerDataset
from scripts.confirm_crf_ensemble import _single_metrics
from scripts.evaluate_crf_ensemble import _collect
from scripts.build_llm_augmentation_cache import sha256_file

scores, artifacts = {}, {}
dataset = None
for mode in ("none", "dropout", "llm_context"):
    paths = list(Path(f"outputs_context_internal/{mode}_seed1234").glob("*/best_model.pt"))
    if len(paths) != 1:
        raise SystemExit(f"Expected one completed {mode} checkpoint")
    checkpoint = paths[0]
    metrics, config = checkpoint.parent / "metrics.csv", checkpoint.parent / "config.json"
    config_values = json.loads(config.read_text(encoding="utf-8"))
    cli = SimpleNamespace(device=config_values["device"], batch_size=4, num_workers=4)
    if dataset is None:
        dataset = MultimodalNerDataset(SimpleNamespace(**config_values), "MNER-MI_val.txt")
    label_map = dataset.processor.get_label_mapping(include_subword_label=True)
    collected = _collect(checkpoint, config_values, dataset, cli, label_map, f"internal-dev/{mode}")
    _, native = _single_metrics(collected, label_map, checkpoint)
    scores[mode] = native["f1"]
    artifacts[mode] = {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "config": config_values, "internal_dev_metrics": native}
    del collected
gains = {name: scores["llm_context"] - scores[name] for name in ("none", "dropout")}
report = {"schema_version": "context-internal-pilot-v1", "split": "training_internal_dev", "dataset": "MI",
          "seed": 1234, "native_dev_f1": scores, "gains": gains, "artifacts": artifacts,
          "minimum_gain_over_both_controls": 0.003, "eligible_for_replication": all(x >= 0.003 for x in gains.values()),
          "test_read": False, "llm_api_requests": 0,
          "note": "Predeclared small pilot, not proof of test performance. Earlier official-test results remain negative."}
output = Path(f"outputs_context_internal/pilot_{datetime.now():%Y%m%d_%H%M%S}.json")
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("\n=== MI training-internal pilot (not official validation/test) ===")
for mode, score in scores.items():
    print(f"{mode:<14} F1={score:.4f}")
print("LLM gains:", gains)
print("Eligible for seed replication:", report["eligible_for_replication"])
print("Report:", output)
PY
