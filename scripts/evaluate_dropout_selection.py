"""Replay the fixed, source-only dropout selection pilot on internal dev."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset import MultimodalNerDataset
from modules.dropout_selection import ARMS, FIXED_TRAINING, DropoutSelectionDataset
from scripts.build_llm_augmentation_cache import sha256_file
from scripts.confirm_crf_ensemble import _single_metrics
from scripts.evaluate_crf_ensemble import _collect


def completed_run(root, arm, cohort_path):
    checkpoints = list((Path(root) / f"{arm}_seed1234").glob("*/best_model.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one checkpoint for {arm}; preserve and inspect existing runs")
    checkpoint = checkpoints[0]
    config_path, complete_path = checkpoint.parent / "config.json", checkpoint.parent / "completed.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed = json.loads(complete_path.read_text(encoding="utf-8"))
    if (completed.get("schema_version") != "dropout-selection-completed-v1" or completed.get("test_read") is not False
            or completed["checkpoint_sha256"] != sha256_file(checkpoint)
            or completed["config_sha256"] != sha256_file(config_path)
            or completed["cohort_sha256"] != sha256_file(Path(cohort_path))
            or config.get("dropout_selection_sha256") != completed["cohort_sha256"]
            or config.get("dropout_selection_arm") != arm):
        raise ValueError("Completed run/hash binding changed")
    if any(config.get(key) != value for key, value in FIXED_TRAINING.items()):
        raise ValueError("Run does not use the fixed dropout selection protocol")
    if Path(config["dropout_selection_manifest"]).resolve() != Path(cohort_path).resolve():
        raise ValueError("Run uses another cohort manifest")
    return checkpoint, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="outputs_dropout_selection_internal")
    parser.add_argument("--cohort", default="dataset/context_internal_mi_seed20261101/dropout_cohorts_seed20261102.json")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()
    if cli.check_only:
        if not cli.arm:
            parser.error("--check-only requires --arm")
        checkpoint, _ = completed_run(cli.runs_root, cli.arm, cli.cohort)
        print("Verified completed run:", checkpoint)
        return
    configs, checkpoints = {}, {}
    for arm in ARMS:
        checkpoints[arm], configs[arm] = completed_run(cli.runs_root, arm, cli.cohort)
    ignored = {"output_dir", "dropout_selection_arm"}
    if ({k: v for k, v in configs[ARMS[0]].items() if k not in ignored}
            != {k: v for k, v in configs[ARMS[1]].items() if k not in ignored}):
        raise ValueError("Training configurations differ beyond arm and output directory")
    scores, artifacts = {}, {}
    for arm in ARMS:
        config, checkpoint = configs[arm], checkpoints[arm]
        args = SimpleNamespace(**config)
        base = MultimodalNerDataset(args, "MNER-MI_train.txt")
        DropoutSelectionDataset(base, cli.cohort, arm, args.internal_manifest, args.augmentation_cache, "MI", args.bert_model)
        dev = MultimodalNerDataset(args, "MNER-MI_val.txt")
        label_map = dev.processor.get_label_mapping(include_subword_label=True)
        collected = _collect(checkpoint, config, dev, SimpleNamespace(device=cli.device, batch_size=4, num_workers=4),
                             label_map, f"internal-dev/{arm}")
        _, native = _single_metrics(collected, label_map, checkpoint)
        scores[arm] = native["f1"]
        artifacts[arm] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                          "config": config, "metrics": native}
        del collected
    gain = scores["llm_selected"] - scores["feature_matched"]
    report = {"schema_version": "dropout-selection-evaluation-v1", "dataset": "MI", "seed": 1234,
              "split": "training_internal_dev", "native_dev_f1": scores, "gain_over_feature_matched": gain,
              "minimum_gain_for_replication": 0.003, "eligible_for_replication": gain >= 0.003,
              "cohort_sha256": sha256_file(Path(cli.cohort)), "artifacts": artifacts,
              "llm_api_requests": 0, "official_val_or_test_read": False,
              "note": "Post-hoc selection pilot; no performance guarantee or significance claim. "
                      "Prior context pilot did not beat dropout. Do not compare new source-only masks "
                      "with the old rewrite-truncated dropout score as if they were identical."}
    path = Path(cli.runs_root) / f"selection_pilot_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("\n=== MI source-only dropout selection (internal dev) ===")
    for arm, score in scores.items():
        print(f"{arm:<18} F1={score:.6f}")
    print(f"LLM selected minus matched: {gain:+.6f}")
    print("Eligible for seed replication:", report["eligible_for_replication"])
    print("Report:", path)


if __name__ == "__main__":
    main()
