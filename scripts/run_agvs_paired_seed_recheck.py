"""Run two fixed AGVS seed pairs and score them on the previously used grouped holdout."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from scripts import evaluate_agvs_text_control_holdout as text_eval
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import train_bio_joint_recheck as train

ROOT = train.ROOT / 'agvs_paired_seeds_v1'
SEEDS = (42, 2026)
ARMS = ('multimodal', 'text')
ARCHIVED = text_eval.OUTPUT / 'evaluation_report.json'
SCOPE = 'supplemental paired-seed comparison on the previously scored grouped holdout'
PILOT = 'agvs-paired-seed-training-v1'
EVALUATION = 'agvs-paired-seed-holdout-v1'
SUMMARY = ROOT / 'paired_seed_summary.json'


def protocol():
    runtime = recheck.baseline.check_baseline_runtime()
    prepared, full = train.load_prepared(train.split.DATA)
    rows, positions, binding = train.training_plan(prepared, full)
    original = train.READ_JSON(recheck.BASELINE_REPORT)
    if original.get('runtime') != runtime or original.get('recheck') != binding:
        raise ValueError('Use the same original AGVS runtime and frozen split as seed 1234')
    config = train.baseline_config(prepared)
    index = {row['source_id']: row for row in full['train']}
    holdout = [index[sid] for sid in train.READ_JSON(train.split.OUTPUT)['source_ids']['recheck_holdout']]
    return runtime, prepared, rows, positions, holdout, binding, config


def seed_dir(seed):
    return ROOT / f'seed{seed}'


def train_arm(seed, arm):
    runtime, _, rows, positions, _, binding, config = protocol()
    import torch
    from torch.utils.data import DataLoader, Subset
    from models.agvs_mner import AgvsMnerModel
    from modules.context_consistency import ContextConsistencyTrainer, ContextViewDataset
    from modules.dataset import MultimodalNerDataset
    from run import configure_logging, set_seed
    from scripts.train_agvs_text_control import TextInputDataset

    output = seed_dir(seed) / arm
    args = SimpleNamespace(**{**config, 'seed': seed, 'output_dir': str(output)})
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Run the AGVS seed pairs with CUDA in Conda base')
    set_seed(seed)
    dataset = TextInputDataset if arm == 'text' else MultimodalNerDataset
    base = dataset(args, 'MNER-MI_train.txt')
    fit = Subset(ContextViewDataset(base, args.internal_manifest, args.augmentation_cache, args.dataset),
                 positions)
    dev = dataset(args, 'MNER-MI_val.txt')
    if ([base.examples['words'][i] for i in positions] != [row['tokens'] for row in rows['train']]
            or dev.examples['words'] != [row['tokens'] for row in rows['dev']]):
        raise ValueError('Training or selection inputs differ from the frozen cohort')
    options = {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    fit_loader = DataLoader(fit, **options, shuffle=True, generator=torch.Generator().manual_seed(seed))
    dev_loader = DataLoader(dev, **options, shuffle=False, generator=torch.Generator().manual_seed(seed + 1))
    output.mkdir(parents=True, exist_ok=False)
    configure_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    logger.info('Paired AGVS seed %d, %s, %d training and %d selection rows', seed, arm, len(fit), len(dev))
    labels = base.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(labels), args)
    trainer = ContextConsistencyTrainer(fit_loader, dev_loader, None, model, labels, args, logger)
    trainer.train()
    report_path = output / 'completed.json'
    artifacts = (trainer.checkpoint_path, trainer.run_directory / 'config.json', trainer.results_path)
    train.WRITE_JSON(report_path, {
        'schema_version': PILOT, 'seed': seed, 'arm': arm, 'recheck': binding, 'runtime': runtime,
        'best_epoch': trainer.best_dev_epoch, 'best_dev_f1': float(trainer.best_dev_f1),
        'checkpoint': str(trainer.checkpoint_path.resolve()),
        'output_sha256': {str(path.resolve()): sha256_file(path) for path in artifacts},
        'runner_sha256': sha256_file(__file__), 'official_val_or_test_read': False,
    })
    print('Completed:', report_path, flush=True)


def read_arm(seed, arm, runtime, binding, config):
    output = seed_dir(seed) / arm
    report_path = output / 'completed.json'
    report = train.READ_JSON(report_path)
    expected = {'schema_version': PILOT, 'seed': seed, 'arm': arm, 'recheck': binding,
                'runtime': runtime, 'runner_sha256': sha256_file(__file__),
                'official_val_or_test_read': False}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Unexpected training report: {report_path}')
    checkpoint = Path(report['checkpoint']).resolve()
    config_path = checkpoint.parent / 'config.json'
    inputs = report['output_sha256']
    if any(str(path) not in inputs for path in (checkpoint, config_path)):
        raise ValueError('Training report must bind the checkpoint and its config')
    recheck.verify_files(inputs)
    saved = train.READ_JSON(config_path)
    if saved != {**config, 'seed': seed, 'output_dir': str(output)}:
        raise ValueError('Training settings differ from the paired protocol')
    return report, checkpoint, saved, {**inputs, str(report_path): sha256_file(report_path)}


def predict(directory, source, source_lines, rows, prepared, arm, checkpoint, config):
    if arm == 'text':
        return text_eval.predict_text(directory, source_lines, rows, checkpoint, config, prepared)
    unlabeled = directory / 'MNER-MI_val.txt'
    recheck.write_unlabeled_inputs(unlabeled, source, rows)
    plan = {'checkpoint': checkpoint, 'config': {**config, 'data_path': str(directory)},
            'manifest': prepared, 'groups': {'full_dev': rows}}
    result = recheck.baseline.collect_predictions(plan, config['device'])
    unlabeled.unlink()
    return result


def score_pair(seed, runtime, prepared, dev, holdout, binding, config):
    output = seed_dir(seed) / 'holdout'
    report_path = output / 'evaluation_report.json'
    if report_path.exists():
        return report_path
    source = Path(verify_internal_manifest(prepared['internal_manifest'], 'MI')['source'])
    source_lines = source.read_text(encoding='utf-8').splitlines()
    inputs = {str(train.split.OUTPUT): train.SPLIT_SHA256, str(Path(__file__).resolve()): sha256_file(__file__)}
    scores, epochs, reproduced, predictions = {}, {}, {}, {}
    with TemporaryDirectory(dir=output.parent, prefix='.paired-evaluation-') as directory:
        scratch = Path(directory)
        for arm in ARMS:
            report, checkpoint, saved, artifacts = read_arm(seed, arm, runtime, binding, config)
            inputs.update(artifacts)
            print(f'Seed {seed}, {arm}: reproducing selection dev...', flush=True)
            _, native, _ = predict(scratch, source, source_lines, dev, prepared, arm, checkpoint, saved)
            if abs(native['f1'] - report['best_dev_f1']) > 0.0005:
                raise ValueError(f'Seed {seed}, {arm}: native selection score failed reproduction')
            reproduced[arm] = native['f1']
            epochs[arm] = report['best_epoch']
            print(f'Seed {seed}, {arm}: scoring {len(holdout)} grouped holdout rows...', flush=True)
            records, _, _ = predict(scratch, source, source_lines, holdout, prepared, arm, checkpoint, saved)
            indexed = recheck.bio.original.checked_records(holdout, records)
            scores[arm] = recheck.bio.original.metrics_for(holdout, indexed, SCOPE)
            path = scratch / f'{arm}_predictions.jsonl'
            recheck.bio.write_records(path, records)
            predictions[path.name] = sha256_file(path)
        recheck.verify_files(inputs)
        train.WRITE_JSON(scratch / report_path.name, {
            'schema_version': EVALUATION, 'scope': SCOPE, 'seed': seed, 'recheck': binding,
            'source_ids': recheck.source_ids(holdout), 'best_epochs': epochs,
            'selection_f1_reproduced': reproduced, 'metrics': scores, 'input_sha256': inputs,
            'output_predictions_sha256': predictions, 'optimizer_steps': 0,
            'new_checkpoint_selection': False, 'official_val_or_test_read': False,
        })
        scratch.rename(output)
    return report_path


def summarize(paths, holdout, binding):
    ids = recheck.source_ids(holdout)
    paired = {}
    for seed, path in paths.items():
        report = train.READ_JSON(path)
        expected_schema = 'agvs-text-control-holdout-v1' if seed == 1234 else EVALUATION
        if (report.get('schema_version') != expected_schema or report.get('source_ids') != ids
                or report.get('recheck') != binding
                or report.get('official_val_or_test_read') is not False
                or report.get('new_checkpoint_selection') is not False
                or (seed != 1234 and report.get('seed') != seed)):
            raise ValueError(f'Report does not match the frozen seed/cohort: {path}')
        recheck.verify_files(report['input_sha256'])
        if seed == 1234:
            text_predictions = path.parent / 'text_predictions.jsonl'
            recheck.verify_files({str(text_predictions): report['text_predictions_sha256']})
            _, multimodal = recheck.load_baseline_predictions(holdout, binding)
            records = {'multimodal': multimodal, 'text': checked_predictions(holdout, text_predictions)}
            scope = recheck.SCOPE
        else:
            predictions = report['output_predictions_sha256']
            recheck.verify_files({str(path.parent / name): predictions[name]
                                  for name in (f'{arm}_predictions.jsonl' for arm in ARMS)})
            records = {arm: checked_predictions(holdout, path.parent / f'{arm}_predictions.jsonl')
                       for arm in ARMS}
            scope = SCOPE
        metrics = {arm: recheck.bio.original.metrics_for(holdout, records[arm], scope) for arm in ARMS}
        if metrics != report['metrics']:
            raise ValueError(f'Saved predictions do not reproduce the scores: {path}')
        f1 = {arm: values['f1'] for arm, values in metrics.items()}
        paired[str(seed)] = {**f1, 'multimodal_minus_text_f1': f1['multimodal'] - f1['text']}
    deltas = [row['multimodal_minus_text_f1'] for row in paired.values()]
    summary = {
        'scope': SCOPE, 'samples_per_seed': len(holdout), 'by_seed': paired,
        'mean_multimodal_minus_text_f1': statistics.mean(deltas),
        'sample_std_of_seed_deltas': statistics.stdev(deltas),
        'positive_seed_count': sum(value > 0 for value in deltas),
        'seed_reports_sha256': {str(path.resolve()): sha256_file(path) for path in paths.values()},
    }
    if SUMMARY.exists():
        if train.READ_JSON(SUMMARY) != summary:
            raise ValueError(f'Existing summary differs: {SUMMARY}')
    else:
        with SUMMARY.open('x', encoding='utf-8') as destination:
            destination.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({**summary, 'report': str(SUMMARY.resolve())}, indent=2), flush=True)


def checked_predictions(rows, path):
    return recheck.bio.original.checked_records(rows, recheck.bio.original.read_records(path))


def score_all():
    runtime, prepared, rows, _, holdout, binding, config = protocol()
    paths = {1234: ARCHIVED}
    for seed in SEEDS:
        paths[seed] = score_pair(seed, runtime, prepared, rows['dev'], holdout, binding, config)
    summarize(paths, holdout, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if (args.train_seed is None) != (args.arm is None):
        parser.error('--train-seed and --arm must be set together')
    if args.train_seed is not None:
        train_arm(args.train_seed, args.arm)
        return
    for seed in SEEDS:
        for arm in ARMS:
            if (seed_dir(seed) / arm / 'completed.json').is_file():
                print(f'Using completed training: seed {seed}, {arm}', flush=True)
                continue
            print(f'Training: seed {seed}, {arm}', flush=True)
            subprocess.run([sys.executable, '-u', __file__, '--train-seed', str(seed), '--arm', arm], check=True)
    score_all()


if __name__ == '__main__':
    main()
