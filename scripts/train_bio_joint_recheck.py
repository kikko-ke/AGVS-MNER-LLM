"""Retrain the AGVS control and BIO candidate on the frozen grouped split."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from modules.vlm_ner import load_prepared
from scripts import evaluate_baseline_fullwords as baseline
from scripts import prepare_bio_joint_recheck as split
from scripts import run_vlm_bio as bio

ROOT = split.OUTPUT.parent
SPLIT_SHA256 = 'd1d0d447603fedaa5936cc1161fc679b6ab7bff1b4aea8408f9ca36e89684dfc'
BASELINE_REFERENCE = Path(
    'outputs_vlm_gold/baseline_fullwords_seed1234_dev_evaluation/comparison_report.json')
BIO_MODEL = '/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct'
BIO_CONTROL = 'outputs_vlm_gold/mi_multi_image_seed1234'
READ_JSON = bio.original.read_json
WRITE_JSON = bio.common.write_json


def training_plan(prepared, rows):
    if sha256_file(split.OUTPUT) != SPLIT_SHA256:
        raise ValueError('The grouped split differs from the accepted server preflight')
    frozen = READ_JSON(split.OUTPUT)
    policy = {'version': 'bio_joint_v1', 'sha256': sha256_file(split.POLICY),
              'bio_codec_sha256': sha256_file(split.BIO_CODEC)}
    if (frozen['ready'] is not True or frozen['frozen_policy'] != policy
            or frozen['prepared_manifest_sha256'] != sha256_file(split.DATA)
            or frozen['split_script_sha256'] != sha256_file(split.__file__)):
        raise ValueError('The accepted split, data or fixed BIO policy changed')

    ids = frozen['source_ids']
    holdout = set(ids['recheck_holdout'])
    positions = [i for i, row in enumerate(rows['train']) if row['source_id'] not in holdout]
    retrain = [rows['train'][i] for i in positions]
    if (len(holdout) != 1096 or len(retrain) != 4388
            or [row['source_id'] for row in retrain] != ids['retrain']
            or [row['source_id'] for row in rows['dev']] != ids['selection_dev']
            or [row['source_id'] for row in rows['dev_panel']] != ids['bio_selection_panel']
            or prepared['panel_source_ids'] != ids['bio_selection_panel']
            or not holdout.isdisjoint(ids['selection_dev'])):
        raise ValueError('Training and selection rows differ from the fixed split')
    binding = {'split_manifest_sha256': SPLIT_SHA256, 'frozen_policy': policy,
               'runner_sha256': sha256_file(__file__),
               'train_samples': len(retrain), 'holdout_samples': len(holdout)}
    return {**rows, 'train': retrain}, positions, binding


def baseline_config(prepared):
    report = READ_JSON(BASELINE_REFERENCE)
    config_path = (Path(report['checkpoint']).parent / 'config.json').resolve()
    config = READ_JSON(config_path)
    if (report['schema_version'] != 'baseline-vlm-fullword-comparison-v1'
            or report['baseline_native_reproduction']['passed'] is not True
            or config != report['baseline_training_config']
            or report['input_bindings'][str(config_path)] != sha256_file(config_path)
            or report['input_bindings'][str(split.DATA.resolve())] != sha256_file(split.DATA)):
        raise ValueError('Expected the archived, reproduced AGVS baseline configuration')
    internal = Path(prepared['internal_manifest'])
    original_split = verify_internal_manifest(internal, 'MI')
    baseline.validate_baseline_config(config, internal, original_split)
    return config


def train_baseline():
    runtime = baseline.check_baseline_runtime()
    import torch
    from torch.utils.data import DataLoader, Subset

    from models.agvs_mner import AgvsMnerModel
    from modules.context_consistency import ContextConsistencyTrainer, ContextViewDataset
    from modules.dataset import MultimodalNerDataset
    from run import configure_logging, set_seed

    prepared, full = load_prepared(split.DATA)
    rows, positions, binding = training_plan(prepared, full)
    config = baseline_config(prepared)
    if not config['device'].startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The AGVS recheck requires CUDA in the baseline environment')
    for key in ('bert_model', 'vit_model'):
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(f'Original local model is missing: {config[key]}')
    output = ROOT / 'baseline'
    args = SimpleNamespace(**{**config, 'output_dir': str(output)})
    set_seed(args.seed)
    base = MultimodalNerDataset(args, 'MNER-MI_train.txt')
    context = ContextViewDataset(base, args.internal_manifest, args.augmentation_cache, args.dataset)
    dev = MultimodalNerDataset(args, 'MNER-MI_val.txt')
    if ([base.examples['words'][i] for i in positions] != [row['tokens'] for row in rows['train']]
            or dev.examples['words'] != [row['tokens'] for row in rows['dev']]):
        raise ValueError('AGVS dataset indices differ from the shared original words')

    # Subset the original aligned dataset before shuffling or fetching training batches.
    train = Subset(context, positions)
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, generator=torch.Generator().manual_seed(args.seed + 1))
    output.mkdir(exist_ok=False)
    configure_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    logger.info('Grouped recheck training rows %d, selection dev rows %d', len(train), len(dev))
    labels = base.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(labels), args)
    trainer = ContextConsistencyTrainer(train_loader, dev_loader, None, model, labels, args, logger)
    trainer.train()

    result = {'schema_version': 'bio-joint-recheck-baseline-v1', 'recheck': binding,
              'checkpoint': str(trainer.checkpoint_path.resolve()),
              'config': str((trainer.run_directory / 'config.json').resolve()),
              'best_epoch': trainer.best_dev_epoch, 'best_dev_f1': trainer.best_dev_f1,
              'runtime': runtime, 'official_val_or_test_read': False}
    result['input_sha256'] = {str(path.resolve()): sha256_file(path) for path in (
        trainer.checkpoint_path, trainer.run_directory / 'config.json', trainer.results_path,
        BASELINE_REFERENCE, Path(args.augmentation_cache))}
    path = output / 'completed.json'
    WRITE_JSON(path, result)
    print(f'AGVS best selection epoch {trainer.best_dev_epoch}, report {path}', flush=True)


def bio_inputs(stage):
    args = SimpleNamespace(stage=stage, model=BIO_MODEL, data=str(split.DATA),
                           control_run=BIO_CONTROL, run=str(ROOT / 'bio'))
    profile, prepared, full, _ = bio.load_profile(args)
    rows, _, binding = training_plan(prepared, full)
    return args, {**profile, 'recheck': binding}, rows


def train_bio(stage):
    args, profile, rows = bio_inputs(stage)
    if stage == 'train':
        smoke = READ_JSON(ROOT / 'bio_smoke' / 'smoke_report.json')
        if (smoke.get('schema_version') != bio.SMOKE_SCHEMA or smoke.get('profile') != profile
                or smoke.get('passed') is not True or smoke.get('optimizer_steps') != 8
                or smoke.get('generation_completed') is not True):
            raise ValueError('Expected the passing BIO smoke from this grouped split')
    if bio.backend.check_runtime() != profile['versions']:
        raise ValueError('Use the pinned original BIO runtime')
    import torch

    output = ROOT / ('bio_smoke' if stage == 'smoke' else 'bio')
    output.mkdir(exist_ok=False)
    WRITE_JSON(output / 'config.json', {'stage': stage, 'profile': profile,
                                       'official_val_or_test_read': False})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    try:
        name, report = bio.train(args, profile, rows, output)
        WRITE_JSON(output / name, report)
    except Exception as error:
        WRITE_JSON(output / 'failure.json', {'stage': stage, 'error': str(error)})
        raise
    print(f'BIO {stage} report {output / name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('baseline', 'bio-smoke', 'bio-train'))
    stage = parser.parse_args().stage
    if stage == 'baseline':
        train_baseline()
    else:
        train_bio(stage.removeprefix('bio-'))


if __name__ == '__main__':
    main()
