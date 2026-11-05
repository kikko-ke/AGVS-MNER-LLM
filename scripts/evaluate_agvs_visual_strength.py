"""Probe the legacy AGVS visual coefficient with the frozen epoch-11 checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import train_bio_joint_recheck as train

OUTPUT = train.ROOT / 'dev_visual_strength_v1.json'
SHUFFLE_SEED = 20261019
REFERENCE_F1 = {'matched': 0.8059550845319204, 'shuffled': 0.8012108980827448}
MIN_MATCHED_DELTA = 0.003
REPRODUCTION_TOLERANCE = 0.0005


def unit_strength(_model, args, kwargs):
    """Keep absent images at zero and set the strength to one for any valid image."""
    strength = kwargs['image_mask'].bool().any(dim=1)
    return args, {**kwargs, 'visual_strength_override': strength}


def comparison(legacy, unit):
    legacy_matched = float(legacy['matched']['f1'])
    unit_matched = float(unit['matched']['f1'])
    matched_gain = unit_matched - legacy_matched
    shuffled_gain = float(unit['shuffled']['f1']) - float(legacy['shuffled']['f1'])
    interaction = matched_gain - shuffled_gain
    return {
        'unit_minus_legacy_matched_f1': matched_gain,
        'unit_minus_legacy_shuffled_f1': shuffled_gain,
        'change_in_matched_minus_shuffled_f1': interaction,
        'minimum_matched_f1_gain': MIN_MATCHED_DELTA,
        'advance_to_training_pilot': (
            unit_matched >= legacy_matched + MIN_MATCHED_DELTA and interaction > 0
        ),
    }


def evaluate_views(evaluator, datasets, mode, loader_options):
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    scores = {}
    for name, dataset in datasets.items():
        progress = tqdm(DataLoader(dataset, **loader_options), desc=f'{mode}/{name}', unit='batch')
        with progress:
            metrics = evaluator._run_epoch(progress, training=False)
        scores[name] = {**metrics, **{key: float(metrics[key])
                                    for key in ('f1', 'precision', 'recall', 'accuracy', 'loss')}}
        print(f'\n=== {mode}/{name} ===\n{metrics["report"]}', flush=True)
    return scores


def print_summary(report, path):
    print(json.dumps({
        'samples': report['samples'],
        'f1': {mode: {name: score['f1'] for name, score in values.items()}
               for mode, values in report['metrics'].items()},
        **report['comparison'], 'report': str(path.resolve()),
    }, indent=2), flush=True)


def recover_partial(path):
    """Recover the four scores from the v1 write that stopped at the NumPy gate."""
    original = path.read_bytes()
    text = original.decode('utf-8').rstrip()
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        if not text.endswith('"advance_to_training_pilot":'):
            raise ValueError('The report does not end at the known serialization failure')
        report = json.loads(text + ' null}}')
    else:
        print_summary(report, path)
        return

    if (report.get('schema_version') != 'agvs-visual-strength-dev-probe-v1'
            or report.get('best_epoch') != 11 or report.get('samples') != 1372
            or report.get('shuffle_seed') != SHUFFLE_SEED
            or report.get('reassigned_donors') != 1372
            or report.get('legacy_reproduction') != {'expected_f1': REFERENCE_F1, 'passed': True}):
        raise ValueError('The partial report does not match the fixed visual-strength probe')
    legacy = report['metrics']['legacy']
    unit = report['metrics']['unit_if_image_present']
    if (any(not 0 <= scores[name]['f1'] <= 1 for scores in (legacy, unit) for name in REFERENCE_F1)
            or any(abs(legacy[name]['f1'] - expected) > REPRODUCTION_TOLERANCE
                   for name, expected in REFERENCE_F1.items())):
        raise ValueError('The saved F1 values fail the reproduction check')
    delta = comparison(legacy, unit)
    for key, value in delta.items():
        if key != 'advance_to_training_pilot' and not math.isclose(
                report['comparison'][key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Saved comparison differs from the four F1 values: {key}')

    backup = path.with_suffix('.partial.json')
    report['comparison'] = delta
    report['recovered_from_partial'] = {'path': str(backup.resolve()), 'sha256': sha256_file(path)}
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError(f'Existing partial backup differs: {backup}')
    else:
        with backup.open('xb') as destination:
            destination.write(original)
    with TemporaryDirectory(dir=path.parent, prefix='.agvs-strength-recovery-') as directory:
        recovered = Path(directory) / path.name
        recovered.write_text(payload, encoding='utf-8')
        if path.read_bytes() != original:
            raise ValueError('The source report changed during recovery')
        recovered.replace(path)
    print_summary(report, path)


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing result: {OUTPUT}')
    runtime = recheck.baseline.check_baseline_runtime()
    import torch
    from scripts import evaluate_visual_ablation as visual

    prepared, full = train.load_prepared(train.split.DATA)
    rows, _, binding = train.training_plan(prepared, full)
    seal = recheck.baseline_seal(binding)
    if (seal['best_epoch'] != 11 or runtime != seal['runtime']
            or abs(seal['best_dev_f1'] - REFERENCE_F1['matched']) > REPRODUCTION_TOLERANCE):
        raise ValueError('Expected the frozen epoch-11 AGVS checkpoint and its original runtime')
    config = train.READ_JSON(seal['config'])
    reference = train.baseline_config(prepared)
    if config != {**reference, 'output_dir': str(train.ROOT / 'baseline')}:
        raise ValueError('Expected the original AGVS legacy configuration')
    args = SimpleNamespace(**config)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The visual-strength probe requires CUDA in Conda base')

    visual.set_seed(args.seed)
    dataset = visual.MultimodalNerDataset(args, 'MNER-MI_val.txt')
    if (len(dataset) != 1372 or dataset.examples['words'] != [row['tokens'] for row in rows['dev']]):
        raise ValueError('Expected the original internal selection dev rows')
    recheck.baseline.audit_images(dataset, prepared)
    print('Building the same seed-20261019 slot-matched image permutation...', flush=True)
    shuffled = visual.ShuffledImageDataset(dataset, seed=SHUFFLE_SEED)
    if shuffled.changed_samples != 1372:
        raise ValueError('Image donor reassignment count differs from the archived ablation')
    datasets = {'matched': dataset, 'shuffled': shuffled}

    labels = dataset.processor.get_label_mapping(include_subword_label=True)
    model = visual.AgvsMnerModel(list(labels), args)
    missing = visual.load_compatible_model_checkpoint(model, Path(seal['checkpoint']), map_location='cpu')
    model.to(args.device)
    model.requires_grad_(False)
    evaluator = visual._evaluator(model, args, labels)
    loader_options = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'shuffle': False}
    inputs = {
        **seal['input_sha256'],
        str(recheck.BASELINE_REPORT): sha256_file(recheck.BASELINE_REPORT),
        str(train.split.OUTPUT): train.SPLIT_SHA256,
        str(visual.__file__): sha256_file(visual.__file__),
        str(Path(__file__).resolve()): sha256_file(__file__),
    }

    legacy = evaluate_views(evaluator, datasets, 'legacy', loader_options)
    if any(abs(legacy[name]['f1'] - expected) > REPRODUCTION_TOLERANCE
           for name, expected in REFERENCE_F1.items()):
        raise ValueError('The two archived legacy image conditions failed reproduction')
    with model.register_forward_pre_hook(unit_strength, with_kwargs=True):
        unit = evaluate_views(evaluator, datasets, 'unit_if_image_present', loader_options)
    recheck.verify_files(inputs)

    delta = comparison(legacy, unit)
    report = {
        'schema_version': 'agvs-visual-strength-dev-probe-v1',
        'scope': 'original internal selection dev, native AGVS scoring',
        'samples': len(dataset), 'checkpoint': seal['checkpoint'], 'best_epoch': seal['best_epoch'],
        'shuffle_seed': SHUFFLE_SEED, 'reassigned_donors': shuffled.changed_samples,
        'legacy_reproduction': {'expected_f1': REFERENCE_F1, 'passed': True},
        'metrics': {'legacy': legacy, 'unit_if_image_present': unit},
        'comparison': delta, 'runtime': runtime, 'optional_missing_checkpoint_keys': missing,
        'input_sha256': inputs, 'optimizer_steps': 0, 'official_val_or_test_read': False,
        'grouped_holdout_scored': False,
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with OUTPUT.open('x', encoding='utf-8') as destination:
        destination.write(payload)
    print_summary(report, OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recover-partial', action='store_true')
    if parser.parse_args().recover_partial:
        recover_partial(OUTPUT)
    else:
        evaluate()


if __name__ == '__main__':
    main()
