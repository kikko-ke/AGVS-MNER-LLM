"""Compare the frozen AGVS text and multimodal controls on the same grouped holdout."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import train_bio_joint_recheck as train

TEXT_REPORT = train.ROOT / 'agvs_text_control' / 'completed.json'
OUTPUT = train.ROOT / 'agvs_text_control_holdout'


def write_text_inputs(path, source_lines, rows):
    with path.open('x', encoding='utf-8') as destination:
        for row in rows:
            raw = ast.literal_eval(source_lines[row['source_id']])
            destination.write(repr({'text': raw['text'], 'images': []}) + '\n')


def predict_text(directory, source_lines, rows, checkpoint, config, prepared):
    unlabeled = directory / 'MNER-MI_val.txt'
    write_text_inputs(unlabeled, source_lines, rows)
    plan = {'checkpoint': checkpoint, 'config': {**config, 'data_path': str(directory)},
            'manifest': prepared, 'groups': {'full_dev': rows}}
    records, native, execution = recheck.baseline.collect_predictions(plan, config['device'])
    unlabeled.unlink()
    if execution['source_images_verified'] != 0:
        raise ValueError('Text-only inference must have no image inputs')
    return records, native, execution


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing supplemental evaluation: {OUTPUT}')
    runtime = recheck.baseline.check_baseline_runtime()
    prepared, full = train.load_prepared(train.split.DATA)
    rows, holdout, binding = recheck.evaluation_plan(prepared, full)
    multimodal_seal = recheck.baseline_seal(binding)
    text_seal = train.READ_JSON(TEXT_REPORT)
    if (text_seal.get('schema_version') != 'agvs-text-control-dev-v1'
            or text_seal.get('recheck') != binding
            or text_seal.get('image_policy') != 'zero pixels and masks in training and selection'
            or text_seal.get('text_best_epoch') != 4 or multimodal_seal.get('best_epoch') != 11
            or text_seal.get('official_val_or_test_read') is not False
            or text_seal.get('grouped_holdout_scored') is not False
            or text_seal.get('runtime') != runtime or multimodal_seal.get('runtime') != runtime):
        raise ValueError('Expected both frozen AGVS controls from the same split and runtime')
    checkpoint = Path(text_seal['checkpoint']).resolve()
    config_path = checkpoint.parent / 'config.json'
    for artifact in (checkpoint, config_path):
        if str(artifact) not in text_seal['output_sha256']:
            raise ValueError(f'Text training report does not bind {artifact}')
    recheck.verify_files({**text_seal['input_sha256'], **text_seal['output_sha256']})
    config = train.READ_JSON(config_path)
    multimodal_config = train.READ_JSON(multimodal_seal['config'])
    if config != {**multimodal_config, 'output_dir': str(TEXT_REPORT.parent)}:
        raise ValueError('Text and multimodal control training settings differ')

    prediction_report, multimodal_records = recheck.load_baseline_predictions(holdout, binding)
    score = recheck.bio.original.metrics_for
    multimodal = score(holdout, multimodal_records, recheck.SCOPE)
    prior_report = recheck.OUTPUT / 'evaluation_report.json'
    prior = train.READ_JSON(prior_report)
    if (prior.get('recheck') != binding or prior.get('source_ids') != recheck.source_ids(holdout)
            or prior['metrics']['baseline'] != multimodal):
        raise ValueError('The saved multimodal predictions do not reproduce the archived holdout score')
    inputs = {
        **text_seal['input_sha256'], **text_seal['output_sha256'],
        str(TEXT_REPORT): sha256_file(TEXT_REPORT),
        str(recheck.BASELINE_OUTPUT / 'prediction_report.json'):
            sha256_file(recheck.BASELINE_OUTPUT / 'prediction_report.json'),
        str(recheck.BASELINE_OUTPUT / 'baseline_predictions.jsonl'): prediction_report['predictions_sha256'],
        str(prior_report): sha256_file(prior_report),
        str(Path(__file__).resolve()): sha256_file(__file__),
    }
    internal = verify_internal_manifest(prepared['internal_manifest'], 'MI')
    source_lines = Path(internal['source']).read_text(encoding='utf-8').splitlines()
    with TemporaryDirectory(dir=train.ROOT, prefix='.agvs-text-holdout-') as directory:
        scratch = Path(directory)
        print('Reproducing the text control selection dev...', flush=True)
        _, native, _ = predict_text(scratch, source_lines, rows['dev'], checkpoint, config, prepared)
        expected = text_seal['text_trained_dev_f1']
        if abs(native['f1'] - expected) > 0.0005:
            raise ValueError(f'Text control selection score failed reproduction: {native["f1"]} vs {expected}')
        print(f'Text selection reproduced: F1={native["f1"]:.6f}. Scoring {len(holdout)} holdout rows.',
              flush=True)
        records, _, execution = predict_text(scratch, source_lines, holdout, checkpoint, config, prepared)
        indexed = recheck.bio.original.checked_records(holdout, records)
        text = score(holdout, indexed, recheck.SCOPE)
        recheck.verify_files(inputs)

        path = scratch / 'text_predictions.jsonl'
        recheck.bio.write_records(path, records)
        scores = {'multimodal': multimodal, 'text': text}
        delta = multimodal['f1'] - text['f1']
        by_type = {label: multimodal['by_type'][label]['f1'] - value['f1']
                   for label, value in text['by_type'].items()}
        report = {
            'schema_version': 'agvs-text-control-holdout-v1',
            'scope': 'supplemental comparison on the previously scored grouped holdout',
            'samples': len(holdout), 'source_ids': recheck.source_ids(holdout), 'recheck': binding,
            'text_best_epoch': text_seal['text_best_epoch'], 'multimodal_best_epoch': multimodal_seal['best_epoch'],
            'text_selection_reproduction': {'passed': True, 'f1': native['f1'], 'expected_f1': expected},
            'metrics': scores, 'multimodal_minus_text_f1': delta, 'by_type_multimodal_minus_text_f1': by_type,
            'text_predictions_sha256': sha256_file(path), 'input_sha256': inputs,
            'runtime': runtime, 'text_execution': execution,
            'new_checkpoint_selection': False, 'optimizer_steps': 0, 'official_val_or_test_read': False,
        }
        train.WRITE_JSON(scratch / 'evaluation_report.json', report)
        scratch.rename(OUTPUT)

    print(json.dumps({
        'samples': len(holdout),
        'metrics': {name: {key: values[key] for key in ('precision', 'recall', 'f1')}
                    for name, values in scores.items()},
        'multimodal_minus_text_f1': delta, 'by_type_multimodal_minus_text_f1': by_type,
        'report': str((OUTPUT / 'evaluation_report.json').resolve()),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
