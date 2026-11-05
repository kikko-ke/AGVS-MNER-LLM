"""Score the frozen BIO joint policy on the grouped recheck holdout."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from scripts import audit_bio_joint_v1 as policy
from scripts import evaluate_baseline_fullwords as baseline
from scripts import run_vlm_bio as bio
from scripts import train_bio_joint_recheck as train

BASELINE_REPORT = train.ROOT / 'baseline' / 'completed.json'
BIO_REPORT = train.ROOT / 'bio' / 'pilot_report.json'
BASELINE_OUTPUT = train.ROOT / 'baseline_holdout'
OUTPUT = train.ROOT / 'holdout_evaluation'
SCOPE = 'grouped recheck holdout from the previously used training corpus'
MIN_DELTA = 0.003
READ_JSON = bio.original.read_json
READ_RECORDS = bio.original.read_records
WRITE_JSON = bio.common.write_json


def source_ids(rows):
    return [row['source_id'] for row in rows]


def evaluation_plan(prepared, full):
    rows, _, binding = train.training_plan(prepared, full)
    by_id = {row['source_id']: row for row in full['train']}
    holdout = [by_id[sid] for sid in READ_JSON(train.split.OUTPUT)['source_ids']['recheck_holdout']]
    return rows, holdout, binding


def verify_files(files):
    for path, digest in files.items():
        if sha256_file(path) != digest:
            raise ValueError(f'Frozen evaluation input changed: {path}')


def baseline_seal(binding):
    report = READ_JSON(BASELINE_REPORT)
    if (report.get('schema_version') != 'bio-joint-recheck-baseline-v1'
            or report.get('official_val_or_test_read') is not False
            or report.get('recheck') != binding):
        raise ValueError('Expected the completed AGVS recheck on this exact split')
    files = report['input_sha256']
    for key in ('checkpoint', 'config'):
        if str(Path(report[key]).resolve()) not in files:
            raise ValueError(f'AGVS training report does not bind its {key}')
    verify_files(files)
    return report


def load_baseline_predictions(holdout, binding):
    report = READ_JSON(BASELINE_OUTPUT / 'prediction_report.json')
    predictions = BASELINE_OUTPUT / 'baseline_predictions.jsonl'
    if (report.get('schema_version') != 'bio-joint-recheck-agvs-predictions-v1'
            or report.get('recheck') != binding
            or report.get('source_ids') != source_ids(holdout)
            or report.get('training_report_sha256') != sha256_file(BASELINE_REPORT)
            or report.get('evaluation_script_sha256') != sha256_file(__file__)
            or report.get('predictions_sha256') != sha256_file(predictions)
            or report['native_selection_reproduction']['passed'] is not True):
        raise ValueError('Saved AGVS predictions do not bind the fixed recheck inputs')
    return report, bio.original.checked_records(holdout, READ_RECORDS(predictions))


def write_unlabeled_inputs(path, source, rows):
    lines = Path(source).read_text(encoding='utf-8').splitlines()
    with path.open('x', encoding='utf-8') as output:
        for row in rows:
            original = ast.literal_eval(lines[row['source_id']])
            output.write(repr({'text': original['text'], 'images': original['images']}) + '\n')


def predict_baseline():
    prepared, full = train.load_prepared(train.split.DATA)
    rows, holdout, binding = evaluation_plan(prepared, full)
    seal = baseline_seal(binding)
    bio_report = READ_JSON(BIO_REPORT)
    if bio_report['profile'].get('recheck') != binding:
        raise ValueError('Expected BIO training to finish on the same fixed split')
    bio.validate_candidate(SimpleNamespace(run=str(BIO_REPORT.parent)), bio_report['profile'], rows)
    if BASELINE_OUTPUT.exists():
        load_baseline_predictions(holdout, binding)
        print('Verified existing AGVS holdout predictions:', BASELINE_OUTPUT, flush=True)
        return

    runtime = baseline.check_baseline_runtime()
    if runtime != seal['runtime']:
        raise ValueError('Use the same Conda base runtime as AGVS training')
    config = READ_JSON(seal['config'])
    reference = train.baseline_config(prepared)
    if config != {**reference, 'output_dir': str(train.ROOT / 'baseline')}:
        raise ValueError('AGVS recheck configuration differs from the locked original configuration')
    seal_hash = sha256_file(BASELINE_REPORT)
    plan = {'checkpoint': Path(seal['checkpoint']), 'config': config, 'manifest': prepared,
            'groups': {'full_dev': rows['dev']}}
    _, native, _ = baseline.collect_predictions(plan, config['device'])
    if abs(native['f1'] - seal['best_dev_f1']) > 0.0005:
        raise ValueError('AGVS native selection score failed reproduction')
    print(f"AGVS selection reproduced: F1={native['f1']:.6f}", flush=True)

    original = verify_internal_manifest(prepared['internal_manifest'], 'MI')
    with TemporaryDirectory(dir=train.ROOT, prefix='.agvs-holdout-') as directory:
        scratch = Path(directory)
        unlabeled = scratch / 'MNER-MI_val.txt'
        write_unlabeled_inputs(unlabeled, original['source'], holdout)
        plan['config'] = {**config, 'data_path': directory}
        plan['groups']['full_dev'] = holdout
        records, _, execution = baseline.collect_predictions(plan, config['device'])
        bio.original.checked_records(holdout, records)
        verify_files({**seal['input_sha256'], str(BASELINE_REPORT): seal_hash})
        unlabeled.unlink()

        path = scratch / 'baseline_predictions.jsonl'
        bio.write_records(path, records)
        WRITE_JSON(scratch / 'prediction_report.json', {
            'schema_version': 'bio-joint-recheck-agvs-predictions-v1',
            'recheck': binding, 'source_ids': source_ids(holdout), 'training_report_sha256': seal_hash,
            'evaluation_script_sha256': sha256_file(__file__), 'predictions_sha256': sha256_file(path),
            'native_selection_reproduction': {'passed': True, 'f1': native['f1'],
                                              'expected_f1': seal['best_dev_f1']},
            'execution': execution, 'official_val_or_test_read': False})
        scratch.rename(BASELINE_OUTPUT)
    print('AGVS holdout predictions:', BASELINE_OUTPUT / 'prediction_report.json', flush=True)


def apply_joint(rows, original, baseline_records, lexicon):
    joint = {}
    for row in rows:
        sid = row['source_id']
        record = original[sid]
        if record['error'] is None:
            _, additions = policy.candidates_for(row['tokens'], record, baseline_records[sid], lexicon)
            spans = sorted([tuple(span) for span in record['predicted']] + additions)
            joint[sid] = {**record, 'predicted': [list(span) for span in spans]}
        else:
            joint[sid] = policy.repair_bio_length(len(row['tokens']), record)
    return joint


def inference_rows(rows):
    return [{key: row[key] for key in ('source_id', 'tokens', 'images')} for row in rows]


def predict_bio_and_score():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing evaluation: {OUTPUT}')
    args = SimpleNamespace(stage='evaluate', model=train.BIO_MODEL, data=str(train.split.DATA),
                           control_run=train.BIO_CONTROL, run=str(BIO_REPORT.parent))
    profile, prepared, full, _ = bio.load_profile(args)
    rows, holdout, binding = evaluation_plan(prepared, full)
    profile = {**profile, 'recheck': binding}
    seal = baseline_seal(binding)
    base_report, base_records = load_baseline_predictions(holdout, binding)
    epoch, adapter, adapter_hashes, reference, inputs = bio.validate_candidate(args, profile, rows)
    base_predictions = BASELINE_OUTPUT / 'baseline_predictions.jsonl'
    inputs.update({str(BASELINE_REPORT): base_report['training_report_sha256'],
                   str(BASELINE_OUTPUT / 'prediction_report.json'):
                       sha256_file(BASELINE_OUTPUT / 'prediction_report.json'),
                   str(base_predictions): base_report['predictions_sha256'],
                   str(train.split.OUTPUT): train.SPLIT_SHA256})
    if bio.backend.check_runtime() != profile['versions']:
        raise ValueError('Use the pinned original BIO runtime')
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    model, processor = bio.original.load_frozen_adapter(args.model, adapter, profile)
    with TemporaryDirectory(dir=train.ROOT, prefix='.bio-holdout-') as directory:
        scratch = Path(directory)
        panel_path = scratch / 'panel_predictions.jsonl'
        panel = bio.generate(model, processor, inference_rows(rows['dev_panel']), profile, panel_path)
        for sid, record in panel.items():
            if (record['predicted'] != reference[sid]['predicted']
                    or (record['error'] is None) != (reference[sid]['error'] is None)):
                raise ValueError(f'BIO selection panel failed reproduction at source_id={sid}')
        print('BIO selection panel reproduced. Generating the grouped holdout.', flush=True)
        panel_path.unlink()
        raw_path = scratch / 'bio_predictions.jsonl'
        original = bio.generate(model, processor, inference_rows(holdout), profile, raw_path)
        verify_files(inputs)
        if bio.backend.model_binding(adapter) != adapter_hashes:
            raise ValueError('BIO adapter changed during holdout evaluation')

        lexicon = policy.train_lexicon(rows['train'])
        joint = apply_joint(holdout, original, base_records, lexicon)
        scores = {name: bio.original.metrics_for(holdout, records, SCOPE) for name, records in
                  (('baseline', base_records), ('bio', original), ('bio_joint_v1', joint))}
        base_f1 = scores['baseline']['f1']
        joint_f1 = scores['bio_joint_v1']['f1']

        joint_path = scratch / 'bio_joint_predictions.jsonl'
        # Keep postprocessed spans separate from the model's raw response.
        bio.write_records(joint_path, [{key: record[key] for key in ('source_id', 'predicted', 'error')}
                                      for record in joint.values()])
        report = {'schema_version': 'bio-joint-recheck-holdout-v1', 'recheck': binding,
                  'evaluation_script_sha256': sha256_file(__file__),
                  'scope': SCOPE, 'source_ids': source_ids(holdout), 'samples': len(holdout),
                  'baseline_best_epoch': seal['best_epoch'], 'bio_best_epoch': epoch,
                  'baseline_native_selection_reproduction': base_report['native_selection_reproduction'],
                  'bio_selection_panel_reproduced': True, 'lexicon_size': len(lexicon),
                  'metrics': scores, 'joint_minus_baseline_f1': joint_f1 - base_f1,
                  'minimum_f1_delta': MIN_DELTA, 'passes_gate': joint_f1 >= base_f1 + MIN_DELTA,
                  'input_sha256': inputs, 'bio_adapter_sha256': adapter_hashes,
                  'output_predictions_sha256': {path.name: sha256_file(path) for path in (raw_path, joint_path)},
                  'official_val_or_test_read': False, 'optimizer_steps': 0, 'new_checkpoint_selection': False}
        WRITE_JSON(scratch / 'evaluation_report.json', report)
        scratch.rename(OUTPUT)
    path = OUTPUT / 'evaluation_report.json'
    print(json.dumps({
        'samples': len(holdout), 'lexicon_size': len(lexicon),
        'f1': {name: result['f1'] for name, result in scores.items()},
        'bio_malformed_responses': scores['bio']['malformed_responses'],
        'joint_malformed_responses': scores['bio_joint_v1']['malformed_responses'],
        'joint_minus_baseline_f1': report['joint_minus_baseline_f1'],
        'minimum_f1_delta': MIN_DELTA, 'passes_gate': report['passes_gate'],
        'report': str(path.resolve()),
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('baseline', 'bio-and-score'))
    if parser.parse_args().stage == 'baseline':
        predict_baseline()
    else:
        predict_bio_and_score()


if __name__ == '__main__':
    main()
