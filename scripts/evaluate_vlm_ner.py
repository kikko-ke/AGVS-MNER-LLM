"""Evaluate a frozen VLM pilot adapter on the full training-internal dev split.

Reproduce the checkpoint-selection panel before evaluating the remaining rows.
No optimizer, training, official validation/test access, or new model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from modules.vlm_ner import load_prepared, parse_entities, span_metrics, target
from modules.vlm_training import check_runtime, generate_evaluation, load_model, model_binding

BOUND_IMPLEMENTATION = (
    'modules/internal_split.py', 'modules/vlm_ner.py',
    'modules/vlm_training.py', 'scripts/run_vlm_ner.py',
)
SCOPES = {
    'full_dev': 'full training-internal dev; includes the checkpoint-selection panel',
    'dev_panel': 'fixed internal dev panel used for checkpoint selection',
    'dev_nonpanel': 'internal dev excluding the checkpoint-selection panel; not an independent final test',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def checked_records(rows, records):
    """Require exactly one prediction per source, with consistent strict parsing."""
    row_index = {row['source_id']: row for row in rows}
    indexed = {}
    for record in records:
        source_id = record['source_id']
        if type(source_id) is not int or source_id not in row_index or source_id in indexed:
            raise ValueError('Prediction source IDs are duplicate or outside the requested split')
        if not isinstance(record['response'], str):
            raise ValueError('Prediction response must be text')
        try:
            entities = parse_entities(record['response'], len(row_index[source_id]['tokens']))
            malformed = False
        except (ValueError, TypeError, KeyError):
            entities, malformed = [], True
        if (record['predicted'] != [list(entity) for entity in entities]
                or (record['error'] is not None) != malformed):
            raise ValueError('Saved predictions/error flags disagree with strict response parsing')
        indexed[source_id] = record
    if len(row_index) != len(rows) or set(indexed) != set(row_index):
        raise ValueError('Every requested source must have exactly one prediction')
    return indexed


def metrics_for(rows, indexed, scope):
    gold = [parse_entities(target(row), len(row['tokens'])) for row in rows]
    predicted = [[tuple(entity) for entity in indexed[row['source_id']]['predicted']] for row in rows]
    malformed = sum(indexed[row['source_id']]['error'] is not None for row in rows)
    return {
        **span_metrics(gold, predicted), 'malformed_responses': malformed,
        'malformed_rate': malformed / len(rows) if rows else 0.,
        'scope': scope + '; full original word spans',
    }


def validate_run(run, model_directory, data):
    report_path, config_path = run / 'pilot_report.json', run / 'config.json'
    report, config = read_json(report_path), read_json(config_path)
    if (report.get('schema_version') != 'vlm-ner-pilot-v1'
            or report.get('official_val_or_test_read') is not False
            or config.get('stage') != 'train' or config.get('official_val_or_test_read') is not False):
        raise ValueError('Expected a completed training-internal VLM pilot run')
    profile = report['profile']
    if config['profile'] != profile or profile.get('schema_version') != 'vlm-qlora-profile-v1':
        raise ValueError('Training configuration and pilot report disagree')
    if profile.get('input_mode') not in {'text', 'multi_image'}:
        raise ValueError('Unsupported pilot input mode')
    history = report['history']
    if [item['epoch'] for item in history] != list(range(profile['epochs'] + 1)):
        raise ValueError('Pilot history is incomplete')
    best = max(history, key=lambda item: item['metrics']['f1'])
    if (best['epoch'] != report['best_epoch']
            or best['metrics']['f1'] != report['best_dev_panel_f1']):
        raise ValueError('Best checkpoint must follow the original panel selection')
    if sha256_file(data) != profile['data_sha256']:
        raise ValueError('Prepared manifest differs from the training run')
    implementation = {name: sha256_file(PROJECT / name) for name in BOUND_IMPLEMENTATION}
    if implementation != profile['implementation_sha256']:
        raise ValueError('Training/generation implementation changed; preserve the original bound files')
    print('Verifying base model hashes...', flush=True)
    if model_binding(model_directory) != profile['model_files_sha256']:
        raise ValueError('Base model files differ from the training run')
    manifest, rows = load_prepared(data)
    dev_ids = [row['source_id'] for row in rows['dev']]
    panel_ids = [row['source_id'] for row in rows['dev_panel']]
    if (len(set(dev_ids)) != len(dev_ids) or len(set(panel_ids)) != len(panel_ids)
            or panel_ids != manifest['panel_source_ids'] or not set(panel_ids) <= set(dev_ids)):
        raise ValueError('Invalid dev/panel partition')
    dev_index = {row['source_id']: row for row in rows['dev']}
    if any(row != dev_index[row['source_id']] for row in rows['dev_panel']):
        raise ValueError('Panel rows must be identical to their full-dev rows')
    nonpanel = [row for row in rows['dev'] if row['source_id'] not in set(panel_ids)]
    if not panel_ids or not nonpanel:
        raise ValueError('Both panel and non-panel dev rows are required')
    reference_path = run / f"dev_epoch{report['best_epoch']}.jsonl"
    reference = checked_records(rows['dev_panel'], read_records(reference_path))
    reference_metrics = metrics_for(rows['dev_panel'], reference, SCOPES['dev_panel'])
    for key in ('tp', 'fp', 'fn', 'by_type', 'samples', 'malformed_responses', 'f1'):
        if reference_metrics[key] != best['metrics'][key]:
            raise ValueError('Saved best-panel predictions disagree with the pilot report')
    adapter = run / 'best_adapter'
    if not (adapter / 'adapter_config.json').is_file() or not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('Saved best_adapter configuration/safetensors are missing')
    bindings = {
        'pilot_report_sha256': sha256_file(report_path),
        'training_config_sha256': sha256_file(config_path),
        'reference_panel_predictions_sha256': sha256_file(reference_path),
        'adapter_files_sha256': model_binding(adapter),
        'evaluation_script_sha256': sha256_file(__file__),
    }
    return report, manifest, rows, nonpanel, reference, bindings


def load_frozen_adapter(model_directory, adapter, profile):
    # Reuse the exact quantization and processor path bound to the training run.
    model, processor = load_model(model_directory, profile['seed'], profile['min_pixels'], profile['max_pixels'])
    result = model.load_adapter(str(adapter), adapter_name='evaluation', is_trainable=False,
                                local_files_only=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f'Adapter weights did not load exactly: {result}')
    original, restored = model.peft_config['default'], model.peft_config['evaluation']
    for name in ('r', 'lora_alpha', 'lora_dropout', 'target_modules', 'bias', 'task_type', 'use_dora'):
        if getattr(original, name) != getattr(restored, name):
            raise ValueError(f'Saved adapter differs from the original LoRA configuration: {name}')
    model.set_adapter('evaluation')
    # PEFT set_adapter enables gradients again, so freeze after activation.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, processor


def evaluate(args):
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError('Evaluation output must be separate from the training run')
    if output.exists():
        raise FileExistsError('Evaluation output already exists; preserve the previous result')
    report, manifest, rows, nonpanel, reference, bindings = validate_run(run, args.model, args.data)
    profile, adapter = report['profile'], run / 'best_adapter'
    if check_runtime() != profile['versions']:
        raise ValueError('Use the exact runtime versions recorded by the training run')
    output.mkdir(parents=True, exist_ok=False)
    config = {
        'schema_version': 'vlm-ner-dev-evaluation-v1', 'run': str(run), 'profile': profile,
        'best_epoch': report['best_epoch'], 'bindings': bindings,
        'official_val_or_test_read': False, 'optimizer_steps': 0, 'new_checkpoint_selection': False,
        'source_ids': {'full_dev': [r['source_id'] for r in rows['dev']],
                       'dev_panel': manifest['panel_source_ids'],
                       'dev_nonpanel': [r['source_id'] for r in nonpanel]},
    }
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        model, processor = load_frozen_adapter(args.model, adapter, profile)
        common = (model, processor)
        settings = (profile['input_mode'], profile['max_length'], profile['max_new_tokens'])
        print('Reproducing the saved best-checkpoint panel before full-dev evaluation...', flush=True)
        generate_evaluation(*common, rows['dev_panel'], *settings, output / 'panel_predictions.jsonl')
        panel = checked_records(rows['dev_panel'], read_records(output / 'panel_predictions.jsonl'))
        mismatches = [source_id for source_id, record in panel.items()
                      if record['predicted'] != reference[source_id]['predicted']
                      or (record['error'] is None) != (reference[source_id]['error'] is None)]
        reproduction = {'passed': not mismatches, 'mismatched_source_ids': mismatches,
                        'metrics': metrics_for(rows['dev_panel'], panel, SCOPES['dev_panel'])}
        (output / 'panel_reproduction.json').write_text(json.dumps(reproduction, indent=2) + '\n', encoding='utf-8')
        if mismatches:
            raise ValueError('Reloaded adapter does not reproduce saved panel predictions; inspect panel_reproduction.json')
        print(f'Panel reproduced. Evaluating the remaining {len(nonpanel)} internal dev samples...', flush=True)
        generate_evaluation(*common, nonpanel, *settings, output / 'nonpanel_predictions.jsonl')
        remaining = checked_records(nonpanel, read_records(output / 'nonpanel_predictions.jsonl'))
        full = {**panel, **remaining}
        checked_records(rows['dev'], list(full.values()))
        ordered = [full[row['source_id']] for row in rows['dev']]
        (output / 'dev_predictions.jsonl').write_text(
            ''.join(json.dumps(record, ensure_ascii=True) + '\n' for record in ordered), encoding='utf-8')
        if model_binding(adapter) != bindings['adapter_files_sha256']:
            raise ValueError('Saved adapter files changed during evaluation')
        metrics = {role: metrics_for(values, full, SCOPES[role]) for role, values in (
            ('full_dev', rows['dev']), ('dev_panel', rows['dev_panel']), ('dev_nonpanel', nonpanel))}
        result = {
            **config, 'metrics': metrics, 'panel_reproduction': reproduction,
            'elapsed_seconds': time.monotonic() - started,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 1024**3,
            'note': 'Frozen best adapter; internal development evaluation only. Non-panel rows are not a pristine final test.',
        }
        (output / 'evaluation_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        for role, value in metrics.items():
            print(f'{role}: ' + json.dumps(value), flush=True)
        print('Report:', output / 'evaluation_report.json', flush=True)
        return result
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'error': str(error)}, indent=2) + '\n', encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs_vlm_gold/mi_multi_image_seed1234')
    parser.add_argument('--model', default='/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--output', default='outputs_vlm_gold/mi_multi_image_seed1234_dev_evaluation')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
