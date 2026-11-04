"""Remove images at inference from the frozen multi-image adapter.

This is an input ablation, separate from the independently trained text adapter.
The original matched-image panel must reproduce before any ablated generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from modules.vlm_ner import LABELS
from scripts import evaluate_vlm_ner as original


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def changed_predictions(first, second):
    return [source_id for source_id, record in first.items()
            if record['predicted'] != second[source_id]['predicted']
            or (record['error'] is None) != (second[source_id]['error'] is None)]


def load_reference(args):
    run, reference = Path(args.run).resolve(), Path(args.reference_evaluation).resolve()
    pilot, manifest, rows, nonpanel, panel, run_bindings = original.validate_run(run, args.model, args.data)
    if pilot['profile']['input_mode'] != 'multi_image':
        raise ValueError('This ablation requires the original multi-image-trained adapter')
    groups = {'full_dev': rows['dev'], 'dev_panel': rows['dev_panel'], 'dev_nonpanel': nonpanel}
    report_path, predictions_path = reference / 'evaluation_report.json', reference / 'dev_predictions.jsonl'
    report = original.read_json(report_path)
    ids = {role: [row['source_id'] for row in values] for role, values in groups.items()}
    if (report.get('schema_version') != 'vlm-ner-dev-evaluation-v1'
            or report.get('panel_reproduction', {}).get('passed') is not True
            or report.get('official_val_or_test_read') is not False
            or report.get('optimizer_steps') != 0 or report.get('new_checkpoint_selection') is not False
            or report.get('profile') != pilot['profile'] or report.get('best_epoch') != pilot['best_epoch']
            or report.get('bindings') != run_bindings or report.get('source_ids') != ids):
        raise ValueError('Reference must be the completed evaluation of the identical frozen multi-image run')
    matched = original.checked_records(rows['dev'], original.read_records(predictions_path))
    if changed_predictions(panel, matched):
        raise ValueError('Reference full-dev predictions disagree with the saved best panel')
    for role, values in groups.items():
        if original.metrics_for(values, matched, original.SCOPES[role]) != report['metrics'][role]:
            raise ValueError('Reference predictions no longer reproduce the saved metrics')
    # The old evaluation did not hash its own final predictions. Use their
    # historical binding from the completed offline audit, rather than treating
    # metric agreement alone as proof that these are the same frozen records.
    audit_path = Path(args.reference_audit) / 'error_audit_report.json'
    audit = original.read_json(audit_path)
    if (audit.get('schema_version') != 'baseline-qlora-error-audit-v1'
            or audit.get('diagnostic_only') is not True or audit.get('source_ids') != ids
            or audit.get('official_val_or_test_read') is not False
            or audit.get('model_inference_runs') != 0 or audit.get('training_runs') != 0):
        raise ValueError('Expected the completed offline audit of the frozen reference predictions')
    for path in (report_path, predictions_path, Path(args.data)):
        if audit.get('input_sha256', {}).get(str(path.resolve())) != sha256_file(path):
            raise ValueError('Reference inputs differ from their historical offline audit binding')
    data = Path(args.data)
    paths = [data, Path(manifest['internal_manifest']), report_path, predictions_path, audit_path,
             run / 'pilot_report.json', run / 'config.json', run / f"dev_epoch{pilot['best_epoch']}.jsonl",
             PROJECT / 'scripts/evaluate_vlm_ner.py', Path(__file__),
             *(PROJECT / name for name in original.BOUND_IMPLEMENTATION),
             *(Path(info['path']) for info in manifest['files'].values())]
    paths.append(Path(original.read_json(manifest['internal_manifest'])['source']))
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    return pilot, groups, panel, matched, report, run_bindings, bindings


def evaluate(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Ablation output already exists; preserve the previous result')
    for source in (args.run, args.reference_evaluation, args.reference_audit, args.model, Path(args.data).parent):
        protected = Path(source).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Ablation output must be separate from the frozen inputs')
    pilot, groups, panel, matched, reference, run_bindings, bindings = load_reference(args)
    profile = pilot['profile']
    image_counts = dict(sorted(Counter(len(row['images']) for row in groups['full_dev']).items()))
    if args.check_only:
        print('Frozen multi-image run and reference predictions verified. No model loaded.', flush=True)
        print('Original image counts:', json.dumps(image_counts), flush=True)
        return {'check_only': True, 'source_ids': reference['source_ids'], 'input_sha256': bindings}
    if original.check_runtime() != profile['versions']:
        raise ValueError('Use the exact runtime versions recorded by the original multi-image run')
    output.mkdir(parents=True, exist_ok=False)
    config = {
        'schema_version': 'vlm-no-image-ablation-v1', 'run': str(Path(args.run).resolve()),
        'reference_evaluation': str(Path(args.reference_evaluation).resolve()),
        'reference_audit': str(Path(args.reference_audit).resolve()),
        'training_profile': profile, 'best_epoch': pilot['best_epoch'],
        'intervention': {'inference_input_mode': 'text', 'all_images_removed': True,
                         'system_prompt_changed': False, 'word_content_changed': False, 'weights_changed': False},
        'source_ids': reference['source_ids'], 'original_image_counts': image_counts,
        'input_sha256': bindings, 'adapter_files_sha256': run_bindings['adapter_files_sha256'],
        'official_val_or_test_read': False, 'optimizer_steps': 0,
        'new_checkpoint_selection': False, 'llm_api_requests': 0,
    }
    write_json(output / 'config.json', config)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        model, processor = original.load_frozen_adapter(args.model, Path(args.run) / 'best_adapter', profile)
        settings = (profile['max_length'], profile['max_new_tokens'])
        print('Reproducing the original matched-image panel...', flush=True)
        original.generate_evaluation(model, processor, groups['dev_panel'], 'multi_image',
                                     *settings, output / 'matched_panel_predictions.jsonl')
        restored = original.checked_records(groups['dev_panel'], original.read_records(
            output / 'matched_panel_predictions.jsonl'))
        mismatches = changed_predictions(panel, restored)
        reproduction = {'passed': not mismatches, 'mismatched_source_ids': mismatches,
                        'metrics': original.metrics_for(groups['dev_panel'], restored, original.SCOPES['dev_panel'])}
        write_json(output / 'panel_reproduction.json', reproduction)
        if mismatches:
            raise ValueError('Original matched-image panel does not reproduce; no ablated generation performed')
        print('Panel reproduced. Removing images from all internal dev inputs...', flush=True)
        # Keep original words/gold/order. Text mode changes only image content;
        # clear the image list as well, so no ablated row can read an image file.
        ablated_rows = [{**row, 'images': []} for row in groups['full_dev']]
        original.generate_evaluation(model, processor, ablated_rows, 'text', *settings,
                                     output / 'no_image_predictions.jsonl')
        ablated = original.checked_records(ablated_rows, original.read_records(output / 'no_image_predictions.jsonl'))
        empty = {row['source_id']: matched[row['source_id']] for row in groups['full_dev'] if not row['images']}
        mismatches = changed_predictions(empty, ablated)
        zero_image_check = {'passed': not mismatches, 'samples': len(empty), 'mismatched_source_ids': mismatches}
        write_json(output / 'zero_image_reproduction.json', zero_image_check)
        if mismatches:
            raise ValueError('Originally image-free inputs changed predictions despite identical inputs')
        comparisons = {}
        for role, values in groups.items():
            left = reference['metrics'][role]
            right = original.metrics_for(values, ablated, original.SCOPES[role])
            comparisons[role] = {
                'matched_images': left, 'no_images': right,
                'matched_f1_minus_no_images': left['f1'] - right['f1'],
                'by_type_f1_delta': {label: left['by_type'][label]['f1'] - right['by_type'][label]['f1']
                                     for label in LABELS},
            }
        for path, expected in bindings.items():
            if sha256_file(path) != expected:
                raise ValueError(f'Frozen input changed during evaluation: {path}')
        if original.model_binding(Path(args.run) / 'best_adapter') != run_bindings['adapter_files_sha256']:
            raise ValueError('Saved adapter changed during evaluation')
        result = {
            **config, 'comparisons': comparisons, 'panel_reproduction': reproduction,
            'zero_image_reproduction': zero_image_check,
            'elapsed_seconds': time.monotonic() - started,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
            'note': 'Frozen multi-image adapter with images removed only at inference. '
                    'This is not an independently trained text control. Matched full-dev predictions '
                    'come from the verified existing report; only its panel is regenerated here. '
                    'Removal tests image-presence sensitivity, not whether the correct image semantics '
                    'are responsible. Internal dev has development history; no new model selection.',
        }
        write_json(output / 'ablation_report.json', result)
        for role, values in comparisons.items():
            print(f"{role}: matched_images={values['matched_images']['f1']:.6f} "
                  f"no_images={values['no_images']['f1']:.6f} "
                  f"matched-minus-no_images={values['matched_f1_minus_no_images']:+.6f}", flush=True)
            if role == 'full_dev':
                for label, delta in values['by_type_f1_delta'].items():
                    print(f'{role}/{label}: matched-minus-no_images={delta:+.6f}', flush=True)
        print('Report:', output / 'ablation_report.json', flush=True)
        return result
    except Exception as error:
        write_json(output / 'failure.json', {'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs_vlm_gold/mi_multi_image_seed1234')
    parser.add_argument('--reference-evaluation', default='outputs_vlm_gold/mi_multi_image_seed1234_dev_evaluation')
    parser.add_argument('--reference-audit', default='outputs_vlm_gold/baseline_qlora_error_audit')
    parser.add_argument('--model', default='/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--output', default='outputs_vlm_gold/mi_multi_image_seed1234_no_images')
    parser.add_argument('--check-only', action='store_true', help='Verify frozen files without GPU/model imports or output')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
