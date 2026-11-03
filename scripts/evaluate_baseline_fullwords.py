"""Compare the original internal-split AGVS baseline with frozen QLoRA predictions.

Keep the baseline's original WordPiece window and native CRF paths. Project
first-piece predictions onto all original words, filling unseen words with O.
Verify its historical truncated-dev F1 before reporting full-word comparisons.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from modules.vlm_ner import load_prepared, span_metrics
from scripts.evaluate_vlm_ner import SCOPES, checked_records, metrics_for, read_json, read_records


def locate_checkpoint(root, explicit=None):
    paths = [Path(explicit)] if explicit else sorted((root / 'none_seed1234').glob('*/best_model.pt'))
    if len(paths) != 1 or not paths[0].is_file():
        raise ValueError('Expected exactly one outputs_context_internal/none_seed1234/*/best_model.pt; '
                         'use --checkpoint to identify the original completed run')
    return paths[0]


def validate_baseline_config(config, internal_path, split):
    expected = {'dataset': 'MI', 'seed': 1234, 'consistency_mode': 'none', 'skip_test': True}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Use the original MI/seed1234/none internal-training baseline, not a full-train or auxiliary arm')
    if (config.get('visual_fusion', 'legacy') != 'legacy'
            or config.get('dropout_selection_manifest') or config.get('dropout_selection_arm')):
        raise ValueError('Expected the original legacy baseline without resampling')
    if (Path(config['internal_manifest']).resolve() != internal_path.resolve()
            or Path(config['data_path']).resolve() != Path(split['partitions']['train']['path']).parent.resolve()):
        raise ValueError('Baseline must have been trained on the same fixed internal split as QLoRA')
    for key in ('llm_token_loss_weight', 'llm_image_loss_weight', 'llm_grounding_loss_weight',
                'llm_evidence_loss_weight', 'llm_utility_gate', 'llm_evidence_protected_adapter',
                'llm_init_checkpoint', 'init_checkpoint'):
        if config.get(key):
            raise ValueError(f'Unexpected auxiliary training or initialization: {key}')
    if type(config.get('max_seq_length')) is not int or config['max_seq_length'] < 3:
        raise ValueError('Invalid original WordPiece window')


def reference_report(root, explicit, checkpoint, config):
    matches = []
    for path in ([Path(explicit)] if explicit else sorted(root.glob('pilot_*.json'))):
        value = read_json(path)
        artifact = value.get('artifacts', {}).get('none', {})
        if (value.get('schema_version') == 'context-internal-pilot-v1'
                and value.get('test_read') is False and value.get('dataset') == 'MI'
                and value.get('seed') == 1234 and artifact.get('checkpoint')
                and Path(artifact['checkpoint']).resolve() == checkpoint.resolve()
                and artifact.get('sha256') == sha256_file(checkpoint) and artifact.get('config') == config):
            matches.append((path, value))
    if len(matches) != 1:
        raise ValueError('Expected one original context-internal pilot report binding this baseline/config; '
                         'use --reference-report if multiple reports exist')
    path, value = matches[0]
    expected = value['native_dev_f1']['none']
    with (checkpoint.parent / 'metrics.csv').open(encoding='utf-8', newline='') as source:
        recorded = [float(row['dev_f1']) for row in csv.DictReader(source) if row.get('dev_f1')]
    if (not recorded or any(not math.isfinite(item) for item in recorded)
            or not math.isfinite(expected) or abs(expected - max(recorded)) > 0.0005):
        raise ValueError('Original reference report and recorded best dev F1 disagree')
    return path, expected, max(recorded)


def prepare(cli):
    root, data = Path(cli.runs_root), Path(cli.data)
    manifest, rows = load_prepared(data)
    internal = Path(manifest['internal_manifest'])
    split = verify_internal_manifest(internal, 'MI')
    checkpoint = locate_checkpoint(root, cli.checkpoint)
    config = read_json(checkpoint.parent / 'config.json')
    validate_baseline_config(config, internal, split)
    reference, expected, recorded = reference_report(root, cli.reference_report, checkpoint, config)
    panel_ids = set(manifest['panel_source_ids'])
    dev_ids = [row['source_id'] for row in rows['dev']]
    if (len(panel_ids) != len(manifest['panel_source_ids'])
            or len(set(dev_ids)) != len(dev_ids)
            or not panel_ids < set(dev_ids)
            or [row['source_id'] for row in rows['dev_panel']] != manifest['panel_source_ids']):
        raise ValueError('Invalid full-dev/panel/non-panel source IDs')
    dev_index = {row['source_id']: row for row in rows['dev']}
    if any(row != dev_index[row['source_id']] for row in rows['dev_panel']):
        raise ValueError('Panel rows differ from their full-dev counterparts')
    groups = {'full_dev': rows['dev'], 'dev_panel': rows['dev_panel'],
              'dev_nonpanel': [row for row in rows['dev'] if row['source_id'] not in panel_ids]}
    vlm_root = Path(cli.vlm_evaluation)
    vlm_report_path = vlm_root / 'evaluation_report.json'
    vlm_report = read_json(vlm_report_path)
    if (vlm_report.get('schema_version') != 'vlm-ner-dev-evaluation-v1'
            or vlm_report.get('official_val_or_test_read') is not False
            or vlm_report.get('new_checkpoint_selection') is not False
            or vlm_report.get('optimizer_steps') != 0
            or vlm_report['profile']['data_sha256'] != sha256_file(data)
            or vlm_report['profile']['input_mode'] != 'multi_image'
            or vlm_report['panel_reproduction']['passed'] is not True):
        raise ValueError('Expected the completed frozen multi-image QLoRA evaluation on identical data')
    vlm_predictions_path = vlm_root / 'dev_predictions.jsonl'
    vlm_predictions = checked_records(rows['dev'], read_records(vlm_predictions_path))
    for role, values in groups.items():
        if vlm_report['source_ids'][role] != [row['source_id'] for row in values]:
            raise ValueError('QLoRA evaluation source IDs/order differ from the common gold data')
        score = metrics_for(values, vlm_predictions, SCOPES[role])
        for key in ('tp', 'fp', 'fn', 'by_type', 'samples', 'f1', 'malformed_responses'):
            if score[key] != vlm_report['metrics'][role][key]:
                raise ValueError('QLoRA predictions do not reproduce the saved full-word report')
    paths = [data, internal, checkpoint, checkpoint.parent / 'config.json',
             checkpoint.parent / 'metrics.csv', reference, vlm_report_path, vlm_predictions_path]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    return {'checkpoint': checkpoint, 'config': config, 'manifest': manifest, 'groups': groups,
            'reference_f1': expected, 'recorded_f1': recorded, 'vlm_report': vlm_report,
            'bindings': bindings}


def word_bio(row):
    labels = ['O'] * len(row['tokens'])
    for entity in row['entities']:
        labels[entity['start']] = 'B-' + entity['label']
        for index in range(entity['start'] + 1, entity['end']):
            labels[index] = 'I-' + entity['label']
    return labels


def bio_entities(labels):
    # This repository's chunk extractor shares the native evaluator's BIO
    # semantics: an isolated I starts an entity, and a type change closes one.
    from ner_evaluate import get_chunks
    mapping = {label: index for index, label in enumerate(dict.fromkeys(['O', *labels]))}
    return sorted((start, end, label) for label, start, end in
                  get_chunks([mapping[label] for label in labels], mapping))


def project_path(row, path, structure, retained_indices, label_mapping, active_length):
    if len(path) != active_length or len(structure) < active_length:
        raise ValueError('Native CRF path must cover every active WordPiece position')
    positions = [index for index in range(active_length) if not structure[index]]
    if (len(positions) != len(retained_indices) or len(set(retained_indices)) != len(retained_indices)
            or list(retained_indices) != sorted(retained_indices)
            or any(type(index) is not int or not 0 <= index < len(row['tokens']) for index in retained_indices)):
        raise ValueError('Invalid tokenizer-derived original-word alignment')
    inverse = {value: key for key, value in label_mapping.items()}
    observed = []
    full = ['O'] * len(row['tokens'])
    for position, original_index in zip(positions, retained_indices):
        label = inverse[path[position]]
        label = 'O' if label == 'X' else label
        if label != 'O' and (label[:2] not in {'B-', 'I-'} or label[2:] not in {'LOC', 'MIS', 'ORG', 'PER'}):
            raise ValueError('Unsupported baseline prediction label')
        observed.append(label)
        full[original_index] = label
    predicted = [list(entity) for entity in bio_entities(full)]
    record = {'source_id': row['source_id'], 'predicted': predicted, 'error': None,
              'response': json.dumps({'entities': [dict(start=a, end=b, label=label)
                                                 for a, b, label in predicted]}),
              'retained_word_indices': list(retained_indices), 'original_words': len(full)}
    gold = word_bio(row)
    return record, bio_entities([gold[index] for index in retained_indices]), bio_entities(observed)


def audit_images(dataset, manifest):
    audit = {Path(path).resolve(): value for path, value in manifest['image_audit'].items()}
    checked = set()
    for names in dataset.examples['images']:
        for name in names[:4]:
            path = dataset._get_image_path(name).resolve()
            if path in checked:
                continue
            expected = audit.get(path)
            if expected is None:
                raise ValueError(f'Baseline image is not in the prepared source audit: {path}')
            actual = sha256_file(path) if path.is_file() else None
            if actual != expected['sha256']:
                raise ValueError(f'Source image changed since QLoRA preparation: {path}')
            checked.add(path)
    return len(checked)


def check_baseline_runtime():
    # Use the original baseline environment; never install into mner-vlm here.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    import torch
    import transformers
    if torch.__version__.split('+')[0] != '2.3.0' or transformers.__version__ != '4.44.2':
        raise RuntimeError(f'Run in the original BERT/ViT environment (torch2.3.0 / transformers4.44.2), '
                           f'not mner-vlm; found torch={torch.__version__}, transformers={transformers.__version__}')
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'pytorch-crf', 'seqeval', 'Pillow')}


def collect_predictions(plan, device):
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from models.agvs_mner import AgvsMnerModel
    from modules.dataset import MultimodalInferenceDataset
    from modules.trainer import load_compatible_model_checkpoint
    from run import set_seed
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in the original baseline environment')
    args = SimpleNamespace(**{**plan['config'], 'device': device})
    for name in ('bert_model', 'vit_model'):
        if not Path(getattr(args, name)).is_dir():
            raise FileNotFoundError(f'Original local {name} directory is missing: {getattr(args, name)}')
    set_seed(args.seed)
    dataset = MultimodalInferenceDataset(args, 'MNER-MI_val.txt')
    rows = plan['groups']['full_dev']
    if dataset.examples['words'] != [row['tokens'] for row in rows]:
        raise ValueError('Inference-only baseline inputs differ from the shared original words')
    image_count = audit_images(dataset, plan['manifest'])
    labels = dataset.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(labels), args)
    missing = load_compatible_model_checkpoint(model, plan['checkpoint'], map_location='cpu')
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    records, native_gold, native_predictions = [], [], []
    with torch.inference_mode():
        for ids, types, mask, pixels, image_mask, indices in tqdm(loader, desc='Baseline full-word evaluation', unit='batch'):
            paths, _, features = model(input_ids=ids.to(device), token_type_ids=types.to(device),
                attention_mask=mask.to(device), image_pixel_values=pixels.to(device),
                image_mask=image_mask.to(device), labels=None, return_features=True)
            if not torch.isfinite(features['token_emissions']).all() or len(paths) != len(indices):
                raise RuntimeError('Non-finite emissions or missing native CRF predictions')
            for offset, index_tensor in enumerate(indices):
                index = int(index_tensor)
                structure, _, retained = dataset.token_structure(index)
                record, gold, prediction = project_path(rows[index], paths[offset], structure.tolist(),
                    retained, labels, int(mask[offset].sum()))
                records.append(record)
                native_gold.append(gold)
                native_predictions.append(prediction)
    return records, span_metrics(native_gold, native_predictions), {
        'optional_missing_checkpoint_keys': missing, 'source_images_verified': image_count,
    }


def evaluate(cli):
    output = Path(cli.output).resolve()
    if output.exists():
        raise FileExistsError('Comparison output already exists; preserve the previous result')
    plan = prepare(cli)
    for protected in (plan['checkpoint'].parent.resolve(), Path(cli.vlm_evaluation).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Comparison output must be separate from the saved source runs')
    print('Verified original internal baseline:', plan['checkpoint'], flush=True)
    print('Original truncated-dev F1:', plan['reference_f1'], flush=True)
    if cli.check_only:
        print('Preflight passed; no model loaded, no outputs written.', flush=True)
        return
    runtime = check_baseline_runtime()
    output.mkdir(parents=True, exist_ok=False)
    try:
        records, native, execution = collect_predictions(plan, cli.device)
        indexed = checked_records(plan['groups']['full_dev'], records)
        (output / 'baseline_predictions.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        if any(abs(native['f1'] - expected) > 0.0005 for expected in (plan['reference_f1'], plan['recorded_f1'])):
            (output / 'native_reproduction.json').write_text(json.dumps(native, indent=2) + '\n', encoding='utf-8')
            raise RuntimeError('Baseline native truncated F1 does not reproduce the original best; stop before comparison')
        for name, expected in plan['bindings'].items():
            if sha256_file(name) != expected:
                raise ValueError(f'Frozen input changed during comparison: {name}')
        scores = {role: metrics_for(values, indexed, SCOPES[role]) for role, values in plan['groups'].items()}
        comparisons = {role: {
            'baseline': value, 'vlm': plan['vlm_report']['metrics'][role],
            'vlm_f1_minus_baseline': plan['vlm_report']['metrics'][role]['f1'] - value['f1'],
        } for role, value in scores.items()}
        coverage = {
            'original_words': sum(record['original_words'] for record in records),
            'observed_words': sum(len(record['retained_word_indices']) for record in records),
            'samples_with_unobserved_words': sum(len(record['retained_word_indices']) < record['original_words'] for record in records),
        }
        coverage['unobserved_words'] = coverage['original_words'] - coverage['observed_words']
        code_files = [Path(__file__), PROJECT / 'scripts/evaluate_vlm_ner.py', PROJECT / 'ner_evaluate.py']
        code_files += [PROJECT / name for name in ('models/agvs_mner.py', 'modules/dataset.py',
                                                   'modules/trainer.py', 'modules/vlm_ner.py', 'modules/internal_split.py')]
        result = {
            'schema_version': 'baseline-vlm-fullword-comparison-v1', 'checkpoint': str(plan['checkpoint']),
            'baseline_training_config': plan['config'], 'input_bindings': plan['bindings'],
            'evaluation_code_sha256': {str(path): sha256_file(path) for path in code_files},
            'source_ids': {role: [row['source_id'] for row in values] for role, values in plan['groups'].items()},
            'baseline_native_reproduction': {'passed': True, 'metrics': native, 'expected_f1': plan['reference_f1']},
            'comparisons': comparisons, 'baseline_word_coverage': coverage,
            'execution': {**execution, 'runtime': runtime},
            'official_val_or_test_read': False, 'optimizer_steps': 0, 'new_checkpoint_selection': False,
            'note': 'Same internal dev samples and full original-word gold spans. Baseline keeps its original '
                    'WordPiece window; unseen words predict O, and all gold entities remain in scoring. '
                    'Original image processors differ across architectures. Baseline was selected on full internal '
                    'dev; QLoRA was selected on the 128-row panel. This is an exploratory comparison of '
                    'different architectures/training protocols, not an AGVS component gain or independent final test.',
        }
        (output / 'comparison_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f"Native reproduction passed: F1={native['f1']:.6f}", flush=True)
        for role, value in comparisons.items():
            print(f"{role}: baseline={value['baseline']['f1']:.6f} QLoRA={value['vlm']['f1']:.6f} "
                  f"QLoRA-minus-baseline={value['vlm_f1_minus_baseline']:+.6f}", flush=True)
        print('Report:', output / 'comparison_report.json', flush=True)
        return result
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'error': str(error)}, indent=2) + '\n', encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', default='outputs_context_internal')
    parser.add_argument('--checkpoint')
    parser.add_argument('--reference-report')
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--vlm-evaluation', default='outputs_vlm_gold/mi_multi_image_seed1234_dev_evaluation')
    parser.add_argument('--output', default='outputs_vlm_gold/baseline_fullwords_seed1234_dev_evaluation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-only', action='store_true')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
