"""Audit frozen baseline/QLoRA predictions offline; no model or GPU imports.

Gold is used only for descriptive error analysis. No fusion, routing policy,
new checkpoint selection, training, inference, or official val/test access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from modules.vlm_ner import LABELS, load_prepared
from scripts.evaluate_vlm_ner import SCOPES, checked_records, metrics_for, read_json, read_records

CHANGE_FIELDS = (
    'common_tp', 'recovered_gold', 'lost_gold', 'both_missed_gold',
    'shared_fp', 'removed_baseline_fp', 'introduced_vlm_fp',
)
OUTCOMES = (
    'both_exact_entities', 'baseline_only_exact_entities', 'vlm_only_exact_entities',
    'both_wrong_same_entities', 'both_wrong_different_entities',
)
METRIC_KEYS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'by_type', 'samples',
               'malformed_responses', 'malformed_rate')


def require_bound(path, bindings):
    path = Path(path).resolve()
    if bindings.get(str(path)) != sha256_file(path):
        raise ValueError(f'Frozen artifact is missing its binding or changed: {path}')


def load_inputs(cli):
    manifest, rows = load_prepared(cli.data)
    panel_ids = set(manifest['panel_source_ids'])
    full_ids = [row['source_id'] for row in rows['dev']]
    if (len(panel_ids) != len(manifest['panel_source_ids']) or len(set(full_ids)) != len(full_ids)
            or not panel_ids < set(full_ids)
            or [row['source_id'] for row in rows['dev_panel']] != manifest['panel_source_ids']):
        raise ValueError('Invalid fixed internal dev/panel partition')
    row_index = {row['source_id']: row for row in rows['dev']}
    if any(row != row_index[row['source_id']] for row in rows['dev_panel']):
        raise ValueError('Panel rows differ from full-dev gold')
    groups = {'full_dev': rows['dev'], 'dev_panel': rows['dev_panel'],
              'dev_nonpanel': [row for row in rows['dev'] if row['source_id'] not in panel_ids]}
    comparison_path = Path(cli.comparison) / 'comparison_report.json'
    baseline_path = Path(cli.comparison) / 'baseline_predictions.jsonl'
    vlm_report_path = Path(cli.vlm_evaluation) / 'evaluation_report.json'
    vlm_path = Path(cli.vlm_evaluation) / 'dev_predictions.jsonl'
    comparison, vlm_report = read_json(comparison_path), read_json(vlm_report_path)
    if (comparison.get('schema_version') != 'baseline-vlm-fullword-comparison-v1'
            or comparison.get('baseline_native_reproduction', {}).get('passed') is not True):
        raise ValueError('A completed comparison with passing baseline reproduction is required')
    if (vlm_report.get('schema_version') != 'vlm-ner-dev-evaluation-v1'
            or vlm_report.get('panel_reproduction', {}).get('passed') is not True
            or vlm_report['profile']['data_sha256'] != sha256_file(cli.data)
            or vlm_report['profile']['input_mode'] != 'multi_image'):
        raise ValueError('A matching frozen multi-image QLoRA evaluation is required')
    for report in (comparison, vlm_report):
        if (report.get('official_val_or_test_read') is not False or report.get('optimizer_steps') != 0
                or report.get('new_checkpoint_selection') is not False):
            raise ValueError('Expected frozen internal-development results without new selection')
        if report['source_ids'] != {role: [row['source_id'] for row in values] for role, values in groups.items()}:
            raise ValueError('Saved evaluation source IDs differ from the common internal gold')
    config = comparison['baseline_training_config']
    if (config.get('dataset') != 'MI' or config.get('seed') != 1234
            or config.get('consistency_mode') != 'none'
            or Path(config['internal_manifest']).resolve() != Path(manifest['internal_manifest']).resolve()):
        raise ValueError('Comparison is not the original baseline on the common internal split')
    for path in (cli.data, manifest['internal_manifest'], vlm_report_path, vlm_path):
        require_bound(path, comparison['input_bindings'])
    scoring_files = [PROJECT / name for name in
                     ('modules/vlm_ner.py', 'modules/internal_split.py', 'scripts/evaluate_vlm_ner.py')]
    for path in scoring_files:
        require_bound(path, comparison['evaluation_code_sha256'])
    baseline = checked_records(rows['dev'], read_records(baseline_path))
    vlm = checked_records(rows['dev'], read_records(vlm_path))
    for row in rows['dev']:
        record = baseline[row['source_id']]
        indices = record['retained_word_indices']
        if (record['error'] is not None or record['original_words'] != len(row['tokens'])
                or any(type(index) is not int or not 0 <= index < len(row['tokens']) for index in indices)
                or indices != sorted(set(indices))):
            raise ValueError('Invalid baseline original-word observation metadata')
        observed = set(indices)
        if any(not set(range(a, b)) <= observed for a, b, _ in record['predicted']):
            raise ValueError('Baseline prediction covers a word outside its observed window')
    metrics = {}
    for role, values in groups.items():
        left, right = (metrics_for(values, predictions, SCOPES[role]) for predictions in (baseline, vlm))
        for key in METRIC_KEYS:
            if (left[key] != comparison['comparisons'][role]['baseline'][key]
                    or right[key] != comparison['comparisons'][role]['vlm'][key]
                    or right[key] != vlm_report['metrics'][role][key]):
                raise ValueError('Raw predictions no longer reproduce the completed comparison metrics')
        delta = right['f1'] - left['f1']
        if abs(delta - comparison['comparisons'][role]['vlm_f1_minus_baseline']) > 1e-12:
            raise ValueError('Saved F1 difference does not match the predictions')
        metrics[role] = {'baseline': left, 'vlm': right, 'vlm_f1_minus_baseline': delta}
    paths = [Path(cli.data), Path(manifest['internal_manifest']), comparison_path,
             baseline_path, vlm_report_path, vlm_path, *scoring_files, Path(__file__)]
    return groups, baseline, vlm, metrics, {str(path.resolve()): sha256_file(path) for path in paths}


def changes(gold, baseline, vlm):
    return {
        'common_tp': gold & baseline & vlm,
        'recovered_gold': (gold & vlm) - baseline,
        'lost_gold': (gold & baseline) - vlm,
        'both_missed_gold': gold - (baseline | vlm),
        'shared_fp': (baseline & vlm) - gold,
        'removed_baseline_fp': (baseline - gold) - vlm,
        'introduced_vlm_fp': (vlm - gold) - baseline,
    }


def outcome(gold, baseline, vlm):
    if baseline == gold and vlm == gold:
        return 'both_exact_entities'
    if baseline == gold:
        return 'baseline_only_exact_entities'
    if vlm == gold:
        return 'vlm_only_exact_entities'
    return 'both_wrong_same_entities' if baseline == vlm else 'both_wrong_different_entities'


def entity_sets(row, baseline, vlm):
    gold = {(entity['start'], entity['end'], entity['label']) for entity in row['entities']}
    return gold, {tuple(entity) for entity in baseline['predicted']}, {tuple(entity) for entity in vlm['predicted']}


def verify_counts(counts, baseline, vlm):
    identities = (
        (baseline['tp'], counts['common_tp'] + counts['lost_gold']),
        (vlm['tp'], counts['common_tp'] + counts['recovered_gold']),
        (baseline['fn'], counts['both_missed_gold'] + counts['recovered_gold']),
        (vlm['fn'], counts['both_missed_gold'] + counts['lost_gold']),
        (baseline['fp'], counts['shared_fp'] + counts['removed_baseline_fp']),
        (vlm['fp'], counts['shared_fp'] + counts['introduced_vlm_fp']),
    )
    if any(left != right for left, right in identities):
        raise ValueError('Entity error decomposition does not conserve the verified TP/FP/FN counts')


def analyse_group(rows, baseline, vlm, metrics):
    total = Counter({name: 0 for name in CHANGE_FIELDS})
    by_type = {label: Counter({name: 0 for name in CHANGE_FIELDS}) for label in LABELS}
    samples = Counter({name: 0 for name in OUTCOMES})
    different = 0
    beyond_coverage = 0
    for row in rows:
        source_id = row['source_id']
        gold, left, right = entity_sets(row, baseline[source_id], vlm[source_id])
        sample_changes = changes(gold, left, right)
        samples[outcome(gold, left, right)] += 1
        different += left != right
        for name, entities in sample_changes.items():
            total[name] += len(entities)
            for _, _, label in entities:
                by_type[label][name] += 1
        observed = set(baseline[source_id]['retained_word_indices'])
        beyond_coverage += sum(not set(range(a, b)) <= observed
                               for a, b, _ in sample_changes['recovered_gold'])
    verify_counts(total, metrics['baseline'], metrics['vlm'])
    classes = {}
    for label, counts in by_type.items():
        left, right = metrics['baseline']['by_type'][label], metrics['vlm']['by_type'][label]
        verify_counts(counts, left, right)
        classes[label] = {'baseline': left, 'vlm': right, 'vlm_f1_minus_baseline': right['f1'] - left['f1'],
                          'entity_changes': dict(counts)}
    return {
        **metrics, 'entity_changes': dict(total), 'by_type': classes,
        'sample_outcomes': dict(samples), 'different_entity_sets': different,
        'malformed_vlm_samples': sum(vlm[row['source_id']]['error'] is not None for row in rows),
        'recovered_gold_touching_unobserved_baseline_words': beyond_coverage,
    }


def annotated(entities, words):
    return [{'start': a, 'end': b, 'label': label, 'surface': ' '.join(words[a:b])}
            for a, b, label in sorted(entities)]


def case_records(rows, baseline, vlm, panel_ids):
    result = []
    for row in rows:
        source_id, words = row['source_id'], row['tokens']
        left_record, right_record = baseline[source_id], vlm[source_id]
        gold, left, right = entity_sets(row, left_record, right_record)
        result.append({
            'source_id': source_id, 'in_selection_panel': source_id in panel_ids, 'tokens': words,
            'sample_outcome': outcome(gold, left, right), 'gold': annotated(gold, words),
            'baseline': annotated(left, words), 'vlm': annotated(right, words),
            'changes': {name: annotated(values, words) for name, values in changes(gold, left, right).items()},
            'baseline_retained_word_indices': left_record['retained_word_indices'],
            'vlm_error': right_record['error'], 'vlm_response': right_record['response'],
        })
    return result


def select_review_cases(cases, count):
    selected = {}
    # Fixed deterministic examples from non-panel dev, not the largest errors.
    for name in CHANGE_FIELDS[1:]:
        candidates = [case for case in cases if not case['in_selection_panel'] and case['changes'][name]]
        candidates.sort(key=lambda case: hashlib.sha256(f"20261103:{name}:{case['source_id']}".encode()).hexdigest())
        for case in candidates[:count]:
            selected.setdefault(case['source_id'], {**case, 'review_buckets': []})['review_buckets'].append(name)
    return [selected[source_id] for source_id in sorted(selected)]


def audit(cli):
    output = Path(cli.output).resolve()
    if output.exists():
        raise FileExistsError('Audit output already exists; preserve the previous result')
    for source in (cli.comparison, cli.vlm_evaluation, Path(cli.data).parent):
        protected = Path(source).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Audit output must be separate from the source artifact directories')
    groups, baseline, vlm, metrics, bindings = load_inputs(cli)
    analysis = {role: analyse_group(values, baseline, vlm, metrics[role]) for role, values in groups.items()}
    panel_ids = {row['source_id'] for row in groups['dev_panel']}
    cases = case_records(groups['full_dev'], baseline, vlm, panel_ids)
    review = select_review_cases(cases, cli.review_per_bucket)
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise ValueError(f'Artifact changed during offline analysis: {path}')
    result = {
        'schema_version': 'baseline-qlora-error-audit-v1', 'analysis': analysis, 'input_sha256': bindings,
        'source_ids': {role: [row['source_id'] for row in values] for role, values in groups.items()},
        'diagnostic_only': True, 'gold_used_for_analysis': True, 'deployment_selector_created': False,
        'model_inference_runs': 0, 'training_runs': 0, 'llm_api_requests': 0, 'official_val_or_test_read': False,
        'review_policy': {'seed': 20261103, 'scope': 'dev_nonpanel', 'per_bucket': cli.review_per_bucket,
                          'buckets': list(CHANGE_FIELDS[1:]), 'selected_cases': len(review)},
        'note': 'Recovered/lost gold and added/removed false positives describe whole-model replacement. '
                'They are gold-aware diagnostics, not a deployable edit policy, fusion score, or AGVS component gain. '
                'Internal dev has historical development use; this is not an independent final test. '
                'The original comparison did not hash-bind baseline_predictions.jsonl; its strict parsed predictions '
                'and all reported metrics were verified here, and its SHA256 is frozen by this audit. '
                'Exact-entity sample categories ignore formatting; malformed responses remain explicit empty predictions.',
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'error_audit_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, values in (('error_cases.jsonl', cases), ('review_cases.jsonl', review)):
        (output / name).write_text(''.join(json.dumps(value, ensure_ascii=True) + '\n' for value in values), encoding='utf-8')
    with (output / 'by_type.csv').open('w', encoding='utf-8', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=['scope', 'label', 'baseline_f1', 'vlm_f1',
                                                        'vlm_f1_minus_baseline', *CHANGE_FIELDS])
        writer.writeheader()
        for role, value in analysis.items():
            for label, detail in value['by_type'].items():
                writer.writerow({'scope': role, 'label': label, 'baseline_f1': detail['baseline']['f1'],
                    'vlm_f1': detail['vlm']['f1'], 'vlm_f1_minus_baseline': detail['vlm_f1_minus_baseline'],
                    **detail['entity_changes']})
    for role, value in analysis.items():
        print(f"{role}: baseline={value['baseline']['f1']:.6f} QLoRA={value['vlm']['f1']:.6f} "
              f"delta={value['vlm_f1_minus_baseline']:+.6f} changes=" + json.dumps(value['entity_changes']), flush=True)
    for label, detail in analysis['full_dev']['by_type'].items():
        print(f"full_dev/{label}: baseline={detail['baseline']['f1']:.6f} QLoRA={detail['vlm']['f1']:.6f} "
              f"delta={detail['vlm_f1_minus_baseline']:+.6f} changes=" + json.dumps(detail['entity_changes']), flush=True)
    print('Report:', output / 'error_audit_report.json', flush=True)
    return result


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--comparison', default='outputs_vlm_gold/baseline_fullwords_seed1234_dev_evaluation')
    parser.add_argument('--vlm-evaluation', default='outputs_vlm_gold/mi_multi_image_seed1234_dev_evaluation')
    parser.add_argument('--output', default='outputs_vlm_gold/baseline_qlora_error_audit')
    parser.add_argument('--review-per-bucket', type=positive_int, default=8)
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
