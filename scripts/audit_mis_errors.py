"""Diagnose MIS errors from frozen predictions without models, GPU, or training.

Categories describe span relationships, not proven semantic causes. FN and FP
views are separate: a single type/boundary mistake can appear in both views.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from scripts import audit_baseline_qlora_errors as previous

CATEGORIES = ('no_overlap', 'type_only', 'boundary_only', 'boundary_and_type',
              'multiple_overlaps', 'malformed_response')
VIEWS = ('baseline_fn', 'vlm_fn', 'baseline_fp', 'vlm_fp', 'lost_gold', 'recovered_gold',
         'introduced_vlm_fp', 'removed_baseline_fp', 'both_missed_gold', 'shared_fp')
REVIEW_BUCKETS = ('lost_gold', 'introduced_vlm_fp', 'recovered_gold')


def classify(anchor, opposing, gold_anchor, malformed=False):
    if anchor in opposing:
        raise ValueError('An exact entity match cannot be classified as an error')
    if malformed:
        if not gold_anchor or opposing:
            raise ValueError('Malformed response must be an empty prediction in the FN view')
        return {'category': 'malformed_response', 'overlaps': []}
    a, b, label = anchor
    overlaps = sorted(entity for entity in opposing if max(a, entity[0]) < min(b, entity[1]))
    exact_span = [entity for entity in overlaps if entity[:2] == anchor[:2]]
    if len(exact_span) == 1 and len(overlaps) == 1:
        category, counterpart = 'type_only', exact_span[0]
    elif not overlaps:
        return {'category': 'no_overlap', 'overlaps': []}
    elif len(overlaps) > 1:
        return {'category': 'multiple_overlaps', 'overlaps': overlaps}
    else:
        counterpart = overlaps[0]
        category = 'boundary_only' if counterpart[2] == label else 'boundary_and_type'
    gold, prediction = (anchor, counterpart) if gold_anchor else (counterpart, anchor)
    result = {'category': category, 'overlaps': overlaps}
    if gold[2] != prediction[2]:
        result['type_confusion'] = gold[2] + '->' + prediction[2]
    if gold[:2] != prediction[:2]:
        if prediction[0] <= gold[0] and prediction[1] >= gold[1]:
            relation = 'prediction_contains_gold'
        elif gold[0] <= prediction[0] and gold[1] >= prediction[1]:
            relation = 'prediction_inside_gold'
        else:
            relation = 'partial_overlap'
        result['boundary_relation'] = relation
    return result


def make_errors(rows, baseline, vlm, panel_ids):
    records = []
    for row in rows:
        source_id, words = row['source_id'], row['tokens']
        gold, left, right = previous.entity_sets(row, baseline[source_id], vlm[source_id])
        for model, prediction, other in (('baseline', left, right), ('vlm', right, left)):
            malformed = model == 'vlm' and vlm[source_id]['error'] is not None
            fn = {entity for entity in gold - prediction if entity[2] == 'MIS'}
            fp = {entity for entity in prediction - gold if entity[2] == 'MIS'}
            for direction, entities in (('fn', fn), ('fp', fp)):
                for anchor in sorted(entities):
                    if direction == 'fn':
                        bucket = ('recovered_gold' if model == 'baseline' else 'lost_gold') if anchor in other else 'both_missed_gold'
                    else:
                        bucket = 'shared_fp' if anchor in other else (
                            'removed_baseline_fp' if model == 'baseline' else 'introduced_vlm_fp')
                    diagnosis = classify(anchor, prediction if direction == 'fn' else gold,
                                         direction == 'fn', malformed and direction == 'fn')
                    records.append({
                        'source_id': source_id, 'in_selection_panel': source_id in panel_ids,
                        'model': model, 'direction': direction, 'relative_bucket': bucket,
                        'anchor': previous.annotated({anchor}, words)[0],
                        'diagnosis': {**diagnosis, 'overlaps': previous.annotated(diagnosis['overlaps'], words)},
                        'tokens': words, 'gold': previous.annotated(gold, words),
                        'baseline': previous.annotated(left, words), 'vlm': previous.annotated(right, words),
                        'vlm_error': vlm[source_id]['error'], 'vlm_response': vlm[source_id]['response'],
                    })
    return records


def summarise(records):
    categories = Counter({category: 0 for category in CATEGORIES})
    exact_types, mixed_types, boundaries = Counter(), Counter(), Counter()
    for record in records:
        diagnosis = record['diagnosis']
        categories[diagnosis['category']] += 1
        if diagnosis['category'] == 'type_only':
            exact_types[diagnosis['type_confusion']] += 1
        elif diagnosis['category'] == 'boundary_and_type':
            mixed_types[diagnosis['type_confusion']] += 1
        if diagnosis.get('boundary_relation'):
            boundaries[diagnosis['boundary_relation']] += 1
    return {'entities': len(records), 'samples': len({record['source_id'] for record in records}),
            'categories': dict(categories), 'exact_span_type_confusions': dict(sorted(exact_types.items())),
            'boundary_and_type_confusions': dict(sorted(mixed_types.items())),
            'boundary_relations': dict(sorted(boundaries.items()))}


def views_for(records):
    result = {f'{model}_{direction}': [record for record in records
                                      if record['model'] == model and record['direction'] == direction]
              for model in ('baseline', 'vlm') for direction in ('fn', 'fp')}
    for bucket in VIEWS[4:]:
        # Shared errors have one record per model. Use the VLM view once here.
        result[bucket] = [record for record in records if record['relative_bucket'] == bucket
                         and (bucket not in {'both_missed_gold', 'shared_fp'} or record['model'] == 'vlm')]
    return result


def analyse_scope(records, source_ids, old_analysis):
    records = [record for record in records if record['source_id'] in source_ids]
    views = views_for(records)
    mis = old_analysis['by_type']['MIS']
    expected = {'baseline_fn': mis['baseline']['fn'], 'vlm_fn': mis['vlm']['fn'],
                'baseline_fp': mis['baseline']['fp'], 'vlm_fp': mis['vlm']['fp'], **mis['entity_changes']}
    summaries = {name: summarise(values) for name, values in views.items()}
    if any(summary['entities'] != expected[name] for name, summary in summaries.items()):
        raise ValueError('MIS diagnosis must retain every error from the frozen audit')
    pair_index = {}
    for record in records:
        if record['relative_bucket'] == 'both_missed_gold':
            anchor = record['anchor']
            key = (record['source_id'], anchor['start'], anchor['end'], anchor['label'])
            pair_index.setdefault(key, {})[record['model']] = record['diagnosis']['category']
    if any(set(value) != {'baseline', 'vlm'} for value in pair_index.values()):
        raise ValueError('Each shared MIS miss needs both model diagnoses')
    pairs = Counter(f"{value['baseline']}->{value['vlm']}" for value in pair_index.values())
    return {'baseline': mis['baseline'], 'vlm': mis['vlm'], 'vlm_f1_minus_baseline': mis['vlm_f1_minus_baseline'],
            'entity_changes': mis['entity_changes'], 'error_views': summaries,
            'common_miss_category_pairs_baseline_to_vlm': dict(sorted(pairs.items()))}


def select_review(records, count):
    selected = []
    for bucket in REVIEW_BUCKETS:
        for category in CATEGORIES:
            values = [record for record in records if not record['in_selection_panel']
                      and record['relative_bucket'] == bucket and record['diagnosis']['category'] == category]
            values.sort(key=lambda record: hashlib.sha256(
                f"20261103:{bucket}:{category}:{record['source_id']}:{record['anchor']['start']}:{record['anchor']['end']}".encode()).hexdigest())
            selected.extend(values[:count])
    return selected


def render_example(record):
    return '\n'.join((
        f"source_id={record['source_id']} bucket={record['relative_bucket']} model={record['model']} "
        f"category={record['diagnosis']['category']}",
        'WORDS: ' + json.dumps(list(enumerate(record['tokens'])), ensure_ascii=True),
        'ANCHOR: ' + json.dumps(record['anchor'], ensure_ascii=True),
        'OVERLAPS: ' + json.dumps(record['diagnosis']['overlaps'], ensure_ascii=True),
        'GOLD: ' + json.dumps(record['gold'], ensure_ascii=True),
        'BASELINE: ' + json.dumps(record['baseline'], ensure_ascii=True),
        'QLORA: ' + json.dumps(record['vlm'], ensure_ascii=True),
    ))


def load_frozen(cli):
    groups, baseline, vlm, metrics, bindings = previous.load_inputs(cli)
    old_path = Path(cli.error_audit) / 'error_audit_report.json'
    old = previous.read_json(old_path)
    if (old.get('schema_version') != 'baseline-qlora-error-audit-v1'
            or old.get('diagnostic_only') is not True or old.get('gold_used_for_analysis') is not True
            or old.get('official_val_or_test_read') is not False or old.get('deployment_selector_created') is not False
            or old.get('model_inference_runs') != 0 or old.get('training_runs') != 0
            or old.get('input_sha256') != bindings):
        raise ValueError('The completed prior error audit or its frozen inputs changed')
    if old['source_ids'] != {role: [row['source_id'] for row in rows] for role, rows in groups.items()}:
        raise ValueError('Prior error audit source IDs changed')
    for role, rows in groups.items():
        if previous.analyse_group(rows, baseline, vlm, metrics[role]) != old['analysis'][role]:
            raise ValueError('Prior error counts do not reproduce from the frozen predictions')
    return groups, baseline, vlm, old, {**bindings, str(old_path.resolve()): sha256_file(old_path),
                                      str(Path(__file__).resolve()): sha256_file(__file__)}


def audit(cli):
    output = Path(cli.output).resolve()
    if output.exists():
        raise FileExistsError('MIS diagnosis output already exists; preserve the previous result')
    for source in (cli.error_audit, cli.comparison, cli.vlm_evaluation, Path(cli.data).parent):
        protected = Path(source).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('MIS diagnosis output must be separate from the frozen artifact directories')
    groups, baseline, vlm, old, bindings = load_frozen(cli)
    records = make_errors(groups['full_dev'], baseline, vlm, {row['source_id'] for row in groups['dev_panel']})
    analysis = {role: analyse_scope(records, {row['source_id'] for row in rows}, old['analysis'][role])
                for role, rows in groups.items()}
    review = select_review(records, cli.review_per_category)
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise ValueError(f'Frozen input changed during MIS diagnosis: {path}')
    result = {
        'schema_version': 'mis-error-diagnosis-v1', 'analysis': analysis, 'input_sha256': bindings,
        'diagnostic_only': True, 'gold_used_for_analysis': True, 'model_inference_runs': 0,
        'training_runs': 0, 'llm_api_requests': 0, 'official_val_or_test_read': False,
        'review_policy': {'scope': 'dev_nonpanel', 'seed': 20261103, 'buckets': list(REVIEW_BUCKETS),
                          'per_category': cli.review_per_category, 'selected_entity_errors': len(review)},
        'note': 'Categories describe overlap of predicted and gold spans, not proven semantic causes. '
                'FN and FP counts are separate views and must not be summed into unique mistakes. '
                'Multiple overlaps remain ambiguous; no greedy entity matching or gold repair is used. '
                'All internal dev data has development history; examples must not become an inference policy.',
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'mis_diagnosis_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, values in (('mis_errors.jsonl', records), ('review_cases.jsonl', review)):
        (output / name).write_text(''.join(json.dumps(record, ensure_ascii=True) + '\n' for record in values), encoding='utf-8')
    (output / 'review_examples.txt').write_text('\n\n'.join(render_example(record) for record in review) + '\n', encoding='utf-8')
    for role, value in analysis.items():
        print(f"{role}: MIS baseline={value['baseline']['f1']:.6f} QLoRA={value['vlm']['f1']:.6f} "
              f"delta={value['vlm_f1_minus_baseline']:+.6f}", flush=True)
        for name in ('lost_gold', 'introduced_vlm_fp'):
            print(f"{role}/{name}: " + json.dumps(value['error_views'][name]), flush=True)
    # Print a small set of actual contexts so the next review can inspect cases,
    # rather than requiring another round of aggregate-only diagnostics.
    for bucket in REVIEW_BUCKETS:
        available = [value for value in review if value['relative_bucket'] == bucket]
        first_by_category = [next(value for value in available if value['diagnosis']['category'] == category)
                             for category in CATEGORIES
                             if any(value['diagnosis']['category'] == category for value in available)]
        preview = (first_by_category + [value for value in available if value not in first_by_category])[:2]
        for record in preview:
            print('\nEXAMPLE ' + render_example(record), flush=True)
    print('Report:', output / 'mis_diagnosis_report.json', flush=True)
    print('Review examples:', output / 'review_examples.txt', flush=True)
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
    parser.add_argument('--error-audit', default='outputs_vlm_gold/baseline_qlora_error_audit')
    parser.add_argument('--output', default='outputs_vlm_gold/mis_error_diagnosis')
    parser.add_argument('--review-per-category', type=positive_int, default=2)
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
