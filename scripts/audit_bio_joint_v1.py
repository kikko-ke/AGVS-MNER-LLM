"""Replay the frozen BIO joint diagnostic on the original internal dev."""
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.vlm_ner import load_prepared
from scripts import audit_baseline_qlora_errors as audit
from scripts import run_vlm_bio as bio


def gold_spans(row):
    return {(entity['start'], entity['end'], entity['label']) for entity in row['entities']}


def train_lexicon(rows):
    supports = defaultdict(lambda: defaultdict(set))
    for row in rows:
        words = tuple(row['tokens'])
        for start, end, label in gold_spans(row):
            supports[words[start:end]][label].add(words)

    lexicon = {}
    for phrase, labels in supports.items():
        if len(labels) == 1:
            label, texts = next(iter(labels.items()))
            if len(texts) >= 3:
                lexicon[phrase] = label

    lengths = defaultdict(set)
    for phrase in lexicon:
        lengths[phrase[0]].add(len(phrase))
    ambiguous = set()
    for row in rows:
        words = tuple(row['tokens'])
        gold = gold_spans(row)
        for start, word in enumerate(words):
            for size in lengths.get(word, ()):
                end = start + size
                if end > len(words):
                    continue
                phrase = words[start:end]
                label = lexicon.get(phrase)
                if label is not None and (start, end, label) not in gold:
                    ambiguous.add(phrase)
    return {phrase: label for phrase, label in lexicon.items() if phrase not in ambiguous}


def candidates_for(words, bio_record, baseline_record, lexicon):
    if bio_record['error'] is not None:
        return [], []
    available = []
    additions = []
    for start, end, label in baseline_record['predicted']:
        if any(start < right and left < end for left, right, _ in bio_record['predicted']):
            continue
        span = (start, end, label)
        available.append(span)
        if lexicon.get(tuple(words[start:end])) == label:
            additions.append(span)
    return available, additions


def repair_bio_length(word_count, record):
    if record['error'] is None:
        return record
    text = record['response'].strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    try:
        value = json.loads(text)
    except ValueError:
        return record
    if not isinstance(value, dict) or set(value) != {'tags'} or not isinstance(value['tags'], str):
        return record
    tags = value['tags']
    if len(tags) == word_count or any(ch not in '012345678' for ch in tags):
        return record
    response = json.dumps({'tags': tags[:word_count].ljust(word_count, '0')})
    spans, error = bio.decoded(response, word_count)
    if error is not None:
        return record
    return {**record, 'response': response, 'predicted': spans, 'error': None}


def main():
    data = Path('dataset/vlm_gold_mi_v1/manifest.json')
    base_dir = Path('outputs_vlm_gold/baseline_fullwords_seed1234_dev_evaluation')
    bio_dir = Path('outputs_vlm_gold/mi_bio_seed1234_dev_evaluation')
    audit_path = Path('outputs_vlm_gold/baseline_qlora_error_audit/error_audit_report.json')
    comparison_path = base_dir / 'comparison_report.json'
    baseline_path = base_dir / 'baseline_predictions.jsonl'
    bio_path = bio_dir / 'dev_predictions.jsonl'
    prior_audit = bio.original.read_json(audit_path)
    for path in (data, comparison_path, baseline_path):
        audit.require_bound(path, prior_audit['input_sha256'])

    manifest, rows = load_prepared(data)
    comparison = bio.original.read_json(comparison_path)
    report = bio.original.read_json(bio_dir / 'evaluation_report.json')
    dev = rows['dev']
    ids = [row['source_id'] for row in dev]
    if (comparison['baseline_native_reproduction']['passed'] is not True
            or comparison['source_ids']['full_dev'] != ids
            or report['source_ids']['full_dev'] != ids
            or report['panel_reproduction']['passed'] is not True
            or report['profile']['data_sha256'] != bio.sha256_file(data)
            or report['profile']['bio_implementation_sha256'] != bio.sha256_file(bio.__file__)
            or report['dev_predictions_sha256'] != bio.sha256_file(bio_path)):
        raise SystemExit('Frozen BIO and baseline artifacts failed verification')

    baseline = bio.original.checked_records(dev, bio.original.read_records(baseline_path))
    original = bio.checked_records(dev, bio.original.read_records(bio_path))
    scope = bio.original.SCOPES['full_dev']
    base_scores = bio.original.metrics_for(dev, baseline, scope)
    saved_base = comparison['comparisons']['full_dev']['baseline']
    if any(base_scores[key] != saved_base[key] for key in audit.METRIC_KEYS):
        raise SystemExit('Baseline metrics failed replay')
    if bio.original.metrics_for(dev, original, scope) != report['metrics']['full_dev']:
        raise SystemExit('BIO metrics failed replay')

    lexicon = train_lexicon(rows['train'])
    eligible = {}
    combined = {}
    tail_only = {}
    joint = {}
    for row in dev:
        sid = row['source_id']
        record = original[sid]
        available, additions = candidates_for(row['tokens'], record, baseline[sid], lexicon)
        eligible[sid] = available
        spans = sorted([tuple(span) for span in record['predicted']] + additions)
        combined[sid] = {**record, 'predicted': [list(span) for span in spans]}
        tail_only[sid] = repair_bio_length(len(row['tokens']), record)
        joint[sid] = combined[sid] if record['error'] is None else tail_only[sid]

    panel_ids = set(manifest['panel_source_ids'])
    groups = {'full_dev': dev, 'dev_nonpanel': [row for row in dev if row['source_id'] not in panel_ids]}
    gate = base_scores['f1'] + 0.003
    result = {
        'audit_version': 'bio_joint_v1',
        'lexicon_size': len(lexicon),
        'minimum_distinct_training_texts': 3,
        'full_dev_gate': gate,
    }
    for name, subset in groups.items():
        description = bio.original.SCOPES[name]
        before = bio.original.metrics_for(subset, original, description)
        after = bio.original.metrics_for(subset, combined, description)
        tail = bio.original.metrics_for(subset, tail_only, description)
        together = bio.original.metrics_for(subset, joint, description)
        for key in ('tp', 'fp', 'fn'):
            if together[key] - before[key] != (after[key] - before[key]) + (tail[key] - before[key]):
                raise SystemExit(f'Disjoint rule counts failed for {name}/{key}')
        result[name] = {
            'nonoverlapping_baseline_spans': sum(len(eligible[row['source_id']]) for row in subset),
            'correct_nonoverlapping_baseline_spans': sum(
                len(set(eligible[row['source_id']]) & gold_spans(row)) for row in subset),
            'added_tp': after['tp'] - before['tp'],
            'added_fp': after['fp'] - before['fp'],
            'bio_f1': before['f1'],
            'candidate_f1': after['f1'],
            'candidate_recall': after['recall'],
            'tail_only_added_tp': tail['tp'] - before['tp'],
            'tail_only_added_fp': tail['fp'] - before['fp'],
            'joint_added_tp': together['tp'] - before['tp'],
            'joint_added_fp': together['fp'] - before['fp'],
            'joint_f1': together['f1'],
            'joint_recall': together['recall'],
        }
    result['passes_gate'] = result['full_dev']['candidate_f1'] >= gate
    result['joint_passes_numeric_gate'] = result['full_dev']['joint_f1'] >= gate
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
