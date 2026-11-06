"""Screen AGVS-first BIO additions against the unchanged multimodal AGVS."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from scripts import audit_bio_joint_v1 as policy
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import train_bio_joint_recheck as train

MIN_GAIN = 0.003


def disjoint(span, chosen):
    start, end, _ = span
    return all(end <= left or right <= start for left, right, _ in chosen)


def nonoverlapping(candidates):
    """Use the same deterministic order in both lexical arms."""
    result = []
    for span in sorted(candidates, key=lambda item: (item[0] - item[1], item[0], item[2])):
        if disjoint(span, result):
            result.append(span)
    return result


def augmented(record, spans):
    return {**record, 'predicted': sorted([*record['predicted'], *map(list, spans)])}


def score_arms(rows, baseline, bio, lexicon):
    lengths = defaultdict(set)
    for phrase in lexicon:
        lengths[phrase[0]].add(len(phrase))
    lexical, assisted, bio_oracle, screened_oracle = {}, {}, {}, {}
    for row in rows:
        sid = row['source_id']
        anchor = baseline[sid]
        words = tuple(row['tokens'])
        candidates = set()
        for start, word in enumerate(words):
            for length in lengths.get(word, ()):
                end = start + length
                label = lexicon.get(words[start:end]) if end <= len(words) else None
                if label is not None and disjoint((start, end, label), anchor['predicted']):
                    candidates.add((start, end, label))

        proposals = ({tuple(span) for span in bio[sid]['predicted']}
                     if bio[sid]['error'] is None else set())
        supported = candidates & proposals
        lexical[sid] = augmented(anchor, nonoverlapping(candidates))
        assisted[sid] = augmented(anchor, nonoverlapping(supported))
        gold = policy.gold_spans(row)
        bio_oracle[sid] = augmented(anchor, (span for span in proposals & gold
                                            if disjoint(span, anchor['predicted'])))
        screened_oracle[sid] = augmented(anchor, supported & gold)

    score = recheck.bio.original.metrics_for
    return {name: score(rows, records, recheck.SCOPE) for name, records in (
        ('agvs', baseline), ('agvs_lexicon', lexical), ('agvs_lexicon_bio', assisted),
        ('bio_add_only_oracle', bio_oracle), ('lexicon_bio_add_only_oracle', screened_oracle),
    )}


def main():
    prepared, full = train.load_prepared(train.split.DATA)
    fit, rows, binding = recheck.evaluation_plan(prepared, full)
    _, baseline = recheck.load_baseline_predictions(rows, binding)
    report_path = recheck.OUTPUT / 'evaluation_report.json'
    report = train.READ_JSON(report_path)
    raw_path = recheck.OUTPUT / 'bio_predictions.jsonl'
    if (report.get('schema_version') != 'bio-joint-recheck-holdout-v1'
            or report.get('recheck') != binding or report.get('source_ids') != recheck.source_ids(rows)
            or report.get('official_val_or_test_read') is not False
            or report['output_predictions_sha256'][raw_path.name] != sha256_file(raw_path)):
        raise ValueError('The BIO predictions do not match the archived evaluation')
    bio = recheck.bio.checked_records(rows, recheck.READ_RECORDS(raw_path))
    score = recheck.bio.original.metrics_for
    for name, records in (('baseline', baseline), ('bio', bio)):
        if score(rows, records, recheck.SCOPE) != report['metrics'][name]:
            raise ValueError(f'Failed to reproduce the archived {name} scores')
    lexicon = policy.train_lexicon(fit['train'])
    scores = score_arms(rows, baseline, bio, lexicon)
    original = scores['agvs']['f1']
    lexical = scores['agvs_lexicon']['f1']
    assisted = scores['agvs_lexicon_bio']['f1']
    print(json.dumps({
        'scope': 'exploratory AGVS-first residual addition on the previously scored grouped holdout',
        'samples': len(rows), 'training_lexicon_size': len(lexicon),
        'scores': {name: {key: metrics[key] for key in ('tp', 'fp', 'fn', 'f1')}
                   for name, metrics in scores.items()},
        'llm_arm_minus_agvs_f1': assisted - original,
        'llm_arm_minus_lexicon_control_f1': assisted - lexical,
        'min_gain_to_continue': MIN_GAIN,
        'passes_exploratory_gate': assisted >= max(original, lexical) + MIN_GAIN,
        'oracles_use_gold_for_diagnostic_ceiling': True,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
