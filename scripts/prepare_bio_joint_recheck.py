"""Plan a grouped training-side recheck of the frozen BIO joint policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from modules.vlm_ner import load_prepared

SEED = 20261105
MIN_HOLDOUT = 1000
DATA = Path('dataset/vlm_gold_mi_v1/manifest.json')
OUTPUT = Path('outputs_vlm_gold/bio_joint_recheck_v1/split_manifest.json')
POLICY = PROJECT / 'scripts/audit_bio_joint_v1.py'
BIO_CODEC = PROJECT / 'scripts/run_vlm_bio.py'
BIO_REPORT = Path('outputs_vlm_gold/mi_bio_seed1234_dev_evaluation/evaluation_report.json')


def grouped_sources(rows):
    parents = {row['source_id']: row['source_id'] for row in rows}
    if len(parents) != len(rows):
        raise ValueError('Source IDs must be unique across the original train and dev')

    def root(sid):
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid

    owners = {}
    for row in rows:
        sid = row['source_id']
        keys = [('text', tuple(word.casefold() for word in row['tokens']))]
        keys.extend(('image', image['sha256']) for image in row['images'])
        for key in keys:
            other = owners.setdefault(key, sid)
            left, right = root(sid), root(other)
            parents[max(left, right)] = min(left, right)

    groups = {}
    for sid in sorted(parents):
        groups.setdefault(root(sid), []).append(sid)
    return list(groups.values())


def split_plan(train, dev, panel_ids):
    train_ids = {row['source_id'] for row in train}
    dev_ids = {row['source_id'] for row in dev}
    if not panel_ids or not set(panel_ids) <= dev_ids:
        raise ValueError('Expected the original checkpoint-selection panel')
    groups = grouped_sources(train + dev)
    eligible = [group for group in groups if dev_ids.isdisjoint(group)]
    ordered = sorted(eligible, key=lambda group: hashlib.sha256(f'{SEED}:{group[0]}'.encode()).digest())
    target = len(train) // 5
    holdout = set()
    for group in ordered:
        if len(holdout) + len(group) <= target:
            holdout.update(group)
        if len(holdout) == target:
            break

    remaining = train_ids - holdout
    if any(holdout.intersection(group) and not holdout.issuperset(group) for group in groups):
        raise ValueError('A text/image group crosses the holdout boundary')
    return {
        'target_holdout': target,
        'minimum_holdout': MIN_HOLDOUT,
        'eligible_train_samples': sum(len(group) for group in eligible),
        'train_samples_linked_to_old_dev': len(train_ids) - sum(len(group) for group in eligible),
        'largest_eligible_group': max(map(len, eligible), default=0),
        'holdout_samples': len(holdout),
        'retrain_samples': len(remaining),
        'ready': len(holdout) >= MIN_HOLDOUT,
        'source_ids': {
            'retrain': [row['source_id'] for row in train if row['source_id'] in remaining],
            'selection_dev': [row['source_id'] for row in dev],
            'bio_selection_panel': panel_ids,
            'recheck_holdout': sorted(holdout),
        },
    }


def main():
    prepared, rows = load_prepared(DATA)
    report = json.loads(BIO_REPORT.read_text(encoding='utf-8'))
    data_digest = sha256_file(DATA)
    codec_digest = sha256_file(BIO_CODEC)
    if (report['profile']['data_sha256'] != data_digest
            or report['profile']['bio_implementation_sha256'] != codec_digest
            or report['panel_reproduction']['passed'] is not True
            or report['source_ids']['full_dev'] != [row['source_id'] for row in rows['dev']]):
        raise ValueError('Expected the archived BIO run and original internal split')
    plan = split_plan(rows['train'], rows['dev'], prepared['panel_source_ids'])
    manifest = {
        'schema_version': 'bio-joint-recheck-split-v1',
        'scope': 'grouped recheck on the previously used training corpus',
        'selection_seed': SEED,
        'grouping': 'connected components by casefolded original words or prepared image SHA256',
        'prepared_manifest': str(DATA),
        'prepared_manifest_sha256': data_digest,
        'split_script_sha256': sha256_file(__file__),
        'frozen_policy': {'version': 'bio_joint_v1', 'sha256': sha256_file(POLICY),
                          'bio_codec_sha256': codec_digest},
        **plan,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text(encoding='utf-8')) != manifest:
            raise ValueError(f'Existing split manifest differs: {OUTPUT}')
    else:
        with OUTPUT.open('x', encoding='utf-8') as destination:
            json.dump(manifest, destination, indent=2)
            destination.write('\n')
    summary = {key: value for key, value in plan.items() if key != 'source_ids'}
    summary['split_manifest'] = str(OUTPUT.resolve())
    summary['split_manifest_sha256'] = sha256_file(OUTPUT)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
