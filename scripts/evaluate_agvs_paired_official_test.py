"""Score frozen AGVS pairs on a second, previously used MNER-MI evaluation split."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from llm.augmentation_schema import extract_protected_entities
from modules.internal_split import sha256_file
from modules.vlm_ner import image_path, normalize_words
from scripts import evaluate_agvs_text_control_holdout as text_eval
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import prepare_bio_joint_recheck as split
from scripts import run_agvs_paired_seed_recheck as paired
from scripts import train_bio_joint_recheck as train

SOURCE = Path('dataset/text/MNER-MI_test.txt')
OUTPUT = train.ROOT / 'agvs_paired_official_test_supplement_v1'
SCOPE = 'previously used official MNER-MI test, excluding groups linked to the original training corpus'


def text_key(words):
    return tuple(word.casefold() for word in words)


def build_cohort(lines, original_rows, original_images, config):
    """Use input identity alone to exclude connected groups touching old data."""
    old_text = {text_key(row['tokens']) for row in original_rows}
    old_paths = {str(Path(path).resolve()) for path in original_images}
    old_hashes = {info['sha256'] for info in original_images.values() if info['sha256'] is not None}
    image_audit, candidates, overlaps = {}, [], set()
    for sid, line in enumerate(lines):
        raw = ast.literal_eval(line)
        words = normalize_words(raw['text'])
        paths = set()
        hashes = set()
        for name in raw['images'][:4]:
            path = image_path(str(name), config['image_path'], config['twitter2017_image_path']).resolve()
            key = str(path)
            if key not in image_audit:
                image_audit[key] = {'sha256': sha256_file(path) if path.is_file() else None}
            paths.add(key)
            digest = image_audit[key]['sha256']
            if digest is not None:
                hashes.add(digest)
        candidates.append({'source_id': sid, 'tokens': words,
                           'images': [{'sha256': digest} for digest in sorted(hashes)]})
        if text_key(words) in old_text or paths & old_paths or hashes & old_hashes:
            overlaps.add(sid)

    groups = split.grouped_sources(candidates)
    excluded = {sid for group in groups if overlaps.intersection(group) for sid in group}
    rows = []
    for row in candidates:
        sid = row['source_id']
        if sid not in excluded:
            gold = extract_protected_entities(row['tokens'], ast.literal_eval(lines[sid])['label'])
            rows.append({**row, 'entities': [dict(start=e.start, end=e.end, label=e.label) for e in gold]})
    if not rows:
        raise ValueError('No eligible official-test rows after excluding overlap groups')
    with_images = sum(bool(row['images']) for row in rows)
    if not with_images:
        raise ValueError('The retained official-test rows have no available image files')
    audit = {'source_samples': len(lines), 'retained_samples': len(rows),
             'direct_overlap_samples': len(overlaps), 'excluded_group_samples': len(excluded),
             'excluded_source_ids': sorted(excluded), 'retained_with_image_files': with_images}
    return rows, {'image_audit': image_audit}, audit


def frozen_1234(runtime, binding, config):
    full = recheck.baseline_seal(binding)
    text = train.READ_JSON(text_eval.TEXT_REPORT)
    if (full.get('runtime') != runtime or full.get('best_epoch') != 11
            or text.get('schema_version') != 'agvs-text-control-dev-v1'
            or text.get('runtime') != runtime or text.get('recheck') != binding
            or text.get('text_best_epoch') != 4
            or text.get('official_val_or_test_read') is not False
            or text.get('image_policy') != 'zero pixels and masks in training and selection'):
        raise ValueError('Expected the archived seed-1234 pair')
    results = {}
    for arm, report, report_path, expected_epoch, expected_dir in (
        ('multimodal', full, recheck.BASELINE_REPORT, 11, train.ROOT / 'baseline'),
        ('text', text, text_eval.TEXT_REPORT, 4, text_eval.TEXT_REPORT.parent),
    ):
        checkpoint = Path(report['checkpoint']).resolve()
        cfg = checkpoint.parent / 'config.json'
        artifacts = full['input_sha256'] if arm == 'multimodal' else text['output_sha256']
        if any(str(path) not in artifacts for path in (checkpoint, cfg)):
            raise ValueError(f'Unbound checkpoint or config for seed 1234, {arm}')
        recheck.verify_files(artifacts)
        saved = train.READ_JSON(cfg)
        if saved != {**config, 'output_dir': str(expected_dir)}:
            raise ValueError(f'Training config differs for seed 1234, {arm}')
        results[arm] = (expected_epoch, checkpoint, saved,
                        {**artifacts, str(report_path): sha256_file(report_path)})
    return results


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing supplemental evaluation: {OUTPUT}')
    runtime, prepared, fit, _, holdout, binding, config = paired.protocol()
    models = {1234: frozen_1234(runtime, binding, config)}
    for seed in paired.SEEDS:
        models[seed] = {}
        for arm in paired.ARMS:
            report, checkpoint, saved, artifacts = paired.read_arm(seed, arm, runtime, binding, config)
            models[seed][arm] = (report['best_epoch'], checkpoint, saved, artifacts)

    original = SOURCE.read_bytes()
    lines = original.decode('utf-8').splitlines()
    rows, manifest, cohort = build_cohort(lines, fit['train'] + fit['dev'] + holdout,
                                          prepared['image_audit'], config)
    print(json.dumps({key: value for key, value in cohort.items() if key != 'excluded_source_ids'}),
          flush=True)
    inputs = {str(SOURCE.resolve()): hashlib.sha256(original).hexdigest(),
              str(train.split.DATA): sha256_file(train.split.DATA),
              str(train.split.OUTPUT): train.SPLIT_SHA256,
              str(Path(__file__).resolve()): sha256_file(__file__)}
    scores, epochs, predictions = {}, {}, {}
    with TemporaryDirectory(dir=OUTPUT.parent, prefix='.agvs-second-split-') as directory:
        scratch = Path(directory)
        for seed, arms in models.items():
            scores[seed], epochs[seed] = {}, {}
            for arm, (epoch, checkpoint, saved, artifacts) in arms.items():
                inputs.update(artifacts)
                epochs[seed][arm] = epoch
                print(f'Seed {seed}, {arm}: scoring {len(rows)} official-test rows...', flush=True)
                records, _, _ = paired.predict(scratch, SOURCE, lines, rows, manifest, arm, checkpoint, saved)
                indexed = recheck.bio.original.checked_records(rows, records)
                scores[seed][arm] = recheck.bio.original.metrics_for(rows, indexed, SCOPE)
                path = scratch / f'seed{seed}_{arm}_predictions.jsonl'
                recheck.bio.write_records(path, records)
                predictions[path.name] = sha256_file(path)

        recheck.verify_files(inputs)
        by_seed = {seed: {
            'multimodal_f1': metrics['multimodal']['f1'],
            'text_f1': metrics['text']['f1'],
            'multimodal_minus_text_f1': metrics['multimodal']['f1'] - metrics['text']['f1'],
        } for seed, metrics in scores.items()}
        deltas = [row['multimodal_minus_text_f1'] for row in by_seed.values()]
        result = {'scope': SCOPE, 'samples_per_seed': len(rows), 'by_seed': by_seed,
                  'mean_multimodal_minus_text_f1': statistics.mean(deltas),
                  'positive_seed_count': sum(delta > 0 for delta in deltas),
                  'sample_std_of_seed_deltas': statistics.stdev(deltas)}
        train.WRITE_JSON(scratch / 'evaluation_report.json', {
            'schema_version': 'agvs-official-test-supplement-v1', **result,
            'previously_used_official_test': True, 'source_ids': recheck.source_ids(rows),
            'cohort': cohort, 'recheck': binding, 'best_epochs': epochs, 'metrics': scores,
            'runtime': runtime, 'input_sha256': inputs, 'output_predictions_sha256': predictions,
            'new_checkpoint_selection': False, 'optimizer_steps': 0,
        })
        scratch.rename(OUTPUT)
    print(json.dumps({**result, 'report': str((OUTPUT / 'evaluation_report.json').resolve())},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
