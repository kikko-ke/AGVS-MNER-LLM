"""Frozen-adapter image-bundle permutation with equal per-sample image counts.

Pairs are fixed from IDs, image counts and hashes only. Entire unmatchable count
buckets are excluded and disclosed; all three conditions use the same cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from modules.vlm_ner import LABELS
from modules.vlm_training import encode
from scripts import evaluate_vlm_no_images as removal

original = removal.original
SEED = 20261104


def rank(role, number, source_id):
    return hashlib.sha256(f'{SEED}:{role}:{number}:{source_id}'.encode()).hexdigest()


def match_bucket(rows):
    """Deterministic bipartite matching; no recursion, labels or model outputs."""
    count = len(rows[0]['images'])
    hashes = {row['source_id']: {image['sha256'] for image in row['images']} for row in rows}
    recipients = sorted(hashes, key=lambda sid: rank('recipient', count, sid))
    donors = sorted(hashes, key=lambda sid: rank('donor', count, sid))
    assigned, owner = {}, {}
    for start in recipients:
        queue, seen, parent, free = deque([start]), {start}, {}, None
        while queue and free is None:
            recipient = queue.popleft()
            for donor in donors:
                if donor in parent or donor == recipient or not hashes[recipient].isdisjoint(hashes[donor]):
                    continue
                parent[donor] = recipient
                if donor not in owner:
                    free = donor
                    break
                if owner[donor] not in seen:
                    seen.add(owner[donor])
                    queue.append(owner[donor])
        if free is None:
            return None
        while free is not None:
            recipient = parent[free]
            old = assigned.get(recipient)
            assigned[recipient], owner[free] = free, recipient
            free = old
    return assigned


def make_plan(rows, panel_ids=frozenset()):
    index = {row['source_id']: row for row in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate source IDs in permutation inputs')
    buckets, mapping, skipped = {}, {}, []
    for row in rows:
        count = len(row['images'])
        if not 0 <= count <= 4:
            raise ValueError('Expected the original zero-to-four image budget')
        partition = 'dev_panel' if row['source_id'] in panel_ids else 'dev_nonpanel'
        buckets.setdefault((partition, count), []).append(row)
    for (partition, count), values in sorted(buckets.items()):
        if count == 0:
            skipped.append({'partition': partition, 'images_per_sample': 0,
                            'samples': len(values), 'reason': 'no_images_to_permute'})
            continue
        assignment = match_bucket(values)
        if assignment is None:
            skipped.append({'partition': partition, 'images_per_sample': count, 'samples': len(values),
                            'reason': 'no_complete_hash_disjoint_permutation'})
        else:
            mapping.update(assignment)
    plan = []
    for source_id in sorted(mapping):
        donor_id = mapping[source_id]
        own, donor = index[source_id]['images'], index[donor_id]['images']
        if (source_id == donor_id or (source_id in panel_ids) != (donor_id in panel_ids) or len(own) != len(donor)
                or {image['sha256'] for image in own} & {image['sha256'] for image in donor}):
            raise ValueError('Invalid counterfactual image assignment')
        plan.append({'source_id': source_id, 'donor_source_id': donor_id,
                     'partition': 'dev_panel' if source_id in panel_ids else 'dev_nonpanel',
                     'original_images': own, 'replacement_images': donor})
    if set(mapping.values()) != set(mapping):
        raise ValueError('Each eligible image bundle must be used exactly once')
    return plan, skipped


def load_controls(args):
    pilot, groups, panel, matched, reference, run_bindings, bindings = removal.load_reference(args)
    directory = Path(args.no_image_evaluation)
    report_path, predictions_path = directory / 'ablation_report.json', directory / 'no_image_predictions.jsonl'
    report = original.read_json(report_path)
    if (report.get('schema_version') != 'vlm-no-image-ablation-v1'
            or report.get('training_profile') != pilot['profile'] or report.get('best_epoch') != pilot['best_epoch']
            or report.get('source_ids') != reference['source_ids'] or report.get('input_sha256') != bindings
            or report.get('adapter_files_sha256') != run_bindings['adapter_files_sha256']
            or report.get('panel_reproduction', {}).get('passed') is not True
            or report.get('zero_image_reproduction', {}).get('passed') is not True
            or report.get('intervention') != {'inference_input_mode': 'text', 'all_images_removed': True,
                'system_prompt_changed': False, 'word_content_changed': False, 'weights_changed': False}
            or report.get('official_val_or_test_read') is not False
            or report.get('optimizer_steps') != 0 or report.get('new_checkpoint_selection') is not False):
        raise ValueError('Expected the completed no-image evaluation of the identical frozen adapter and inputs')
    no_images = original.checked_records(groups['full_dev'], original.read_records(predictions_path))
    for role, rows in groups.items():
        item = report['comparisons'][role]
        left = original.metrics_for(rows, matched, original.SCOPES[role])
        right = original.metrics_for(rows, no_images, original.SCOPES[role])
        if (item['matched_images'] != left or item['no_images'] != right
                or abs(item['matched_f1_minus_no_images'] - (left['f1'] - right['f1'])) > 1e-12
                or item['by_type_f1_delta'] != {label: left['by_type'][label]['f1'] - right['by_type'][label]['f1']
                                               for label in LABELS}):
            raise ValueError('No-image predictions no longer reproduce the completed ablation metrics')
    empty = {row['source_id']: matched[row['source_id']] for row in groups['full_dev'] if not row['images']}
    if removal.changed_predictions(empty, no_images):
        raise ValueError('Originally image-free inputs disagree across the saved controls')
    bindings = {**bindings, **{str(path.resolve()): sha256_file(path)
                              for path in (report_path, predictions_path, Path(__file__))}}
    return pilot, groups, panel, matched, no_images, run_bindings, bindings


def compare(rows, matched, no_images, mismatched, scope):
    scores = {name: original.metrics_for(rows, values, scope)
              for name, values in (('matched_images', matched), ('no_images', no_images),
                                   ('mismatched_images', mismatched))}
    return {**scores,
            'matched_f1_minus_mismatched': scores['matched_images']['f1'] - scores['mismatched_images']['f1'],
            'matched_f1_minus_no_images': scores['matched_images']['f1'] - scores['no_images']['f1'],
            'by_type_matched_minus_mismatched': {
                label: scores['matched_images']['by_type'][label]['f1'] - scores['mismatched_images']['by_type'][label]['f1']
                for label in LABELS}}


def evaluate(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Mismatch output already exists; preserve the previous result')
    for source in (args.run, args.reference_evaluation, args.reference_audit, args.no_image_evaluation,
                   args.model, Path(args.data).parent):
        protected = Path(source).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Mismatch output must be separate from frozen input directories')
    pilot, groups, panel, matched, no_images, run_bindings, bindings = load_controls(args)
    plan, skipped = make_plan(groups['full_dev'], {row['source_id'] for row in groups['dev_panel']})
    if not plan:
        raise ValueError(f'No eligible image-count bucket can be permuted: {skipped}')
    assignments = {item['source_id']: item for item in plan}
    cohorts = {role: [row for row in rows if row['source_id'] in assignments] for role, rows in groups.items()}
    print(f'Fixed input-only permutation: {len(plan)}/{len(groups["full_dev"])} eligible samples; seed={SEED}', flush=True)
    print('Excluded whole image-count buckets:', skipped, flush=True)
    if args.check_only:
        return {'check_only': True, 'plan': plan, 'skipped_buckets': skipped, 'input_sha256': bindings}
    profile = pilot['profile']
    if original.check_runtime() != profile['versions']:
        raise ValueError('Use the exact runtime versions recorded by the original multi-image run')
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / 'image_permutation.json'
    removal.write_json(plan_path, {'seed': SEED, 'assignments': plan, 'skipped_buckets': skipped})
    bindings[str(plan_path)] = sha256_file(plan_path)
    config = {
        'schema_version': 'vlm-mismatched-image-ablation-v1', 'training_profile': profile,
        'best_epoch': pilot['best_epoch'], 'input_sha256': bindings,
        'adapter_files_sha256': run_bindings['adapter_files_sha256'],
        'source_ids': {role: [row['source_id'] for row in rows] for role, rows in cohorts.items()},
        'original_source_ids': {role: [row['source_id'] for row in rows] for role, rows in groups.items()},
        'permutation_seed': SEED, 'skipped_buckets': skipped,
        'eligible_image_counts': dict(sorted(Counter(len(row['images']) for row in cohorts['full_dev']).items())),
        'eligibility_uses': ['source_id', 'fixed_panel_membership', 'image_count', 'image_sha256'],
        'per_sample_image_count_preserved': True, 'eligible_bundle_multiset_preserved': True,
        'bundles_permuted_within_panel_and_nonpanel_separately': True,
        'per_sample_image_resolution_preserved': False,
        'official_val_or_test_read': False, 'optimizer_steps': 0,
        'new_checkpoint_selection': False, 'llm_api_requests': 0,
    }
    removal.write_json(output / 'config.json', config)
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
        changed = removal.changed_predictions(panel, restored)
        reproduction = {'passed': not changed, 'mismatched_source_ids': changed}
        removal.write_json(output / 'panel_reproduction.json', reproduction)
        if changed:
            raise ValueError('Original matched-image panel does not reproduce; mismatch generation stopped')
        permuted = [{**row, 'images': assignments[row['source_id']]['replacement_images']}
                    for row in cohorts['full_dev']]
        from tqdm.auto import tqdm
        maximum = 0
        for row in tqdm(permuted, desc='Audit mismatched prompt budgets', unit='sample'):
            batch = encode(processor, row, 'multi_image', False, settings[0] - settings[1])
            maximum = max(maximum, batch['input_ids'].shape[1])
            del batch
        original.generate_evaluation(model, processor, permuted, 'multi_image', *settings,
                                     output / 'mismatched_predictions.jsonl')
        mismatched = original.checked_records(permuted, original.read_records(output / 'mismatched_predictions.jsonl'))
        comparisons = {role: compare(rows, matched, no_images, mismatched,
                        original.SCOPES[role] + '; input-only eligible image-count buckets')
                       for role, rows in cohorts.items()}
        for path, expected in bindings.items():
            if sha256_file(path) != expected:
                raise ValueError(f'Frozen input or permutation changed during evaluation: {path}')
        if original.model_binding(Path(args.run) / 'best_adapter') != run_bindings['adapter_files_sha256']:
            raise ValueError('Saved adapter changed during evaluation')
        result = {
            **config, 'comparisons': comparisons, 'panel_reproduction': reproduction,
            'maximum_mismatched_prompt_tokens': maximum,
            'elapsed_seconds': time.monotonic() - started,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
            'note': 'All three conditions are scored on identical input-only eligible subsets. '
                    'Matched and no-image predictions are reused after verification. '
                    'Each eligible bundle is used once, with identical per-recipient image count and '
                    'no shared image SHA256; panel and nonpanel bundles never mix. Recipient texts and gold are unchanged. '
                    'Individual image resolutions/visual token counts may change; only their cohort '
                    'bundle distribution is preserved. A different source is not proof of semantic '
                    'irrelevance. Single fixed permutation and development data; no significance or '
                    'AGVS component claim. No model or checkpoint is selected from these scores.',
        }
        removal.write_json(output / 'mismatch_report.json', result)
        for role, values in comparisons.items():
            print(f"eligible_{role}: samples={len(cohorts[role])}/{len(groups[role])} "
                  f"matched={values['matched_images']['f1']:.6f} "
                  f"no_images={values['no_images']['f1']:.6f} "
                  f"mismatched={values['mismatched_images']['f1']:.6f} "
                  f"matched-minus-mismatched={values['matched_f1_minus_mismatched']:+.6f}", flush=True)
            if role == 'full_dev':
                for label, delta in values['by_type_matched_minus_mismatched'].items():
                    print(f'eligible_{role}/{label}: matched-minus-mismatched={delta:+.6f}', flush=True)
        print('Report:', output / 'mismatch_report.json', flush=True)
        return result
    except Exception as error:
        removal.write_json(output / 'failure.json', {'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs_vlm_gold/mi_multi_image_seed1234')
    parser.add_argument('--reference-evaluation', default='outputs_vlm_gold/mi_multi_image_seed1234_dev_evaluation')
    parser.add_argument('--reference-audit', default='outputs_vlm_gold/baseline_qlora_error_audit')
    parser.add_argument('--no-image-evaluation', default='outputs_vlm_gold/mi_multi_image_seed1234_no_images')
    parser.add_argument('--model', default='/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--output', default='outputs_vlm_gold/mi_multi_image_seed1234_mismatched_images')
    parser.add_argument('--check-only', action='store_true', help='Verify controls and permutation without model/GPU imports')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
