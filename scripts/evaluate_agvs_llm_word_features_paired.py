"""Repeat the frozen BERT/Qwen feature comparison for two additional AGVS seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from scripts import evaluate_agvs_llm_word_features_holdout as frozen
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import run_agvs_llm_word_features as pilot
from scripts import run_agvs_paired_seed_recheck as paired
from scripts import train_bio_joint_recheck as train

SEEDS = (42, 2026)
ARMS = ('agvs', 'agvs_bert_head', 'agvs_qwen_head')
ROOT = pilot.ROOT / 'paired_seed_holdout_v1'
HEADS = ROOT / 'heads'
OUTPUT = ROOT / 'evaluation'
SUMMARY_SCOPE = 'three paired seeds on the previously scored grouped holdout; fusion weights frozen on seed 1234 dev'


def protocol():
    prepared, fit, holdout, config, manifest, freeze = frozen.frozen_inputs()
    binding = {**freeze, 'seeds': list(SEEDS), 'fit_source_ids': recheck.source_ids(fit['train']),
               'paired_seed_runner_sha256': sha256_file(__file__)}
    return prepared, fit, holdout, config, manifest, binding


def train_seed_head(features, rows, count, seed, device='cuda:0'):
    # The original extraction projection stays fixed. Only head initialization and
    # training order are seeded here, after the shared frozen features are built.
    original_seed = pilot.SEED
    try:
        pilot.SEED = seed
        return pilot.train_head(features, rows, count, device=device)
    finally:
        pilot.SEED = original_seed


def prepare_heads():
    if HEADS.exists():
        print('Reusing saved paired-seed heads for verification:', HEADS, flush=True)
        return
    import torch
    from modules import vlm_training as backend

    runtime = backend.check_runtime()
    _, fit, holdout, config, manifest, binding = protocol()
    if manifest['runtime'] != runtime:
        raise ValueError('Use the same pinned encoder runtime as the original Qwen pilot')
    qwen, bert = Path(train.BIO_MODEL), Path(config['bert_model'])
    if backend.model_binding(qwen) != manifest['qwen_weights']:
        raise ValueError('Expected the frozen Qwen weights from the original pilot')
    recheck.verify_files(manifest['bert_files'])
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = fit['train']
    with TemporaryDirectory(dir=ROOT, prefix='.paired-seed-heads-') as directory:
        scratch = Path(directory)
        hashes, coverage = {}, {}
        for name, location in (('bert', bert), ('qwen', qwen)):
            features, valid = pilot.extract(rows + holdout, location, name)
            coverage[name] = sum(int(mask.sum()) for mask in valid[len(rows):])
            for seed in SEEDS:
                print(f'Fitting {name} word head for seed {seed} on {len(rows)} training rows...', flush=True)
                logits, state = train_seed_head(features, rows, len(holdout), seed)
                path = scratch / f'seed{seed}_{name}.pt'
                torch.save({'dev_logits': logits, 'dev_alignment': valid[len(rows):], 'head': state}, path)
                hashes[path.name] = sha256_file(path)
            del features, valid, logits, state
        if backend.model_binding(qwen) != manifest['qwen_weights']:
            raise ValueError('Qwen weights changed during paired head training')
        recheck.verify_files(manifest['bert_files'])
        recheck.verify_files(binding['input_sha256'])
        train.WRITE_JSON(scratch / 'manifest.json', {
            'schema_version': 'agvs-llm-word-paired-heads-v1', **binding, 'outputs_sha256': hashes,
            'holdout_aligned_words': coverage, 'official_val_or_test_read': False,
        })
        scratch.rename(HEADS)
    print('Paired-seed heads:', HEADS, flush=True)


def original_seed_scores(holdout, binding):
    path = frozen.OUTPUT / 'evaluation_report.json'
    report = train.READ_JSON(path)
    if (report.get('schema_version') != 'agvs-frozen-llm-word-fusion-holdout-v1'
            or report.get('recheck') != binding['recheck'] or report.get('source_ids') != binding['source_ids']
            or report.get('coefficients') != binding['coefficients']
            or report.get('weights_selected_on_holdout') is not False
            or report.get('passes_grouped_holdout_gate') is not True):
        raise ValueError('Expected the archived seed-1234 frozen comparison')
    files = {str(path): sha256_file(path)}
    for arm in ARMS:
        predictions = path.parent / f'{arm}_predictions.jsonl'
        files[str(predictions)] = report['output_predictions_sha256'][predictions.name]
    recheck.verify_files(files)
    scores = {arm: recheck.bio.original.metrics_for(
        holdout, paired.checked_predictions(holdout, path.parent / f'{arm}_predictions.jsonl'), frozen.SCOPE)
        for arm in ARMS}
    if scores != report['metrics']:
        raise ValueError('Seed-1234 predictions failed score reproduction')
    return scores, files


def baseline_archive(seed, holdout, binding):
    directory = paired.seed_dir(seed) / 'holdout'
    path = directory / 'evaluation_report.json'
    report = train.READ_JSON(path)
    if (report.get('schema_version') != paired.EVALUATION or report.get('seed') != seed
            or report.get('recheck') != binding['recheck'] or report.get('source_ids') != binding['source_ids']
            or report.get('new_checkpoint_selection') is not False):
        raise ValueError(f'Expected the archived AGVS baseline for seed {seed}')
    predictions = directory / 'multimodal_predictions.jsonl'
    files = {str(path): sha256_file(path),
             str(predictions): report['output_predictions_sha256'][predictions.name]}
    recheck.verify_files(files)
    indexed = paired.checked_predictions(holdout, predictions)
    if recheck.bio.original.metrics_for(holdout, indexed, paired.SCOPE) != report['metrics']['multimodal']:
        raise ValueError(f'Archived AGVS predictions failed score reproduction for seed {seed}')
    return indexed, files


def score_seed(seed, native, mappings, holdout, labels, archived, coefficients):
    import torch
    from torchcrf import CRF

    lengths = native['mask'].sum(1).tolist()
    decoder = CRF(len(labels), batch_first=True)
    decoder.load_state_dict(native['crf'])

    def project(paths):
        if len(paths) != len(holdout):
            raise ValueError('Expected one decoded path per holdout sample')
        records = {}
        for row, path, (mask, _, indices), length in zip(holdout, paths, mappings, lengths):
            record, _, _ = recheck.baseline.project_path(row, path, mask.tolist(), indices, labels, length)
            records[row['source_id']] = record
        return records

    baseline = project(native['native_paths'])
    if baseline != archived:
        raise ValueError(f'Frozen AGVS output did not replay exactly for seed {seed}')
    records = {'agvs': baseline}
    for name in frozen.ENCODERS:
        bundle = torch.load(HEADS / f'seed{seed}_{name}.pt', map_location='cpu', weights_only=True)
        residual = pilot.word_residual(native['emissions'], mappings, lengths, holdout, bundle, labels)
        with torch.inference_mode():
            paths = decoder.decode(native['emissions'] + coefficients[name] * residual,
                                   mask=native['mask'].bool())
        records[f'agvs_{name}_head'] = project(paths)
    return records


def summarize(scores):
    by_seed = {}
    for seed, arms in scores.items():
        f1 = {name: arms[name]['f1'] for name in ARMS}
        by_seed[str(seed)] = {**f1, 'qwen_minus_agvs_f1': f1['agvs_qwen_head'] - f1['agvs'],
                             'qwen_minus_bert_f1': f1['agvs_qwen_head'] - f1['agvs_bert_head']}
    base = [row['qwen_minus_agvs_f1'] for row in by_seed.values()]
    bert = [row['qwen_minus_bert_f1'] for row in by_seed.values()]
    return {'by_seed': by_seed, 'mean_qwen_minus_agvs_f1': statistics.mean(base),
            'mean_qwen_minus_bert_f1': statistics.mean(bert),
            'positive_seeds_vs_agvs': sum(value > 0 for value in base),
            'positive_seeds_vs_bert': sum(value > 0 for value in bert),
            'seeds_with_both_gains_at_least_0_003': sum(
                min(a, b) + 1e-12 >= pilot.MIN_GAIN for a, b in zip(base, bert))}


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing paired-seed report: {OUTPUT}')
    runtime = recheck.baseline.check_baseline_runtime()
    from modules.dataset import MultimodalInferenceDataset, MultimodalNerDataset
    from scripts import evaluate_crf_ensemble as ensemble

    prepared, _, holdout, config, _, binding = protocol()
    path = HEADS / 'manifest.json'
    manifest = train.READ_JSON(path)
    if (manifest.get('schema_version') != 'agvs-llm-word-paired-heads-v1'
            or any(manifest.get(key) != value for key, value in binding.items())):
        raise ValueError('Paired heads do not bind the frozen coefficients and sample IDs')
    inputs = {**binding['input_sha256'], str(path): sha256_file(path)}
    for seed in SEEDS:
        for name in frozen.ENCODERS:
            file = HEADS / f'seed{seed}_{name}.pt'
            inputs[str(file)] = manifest['outputs_sha256'][file.name]
    recheck.verify_files(inputs)
    first, prior_files = original_seed_scores(holdout, binding)
    inputs.update(prior_files)
    scores, hashes = {1234: first}, {}
    original = verify_internal_manifest(prepared['internal_manifest'], 'MI')
    lines = Path(original['source']).read_text(encoding='utf-8').splitlines()
    with TemporaryDirectory(dir=ROOT, prefix='.paired-seed-evaluation-') as directory:
        scratch = Path(directory)
        data_path = scratch / 'MNER-MI_val.txt'
        with data_path.open('x', encoding='utf-8') as destination:
            for row in holdout:
                destination.write(lines[row['source_id']] + '\n')
        args = SimpleNamespace(**{**config, 'data_path': str(scratch)})
        data = MultimodalNerDataset(args, data_path.name)
        structure = MultimodalInferenceDataset(args, data_path.name)
        if data.examples['words'] != [row['tokens'] for row in holdout]:
            raise ValueError('AGVS input words differ from the paired cohort')
        recheck.baseline.audit_images(data, prepared)
        mappings = [structure.token_structure(i) for i in range(len(holdout))]
        labels = data.processor.get_label_mapping(include_subword_label=True)
        cli = SimpleNamespace(device=args.device, batch_size=args.batch_size, num_workers=0)
        for seed in SEEDS:
            _, checkpoint, saved, artifacts = paired.read_arm(seed, 'multimodal', runtime, binding['recheck'], config)
            archived, archive_files = baseline_archive(seed, holdout, binding)
            inputs.update(artifacts)
            inputs.update(archive_files)
            native = ensemble._collect(checkpoint, saved, data, cli, labels, f'AGVS seed {seed}')
            records = score_seed(seed, native, mappings, holdout, labels, archived, binding['coefficients'])
            scores[seed] = {arm: recheck.bio.original.metrics_for(holdout, values, frozen.SCOPE)
                            for arm, values in records.items()}
            for arm, values in records.items():
                path = scratch / f'seed{seed}_{arm}_predictions.jsonl'
                recheck.bio.write_records(path, values.values())
                hashes[path.name] = sha256_file(path)
        data_path.unlink()
        recheck.verify_files(inputs)
        result = {'scope': SUMMARY_SCOPE, 'samples_per_seed': len(holdout),
                  'coefficients': binding['coefficients'], **summarize(scores)}
        train.WRITE_JSON(scratch / 'evaluation_report.json', {
            'schema_version': 'agvs-llm-word-paired-seeds-v1', **result, 'recheck': binding['recheck'],
            'source_ids': binding['source_ids'], 'metrics': scores, 'input_sha256': inputs,
            'output_predictions_sha256': hashes, 'runtime': runtime,
            'weights_selected_on_holdout': False, 'official_val_or_test_read': False,
        })
        scratch.rename(OUTPUT)
    print(json.dumps({**result, 'report': str((OUTPUT / 'evaluation_report.json').resolve())},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('heads', 'evaluate'))
    if parser.parse_args().stage == 'heads':
        prepare_heads()
    else:
        evaluate()


if __name__ == '__main__':
    main()
