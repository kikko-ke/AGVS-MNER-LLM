"""Freeze the successful Qwen word-feature pilot and replay it on the grouped holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file, verify_internal_manifest
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import run_agvs_llm_word_features as pilot
from scripts import train_bio_joint_recheck as train

ENCODERS = ('bert', 'qwen')
COEFFICIENTS = {'bert': 0.5, 'qwen': 1.0}
HOLDOUT_HEADS = pilot.ROOT / 'frozen_holdout_heads'
OUTPUT = pilot.ROOT / 'frozen_holdout_evaluation'
SCOPE = 'frozen word-feature pilot on the previously scored grouped holdout'


def selected_coefficients(report):
    if (report.get('schema_version') != 'agvs-llm-word-fusion-dev-v1'
            or report.get('passes_development_gate') is not True
            or report.get('minimum_gain') != pilot.MIN_GAIN
            or report.get('grouped_holdout_scored') is not False
            or report.get('official_val_or_test_read') is not False):
        raise ValueError('Expected the development pilot that passed before holdout evaluation')
    if set(report['trials']) != set(ENCODERS):
        raise ValueError('Expected the two matched development arms')
    best = {}
    for name in ENCODERS:
        trials = report['trials'][name]
        if tuple(item['coefficient'] for item in trials) != pilot.COEFFICIENTS:
            raise ValueError('Unexpected fusion-weight search in the development report')
        best[name] = max(trials, key=lambda item: item['metrics']['f1'])
    coefficients = {name: item['coefficient'] for name, item in best.items()}
    if (coefficients != COEFFICIENTS or report['best_coefficients'] != coefficients
            or report['baseline_metrics']['f1'] != report['agvs_f1']
            or best['bert']['metrics']['f1'] != report['agvs_bert_head_f1']
            or best['qwen']['metrics']['f1'] != report['agvs_qwen_head_f1']
            or report['agvs_qwen_head_f1'] < max(report['agvs_f1'], report['agvs_bert_head_f1']) + pilot.MIN_GAIN):
        raise ValueError('The selected coefficients and scores differ from the successful development run')
    return coefficients


def frozen_inputs():
    prepared, full = train.load_prepared(train.split.DATA)
    fit, holdout, binding = recheck.evaluation_plan(prepared, full)
    config = train.baseline_config(prepared)
    dev_report = train.READ_JSON(pilot.REPORT)
    manifest_path = pilot.HEADS / 'manifest.json'
    manifest = train.READ_JSON(manifest_path)
    coefficients = selected_coefficients(dev_report)
    if (manifest.get('schema_version') != pilot.HEADS_SCHEMA
            or manifest.get('runner_sha256') != sha256_file(pilot.__file__)
            or manifest.get('recheck') != binding or dev_report.get('recheck') != binding
            or dev_report.get('heads_manifest_sha256') != sha256_file(manifest_path)
            or manifest.get('labels') != list(pilot.CODES)
            or manifest.get('fit_source_ids') != recheck.source_ids(fit['train'])
            or manifest.get('dev_source_ids') != recheck.source_ids(fit['dev'])):
        raise ValueError('Training heads and development report do not bind the current fixed split')
    inputs = {str(train.split.OUTPUT): train.SPLIT_SHA256,
              str(pilot.REPORT): sha256_file(pilot.REPORT), str(manifest_path): sha256_file(manifest_path),
              str(Path(__file__).resolve()): sha256_file(__file__)}
    for name in ENCODERS:
        file = pilot.HEADS / f'{name}.pt'
        inputs[str(file)] = manifest['outputs_sha256'][file.name]
    recheck.verify_files(inputs)
    freeze = {'recheck': binding, 'coefficients': coefficients, 'source_ids': recheck.source_ids(holdout),
              'input_sha256': inputs}
    return prepared, fit, holdout, config, manifest, freeze


def head_logits(features, state, device='cuda:0'):
    """Rebuild the serialized pilot head without changing the hash-bound training script."""
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

    head = nn.ModuleDict({
        'norm': nn.LayerNorm(pilot.WIDTH),
        'context': nn.GRU(pilot.WIDTH, 128, batch_first=True, bidirectional=True),
        'out': nn.Linear(256, len(pilot.CODES)),
    }).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(features), pilot.BATCH_SIZE):
            sequences = features[start:start + pilot.BATCH_SIZE]
            lengths = [len(item) for item in sequences]
            words = pad_sequence(sequences, batch_first=True).to(device, torch.float32)
            packed = pack_padded_sequence(head['norm'](words), lengths, batch_first=True, enforce_sorted=False)
            encoded, _ = head['context'](packed)
            context, _ = pad_packed_sequence(encoded, batch_first=True)
            output = head['out'](context)
            logits.extend(row[:length].cpu().clone() for row, length in zip(output, lengths))
    return logits


def prepare_heads():
    if HOLDOUT_HEADS.exists():
        print('Reusing existing frozen holdout scores:', HOLDOUT_HEADS, flush=True)
        return
    import torch
    from modules import vlm_training as backend

    runtime = backend.check_runtime()
    _, fit, holdout, config, manifest, freeze = frozen_inputs()
    if manifest['runtime'] != runtime:
        raise ValueError('Use the same pinned encoder runtime as the development pilot')
    qwen, bert = Path(train.BIO_MODEL), Path(config['bert_model'])
    if backend.model_binding(qwen) != manifest['qwen_weights']:
        raise ValueError('Frozen Qwen weights differ from the development run')
    recheck.verify_files(manifest['bert_files'])

    dev = fit['dev']
    remainder = len(fit['train']) % pilot.BATCH_SIZE
    prefix = fit['train'][-remainder:] if remainder else []
    dev_end = len(prefix) + len(dev)
    if dev_end % pilot.BATCH_SIZE:
        raise ValueError('The fixed feature batches no longer reproduce the saved dev extraction')
    with TemporaryDirectory(dir=pilot.ROOT, prefix='.frozen-holdout-heads-') as directory:
        scratch = Path(directory)
        hashes, replay = {}, {}
        for name, location in (('bert', bert), ('qwen', qwen)):
            bundle = torch.load(pilot.HEADS / f'{name}.pt', map_location='cpu', weights_only=True)
            print(f'{name}: extracting frozen dev-replay and holdout features...', flush=True)
            features, aligned = pilot.extract(prefix + dev + holdout, location, name)
            actual = head_logits(features[len(prefix):dev_end], bundle['head'])
            if not (len(actual) == len(dev) == len(bundle['dev_logits']) == len(bundle['dev_alignment'])):
                raise ValueError('Frozen dev replay does not cover every original row')
            largest = 0.
            for row, new, old, valid, expected in zip(
                    dev, actual, bundle['dev_logits'], aligned[len(prefix):dev_end], bundle['dev_alignment']):
                if (not torch.equal(valid, expected) or new.shape != old.shape
                        or not torch.allclose(new, old, atol=0.001, rtol=0.0001)):
                    raise ValueError(f'{name} saved dev word scores did not replay at source_id={row["source_id"]}')
                largest = max(largest, float((new - old).abs().max()))
            replay[name] = {'passed': True, 'maximum_absolute_logit_difference': largest}
            path = scratch / f'{name}.pt'
            torch.save({'dev_logits': head_logits(features[dev_end:], bundle['head']),
                        'dev_alignment': aligned[dev_end:]}, path)
            hashes[path.name] = sha256_file(path)
            del features, aligned, actual, bundle
        if backend.model_binding(qwen) != manifest['qwen_weights']:
            raise ValueError('Qwen files changed during inference')
        recheck.verify_files(manifest['bert_files'])
        recheck.verify_files(freeze['input_sha256'])
        train.WRITE_JSON(scratch / 'manifest.json', {
            'schema_version': 'frozen-agvs-word-head-holdout-v1', **freeze,
            'dev_logits_replay': replay, 'outputs_sha256': hashes, 'optimizer_steps': 0,
        })
        scratch.rename(HOLDOUT_HEADS)
    print('Frozen holdout scores:', HOLDOUT_HEADS, flush=True)


def evaluate():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing holdout evaluation: {OUTPUT}')
    runtime = recheck.baseline.check_baseline_runtime()
    import torch
    from torchcrf import CRF
    from modules.dataset import MultimodalInferenceDataset, MultimodalNerDataset
    from scripts import evaluate_crf_ensemble as ensemble

    prepared, _, holdout, config, _, freeze = frozen_inputs()
    binding = freeze['recheck']
    seal = recheck.baseline_seal(binding)
    if seal.get('runtime') != runtime or seal.get('best_epoch') != 11:
        raise ValueError('Expected the original frozen AGVS epoch-11 checkpoint and runtime')
    base_report, archived = recheck.load_baseline_predictions(holdout, binding)
    manifest_path = HOLDOUT_HEADS / 'manifest.json'
    manifest = train.READ_JSON(manifest_path)
    if (manifest.get('schema_version') != 'frozen-agvs-word-head-holdout-v1'
            or any(manifest.get(key) != value for key, value in freeze.items())
            or set(manifest['dev_logits_replay']) != set(ENCODERS)
            or any(item.get('passed') is not True for item in manifest['dev_logits_replay'].values())
            or manifest.get('optimizer_steps') != 0):
        raise ValueError('Expected frozen holdout logits from the same development heads and weights')
    inputs = {**freeze['input_sha256'], **seal['input_sha256'],
              str(manifest_path): sha256_file(manifest_path),
              str(recheck.BASELINE_OUTPUT / 'baseline_predictions.jsonl'): base_report['predictions_sha256']}
    for name in ENCODERS:
        file = HOLDOUT_HEADS / f'{name}.pt'
        inputs[str(file)] = manifest['outputs_sha256'][file.name]
    recheck.verify_files(inputs)

    original = verify_internal_manifest(prepared['internal_manifest'], 'MI')
    lines = Path(original['source']).read_text(encoding='utf-8').splitlines()
    with TemporaryDirectory(dir=pilot.ROOT, prefix='.frozen-holdout-evaluation-') as directory:
        scratch = Path(directory)
        source = scratch / 'MNER-MI_val.txt'
        with source.open('x', encoding='utf-8') as destination:
            for row in holdout:
                destination.write(lines[row['source_id']] + '\n')
        args = SimpleNamespace(**{**config, 'data_path': str(scratch)})
        data = MultimodalNerDataset(args, source.name)
        structure = MultimodalInferenceDataset(args, source.name)
        if data.examples['words'] != [row['tokens'] for row in holdout]:
            raise ValueError('AGVS input ordering differs from the frozen cohort')
        recheck.baseline.audit_images(data, prepared)
        labels = data.processor.get_label_mapping(include_subword_label=True)
        cli = SimpleNamespace(device=args.device, batch_size=args.batch_size, num_workers=0)
        native = ensemble._collect(Path(seal['checkpoint']), config, data, cli, labels, 'frozen AGVS')
        mappings = [structure.token_structure(i) for i in range(len(holdout))]
        lengths = native['mask'].sum(1).tolist()
        source.unlink()
        decoder = CRF(len(labels), batch_first=True)
        decoder.load_state_dict(native['crf'])

        def project(paths):
            if len(paths) != len(holdout):
                raise ValueError('AGVS decoder must produce one path per holdout row')
            records = {}
            for row, path, (mask, _, indices), length in zip(holdout, paths, mappings, lengths):
                record, _, _ = recheck.baseline.project_path(row, path, mask.tolist(), indices, labels, length)
                records[row['source_id']] = record
            return records

        baseline = project(native['native_paths'])
        if baseline != archived:
            raise ValueError('Original AGVS predictions differ from the archived grouped holdout')
        records = {'agvs': baseline}
        for name in ENCODERS:
            bundle = torch.load(HOLDOUT_HEADS / f'{name}.pt', map_location='cpu', weights_only=True)
            residual = pilot.word_residual(native['emissions'], mappings, lengths, holdout, bundle, labels)
            with torch.inference_mode():
                paths = decoder.decode(native['emissions'] + freeze['coefficients'][name] * residual,
                                       mask=native['mask'].bool())
            records[f'agvs_{name}_head'] = project(paths)
        score = recheck.bio.original.metrics_for
        scores = {name: score(holdout, predictions, SCOPE) for name, predictions in records.items()}
        full = scores['agvs_qwen_head']['f1']
        base = scores['agvs']['f1']
        bert = scores['agvs_bert_head']['f1']
        result = {'scope': SCOPE, 'samples': len(holdout), 'coefficients': freeze['coefficients'],
                  'f1': {name: item['f1'] for name, item in scores.items()},
                  'qwen_minus_agvs_f1': full - base, 'qwen_minus_bert_control_f1': full - bert,
                  'minimum_gain': pilot.MIN_GAIN,
                  'passes_grouped_holdout_gate': full >= max(base, bert) + pilot.MIN_GAIN}
        hashes = {}
        for name, indexed in records.items():
            path = scratch / f'{name}_predictions.jsonl'
            recheck.bio.write_records(path, indexed.values())
            hashes[path.name] = sha256_file(path)
        recheck.verify_files(inputs)
        train.WRITE_JSON(scratch / 'evaluation_report.json', {
            'schema_version': 'agvs-frozen-llm-word-fusion-holdout-v1', **result,
            'recheck': binding, 'source_ids': freeze['source_ids'], 'metrics': scores,
            'agvs_best_epoch': seal['best_epoch'], 'agvs_replayed_exactly': True,
            'dev_logits_replay': manifest['dev_logits_replay'], 'input_sha256': inputs,
            'output_predictions_sha256': hashes, 'runtime': runtime, 'optimizer_steps': 0,
            'weights_selected_on_holdout': False, 'official_val_or_test_read': False,
        })
        scratch.rename(OUTPUT)
    print(json.dumps({**result, 'report': str((OUTPUT / 'evaluation_report.json').resolve())},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'evaluate'))
    if parser.parse_args().stage == 'prepare':
        prepare_heads()
    else:
        evaluate()


if __name__ == '__main__':
    main()
