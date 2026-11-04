"""Fixed 25% train-only image-bundle dropout: smoke, train, frozen evaluation.

Uses the original two-epoch multi-image pilot as a hash-bound control. The
control code and checkpoints remain unchanged; this run starts from the base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from modules import vlm_training as backend
from scripts import evaluate_vlm_ner as original

DROP_SEED = 20261105
PROFILE_SCHEMA = 'vlm-image-dropout-profile-v1'
SMOKE_SCHEMA = 'vlm-image-dropout-smoke-v1'
PILOT_SCHEMA = 'vlm-image-dropout-pilot-v1'
TRAIN_DIR = 'outputs_vlm_gold/mi_image_drop25_seed1234'
SMOKE_DIR = 'outputs_vlm_gold/mi_image_drop25_smoke'
BASE_SETTINGS = {
    'input_mode': 'multi_image', 'seed': 1234, 'min_pixels': 4*28*28, 'max_pixels': 128*28*28,
    'max_length': 2048, 'max_new_tokens': 256, 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': .05,
    'quantization': 'nf4-double-quant-bf16', 'batch_size': 1, 'epochs': 2,
    'gradient_accumulation': 8, 'lr': 1e-4, 'warmup_ratio': .03, 'weight_decay': .01,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def should_drop(source_id, epoch):
    if type(source_id) is not int or source_id < 0 or epoch not in (1, 2):
        raise ValueError('Expected a source ID and training epoch 1 or 2')
    digest = hashlib.sha256(f'{DROP_SEED}:{epoch}:{source_id}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') < 2**62


def training_view(row, dropped):
    return {**row, 'images': []} if dropped else row


def dropout_plan(rows):
    ids = [row['source_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training source IDs')
    return {'selection_seed': DROP_SEED, 'probability': .25, 'train_source_ids': ids,
            'dropped_source_ids': {str(epoch): [sid for sid in ids if should_drop(sid, epoch)] for epoch in (1, 2)}}


def load_control(args):
    control, manifest, rows, nonpanel, _, bindings = original.validate_run(
        Path(args.control_run).resolve(), args.model, args.data)
    base = control['profile']
    if any(base.get(name) != value for name, value in BASE_SETTINGS.items()):
        raise ValueError('Control is not the fixed seed1234 two-epoch multi-image pilot')
    profile = {**base, 'schema_version': PROFILE_SCHEMA,
               'control_run': str(Path(args.control_run).resolve()), 'control_bindings': bindings,
               'intervention': {'method': 'train_only_image_bundle_dropout', 'probability': .25,
                   'selection_seed': DROP_SEED, 'rule': 'sha256(seed:epoch:source_id) first uint64 < 2**62',
                   'inference_dropout': False},
               'intervention_implementation_sha256': {'scripts/run_vlm_image_dropout.py': sha256_file(__file__)}}
    return profile, manifest, rows, nonpanel


def protect_output(output, args, extra=()):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; preserve the previous run')
    for source in (args.control_run, args.model, Path(args.data).parent, *extra):
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output must be separate from all input/control directories')
    return output


def check_smoke(path, profile, plan):
    report = original.read_json(path)
    plan_path = Path(path).parent / 'dropout_plan.json'
    if (report.get('schema_version') != SMOKE_SCHEMA or report.get('passed') is not True
            or report.get('profile') != profile or report.get('optimizer_steps') != 8
            or report.get('samples') != 8 or report.get('generation_conditions') != ['original_images', 'no_images']
            or report.get('finite_nonzero_gradients') is not True
            or len(report.get('smoke_views', [])) != 8
            or sum(view['images_removed'] is True for view in report.get('smoke_views', [])) != 4
            or report['smoke_views'][0]['used_image_count'] < 1
            or report.get('heldout_accuracy_evaluated') is not False
            or original.read_json(plan_path) != plan or report.get('dropout_plan_sha256') != sha256_file(plan_path)):
        raise ValueError('A passing image-dropout GPU smoke with identical profile/plan is required')


def run_training(args, profile, manifest, rows):
    plan = dropout_plan(rows['train'])
    if args.stage == 'train':
        check_smoke(args.smoke_report, profile, plan)
    if backend.check_runtime() != profile['versions']:
        raise ValueError('Use the exact pinned runtime of the original control')
    output = protect_output(args.output, args, (Path(args.smoke_report).parent,) if args.stage == 'train' else ())
    output.mkdir(parents=True)
    plan_path = output / 'dropout_plan.json'
    write_json(plan_path, plan)
    write_json(output / 'config.json', {
        'stage': args.stage, 'profile': profile, 'data': str(args.data), 'model': str(args.model),
        'official_val_or_test_read': False, 'llm_api_requests': 0, 'gold_scope': manifest['gold_scope'],
        'panel_source_ids': manifest['panel_source_ids'],
    })
    import torch
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        # Fresh base + fresh LoRA, never the already-trained control adapter.
        model, processor = backend.load_model(args.model, profile['seed'], profile['min_pixels'], profile['max_pixels'])
        selected = backend.smoke_rows(rows['train']) if args.stage == 'smoke' else rows['train']
        if args.stage == 'smoke' and (len(selected) != 8 or not selected[0]['images']):
            raise ValueError('Smoke requires eight training rows, including an original image input')
        accumulation = 1 if args.stage == 'smoke' else profile['gradient_accumulation']
        epochs = 1 if args.stage == 'smoke' else profile['epochs']
        steps = math.ceil(len(selected) / accumulation) * epochs
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=profile['lr'], weight_decay=profile['weight_decay'])
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * profile['warmup_ratio']), steps)
        history, gradients, trace, token_max, best, best_epoch = [], [], [], 0, -1., None
        if args.stage == 'train':
            # Validate both training views before any updates; dev stays original.
            for row in tqdm(rows['train'] + rows['dev_panel'], desc='Audit original token budgets', unit='sample'):
                batch = backend.encode(processor, row, 'multi_image', True, profile['max_length'])
                token_max = max(token_max, batch['input_ids'].shape[1])
                del batch
            for row in tqdm(rows['train'], desc='Audit no-image target masks', unit='sample'):
                batch = backend.encode(processor, training_view(row, True), 'multi_image', True, profile['max_length'])
                del batch
            for row in rows['dev_panel']:
                batch = backend.encode(processor, row, 'multi_image', False, profile['max_length'] - profile['max_new_tokens'])
                del batch
            metrics = backend.generate_evaluation(model, processor, rows['dev_panel'], 'multi_image',
                profile['max_length'], profile['max_new_tokens'], output / 'dev_epoch0.jsonl')
            history.append({'epoch': 0, 'metrics': metrics})
            best, best_epoch = metrics['f1'], 0
            model.save_pretrained(output / 'best_adapter', safe_serialization=True, save_embedding_layers=False)
            processor.save_pretrained(output / 'best_adapter')
            print('Epoch 0 dev panel:', json.dumps(metrics), flush=True)
        for epoch in range(1, epochs + 1):
            backend.train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            order = list(range(8)) if args.stage == 'smoke' else torch.randperm(
                len(selected), generator=torch.Generator().manual_seed(profile['seed'] + epoch)).tolist()
            losses, dropped_ids, used_without_images = [], [], 0
            for index, position in enumerate(tqdm(order, desc=f'Image-dropout epoch {epoch}/{epochs}', unit='sample')):
                row = selected[position]
                # Smoke deliberately exercises four of each view, not a 25% estimate.
                dropped = bool(index % 2) if args.stage == 'smoke' else should_drop(row['source_id'], epoch)
                view = training_view(row, dropped)
                if dropped:
                    dropped_ids.append(row['source_id'])
                used_without_images += not bool(view['images'])
                if args.stage == 'smoke':
                    trace.append({'source_id': row['source_id'], 'images_removed': dropped,
                                  'original_image_count': len(row['images']), 'used_image_count': len(view['images'])})
                batch = backend.move_to_cuda(backend.encode(processor, view, 'multi_image', True, profile['max_length']))
                token_max = max(token_max, batch['input_ids'].shape[1])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(**batch, use_cache=False)
                    loss = result.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at source_id={row["source_id"]}')
                losses.append(float(loss.detach()))
                (loss / backend.accumulation_divisor(index, len(selected), accumulation)).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(selected):
                    gradients.append(backend.optimizer_update(model, optimizer, scheduler))
                if index % 20 == 0 or args.stage == 'smoke':
                    print(f'[step] epoch={epoch} sample={index+1}/{len(selected)} loss={losses[-1]:.4f} '
                          f'images_removed={dropped} peak_allocated_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}', flush=True)
                del batch, result, loss
            if args.stage == 'train':
                if set(dropped_ids) != set(plan['dropped_source_ids'][str(epoch)]):
                    raise RuntimeError('Executed dropout schedule differs from the frozen plan')
                metrics = backend.generate_evaluation(model, processor, rows['dev_panel'], 'multi_image',
                    profile['max_length'], profile['max_new_tokens'], output / f'dev_epoch{epoch}.jsonl')
                history.append({'epoch': epoch, 'mean_train_loss': sum(losses) / len(losses),
                                'images_removed_samples': len(dropped_ids), 'used_without_images': used_without_images,
                                'metrics': metrics})
                if metrics['f1'] > best:
                    best, best_epoch = metrics['f1'], epoch
                    model.save_pretrained(output / 'best_adapter', safe_serialization=True, save_embedding_layers=False)
                write_json(output / 'history.json', history)
                print(f'Epoch {epoch} dev panel:', json.dumps(metrics), flush=True)
        if len(gradients) != steps:
            raise RuntimeError('Unexpected optimizer update budget')
        if args.stage == 'smoke':
            malformed = 0
            for name, removed in (('original_images', False), ('no_images', True)):
                metrics = backend.generate_evaluation(model, processor, [training_view(selected[0], removed)],
                    'multi_image', profile['max_length'], 32, output / f'smoke_{name}.jsonl')
                malformed += metrics['malformed_responses']
            report = {'schema_version': SMOKE_SCHEMA, 'passed': True, 'profile': profile,
                'samples': 8, 'optimizer_steps': len(gradients), 'smoke_views': trace,
                'finite_nonzero_gradients': True, 'generation_conditions': ['original_images', 'no_images'],
                'malformed_smoke_outputs': malformed, 'heldout_accuracy_evaluated': False,
                'GPU': torch.cuda.get_device_name(0),
                'note': 'Eight real updates with four forced views of each kind; training uses hash-based 25% dropout.'}
            name = 'smoke_report.json'
        else:
            report = {'schema_version': PILOT_SCHEMA, 'profile': profile, 'best_epoch': best_epoch,
                'best_dev_panel_f1': best, 'history': history, 'optimizer_steps': len(gradients),
                'best_adapter_sha256': backend.model_binding(output / 'best_adapter'),
                'official_val_or_test_read': False, 'llm_api_requests': 0,
                'note': 'One fixed train-only image-dropout candidate; validation uses original images. No gain claim.'}
            name = 'pilot_report.json'
        report.update({'dropout_plan_sha256': sha256_file(plan_path), 'maximum_tokens': token_max,
                       'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                       'peak_reserved_GiB': torch.cuda.max_memory_reserved()/1024**3,
                       'elapsed_seconds': time.monotonic() - started})
        write_json(output / name, report)
        print('Report:', output / name, flush=True)
        return report
    except Exception as error:
        write_json(output / 'failure.json', {'stage': args.stage, 'error': str(error)})
        raise


def validate_candidate(args, profile, rows):
    run = Path(args.run)
    report, config = original.read_json(run / 'pilot_report.json'), original.read_json(run / 'config.json')
    plan_path = run / 'dropout_plan.json'
    if (report.get('schema_version') != PILOT_SCHEMA or report.get('profile') != profile
            or config.get('stage') != 'train' or config.get('profile') != profile
            or config.get('official_val_or_test_read') is not False or report.get('official_val_or_test_read') is not False
            or report.get('optimizer_steps') != math.ceil(len(rows['train']) / profile['gradient_accumulation']) * profile['epochs']
            or original.read_json(plan_path) != dropout_plan(rows['train'])
            or report.get('dropout_plan_sha256') != sha256_file(plan_path)):
        raise ValueError('Expected the completed hash-bound image-dropout training run')
    history = report['history']
    if [item['epoch'] for item in history] != [0, 1, 2]:
        raise ValueError('Incomplete image-dropout history')
    best = max(history, key=lambda item: item['metrics']['f1'])
    if best['epoch'] != report['best_epoch'] or best['metrics']['f1'] != report['best_dev_panel_f1']:
        raise ValueError('Checkpoint must follow the original-image panel rule, with earliest tie')
    for item in history[1:]:
        if item['images_removed_samples'] != len(dropout_plan(rows['train'])['dropped_source_ids'][str(item['epoch'])]):
            raise ValueError('Reported training dropout counts differ from plan')
    reference_path = run / f'dev_epoch{report["best_epoch"]}.jsonl'
    reference = original.checked_records(rows['dev_panel'], original.read_records(reference_path))
    metrics = original.metrics_for(rows['dev_panel'], reference, original.SCOPES['dev_panel'])
    for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'by_type', 'samples', 'malformed_responses', 'malformed_rate'):
        if metrics[key] != best['metrics'][key]:
            raise ValueError('Best-panel predictions disagree with training report')
    adapter = run / 'best_adapter'
    if not (adapter / 'adapter_config.json').is_file() or not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('Saved adapter is incomplete')
    if backend.model_binding(adapter) != report.get('best_adapter_sha256'):
        raise ValueError('Saved adapter differs from the completed training report')
    bindings = {str(path.resolve()): sha256_file(path) for path in
                (run / 'pilot_report.json', run / 'config.json', plan_path, reference_path)}
    return report, reference, bindings


def evaluate_candidate(args, profile, manifest, rows, nonpanel):
    output = protect_output(args.output, args, (args.run,))
    report, reference, bindings = validate_candidate(args, profile, rows)
    adapter = Path(args.run) / 'best_adapter'
    adapter_binding = backend.model_binding(adapter)
    if backend.check_runtime() != profile['versions']:
        raise ValueError('Use the exact pinned runtime of the original control')
    output.mkdir(parents=True)
    config = {'schema_version': 'vlm-image-dropout-dev-evaluation-v1', 'profile': profile,
        'best_epoch': report['best_epoch'], 'input_sha256': bindings, 'adapter_files_sha256': adapter_binding,
        'official_val_or_test_read': False, 'optimizer_steps': 0, 'new_checkpoint_selection': False,
        'source_ids': {'full_dev': [r['source_id'] for r in rows['dev']], 'dev_panel': manifest['panel_source_ids'],
                       'dev_nonpanel': [r['source_id'] for r in nonpanel]}}
    write_json(output / 'config.json', config)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        model, processor = original.load_frozen_adapter(args.model, adapter, profile)
        settings = ('multi_image', profile['max_length'], profile['max_new_tokens'])
        backend.generate_evaluation(model, processor, rows['dev_panel'], *settings, output / 'panel_predictions.jsonl')
        panel = original.checked_records(rows['dev_panel'], original.read_records(output / 'panel_predictions.jsonl'))
        changed = [sid for sid in panel if panel[sid]['predicted'] != reference[sid]['predicted']
                   or (panel[sid]['error'] is None) != (reference[sid]['error'] is None)]
        reproduction = {'passed': not changed, 'mismatched_source_ids': changed}
        write_json(output / 'panel_reproduction.json', reproduction)
        if changed:
            raise ValueError('Candidate original-image panel does not reproduce; full-dev evaluation stopped')
        print('Panel reproduced. Evaluating remaining internal dev with original images...', flush=True)
        backend.generate_evaluation(model, processor, nonpanel, *settings, output / 'nonpanel_predictions.jsonl')
        full = {**panel, **original.checked_records(nonpanel, original.read_records(output / 'nonpanel_predictions.jsonl'))}
        ordered = [full[row['source_id']] for row in rows['dev']]
        original.checked_records(rows['dev'], ordered)
        predictions_path = output / 'dev_predictions.jsonl'
        predictions_path.write_text(''.join(json.dumps(record) + '\n' for record in ordered), encoding='utf-8')
        for path, expected in bindings.items():
            if sha256_file(path) != expected:
                raise ValueError('Candidate training inputs changed during evaluation')
        if backend.model_binding(adapter) != adapter_binding:
            raise ValueError('Candidate adapter changed during evaluation')
        metrics = {role: original.metrics_for(values, full, original.SCOPES[role]) for role, values in
                   (('full_dev', rows['dev']), ('dev_panel', rows['dev_panel']), ('dev_nonpanel', nonpanel))}
        result = {**config, 'metrics': metrics, 'panel_reproduction': reproduction,
                  'dev_predictions_sha256': sha256_file(predictions_path),
                  'elapsed_seconds': time.monotonic() - started,
                  'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3}
        write_json(output / 'evaluation_report.json', result)
        for role, value in metrics.items():
            print(f'{role}: ' + json.dumps(value), flush=True)
        print('Report:', output / 'evaluation_report.json', flush=True)
        return result
    except Exception as error:
        write_json(output / 'failure.json', {'stage': 'evaluate', 'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'train', 'evaluate'), required=True)
    parser.add_argument('--model', default='/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--control-run', default='outputs_vlm_gold/mi_multi_image_seed1234')
    parser.add_argument('--smoke-report', default=SMOKE_DIR + '/smoke_report.json')
    parser.add_argument('--run', default=TRAIN_DIR)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.output is None:
        args.output = {'smoke': SMOKE_DIR, 'train': TRAIN_DIR, 'evaluate': TRAIN_DIR + '_dev_evaluation'}[args.stage]
    state = load_control(args)
    if args.stage == 'evaluate':
        return evaluate_candidate(args, *state)
    return run_training(args, *state[:3])


if __name__ == '__main__':
    main()
