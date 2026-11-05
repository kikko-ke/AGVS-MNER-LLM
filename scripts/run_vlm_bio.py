"""Train and evaluate Qwen2.5-VL with a compact BIO tag per original word."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules import vlm_training as backend
from modules.internal_split import sha256_file
from modules.vlm_ner import parse_entities, target
from scripts import evaluate_vlm_ner as original
from scripts import run_vlm_image_dropout as common

SMOKE = 'outputs_vlm_gold/mi_bio_smoke'
RUN = 'outputs_vlm_gold/mi_bio_seed1234'
PROFILE = 'vlm-bio-profile-v1'
SMOKE_SCHEMA = 'vlm-bio-smoke-v1'
TRAIN_SCHEMA = 'vlm-bio-pilot-v1'
CODES = ('O', 'B-LOC', 'I-LOC', 'B-MIS', 'I-MIS', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER')
SYSTEM = (
    'Extract named entities from the numbered WORDS and any supplied images. '
    'Return only a JSON object with a tags string, with exactly one ASCII digit per word in WORDS order. '
    'Use 0 for outside any entity. Use 1=B-LOC, 2=I-LOC, 3=B-MIS, 4=I-MIS, '
    '5=B-ORG, 6=I-ORG, 7=B-PER, 8=I-PER. '
    'B begins an entity and I continues an entity of the same type. '
    'LOC means location, MIS other named entity, ORG organization, and PER person. '
    'For two words forming one person name, the answer is {"tags":"78"}.'
)


def bio_target(row):
    length = len(row['tokens'])
    tags = ['0'] * length
    for start, end, label in parse_entities(target(row), length):
        begin = CODES.index('B-' + label)
        tags[start:end] = [str(begin)] + [str(begin + 1)] * (end - start - 1)
    return json.dumps({'tags': ''.join(tags)}, separators=(',', ':'))


def parse_bio(response, length):
    if not isinstance(response, str):
        raise ValueError('Expected a text BIO response')
    response = response.strip()
    if response.startswith('```json\n') and response.endswith('```'):
        response = response[8:-3].strip()
    value = json.loads(response)
    if not isinstance(value, dict) or set(value) != {'tags'} or not isinstance(value['tags'], str):
        raise ValueError('Expected a JSON tags string')
    tags = value['tags']
    if len(tags) != length or any(ch not in '012345678' for ch in tags):
        raise ValueError('Expected one ASCII BIO digit per original word')
    spans, active, start = [], None, 0
    for position, digit in enumerate(tags + '0'):
        code = CODES[int(digit)]
        if code.startswith('I-'):
            if active != code[2:]:
                raise ValueError(f'Invalid inside tag at word {position}')
            continue
        if active is not None:
            spans.append((start, position, active))
        active, start = (None if code == 'O' else code[2:]), position
    return spans


class BioChat:
    """Adapt only the two chat text fields while reusing image encoding and masking."""

    def __init__(self, processor, answer):
        self.processor = processor
        self.answer = answer

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def apply_chat_template(self, chat, **kwargs):
        messages = [{'role': 'system', 'content': SYSTEM}, chat[1]]
        if len(chat) == 3:
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.answer}]})
        return self.processor.apply_chat_template(messages, **kwargs)


def encode(processor, row, profile, training):
    view = BioChat(processor, bio_target(row) if training else None)
    prompt_budget = profile['max_length'] - profile['max_new_tokens']
    result = backend.encode(view, row, 'multi_image', training,
                            profile['max_length'] if training else prompt_budget)
    if training:
        answer_tokens = int(result['labels'].ne(-100).sum().item())
        prompt_tokens = result['input_ids'].shape[1] - answer_tokens
        if answer_tokens > profile['max_new_tokens'] or prompt_tokens > prompt_budget:
            raise ValueError(f'BIO budgets exceeded for source_id={row["source_id"]}: '
                             f'prompt={prompt_tokens}, answer={answer_tokens}')
    return result


def decoded(response, length):
    try:
        return [list(span) for span in parse_bio(response, length)], None
    except (ValueError, TypeError) as error:
        return [], str(error)


def checked_records(rows, records):
    lengths = {row['source_id']: len(row['tokens']) for row in rows}
    indexed = {}
    for record in records:
        sid = record['source_id']
        if type(sid) is not int or sid not in lengths or sid in indexed:
            raise ValueError('BIO prediction source IDs are invalid or duplicated')
        spans, error = decoded(record['response'], lengths[sid])
        if spans != record['predicted'] or (error is None) != (record['error'] is None):
            raise ValueError('Saved predictions disagree with the BIO response')
        indexed[sid] = record
    if set(indexed) != set(lengths):
        raise ValueError('Expected one BIO prediction per requested source ID')
    return indexed


def write_records(path, records):
    path.write_text(''.join(json.dumps(record, ensure_ascii=True) + '\n' for record in records), encoding='utf-8')


def generate(model, processor, rows, profile, path):
    import torch
    from tqdm.auto import tqdm

    records = []
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for row in tqdm(rows, desc='Generate BIO tags', unit='sample'):
            batch = backend.move_to_cuda(encode(processor, row, profile, False))
            prefix_length = batch['input_ids'].shape[1]
            output = model.generate(**batch, max_new_tokens=profile['max_new_tokens'],
                                    do_sample=False, num_beams=1, use_cache=True)
            response = processor.tokenizer.decode(output[0, prefix_length:], skip_special_tokens=True)
            spans, error = decoded(response, len(row['tokens']))
            records.append({'source_id': row['source_id'], 'response': response, 'predicted': spans, 'error': error})
            del batch, output
    write_records(path, records)
    return checked_records(rows, records)


def load_profile(args):
    control, manifest, rows, nonpanel, _, bindings = original.validate_run(
        Path(args.control_run).resolve(), args.model, args.data)
    base = control['profile']
    if any(base.get(key) != value for key, value in common.BASE_SETTINGS.items()):
        raise ValueError('Expected the original fixed two-epoch multi-image control')
    profile = {**base, 'schema_version': PROFILE, 'control_run': str(Path(args.control_run).resolve()),
               'control_bindings': bindings, 'tag_codes': list(CODES),
               'bio_implementation_sha256': sha256_file(__file__),
               'shared_stage_helpers_sha256': sha256_file(common.__file__)}
    return profile, manifest, rows, nonpanel


def updates(profile, samples):
    return math.ceil(samples / profile['gradient_accumulation']) * profile['epochs']


def train(args, profile, rows, output):
    import torch
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup

    smoke = args.stage == 'smoke'
    model, processor = backend.load_model(args.model, profile['seed'], profile['min_pixels'], profile['max_pixels'])
    selected = backend.smoke_rows(rows['train']) if smoke else rows['train']
    if smoke:
        longest = max(rows['train'], key=lambda row: (len(row['tokens']), -row['source_id']))
        if len(selected) != 8 or not selected[0]['images']:
            raise ValueError('Smoke requires eight training samples including an original image input')
        if longest not in selected:
            selected[-1] = longest
    audit_rows = selected if smoke else rows['train'] + rows['dev']
    max_prompt, max_answer = 0, 0
    for row in tqdm(audit_rows, desc='Audit BIO prompt and answer budgets', unit='sample'):
        if parse_bio(bio_target(row), len(row['tokens'])) != parse_entities(target(row), len(row['tokens'])):
            raise ValueError(f'Gold BIO round-trip failed for source_id={row["source_id"]}')
        batch = encode(processor, row, profile, True)
        answer = int(batch['labels'].ne(-100).sum().item())
        max_prompt = max(max_prompt, batch['input_ids'].shape[1] - answer)
        max_answer = max(max_answer, answer)
        del batch

    epochs = 1 if smoke else profile['epochs']
    accumulation = 1 if smoke else profile['gradient_accumulation']
    steps = 8 if smoke else updates(profile, len(selected))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=profile['lr'], weight_decay=profile['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * profile['warmup_ratio']), steps)
    adapter = output / 'best_adapter'
    history, best_epoch, best_f1, completed = [], None, -1., 0
    panel = rows['dev_panel']
    for epoch in range(epochs + 1):
        losses = []
        if epoch:
            backend.train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(selected), generator=torch.Generator().manual_seed(profile['seed'] + epoch)).tolist()
            for index, position in enumerate(tqdm(order, desc=f'BIO epoch {epoch}/{epochs}', unit='sample')):
                row = selected[position]
                batch = backend.move_to_cuda(encode(processor, row, profile, True))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(**batch, use_cache=False)
                    loss = result.loss
                if not torch.isfinite(loss):
                    raise ValueError(f'Non-finite BIO loss for source_id={row["source_id"]}')
                losses.append(float(loss.detach()))
                (loss / backend.accumulation_divisor(index, len(selected), accumulation)).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(selected):
                    backend.optimizer_update(model, optimizer, scheduler)
                    completed += 1
                del batch, result, loss
        if not smoke:
            indexed = generate(model, processor, panel, profile, output / f'dev_epoch{epoch}.jsonl')
            metrics = original.metrics_for(panel, indexed, original.SCOPES['dev_panel'])
            item = {'epoch': epoch, 'metrics': metrics}
            if losses:
                item['mean_train_loss'] = sum(losses) / len(losses)
            history.append(item)
            if metrics['f1'] > best_f1:
                best_epoch, best_f1 = epoch, metrics['f1']
                model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
                if epoch == 0:
                    processor.save_pretrained(adapter)
            common.write_json(output / 'history.json', history)
            print(f'Epoch {epoch} dev panel:', json.dumps(metrics), flush=True)
    if completed != steps:
        raise ValueError('BIO run used an unexpected optimizer update budget')
    report = {'profile': profile, 'optimizer_steps': completed, 'official_val_or_test_read': False,
              'audited_samples': len(audit_rows), 'maximum_prompt_tokens': max_prompt,
              'maximum_answer_tokens': max_answer,
              'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3}
    if smoke:
        generation_rows = selected[:2] if selected[0] == longest else [selected[0], longest]
        predictions = generate(model, processor, generation_rows, profile, output / 'smoke_predictions.jsonl')
        return 'smoke_report.json', {**report, 'schema_version': SMOKE_SCHEMA, 'passed': True,
            'generation_completed': True, 'smoke_samples': len(selected),
            'malformed_smoke_outputs': sum(record['error'] is not None for record in predictions.values()),
            'heldout_accuracy_evaluated': False}
    return 'pilot_report.json', {**report, 'schema_version': TRAIN_SCHEMA, 'history': history,
        'best_epoch': best_epoch, 'best_dev_panel_f1': best_f1,
        'best_panel_sha256': sha256_file(output / f'dev_epoch{best_epoch}.jsonl'),
        'best_adapter_sha256': backend.model_binding(adapter)}


def validate_candidate(args, profile, rows):
    run = Path(args.run)
    report_path = run / 'pilot_report.json'
    report = original.read_json(report_path)
    if (report.get('schema_version') != TRAIN_SCHEMA or report.get('profile') != profile
            or report.get('official_val_or_test_read') is not False
            or report.get('optimizer_steps') != updates(profile, len(rows['train']))):
        raise ValueError('Expected the completed hash-bound BIO training run')
    history = report['history']
    if [item['epoch'] for item in history] != [0, 1, 2]:
        raise ValueError('BIO training history is incomplete')
    best = max(history, key=lambda item: item['metrics']['f1'])
    if (best['epoch'], best['metrics']['f1']) != (report['best_epoch'], report['best_dev_panel_f1']):
        raise ValueError('Saved adapter must follow the fixed panel selection rule')
    adapter = run / 'best_adapter'
    binding = backend.model_binding(adapter)
    if binding != report['best_adapter_sha256']:
        raise ValueError('BIO adapter files differ from training report')
    panel = rows['dev_panel']
    panel_path = run / f'dev_epoch{best["epoch"]}.jsonl'
    if sha256_file(panel_path) != report.get('best_panel_sha256'):
        raise ValueError('Saved BIO panel differs from training report')
    reference = checked_records(panel, original.read_records(panel_path))
    if original.metrics_for(panel, reference, original.SCOPES['dev_panel']) != best['metrics']:
        raise ValueError('Saved BIO panel scores differ from training history')
    inputs = {str(path.resolve()): sha256_file(path) for path in (report_path, panel_path)}
    return best['epoch'], adapter, binding, reference, inputs


def evaluate(args, profile, rows, nonpanel, output, candidate):
    epoch, adapter, binding, reference, inputs = candidate
    panel = rows['dev_panel']
    model, processor = original.load_frozen_adapter(args.model, adapter, profile)
    indexed = generate(model, processor, panel, profile, output / 'panel_predictions.jsonl')
    for sid, record in indexed.items():
        if (record['predicted'] != reference[sid]['predicted']
                or (record['error'] is None) != (reference[sid]['error'] is None)):
            raise ValueError(f'BIO panel reproduction failed at source_id={sid}')
    print('Panel reproduced. Evaluating the remaining original-image dev rows.', flush=True)
    indexed.update(generate(model, processor, nonpanel, profile, output / 'nonpanel_predictions.jsonl'))
    predictions_path = output / 'dev_predictions.jsonl'
    write_records(predictions_path, [indexed[row['source_id']] for row in rows['dev']])
    if backend.model_binding(adapter) != binding:
        raise ValueError('BIO adapter changed during evaluation')
    for path, digest in inputs.items():
        if sha256_file(path) != digest:
            raise ValueError('BIO training report or panel predictions changed during evaluation')
    scores = {scope: original.metrics_for(group, indexed, original.SCOPES[scope]) for scope, group in
              (('full_dev', rows['dev']), ('dev_panel', panel), ('dev_nonpanel', nonpanel))}
    for scope, metrics in scores.items():
        print(f'{scope}: {json.dumps(metrics)}', flush=True)
    return 'evaluation_report.json', {'schema_version': 'vlm-bio-dev-evaluation-v1',
        'profile': profile, 'best_epoch': epoch, 'metrics': scores, 'input_sha256': inputs,
        'panel_reproduction': {'passed': True}, 'adapter_files_sha256': binding,
        'dev_predictions_sha256': sha256_file(predictions_path), 'official_val_or_test_read': False,
        'source_ids': {'full_dev': [r['source_id'] for r in rows['dev']],
                       'dev_panel': [r['source_id'] for r in panel],
                       'dev_nonpanel': [r['source_id'] for r in nonpanel]},
        'optimizer_steps': 0, 'new_checkpoint_selection': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('smoke', 'train', 'evaluate'))
    parser.add_argument('--model', default='/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--data', default='dataset/vlm_gold_mi_v1/manifest.json')
    parser.add_argument('--control-run', default='outputs_vlm_gold/mi_multi_image_seed1234')
    parser.add_argument('--smoke-report', default=SMOKE + '/smoke_report.json')
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--output')
    args = parser.parse_args()
    profile, _, rows, nonpanel = load_profile(args)
    if args.stage == 'train':
        smoke = original.read_json(args.smoke_report)
        if (smoke.get('schema_version') != SMOKE_SCHEMA or smoke.get('profile') != profile
                or smoke.get('passed') is not True or smoke.get('optimizer_steps') != 8
                or smoke.get('generation_completed') is not True):
            raise ValueError('Expected a passing BIO GPU smoke with the same profile')
    defaults = {'smoke': SMOKE, 'train': RUN, 'evaluate': RUN + '_dev_evaluation'}
    protected = {'smoke': (), 'train': (Path(args.smoke_report).parent,), 'evaluate': (args.run,)}
    output = common.protect_output(args.output or defaults[args.stage], args, protected[args.stage])
    candidate = validate_candidate(args, profile, rows) if args.stage == 'evaluate' else None
    if backend.check_runtime() != profile['versions']:
        raise ValueError('Use the pinned runtime from the original control')
    output.mkdir(parents=True, exist_ok=False)
    common.write_json(output / 'config.json', {'stage': args.stage, 'profile': profile,
                      'official_val_or_test_read': False})
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    try:
        if args.stage == 'evaluate':
            name, report = evaluate(args, profile, rows, nonpanel, output, candidate)
        else:
            name, report = train(args, profile, rows, output)
        common.write_json(output / name, report)
        print('Report:', output / name, flush=True)
    except Exception as error:
        common.write_json(output / 'failure.json', {'stage': args.stage, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
