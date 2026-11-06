"""Gold-supervised frozen Qwen word features as a score residual for AGVS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.internal_split import sha256_file
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import train_bio_joint_recheck as train

ROOT = train.ROOT / 'agvs_llm_word_feature_pilot_v1'
HEADS = ROOT / 'heads'
REPORT = ROOT / 'dev_fusion_report.json'
HEADS_SCHEMA = 'agvs-llm-word-heads-v1'
SCOPE = 'original internal selection dev, exploratory fusion-weight selection'
SEED = 1234
WIDTH = 768
BATCH_SIZE = 16
EPOCHS = 6
COEFFICIENTS = (0.25, 0.5, 1.0)
MIN_GAIN = 0.003
CODES = recheck.bio.CODES


def corpus():
    prepared, full = train.load_prepared(train.split.DATA)
    rows, _, binding = train.training_plan(prepared, full)
    config = train.baseline_config(prepared)
    return prepared, rows, binding, config


def join_words(words):
    position = 0
    spans = []
    for word in words:
        spans.append((position, position + len(word)))
        position += len(word) + 1
    return ' '.join(words), spans


def word_indices(spans, offsets):
    """Last subtoken intersecting each original word, with -1 for skipped words."""
    return [max((i for i, (a, b) in enumerate(offsets) if a < b and start < end
                 and a < end and start < b), default=-1) for start, end in spans]


def extract(rows, directory, encoder):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, BertModel, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('The cached word features require tokenizer offset mappings')
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if encoder == 'qwen':
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                  llm_int8_skip_modules=['visual', 'lm_head'])
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            directory, local_files_only=True, torch_dtype=torch.bfloat16, quantization_config=quant,
            device_map={'': 0}, attn_implementation='sdpa').eval()
        model = vlm.get_decoder()
        del vlm
    else:
        model = BertModel.from_pretrained(directory, local_files_only=True).to('cuda:0').eval()

    features, aligned = [], []
    projection = None
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), BATCH_SIZE), desc=f'{encoder} frozen features', unit='batch'):
            batch = rows[start:start + BATCH_SIZE]
            texts, spans = zip(*(join_words(row['tokens']) for row in batch))
            encoded = tokenizer(list(texts), padding=True, return_tensors='pt', return_offsets_mapping=True)
            offsets = encoded.pop('offset_mapping').tolist()
            if encoded['input_ids'].shape[1] > 512:
                raise ValueError('Frozen encoders require at most 512 subtokens; preserve original words')
            model_inputs = {'input_ids': encoded['input_ids'].to('cuda:0'),
                            'attention_mask': encoded['attention_mask'].to('cuda:0')}
            hidden = model(**model_inputs, **({'use_cache': False} if encoder == 'qwen' else {}),
                           return_dict=True).last_hidden_state
            if hidden.size(-1) != WIDTH and projection is None:
                generator = torch.Generator().manual_seed(SEED)
                projection = (torch.randn(hidden.size(-1), WIDTH, generator=generator) /
                              WIDTH**0.5).to(hidden.device)
            for i, bounds in enumerate(spans):
                positions = torch.tensor(word_indices(bounds, offsets[i]), device=hidden.device)
                valid = positions.ge(0)
                states = hidden[i, positions.clamp_min(0)].float()
                if projection is not None:
                    states = states @ projection
                states = states.masked_fill(~valid.unsqueeze(-1), 0)
                if not torch.isfinite(states).all():
                    raise ValueError(f'Non-finite word features at source_id={batch[i]["source_id"]}')
                features.append(states.to(device='cpu', dtype=torch.float16))
                aligned.append(valid.cpu())
    del model
    torch.cuda.empty_cache()
    return features, aligned


def train_head(features, rows, dev_count, device='cuda:0'):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

    class WordHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(WIDTH)
            self.context = nn.GRU(WIDTH, 128, batch_first=True, bidirectional=True)
            self.out = nn.Linear(256, len(CODES))

        def forward(self, words, lengths):
            packed = pack_padded_sequence(self.norm(words), lengths, batch_first=True, enforce_sorted=False)
            encoded, _ = self.context(packed)
            context, _ = pad_packed_sequence(encoded, batch_first=True)
            return self.out(context)

    def batch(indices):
        sequences = [features[i] for i in indices]
        return pad_sequence(sequences, batch_first=True).to(device, torch.float32), [len(x) for x in sequences]

    torch.manual_seed(SEED)
    model = WordHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    tags = [torch.tensor([CODES.index(tag) for tag in recheck.baseline.word_bio(row)]) for row in rows]
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            x, lengths = batch(indices)
            y = pad_sequence([tags[i] for i in indices], batch_first=True, padding_value=-100).to(device)
            logits = model(x, lengths)
            loss = F.cross_entropy(logits.transpose(1, 2), y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite word head loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f'Word head epoch {epoch + 1}/{EPOCHS}, mean loss {sum(losses)/len(losses):.5f}', flush=True)

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(len(rows), len(rows) + dev_count, BATCH_SIZE):
            indices = range(start, min(start + BATCH_SIZE, len(rows) + dev_count))
            x, lengths = batch(indices)
            logits = model(x, lengths)
            predictions.extend(row[:length].cpu().clone() for row, length in zip(logits, lengths))
    return predictions, {key: value.detach().cpu() for key, value in model.state_dict().items()}


def make_heads():
    if HEADS.exists():
        print(f'Reusing feature heads for validation by the fusion stage: {HEADS}', flush=True)
        return
    import torch
    from modules import vlm_training as backend

    runtime = backend.check_runtime()
    _, rows, binding, config = corpus()
    fit, dev = rows['train'], rows['dev']
    qwen = Path(train.BIO_MODEL)
    bert = Path(config['bert_model'])
    qwen_hashes = backend.model_binding(qwen)
    expected = train.READ_JSON(Path(train.BIO_CONTROL) / 'pilot_report.json')['profile']['model_files_sha256']
    if qwen_hashes != expected:
        raise ValueError('Qwen files differ from the approved local weights')
    bert_hashes = {str(path.resolve()): sha256_file(path) for path in bert.iterdir() if path.is_file()}
    ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ROOT, prefix='.word-heads-') as directory:
        scratch = Path(directory)
        outputs, coverage = {}, {}
        for name, location in (('bert', bert), ('qwen', qwen)):
            print(f'Extracting {name} word features and fitting the gold-only head...', flush=True)
            features, valid = extract(fit + dev, location, name)
            logits, state = train_head(features, fit, len(dev))
            path = scratch / f'{name}.pt'
            torch.save({'dev_logits': logits, 'dev_alignment': valid[len(fit):], 'head': state}, path)
            outputs[path.name] = sha256_file(path)
            coverage[name] = {'aligned_words': sum(int(mask.sum()) for mask in valid),
                              'total_words': sum(len(mask) for mask in valid)}
            del features, valid, logits, state
        if backend.model_binding(qwen) != qwen_hashes:
            raise ValueError('Qwen weights changed during feature extraction')
        recheck.verify_files(bert_hashes)
        train.WRITE_JSON(scratch / 'manifest.json', {
            'schema_version': HEADS_SCHEMA, 'recheck': binding, 'runner_sha256': sha256_file(__file__),
            'fit_source_ids': recheck.source_ids(fit), 'dev_source_ids': recheck.source_ids(dev),
            'features': coverage, 'runtime': runtime, 'qwen_weights': qwen_hashes, 'bert_files': bert_hashes,
            'labels': list(CODES), 'outputs_sha256': outputs, 'official_val_or_test_read': False,
        })
        scratch.rename(HEADS)
    print('Frozen head artifacts:', HEADS, flush=True)


def word_residual(emissions, mappings, active_lengths, dev, bundle, labels):
    import torch

    if len(bundle['dev_logits']) != len(dev) or len(bundle['dev_alignment']) != len(dev):
        raise ValueError('Expected word scores and masks for all dev rows')
    residual = torch.zeros_like(emissions)
    order = [CODES.index(tag if tag != 'X' else 'O')
             for tag in sorted(labels, key=labels.__getitem__)]
    for i, ((mask, _, indices), length, row, logits, valid) in enumerate(
            zip(mappings, active_lengths, dev, bundle['dev_logits'], bundle['dev_alignment'])):
        if logits.shape != (len(row['tokens']), len(CODES)) or valid.shape != (len(row['tokens']),):
            raise ValueError('Word scores or masks do not align to the original words')
        positions = (~mask[:length]).nonzero().flatten()
        centered = logits - logits.mean(1, keepdim=True)
        residual[i, positions] = (centered.clamp(-8, 8) * valid.unsqueeze(-1))[indices][:, order]
    if not torch.isfinite(residual).all():
        raise ValueError('Non-finite word-score residual')
    return residual


def score_fusion():
    if REPORT.exists():
        raise FileExistsError(f'Preserve the existing pilot result: {REPORT}')
    recheck.baseline.check_baseline_runtime()
    import torch
    from torchcrf import CRF
    from scripts import evaluate_crf_ensemble as ensemble
    from modules.dataset import MultimodalInferenceDataset, MultimodalNerDataset

    prepared, rows, binding, config = corpus()
    seal = recheck.baseline_seal(binding)
    dev = rows['dev']
    manifest = train.READ_JSON(HEADS / 'manifest.json')
    if (manifest.get('schema_version') != HEADS_SCHEMA or manifest.get('recheck') != binding
            or manifest.get('runner_sha256') != sha256_file(__file__)
            or manifest.get('labels') != list(CODES)
            or manifest.get('dev_source_ids') != recheck.source_ids(dev)
            or manifest.get('fit_source_ids') != recheck.source_ids(rows['train'])):
        raise ValueError('Word-head artifacts differ from the fixed training or dev inputs')
    recheck.verify_files({str(HEADS / name): digest for name, digest in manifest['outputs_sha256'].items()})
    args = SimpleNamespace(**config)
    labeled = MultimodalNerDataset(args, 'MNER-MI_val.txt')
    structure = MultimodalInferenceDataset(args, 'MNER-MI_val.txt')
    if labeled.examples['words'] != [row['tokens'] for row in dev]:
        raise ValueError('AGVS dev rows differ from the word-feature rows')
    recheck.baseline.audit_images(labeled, prepared)
    labels = labeled.processor.get_label_mapping(include_subword_label=True)
    cli = SimpleNamespace(device=args.device, batch_size=args.batch_size, num_workers=0)
    native = ensemble._collect(Path(seal['checkpoint']), config, labeled, cli, labels, 'AGVS')
    decoder = CRF(len(labels), batch_first=True)
    decoder.load_state_dict(native['crf'])
    mappings = [structure.token_structure(i) for i in range(len(dev))]
    active_lengths = native['mask'].sum(1).tolist()

    def score_paths(paths):
        records, observed_gold, observed_predictions = {}, [], []
        for row, path, (mask, _, indices), length in zip(dev, paths, mappings, active_lengths):
            record, gold, predicted = recheck.baseline.project_path(
                row, path, mask.tolist(), indices, labels, length)
            records[row['source_id']] = record
            observed_gold.append(gold)
            observed_predictions.append(predicted)
        return (recheck.bio.original.metrics_for(dev, records, SCOPE),
                recheck.baseline.span_metrics(observed_gold, observed_predictions))

    base, native_score = score_paths(native['native_paths'])
    if abs(native_score['f1'] - seal['best_dev_f1']) > 0.0005:
        raise ValueError('Frozen AGVS selection dev F1 failed reproduction')
    trials = {}
    for name in ('bert', 'qwen'):
        path = HEADS / f'{name}.pt'
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        residual = word_residual(native['emissions'], mappings, active_lengths, dev, bundle, labels)
        trials[name] = []
        for coefficient in COEFFICIENTS:
            with torch.inference_mode():
                paths = decoder.decode(native['emissions'] + coefficient * residual, mask=native['mask'].bool())
            scores, _ = score_paths(paths)
            trials[name].append({'coefficient': coefficient, 'metrics': scores})
    best = {name: max(values, key=lambda entry: entry['metrics']['f1']) for name, values in trials.items()}
    bert_f1 = best['bert']['metrics']['f1']
    qwen_f1 = best['qwen']['metrics']['f1']
    passed = qwen_f1 >= max(base['f1'], bert_f1) + MIN_GAIN
    result = {'scope': SCOPE, 'agvs_f1': base['f1'], 'agvs_bert_head_f1': bert_f1,
              'agvs_qwen_head_f1': qwen_f1, 'qwen_minus_agvs_f1': qwen_f1 - base['f1'],
              'qwen_minus_bert_control_f1': qwen_f1 - bert_f1,
              'best_coefficients': {name: item['coefficient'] for name, item in best.items()},
              'minimum_gain': MIN_GAIN, 'passes_development_gate': passed}
    train.WRITE_JSON(REPORT, {'schema_version': 'agvs-llm-word-fusion-dev-v1', **result,
                             'recheck': binding, 'baseline_metrics': base, 'trials': trials,
                             'heads_manifest_sha256': sha256_file(HEADS / 'manifest.json'),
                             'official_val_or_test_read': False, 'grouped_holdout_scored': False})
    print(json.dumps({**result, 'report': str(REPORT.resolve())}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('heads', 'fusion'))
    if parser.parse_args().stage == 'heads':
        make_heads()
    else:
        score_fusion()


if __name__ == '__main__':
    main()
