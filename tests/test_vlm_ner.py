from __future__ import annotations

import contextlib
import json
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image

from modules.vlm_ner import (
    prepare_data, load_prepared, messages, target, parse_entities, span_metrics, open_image,
)
from modules.vlm_training import encode, accumulation_divisor, text_lora_targets, optimizer_update, smoke_rows
from scripts.create_context_internal_split import create_split


def data_fixture(tmp_path):
    Image.new('RGB', (8, 8), 'white').save(tmp_path / 'ok.png')
    (tmp_path / 'broken.jpg').write_bytes(b'not an image')
    rows = [{'text':['Acme', 'hired', 'Kim', f'today{i}'],
             'label':['B-ORG', 'O', 'B-PER', 'O'],
             'images':['ok.png', 'missing.jpg', 'broken.jpg']} for i in range(20)]
    source = tmp_path / 'MNER-MI_train.txt'
    source.write_text(''.join(repr(row)+'\n' for row in rows), encoding='utf-8')
    split = create_split(source, tmp_path / 'split', 'MI')
    data = prepare_data(split, tmp_path / 'prepared', tmp_path, tmp_path, panel_size=2)
    manifest, rows = load_prepared(data)
    return data, manifest, rows


def test_gold_data_preserves_all_words_and_uses_only_the_fixed_train_dev_split(tmp_path):
    path, manifest, rows = data_fixture(tmp_path)
    assert len(rows['train'])+len(rows['dev']) == 20
    assert not {r['source_id'] for r in rows['train']} & {r['source_id'] for r in rows['dev']}
    assert len(rows['dev_panel']) == 2 and manifest['panel_selection_uses_labels'] is False
    for row in rows['train']+rows['dev']:
        assert len(row['tokens']) == 4 and len(row['images']) == 1
        assert parse_entities(target(row), 4) == [(0,1,'ORG'), (2,3,'PER')]
    assert manifest['official_val_or_test_read'] is False
    assert not list(tmp_path.rglob('*test*txt'))
    with pytest.raises(FileExistsError):
        prepare_data(manifest['internal_manifest'], path.parent, tmp_path, tmp_path)
    original = Path(json.loads(Path(manifest['internal_manifest']).read_text())['source'])
    original.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source changed'):
        load_prepared(path)


@pytest.mark.parametrize('payload', [
    {'entities':[{'start':True,'end':1,'label':'PER'}]},
    {'entities':[{'start':0,'end':4,'label':'PER'}]},
    {'entities':[{'start':0,'end':1,'label':'TITLE'}]},
    {'entities':[{'start':0,'end':2,'label':'ORG'}, {'start':1,'end':2,'label':'PER'}]},
    {'entities':[{'start':0,'end':1,'label':'PER'}, {'start':0,'end':1,'label':'PER'}]},
    {'entities':[], 'explanation':'extra text'},
])
def test_span_output_rejects_invalid_or_overlapping_entities(payload):
    with pytest.raises(ValueError):
        parse_entities(json.dumps(payload), 3)


def test_strict_span_metrics_include_empty_malformed_fallbacks_as_errors():
    gold = [[(0,1,'PER')], [(1,2,'ORG')]]
    result = span_metrics(gold, [[(0,1,'PER')], []])
    assert result['tp'] == 1 and result['fn'] == 1 and result['fp'] == 0
    assert result['precision'] == 1 and result['recall'] == 0.5
    assert result['f1'] == pytest.approx(2/3)
    with pytest.raises(ValueError):
        span_metrics(gold, [[]])


def test_small_images_are_upsampled_before_qwen_processing_and_extreme_shapes_rejected():
    stream = BytesIO()
    Image.new('RGB', (190,21), 'white').save(stream, format='PNG')
    result = open_image(stream.getvalue())
    assert min(result.size) >= 28 and result.size[0] / result.size[1] == pytest.approx(190/21, abs=0.05)
    stream = BytesIO()
    Image.new('RGB', (4000,1), 'white').save(stream, format='PNG')
    with pytest.raises(ValueError):
        open_image(stream.getvalue())


def test_inference_messages_cannot_read_gold_or_use_literal_special_tokens():
    class NoGold(dict):
        def __getitem__(self, key):
            assert key != 'entities', 'Gold leaked to inference'
            return super().__getitem__(key)
    row = NoGold(tokens=['<|im_start|>', 'Kim'], images=[{'path':'image.png'}], entities='SECRET')
    chat = messages(row, 'multi_image')
    assert chat[1]['content'][0]['type'] == 'image'
    assert 'SECRET' not in json.dumps(chat)
    assert '<|im_start|>' not in chat[1]['content'][-1]['text']
    assert messages(row, 'text')[1]['content'][0]['type'] == 'text'


class Processor:
    broken_prefix = False

    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        text = 'PROMPT ' + json.dumps(chat[:2]) + '\nASSISTANT\n'
        if not add_generation_prompt:
            text += chat[-1]['content'][0]['text'] + '<eos>'
            if self.broken_prefix:
                text = 'WRONG ' + text
        return text

    def __call__(self, text, return_tensors, padding, images=None):
        ids = torch.tensor([[ord(c) % 100 + 1 for c in text[0]]])
        return {'input_ids':ids, 'attention_mask':torch.ones_like(ids)}


def test_assistant_only_labels_match_multimodal_prompt_and_preserve_all_target_tokens(tmp_path):
    _, _, rows = data_fixture(tmp_path)
    row, processor = rows['train'][0], Processor()
    batch = encode(processor, row, 'multi_image', True, 5000)
    prefix = encode(processor, row, 'multi_image', False, 5000)['input_ids'].shape[1]
    assert batch['labels'][0,:prefix].eq(-100).all()
    assert torch.equal(batch['labels'][:,prefix:], batch['input_ids'][:,prefix:])
    with pytest.raises(ValueError, match='budget'):
        encode(processor, row, 'multi_image', True, 10)
    processor.broken_prefix = True
    with pytest.raises(ValueError, match='prefix'):
        encode(processor, row, 'multi_image', True, 5000)


def test_optimizer_and_partial_accumulation_do_not_weaken_the_final_group():
    assert [accumulation_divisor(i,10,8) for i in range(10)] == [8]*8+[2]*2
    model = nn.Linear(2,2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1)
    before = model.weight.detach().clone()
    model(torch.ones(1,2)).sum().backward()
    assert optimizer_update(model, optimizer, scheduler) > 0
    assert not torch.equal(before, model.weight)


def test_generation_is_label_free_and_malformed_outputs_remain_in_metrics(tmp_path, monkeypatch):
    import modules.vlm_training as backend
    _, _, rows = data_fixture(tmp_path)
    processor = Processor()

    class Decoder:
        index = 0
        def decode(self, tokens, skip_special_tokens):
            responses = ['{"entities":[{"start":0,"end":1,"label":"ORG"},{"start":2,"end":3,"label":"PER"}]}',
                         'not valid JSON']
            text = responses[self.index]
            self.index += 1
            return text

    class Generator:
        def eval(self):
            pass
        def generate(self, input_ids, **kwargs):
            assert 'labels' not in kwargs and kwargs['do_sample'] is False
            return torch.cat([input_ids, torch.tensor([[7]])], dim=1)

    processor.tokenizer = Decoder()
    monkeypatch.setattr(backend, 'move_to_cuda', lambda batch:batch)
    monkeypatch.setattr(torch, 'autocast', lambda *args,**kwargs:contextlib.nullcontext())
    output = tmp_path/'predictions.jsonl'
    result = backend.generate_evaluation(Generator(), processor, rows['dev_panel'], 'multi_image', 5000, 32, output)
    assert result['samples'] == 2 and result['malformed_responses'] == 1
    assert result['tp'] == 2 and result['fn'] == 2 and result['f1'] == pytest.approx(2/3)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2 and records[1]['error'] and records[1]['predicted'] == []


def test_lora_target_selection_excludes_vision_and_smoke_selection_covers_image_counts():
    model = nn.Module()
    model.visual = nn.ModuleDict({'q_proj':nn.Linear(2,2)})
    model.language = nn.ModuleDict({'q_proj':nn.Linear(2,2), 'v_proj':nn.Linear(2,2)})
    model.lm_head = nn.Linear(2,2)
    assert set(text_lora_targets(model)) == {'language.q_proj','language.v_proj'}
    rows = [{'source_id':i, 'tokens':['longword']*i, 'images':[{}]*n}
            for i,n in enumerate([4,4,4,4,4,4,4,4,3,2,1,0])]
    picked = smoke_rows(rows)
    assert len(picked) == 8 and {len(r['images']) for r in picked} == {0,1,2,3,4}
    assert len({r['source_id'] for r in picked}) == 8


def test_smoke_to_training_entrypoint_requires_real_gate_and_saves_only_adapter(tmp_path, monkeypatch):
    from scripts import run_vlm_ner as entry
    data, _, _ = data_fixture(tmp_path)
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'model.safetensors').write_bytes(b'fake test-only model')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2,2)
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(loss=self.linear(torch.ones(1,2)).square().mean())
        def save_pretrained(self, path, **kwargs):
            Path(path).mkdir(exist_ok=True)
            (Path(path)/'adapter_model.safetensors').write_bytes(b'fake test adapter')

    class SaveProcessor:
        def save_pretrained(self, path):
            pass

    monkeypatch.setattr(entry, 'check_runtime', lambda:{'test':'fake'})
    monkeypatch.setattr(entry, 'load_model', lambda *args:(Model(), SaveProcessor()))
    monkeypatch.setattr(entry, 'encode', lambda *args:{'input_ids':torch.ones(1,2, dtype=torch.long)})
    monkeypatch.setattr(entry, 'move_to_cuda', lambda batch:batch)
    monkeypatch.setattr(torch, 'autocast', lambda *args,**kwargs:contextlib.nullcontext())
    for name in ('reset_peak_memory_stats','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda:0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'test GPU stub')
    monkeypatch.setattr(entry, 'generate_evaluation', lambda *args:{'f1':0.2,'malformed_responses':0})
    smoke = tmp_path / 'smoke'
    common = ['run_vlm_ner.py','--model',str(model_dir),'--data',str(data)]
    monkeypatch.setattr('sys.argv', common+['--stage','smoke','--output',str(smoke)])
    entry.main()
    report = smoke / 'smoke_report.json'
    value = json.loads(report.read_text())
    assert value['passed'] is True and value['heldout_accuracy_evaluated'] is False
    assert value['optimizer_steps'] == 8
    trained = tmp_path / 'trained'
    monkeypatch.setattr('sys.argv', common+['--stage','train','--smoke-report',str(report),'--output',str(trained)])
    entry.main()
    assert (trained/'pilot_report.json').is_file()
    assert (trained/'best_adapter'/'adapter_model.safetensors').is_file()
    assert not (trained/'model.safetensors').exists()
    value['passed'] = False
    report.write_text(json.dumps(value), encoding='utf-8')
    monkeypatch.setattr('sys.argv', common+['--stage','train','--smoke-report',str(report),'--output',str(tmp_path/'blocked')])
    with pytest.raises(ValueError, match='passing GPU smoke'):
        entry.main()
