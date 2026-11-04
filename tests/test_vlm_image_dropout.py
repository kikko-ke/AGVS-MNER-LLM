"""Fixed dropout and stage gates; optimizer checks use a tiny real CPU model."""
import contextlib
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from modules.vlm_ner import target
from scripts import run_vlm_image_dropout as entry

try:
    import torch
except ImportError:
    torch = None


def predictions(rows, bad_id=None):
    return [{'source_id': row['source_id'], 'response': 'bad JSON' if row['source_id'] == bad_id else target(row),
             'error': 'invalid' if row['source_id'] == bad_id else None,
             'predicted': [] if row['source_id'] == bad_id else [[0, 1, 'MIS']]} for row in rows]


@contextlib.contextmanager
def temporary_module(name, module):
    # Restore just the substituted key. Restoring all of sys.modules unloads
    # lazily imported Torch modules while leaving their C++ registrations alive.
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class SelectionTests(unittest.TestCase):
    def test_dropout_decisions_use_only_ids_and_do_not_advance_rng(self):
        class OnlyID(dict):
            def __getitem__(self, name):
                if name != 'source_id':
                    raise AssertionError('Dropout read data/gold: ' + name)
                return super().__getitem__(name)
        rows = [OnlyID(source_id=index) for index in range(10000)]
        before = random.getstate()
        plan = entry.dropout_plan(rows)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(plan, entry.dropout_plan(rows))
        self.assertNotEqual(plan['dropped_source_ids']['1'], plan['dropped_source_ids']['2'])
        for ids in plan['dropped_source_ids'].values():
            self.assertGreater(len(ids), 2300)
            self.assertLess(len(ids), 2700)

    def test_training_view_keeps_all_words_gold_and_original_rows(self):
        row = {'source_id': 1, 'tokens': ['Named'], 'entities': [{'start': 0, 'end': 1, 'label': 'MIS'}],
               'images': [{'path': 'image.png', 'sha256': 'original'}]}
        copied = copy.deepcopy(row)
        removed = entry.training_view(row, True)
        self.assertEqual(removed['images'], [])
        self.assertIs(removed['tokens'], row['tokens'])
        self.assertIs(removed['entities'], row['entities'])
        self.assertIs(entry.training_view(row, False), row)
        self.assertEqual(row, copied)

    def test_invalid_ids_epochs_and_duplicate_training_rows_are_rejected(self):
        for sid, epoch in ((True, 1), (-1, 1), (1, 0), (1, 3)):
            with self.assertRaises(ValueError):
                entry.should_drop(sid, epoch)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            entry.dropout_plan([{'source_id': 1}, {'source_id': 1}])


@unittest.skipIf(torch is None, 'Real CPU torch is required for optimizer/stage tests')
class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        quiet = patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        self.rows = {'train': [self.row(i) for i in range(12)], 'dev': [self.row(i) for i in range(12, 18)]}
        self.rows['dev_panel'] = self.rows['dev'][:2]
        self.nonpanel = self.rows['dev'][2:]
        self.manifest = {'gold_scope': 'all original words', 'panel_source_ids': [12, 13]}
        self.base_dir = self.root / 'base'
        self.base_dir.mkdir()
        (self.base_dir / 'model.safetensors').write_bytes(b'fake unit-test base')
        self.control = self.root / 'control'
        self.control.mkdir()
        (self.control / 'keep.txt').write_text('original control', encoding='utf-8')
        data = self.root / 'prepared' / 'manifest.json'
        data.parent.mkdir()
        entry.write_json(data, {'test': 'fixture'})
        self.base_profile = {**entry.BASE_SETTINGS, 'schema_version': 'vlm-qlora-profile-v1',
            'versions': {'test': 'fake runtime'}, 'data_sha256': entry.sha256_file(data),
            'model_files_sha256': entry.backend.model_binding(self.base_dir),
            'implementation_sha256': {name: entry.sha256_file(entry.PROJECT / name)
                                      for name in entry.original.BOUND_IMPLEMENTATION}}
        self.control_report = {'profile': self.base_profile}
        self.control_bindings = {'training_config_sha256': entry.sha256_file(self.control / 'keep.txt')}
        validator = patch.object(entry.original, 'validate_run', return_value=(
            self.control_report, self.manifest, self.rows, self.nonpanel, {}, self.control_bindings))
        validator.start()
        self.addCleanup(validator.stop)
        self.args = SimpleNamespace(stage='smoke', control_run=str(self.control), model=str(self.base_dir), data=str(data),
            output=str(self.root / 'smoke'), smoke_report=str(self.root / 'smoke' / 'smoke_report.json'),
            run=str(self.root / 'candidate'))
        self.state = entry.load_control(self.args)
        self.profile = self.state[0]
        self.forwards, self.generations, self.encodings = [], [], []
        owner = self
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[.3, .4]]))
            def forward(self, input_ids, use_cache):
                owner.forwards.append((int(input_ids[0, 0]) - 1, int(input_ids[0, 1]) == 2))
                return SimpleNamespace(loss=(self.weight * input_ids.float()).square().mean())
            def save_pretrained(self, path, safe_serialization, save_embedding_layers):
                owner.assertTrue(safe_serialization)
                owner.assertFalse(save_embedding_layers)
                path = Path(path)
                path.mkdir(exist_ok=True)
                entry.write_json(path / 'adapter_config.json', {'test': 'fake CPU adapter'})
                # Intentionally fake serialization, never consumed by PEFT in tests.
                (path / 'adapter_model.safetensors').write_bytes(repr(self.weight.detach().tolist()).encode())
        class Processor:
            def save_pretrained(self, path):
                entry.write_json(Path(path) / 'processor_config.json', {'test': 'fake processor'})
        self.model, self.processor = TinyModel(), Processor()
        self.initial = self.model.weight.detach().clone()
        self.loaded_weights = []
        self.schedule = Mock(side_effect=lambda optimizer, warmup, steps:
                             torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.))

    @staticmethod
    def row(sid):
        return {'source_id': sid, 'tokens': ['Named', f'word{sid}'],
                'entities': [{'start': 0, 'end': 1, 'label': 'MIS'}],
                'images': [{'path': f'fake-image{sid}.png', 'sha256': str(sid)}]}

    def encode(self, processor, row, mode, training, maximum):
        self.encodings.append((row['source_id'], bool(row['images']), training))
        self.assertEqual(mode, 'multi_image')
        return {'input_ids': torch.tensor([[row['source_id'] + 1, 2 if row['images'] else 1]])}

    def generate(self, model, processor, rows, mode, maximum, new_tokens, path):
        self.generations.append((Path(path).name, copy.deepcopy(rows)))
        self.assertEqual(mode, 'multi_image')
        bad = rows[0]['source_id'] if Path(path).name == 'dev_epoch0.jsonl' else None
        if Path(path).name == 'nonpanel_predictions.jsonl':
            bad = rows[-1]['source_id']
        values = predictions(rows, bad)
        Path(path).write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')
        indexed = entry.original.checked_records(rows, values)
        return entry.original.metrics_for(rows, indexed, 'test original-image panel')

    def frozen(self, *args):
        self.model.requires_grad_(False)
        self.model.eval()
        return self.model, self.processor

    def fresh_model(self, *args):
        self.model.requires_grad_(True)
        with torch.no_grad():
            self.model.weight.copy_(self.initial)
        self.loaded_weights.append(self.model.weight.detach().clone())
        return self.model, self.processor

    @contextlib.contextmanager
    def runtime(self, generator=None):
        transformers = ModuleType('transformers')
        transformers.get_linear_schedule_with_warmup = self.schedule
        with contextlib.ExitStack() as stack:
            stack.enter_context(temporary_module('transformers', transformers))
            stack.enter_context(patch.object(entry.backend, 'check_runtime', return_value=self.profile['versions']))
            load = stack.enter_context(patch.object(entry.backend, 'load_model', side_effect=self.fresh_model))
            stack.enter_context(patch.object(entry.original, 'load_frozen_adapter', side_effect=self.frozen))
            stack.enter_context(patch.object(entry.backend, 'encode', side_effect=self.encode))
            stack.enter_context(patch.object(entry.backend, 'move_to_cuda', side_effect=lambda batch: batch))
            gen = stack.enter_context(patch.object(entry.backend, 'generate_evaluation', side_effect=generator or self.generate))
            stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a, **k: contextlib.nullcontext()))
            for name in ('reset_peak_memory_stats', 'max_memory_allocated', 'max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='fake CPU test runtime'))
            yield load, gen

    def smoke(self):
        with self.runtime():
            return entry.run_training(self.args, *self.state[:3])

    def train(self):
        self.smoke()
        self.args.stage, self.args.output = 'train', self.args.run
        self.forwards.clear()
        with self.runtime():
            return entry.run_training(self.args, *self.state[:3])

    def test_smoke_performs_eight_real_updates_and_both_generation_conditions(self):
        before = copy.deepcopy(self.rows)
        result = self.smoke()
        self.assertTrue(result['passed'])
        self.assertEqual(result['optimizer_steps'], 8)
        self.assertEqual(len(self.forwards), 8)
        self.assertEqual(sum(view['images_removed'] for view in result['smoke_views']), 4)
        self.assertFalse(torch.equal(self.initial, self.model.weight.detach()))
        self.assertEqual(result['generation_conditions'], ['original_images', 'no_images'])
        self.assertTrue(self.generations[0][1][0]['images'])
        self.assertEqual(self.generations[1][1][0]['images'], [])
        self.assertEqual(self.rows, before)
        self.assertFalse(result['heldout_accuracy_evaluated'])

    def test_training_keeps_every_row_once_per_epoch_uses_fixed_plan_and_original_dev_images(self):
        before = copy.deepcopy(self.rows)
        result = self.train()
        self.assertEqual(result['optimizer_steps'], math.ceil(12/8) * 2)
        self.assertEqual(result['best_epoch'], 1)  # Epochs 1 and 2 tie; keep the first.
        for epoch in (1, 2):
            observed = self.forwards[(epoch-1)*12:epoch*12]
            order = torch.randperm(12, generator=torch.Generator().manual_seed(1234 + epoch)).tolist()
            self.assertEqual([sid for sid, _ in observed], order)
            self.assertEqual(sorted(sid for sid, _ in observed), list(range(12)))
            self.assertEqual({sid for sid, images in observed if not images},
                             {sid for sid in range(12) if entry.should_drop(sid, epoch)})
        for name, rows in self.generations:
            if name.startswith('dev_epoch'):
                self.assertTrue(all(row['images'] for row in rows))
        self.assertEqual(self.rows, before)
        self.assertEqual((self.control / 'keep.txt').read_text(encoding='utf-8'), 'original control')
        self.assertFalse((Path(self.args.run) / 'model.safetensors').exists())
        self.assertEqual(result['best_adapter_sha256'], entry.backend.model_binding(Path(self.args.run) / 'best_adapter'))
        self.assertEqual(len(self.loaded_weights), 2)
        self.assertTrue(all(torch.equal(value, self.initial) for value in self.loaded_weights))

    def test_inconsistent_or_legacy_smoke_reports_cannot_open_training_gate(self):
        result = self.smoke()
        path = Path(self.args.smoke_report)
        for key, value in (('schema_version', 'vlm-ner-smoke-v1'), ('passed', False), ('optimizer_steps', 7),
                            ('profile', {}), ('finite_nonzero_gradients', False), ('smoke_views', []),
                            ('generation_conditions', ['original_images'])):
            with self.subTest(key=key):
                entry.write_json(path, {**result, key: value})
                with self.assertRaisesRegex(ValueError, 'passing image-dropout GPU smoke'):
                    entry.check_smoke(path, self.profile, entry.dropout_plan(self.rows['train']))
        self.args.stage, self.args.output = 'train', self.args.run
        with patch.object(entry.backend, 'check_runtime', side_effect=AssertionError('Gate must fail before GPU probe')):
            with self.assertRaises(ValueError):
                entry.run_training(self.args, *self.state[:3])
        self.assertFalse(Path(self.args.output).exists())

    def test_changed_smoke_plan_is_rejected(self):
        self.smoke()
        entry.write_json(Path(self.args.smoke_report).parent / 'dropout_plan.json', {})
        with self.assertRaisesRegex(ValueError, 'identical profile/plan'):
            entry.check_smoke(self.args.smoke_report, self.profile, entry.dropout_plan(self.rows['train']))

    def test_frozen_evaluation_reproduces_panel_and_generates_each_original_dev_row_once(self):
        self.train()
        self.args.output = str(self.root / 'evaluation')
        self.generations.clear()
        with self.runtime(), patch.object(torch.optim, 'AdamW', side_effect=AssertionError('No optimizer in evaluation')):
            result = entry.evaluate_candidate(self.args, *self.state)
        self.assertEqual(len(self.generations), 2)
        ids = [row['source_id'] for _, rows in self.generations for row in rows]
        self.assertEqual(sorted(ids), list(range(12, 18)))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(row['images'] for _, rows in self.generations for row in rows))
        self.assertTrue(result['panel_reproduction']['passed'])
        self.assertEqual(result['metrics']['full_dev']['malformed_responses'], 1)
        self.assertEqual(result['metrics']['full_dev']['tp'], 5)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertFalse(result['new_checkpoint_selection'])
        self.assertFalse(result['official_val_or_test_read'])
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))
        path = Path(self.args.output) / 'dev_predictions.jsonl'
        self.assertEqual(result['dev_predictions_sha256'], entry.sha256_file(path))

    def test_panel_mismatch_stops_before_nonpanel_evaluation(self):
        self.train()
        self.args.output = str(self.root / 'failed_evaluation')
        def bad_panel(model, processor, rows, mode, maximum, new_tokens, path):
            path.write_text(''.join(json.dumps(r) + '\n' for r in predictions(rows, rows[0]['source_id'])), encoding='utf-8')
        with self.runtime(generator=bad_panel) as (_, generated):
            with self.assertRaisesRegex(ValueError, 'full-dev evaluation stopped'):
                entry.evaluate_candidate(self.args, *self.state)
        generated.assert_called_once()
        self.assertFalse((Path(self.args.output) / 'nonpanel_predictions.jsonl').exists())

    def test_saved_adapter_tampering_is_rejected_before_gpu_loading(self):
        self.train()
        (Path(self.args.run) / 'best_adapter' / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs from the completed training report'):
            entry.validate_candidate(self.args, self.profile, self.rows)

    def test_wrong_checkpoint_rule_plan_and_budget_are_rejected(self):
        report = self.train()
        path = Path(self.args.run) / 'pilot_report.json'
        for name, value in (('best_epoch', 2), ('optimizer_steps', 99), ('profile', {})):
            entry.write_json(path, {**report, name: value})
            with self.assertRaises(ValueError):
                entry.validate_candidate(self.args, self.profile, self.rows)
        entry.write_json(path, report)
        entry.write_json(Path(self.args.run) / 'dropout_plan.json', {})
        with self.assertRaises(ValueError):
            entry.validate_candidate(self.args, self.profile, self.rows)

    def test_original_control_hyperparameters_must_match_fixed_experiment(self):
        self.base_profile['lr'] = 1e-3
        with self.assertRaisesRegex(ValueError, 'fixed seed1234 two-epoch'):
            entry.load_control(self.args)

    def test_changed_runtime_and_protected_output_stop_without_overwrite(self):
        with patch.object(entry.backend, 'check_runtime', return_value={'wrong': 'runtime'}):
            with self.assertRaisesRegex(ValueError, 'exact pinned runtime'):
                entry.run_training(self.args, *self.state[:3])
        self.assertFalse(Path(self.args.output).exists())
        for source in (self.args.control_run, self.args.model, Path(self.args.data).parent):
            with self.assertRaisesRegex(ValueError, 'separate from all input'):
                entry.protect_output(Path(source) / 'nested', self.args)
        with self.assertRaises(FileExistsError):
            entry.protect_output(self.control, self.args)


if __name__ == '__main__':
    unittest.main()
