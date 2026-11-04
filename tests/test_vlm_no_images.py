"""Frozen-reference and no-image inference orchestration, using a fake GPU/model."""
import builtins
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from modules.vlm_ner import load_prepared, prepare_data, target
from scripts.create_context_internal_split import create_split
from scripts import evaluate_vlm_no_images as ablation


def records(rows, malformed_id=None):
    return [{'source_id': row['source_id'], 'response': 'bad JSON' if row['source_id'] == malformed_id else target(row),
             'error': 'bad JSON' if row['source_id'] == malformed_id else None,
             'predicted': [] if row['source_id'] == malformed_id else
             [[e['start'], e['end'], e['label']] for e in row['entities']]} for row in rows]


class NoImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        quiet = patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        original = ablation.original
        source = self.root / 'original_train.txt'
        raw = [{'text': ['Kim', f'word{i}'], 'label': ['B-PER', 'O'], 'images': []} for i in range(20)]
        source.write_text(''.join(repr(row) + '\n' for row in raw), encoding='utf-8')
        split = create_split(source, self.root / 'split', 'MI')
        self.data = prepare_data(split, self.root / 'prepared', self.root, self.root, panel_size=2)
        self.manifest, self.rows = load_prepared(self.data)
        self.model_dir = self.root / 'model'
        self.model_dir.mkdir()
        (self.model_dir / 'model.safetensors').write_bytes(b'fake base model')
        self.run = self.root / 'run'
        self.adapter = self.run / 'best_adapter'
        self.adapter.mkdir(parents=True)
        ablation.write_json(self.adapter / 'adapter_config.json', {'test': 'fake'})
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'fake adapter')
        self.profile = {
            'schema_version': 'vlm-qlora-profile-v1', 'epochs': 2, 'input_mode': 'multi_image',
            'data_sha256': original.sha256_file(self.data),
            'model_files_sha256': original.model_binding(self.model_dir),
            'implementation_sha256': {name: original.sha256_file(original.PROJECT / name)
                                      for name in original.BOUND_IMPLEMENTATION},
            'seed': 1234, 'min_pixels': 3136, 'max_pixels': 100352,
            'max_length': 2048, 'max_new_tokens': 256, 'versions': {'test': 'mock runtime'},
        }
        self.save_records(self.run / 'dev_epoch2.jsonl', records(self.rows['dev_panel']))
        panel = original.checked_records(self.rows['dev_panel'], records(self.rows['dev_panel']))
        panel_metrics = original.metrics_for(self.rows['dev_panel'], panel, original.SCOPES['dev_panel'])
        pilot = {'schema_version': 'vlm-ner-pilot-v1', 'profile': self.profile,
                 'official_val_or_test_read': False, 'best_epoch': 2, 'best_dev_panel_f1': 1.,
                 'history': [{'epoch': 0, 'metrics': {'f1': 0.}}, {'epoch': 1, 'metrics': {'f1': .5}},
                             {'epoch': 2, 'metrics': panel_metrics}]}
        ablation.write_json(self.run / 'pilot_report.json', pilot)
        ablation.write_json(self.run / 'config.json', {'stage': 'train', 'profile': self.profile,
                                                      'official_val_or_test_read': False})
        _, _, _, nonpanel, _, bindings = original.validate_run(self.run, self.model_dir, self.data)
        self.groups = {'full_dev': self.rows['dev'], 'dev_panel': self.rows['dev_panel'], 'dev_nonpanel': nonpanel}
        matched = original.checked_records(self.rows['dev'], records(self.rows['dev']))
        self.reference = self.root / 'reference'
        self.reference.mkdir()
        self.report_path = self.reference / 'evaluation_report.json'
        self.report = {'schema_version': 'vlm-ner-dev-evaluation-v1', 'profile': self.profile, 'best_epoch': 2,
                       'bindings': bindings, 'source_ids': {role: [r['source_id'] for r in rows]
                                                          for role, rows in self.groups.items()},
                       'official_val_or_test_read': False, 'optimizer_steps': 0, 'new_checkpoint_selection': False,
                       'panel_reproduction': {'passed': True},
                       'metrics': {role: original.metrics_for(rows, matched, original.SCOPES[role])
                                   for role, rows in self.groups.items()}}
        ablation.write_json(self.report_path, self.report)
        self.save_records(self.reference / 'dev_predictions.jsonl', records(self.rows['dev']))
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        ablation.write_json(self.audit / 'error_audit_report.json', {
            'schema_version': 'baseline-qlora-error-audit-v1', 'diagnostic_only': True,
            'source_ids': self.report['source_ids'], 'official_val_or_test_read': False,
            'model_inference_runs': 0, 'training_runs': 0,
            'input_sha256': {str(path.resolve()): original.sha256_file(path)
                             for path in (self.data, self.report_path, self.reference / 'dev_predictions.jsonl')},
        })
        self.args = SimpleNamespace(run=str(self.run), reference_evaluation=str(self.reference),
                                    reference_audit=str(self.audit),
                                    model=str(self.model_dir), data=str(self.data),
                                    output=str(self.root / 'ablation'), check_only=False)
        self.fake_model, self.processor = object(), object()
        self.fake_torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
            cuda=SimpleNamespace(reset_peak_memory_stats=Mock(), max_memory_allocated=lambda: 0))

    @staticmethod
    def save_records(path, values):
        path.write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')

    @contextlib.contextmanager
    def runtime(self, generator):
        with patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(ablation.original, 'check_runtime', return_value=self.profile['versions']), \
                patch.object(ablation.original, 'load_frozen_adapter', return_value=(self.fake_model, self.processor)) as load, \
                patch.object(ablation.original, 'generate_evaluation', side_effect=generator) as generate:
            yield load, generate

    def generator(self, malformed_id=None, modify=None):
        def generate(model, processor, rows, mode, maximum, new_tokens, path):
            self.assertIs(model, self.fake_model)
            self.assertIs(processor, self.processor)
            self.assertEqual((maximum, new_tokens), (2048, 256))
            if mode == 'text':
                self.assertTrue(all(row['images'] == [] for row in rows))
            values = records(rows, malformed_id if mode == 'text' else None)
            self.save_records(path, values)
            if modify:
                modify(mode, path)
        return generate

    def test_check_only_verifies_real_fixture_reports_without_gpu_import_or_output(self):
        self.args.check_only = True
        imported = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'PIL', 'numpy'}:
                raise AssertionError('Heavy dependency imported by check-only: ' + name)
            return imported(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guard):
            result = ablation.evaluate(self.args)
        self.assertTrue(result['check_only'])
        self.assertEqual(result['source_ids'], self.report['source_ids'])
        self.assertFalse(Path(self.args.output).exists())

    def test_generates_matched_panel_then_all_no_image_inputs_once_and_keeps_malformed(self):
        # Give one synthetic nonpanel row a poison image descriptor in memory.
        # This tests orchestration only; fake generation never opens images.
        inputs = ablation.load_reference(self.args)
        nonpanel_row = inputs[1]['dev_nonpanel'][0]
        nonpanel_row['images'] = [{'path': 'must-not-be-opened.png', 'sha256': 'fake'}]
        before = {str(path): path.read_bytes() for directory in (self.run, self.reference)
                  for path in directory.rglob('*') if path.is_file()}
        with patch.object(ablation, 'load_reference', return_value=inputs), \
                self.runtime(self.generator(malformed_id=nonpanel_row['source_id'])) as (load, generate):
            result = ablation.evaluate(self.args)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual([call.args[3] for call in generate.call_args_list], ['multi_image', 'text'])
        self.assertEqual([r['source_id'] for r in generate.call_args_list[1].args[2]], self.report['source_ids']['full_dev'])
        self.assertEqual(nonpanel_row['images'][0]['path'], 'must-not-be-opened.png')
        load.assert_called_once_with(self.args.model, self.adapter, self.profile)
        full = result['comparisons']['full_dev']
        self.assertEqual((full['no_images']['tp'], full['no_images']['fp'], full['no_images']['fn']), (3, 0, 1))
        self.assertEqual(full['no_images']['malformed_responses'], 1)
        self.assertAlmostEqual(full['matched_f1_minus_no_images'], 1 - 6/7)
        self.assertEqual(result['training_profile']['input_mode'], 'multi_image')
        self.assertEqual(result['intervention']['inference_input_mode'], 'text')
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertFalse(result['new_checkpoint_selection'])
        self.assertFalse(result['official_val_or_test_read'])
        self.assertTrue(result['zero_image_reproduction']['passed'])
        self.assertEqual(before, {str(path): path.read_bytes() for directory in (self.run, self.reference)
                                  for path in directory.rglob('*') if path.is_file()})
        self.assertTrue((Path(self.args.output) / 'ablation_report.json').is_file())

    def test_changed_reference_binding_profile_scope_and_flags_stop_before_loading(self):
        for key, value in (('bindings', {}), ('profile', {}), ('source_ids', {}), ('best_epoch', 1),
                           ('optimizer_steps', 1), ('new_checkpoint_selection', True),
                           ('official_val_or_test_read', True), ('panel_reproduction', {'passed': False})):
            with self.subTest(key=key):
                changed = {**self.report, key: value}
                ablation.write_json(self.report_path, changed)
                with self.assertRaisesRegex(ValueError, 'identical frozen multi-image run'):
                    ablation.load_reference(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_reference_metrics_and_raw_prediction_integrity_are_checked(self):
        self.report['metrics']['full_dev']['f1'] = .1
        ablation.write_json(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, 'saved metrics'):
            ablation.load_reference(self.args)
        for values in (records(self.rows['dev'])[:-1], records(self.rows['dev']) + records(self.rows['dev'])[:1]):
            self.save_records(self.reference / 'dev_predictions.jsonl', values)
            with self.assertRaises(ValueError):
                ablation.load_reference(self.args)

    def test_changed_panel_in_full_reference_is_rejected_even_if_report_metrics_were_recomputed(self):
        values = records(self.rows['dev'], self.rows['dev_panel'][0]['source_id'])
        self.save_records(self.reference / 'dev_predictions.jsonl', values)
        with self.assertRaisesRegex(ValueError, 'saved best panel'):
            ablation.load_reference(self.args)

    def test_historical_hash_binding_rejects_changed_reference_bytes_even_with_identical_metrics(self):
        self.save_records(self.reference / 'dev_predictions.jsonl', list(reversed(records(self.rows['dev']))))
        with self.assertRaisesRegex(ValueError, 'historical offline audit binding'):
            ablation.load_reference(self.args)

    def test_text_trained_run_is_rejected(self):
        for name in ('pilot_report.json', 'config.json'):
            value = ablation.original.read_json(self.run / name)
            value['profile']['input_mode'] = 'text'
            ablation.write_json(self.run / name, value)
        with self.assertRaisesRegex(ValueError, 'multi-image-trained adapter'):
            ablation.load_reference(self.args)

    def test_panel_mismatch_stops_before_ablated_generation(self):
        def mismatch(mode, path):
            if mode == 'multi_image':
                self.save_records(path, records(self.rows['dev_panel'], self.rows['dev_panel'][0]['source_id']))
        with self.runtime(self.generator(modify=mismatch)) as (_, generate):
            with self.assertRaisesRegex(ValueError, 'no ablated generation'):
                ablation.evaluate(self.args)
        generate.assert_called_once()
        self.assertFalse((Path(self.args.output) / 'no_image_predictions.jsonl').exists())
        self.assertTrue((Path(self.args.output) / 'failure.json').is_file())

    def test_originally_image_free_rows_must_keep_predictions(self):
        with self.runtime(self.generator(malformed_id=self.rows['dev'][0]['source_id'])):
            with self.assertRaisesRegex(ValueError, 'Originally image-free inputs changed'):
                ablation.evaluate(self.args)
        self.assertFalse((Path(self.args.output) / 'ablation_report.json').exists())

    def test_inputs_changed_during_generation_are_detected(self):
        def tamper(mode, path):
            if mode == 'text':
                self.report_path.write_text('changed', encoding='utf-8')
        with self.runtime(self.generator(modify=tamper)):
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                ablation.evaluate(self.args)
        self.assertFalse((Path(self.args.output) / 'ablation_report.json').exists())

    def test_adapter_changed_during_generation_is_detected(self):
        def tamper(mode, path):
            if mode == 'text':
                (self.adapter / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.runtime(self.generator(modify=tamper)):
            with self.assertRaisesRegex(ValueError, 'adapter changed'):
                ablation.evaluate(self.args)

    def test_wrong_runtime_stops_before_creating_output(self):
        with patch.object(ablation.original, 'check_runtime', return_value={'wrong': 'runtime'}):
            with self.assertRaisesRegex(ValueError, 'exact runtime'):
                ablation.evaluate(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_existing_outputs_and_all_input_directories_are_protected(self):
        for directory in (self.run, self.reference, self.audit, self.model_dir, self.data.parent):
            self.args.output = str(directory / 'nested_ablation')
            with self.assertRaisesRegex(ValueError, 'separate from the frozen inputs'):
                ablation.evaluate(self.args)
        existing = self.root / 'existing'
        existing.mkdir()
        marker = existing / 'keep.txt'
        marker.write_text('keep', encoding='utf-8')
        self.args.output = str(existing)
        with self.assertRaises(FileExistsError):
            ablation.evaluate(self.args)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
