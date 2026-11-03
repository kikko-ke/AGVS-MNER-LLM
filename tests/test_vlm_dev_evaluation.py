"""Frozen-adapter evaluation checks; use tiny internal data and a fake GPU/model."""
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from modules.vlm_ner import load_prepared, prepare_data, target
from scripts.create_context_internal_split import create_split
from scripts import evaluate_vlm_ner as evaluation


def save_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def predictions(rows, malformed_id=None):
    return [{'source_id': row['source_id'],
             'response': 'invalid JSON' if row['source_id'] == malformed_id else target(row),
             'error': 'bad JSON' if row['source_id'] == malformed_id else None,
             'predicted': [] if row['source_id'] == malformed_id else
             [[entity['start'], entity['end'], entity['label']] for entity in row['entities']]}
            for row in rows]


class FakeModel:
    def __init__(self, missing_keys=(), unexpected_keys=()):
        self.parameter = SimpleNamespace(requires_grad=True)
        self.parameter.requires_grad_ = lambda value: setattr(self.parameter, 'requires_grad', value)
        self.peft_config = {'default': SimpleNamespace(
            r=8, lora_alpha=16, lora_dropout=0.05, target_modules={'language.q_proj'},
            bias='none', task_type='CAUSAL_LM', use_dora=False)}
        self.active_adapter = 'default'
        self.result = SimpleNamespace(missing_keys=list(missing_keys), unexpected_keys=list(unexpected_keys))
        self.load_adapter = Mock(side_effect=self.restore)
        self.eval = Mock()

    def restore(self, path, adapter_name, **kwargs):
        self.peft_config[adapter_name] = copy.deepcopy(self.peft_config['default'])
        return self.result

    def set_adapter(self, name):
        self.active_adapter = name
        self.parameter.requires_grad = True

    def parameters(self):
        return [self.parameter]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.quiet = patch('builtins.print')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)
        source = self.root / 'original_training.txt'
        raw = [{'text': ['Acme', f'today{i}'], 'label': ['B-ORG', 'O'], 'images': []}
               for i in range(20)]
        source.write_text(''.join(repr(row) + '\n' for row in raw), encoding='utf-8')
        split = create_split(source, self.root / 'split', 'MI')
        self.data = prepare_data(split, self.root / 'prepared', self.root, self.root, panel_size=2)
        self.manifest, self.rows = load_prepared(self.data)
        self.model_dir = self.root / 'base_model'
        self.model_dir.mkdir()
        (self.model_dir / 'model.safetensors').write_bytes(b'fake base model')
        self.run = self.root / 'trained'
        self.adapter = self.run / 'best_adapter'
        self.adapter.mkdir(parents=True)
        save_json(self.adapter / 'adapter_config.json', {'test': 'fake'})
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'fake trained adapter')
        self.profile = {
            'schema_version': 'vlm-qlora-profile-v1', 'epochs': 2,
            'data_sha256': evaluation.sha256_file(self.data),
            'model_files_sha256': evaluation.model_binding(self.model_dir),
            'implementation_sha256': {name: evaluation.sha256_file(evaluation.PROJECT / name)
                                      for name in evaluation.BOUND_IMPLEMENTATION},
            'input_mode': 'multi_image', 'seed': 1234, 'min_pixels': 3136, 'max_pixels': 100352,
            'max_length': 2048, 'max_new_tokens': 256, 'versions': {'test': 'mock runtime'},
        }
        records = predictions(self.rows['dev_panel'])
        self.reference = self.run / 'dev_epoch2.jsonl'
        self.reference.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        metrics = evaluation.metrics_for(self.rows['dev_panel'],
                    evaluation.checked_records(self.rows['dev_panel'], records), 'test')
        self.report = {
            'schema_version': 'vlm-ner-pilot-v1', 'profile': self.profile,
            'official_val_or_test_read': False, 'best_epoch': 2, 'best_dev_panel_f1': metrics['f1'],
            'history': [{'epoch': 0, 'metrics': {'f1': 0.}},
                        {'epoch': 1, 'metrics': {'f1': 0.5}}, {'epoch': 2, 'metrics': metrics}],
        }
        save_json(self.run / 'pilot_report.json', self.report)
        save_json(self.run / 'config.json', {
            'stage': 'train', 'profile': self.profile, 'official_val_or_test_read': False,
        })
        self.args = SimpleNamespace(run=str(self.run), model=str(self.model_dir), data=str(self.data),
                                    output=str(self.root / 'evaluation'))
        self.fake_model = FakeModel()
        self.fake_torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
            cuda=SimpleNamespace(reset_peak_memory_stats=Mock(),
                                 max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0))

    def validate(self):
        return evaluation.validate_run(self.run, self.model_dir, self.data)

    @contextlib.contextmanager
    def runtime(self, generator):
        with patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(evaluation, 'check_runtime', return_value=self.profile['versions']), \
                patch.object(evaluation, 'load_model', return_value=(self.fake_model, object())), \
                patch.object(evaluation, 'generate_evaluation', side_effect=generator) as generate:
            yield generate

    def generator(self, malformed_id=None):
        def generate(model, processor, rows, mode, maximum, new_tokens, path):
            self.assertEqual(model.active_adapter, 'evaluation')
            self.assertFalse(model.parameter.requires_grad)
            self.assertEqual((mode, maximum, new_tokens), ('multi_image', 2048, 256))
            path.write_text(''.join(json.dumps(record) + '\n' for record in
                                   predictions(rows, malformed_id)), encoding='utf-8')
            return {'scope': 'ignored original panel scope'}
        return generate

    def test_evaluates_panel_and_remainder_once_with_strict_full_dev_metrics(self):
        panel_ids = set(self.manifest['panel_source_ids'])
        nonpanel = [row for row in self.rows['dev'] if row['source_id'] not in panel_ids]
        before = {str(path): path.read_bytes() for path in self.run.rglob('*') if path.is_file()}
        with self.runtime(self.generator(malformed_id=nonpanel[0]['source_id'])) as generate:
            result = evaluation.evaluate(self.args)
        self.assertEqual(generate.call_count, 2)
        called_ids = [row['source_id'] for call in generate.call_args_list for row in call.args[2]]
        self.assertEqual(set(called_ids), {row['source_id'] for row in self.rows['dev']})
        self.assertEqual(len(called_ids), len(set(called_ids)))
        self.assertTrue(result['panel_reproduction']['passed'])
        self.assertEqual(result['best_epoch'], 2)
        self.assertEqual(result['optimizer_steps'], 0)
        metrics = result['metrics']
        self.assertEqual(metrics['full_dev']['samples'], 4)
        self.assertEqual(metrics['full_dev']['malformed_responses'], 1)
        self.assertEqual((metrics['full_dev']['tp'], metrics['full_dev']['fn']), (3, 1))
        self.assertAlmostEqual(metrics['dev_nonpanel']['f1'], 2 / 3)
        self.assertEqual(metrics['dev_panel']['f1'], 1.)
        self.assertFalse(result['official_val_or_test_read'])
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.run.rglob('*') if path.is_file()})
        self.fake_model.load_adapter.assert_called_once_with(
            str(self.adapter.resolve()), adapter_name='evaluation', is_trainable=False, local_files_only=True)
        output = Path(self.args.output)
        records = evaluation.read_records(output / 'dev_predictions.jsonl')
        self.assertEqual([r['source_id'] for r in records], [r['source_id'] for r in self.rows['dev']])
        self.assertTrue((output / 'evaluation_report.json').is_file())

    def test_reproduction_mismatch_stops_before_nonpanel_generation(self):
        with self.runtime(self.generator(malformed_id=self.rows['dev_panel'][0]['source_id'])) as generate:
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                evaluation.evaluate(self.args)
        generate.assert_called_once()
        output = Path(self.args.output)
        self.assertFalse(evaluation.read_json(output / 'panel_reproduction.json')['passed'])
        self.assertTrue((output / 'failure.json').is_file())
        self.assertFalse((output / 'evaluation_report.json').exists())

    def test_incomplete_adapter_load_is_rejected(self):
        self.fake_model = FakeModel(missing_keys=['language.q_proj.lora_A.evaluation.weight'])
        with self.runtime(self.generator()) as generate:
            with self.assertRaisesRegex(ValueError, 'did not load exactly'):
                evaluation.evaluate(self.args)
        generate.assert_not_called()

    def test_base_model_change_is_rejected_before_output_creation(self):
        (self.model_dir / 'model.safetensors').write_bytes(b'changed base model')
        with self.assertRaisesRegex(ValueError, 'Base model files differ'):
            evaluation.evaluate(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_training_implementation_and_manifest_changes_are_rejected(self):
        self.profile['implementation_sha256']['modules/vlm_training.py'] = 'changed'
        save_json(self.run / 'pilot_report.json', self.report)
        save_json(self.run / 'config.json', {'stage': 'train', 'profile': self.profile,
                                           'official_val_or_test_read': False})
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            self.validate()
        self.data.write_text(self.data.read_text() + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            self.validate()

    def test_changed_panel_reference_and_wrong_checkpoint_selection_are_rejected(self):
        changed = predictions(self.rows['dev_panel'], self.rows['dev_panel'][0]['source_id'])
        self.reference.write_text(''.join(json.dumps(record) + '\n' for record in changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'disagree with the pilot report'):
            self.validate()
        self.report['best_epoch'] = 1
        save_json(self.run / 'pilot_report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'original panel selection'):
            self.validate()

    def test_missing_duplicate_and_inconsistent_predictions_are_rejected(self):
        rows = self.rows['dev']
        records = predictions(rows)
        for values in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                evaluation.checked_records(rows, values)
        records[0]['predicted'] = []
        with self.assertRaisesRegex(ValueError, 'strict response parsing'):
            evaluation.checked_records(rows, records)

    def test_training_and_existing_evaluation_outputs_are_protected(self):
        self.args.output = str(self.run / 'evaluation')
        with self.assertRaisesRegex(ValueError, 'separate from the training run'):
            evaluation.evaluate(self.args)
        self.args.output = str(self.root / 'existing')
        output = Path(self.args.output)
        output.mkdir()
        marker = output / 'preserve.txt'
        marker.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            evaluation.evaluate(self.args)
        self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
