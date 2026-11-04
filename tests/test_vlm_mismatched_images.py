"""Input-only permutation and frozen-control orchestration; no real ML model."""
import builtins
import contextlib
import copy
from itertools import permutations, product
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from modules.vlm_ner import target
from scripts import evaluate_vlm_mismatched_images as mismatch


def row(source_id, hashes):
    return {'source_id': source_id, 'tokens': ['Named', f'word{source_id}'],
            'entities': [{'start': 0, 'end': 1, 'label': 'MIS'}],
            'images': [{'path': f'{value}.png', 'sha256': value} for value in hashes]}


def records(rows, bad_id=None):
    return [{'source_id': value['source_id'],
             'response': 'invalid JSON' if value['source_id'] == bad_id else target(value),
             'error': 'invalid' if value['source_id'] == bad_id else None,
             'predicted': [] if value['source_id'] == bad_id else [[0, 1, 'MIS']]} for value in rows]


class PermutationTests(unittest.TestCase):
    def test_deterministic_input_only_mapping_preserves_bundles_counts_and_no_shared_content(self):
        class InputOnly(dict):
            def __getitem__(self, name):
                if name not in {'source_id', 'images'}:
                    raise AssertionError('Permutation read text/gold: ' + name)
                return super().__getitem__(name)
        rows = [InputOnly(row(index, [f'a{index}', f'b{index}'])) for index in range(8)]
        plan, skipped = mismatch.make_plan(rows)
        again, _ = mismatch.make_plan(list(reversed(rows)))
        self.assertEqual(plan, again)
        self.assertEqual(skipped, [])
        self.assertEqual({item['source_id'] for item in plan}, {item['donor_source_id'] for item in plan})
        self.assertCountEqual([item['original_images'] for item in plan], [item['replacement_images'] for item in plan])
        for item in plan:
            self.assertNotEqual(item['source_id'], item['donor_source_id'])
            self.assertEqual(len(item['original_images']), len(item['replacement_images']))
            self.assertFalse({i['sha256'] for i in item['original_images']} & {i['sha256'] for i in item['replacement_images']})

    def test_matching_agrees_with_exhaustive_small_graph_solutions(self):
        for size in range(1, 6):
            for labels in product('abc', repeat=size):
                rows = [row(index, [label]) for index, label in enumerate(labels)]
                possible = any(all(labels[index] != labels[donor] for index, donor in enumerate(order))
                               for order in permutations(range(size)))
                result = mismatch.match_bucket(rows)
                self.assertEqual(result is not None, possible, labels)
                if result:
                    self.assertEqual(set(result), set(result.values()))
                    self.assertTrue(all(labels[a] != labels[b] for a, b in result.items()))

    def test_panel_and_nonpanel_bundles_are_permuted_separately(self):
        rows = [row(index, [str(index)]) for index in range(6)]
        panel_ids = {0, 1}
        plan, skipped = mismatch.make_plan(rows, panel_ids)
        self.assertFalse(skipped)
        for item in plan:
            self.assertEqual(item['source_id'] in panel_ids, item['donor_source_id'] in panel_ids)
        for selected in (panel_ids, {2, 3, 4, 5}):
            subset = [item for item in plan if item['source_id'] in selected]
            self.assertCountEqual([item['original_images'] for item in subset],
                                  [item['replacement_images'] for item in subset])

    def test_entire_unmatchable_buckets_and_zero_image_rows_are_disclosed(self):
        rows = [row(0, []), row(1, ['a']), row(2, ['b']),
                row(3, ['x', 'y']), row(4, ['x', 'z']), row(5, ['u', 'v']),
                row(6, ['only', 'one', 'bundle'])]
        plan, skipped = mismatch.make_plan(rows)
        self.assertEqual({item['source_id'] for item in plan}, {1, 2})
        self.assertEqual([(item['images_per_sample'], item['samples']) for item in skipped], [(0, 1), (2, 3), (3, 1)])
        self.assertEqual(skipped[0]['reason'], 'no_images_to_permute')

    def test_duplicate_ids_and_changed_image_budget_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate source IDs'):
            mismatch.make_plan([row(1, ['a']), row(1, ['b'])])
        with self.assertRaisesRegex(ValueError, 'zero-to-four'):
            mismatch.make_plan([row(1, list('abcde'))])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        quiet = patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        self.rows = [row(index, [f'image{index}']) for index in range(4)] + [row(4, []), row(5, ['x', 'y'])]
        self.groups = {'full_dev': self.rows, 'dev_panel': self.rows[:2], 'dev_nonpanel': self.rows[2:]}
        self.matched = {r['source_id']: r for r in records(self.rows)}
        self.no_images = {r['source_id']: r for r in records(self.rows, bad_id=2)}
        self.profile = {'input_mode': 'multi_image', 'seed': 1234, 'max_length': 2048,
                        'max_new_tokens': 256, 'versions': {'test': 'fake runtime'}}
        self.pilot = {'profile': self.profile, 'best_epoch': 2}
        self.reference = {'source_ids': {role: [r['source_id'] for r in rows] for role, rows in self.groups.items()}}
        self.run = self.root / 'run'
        self.adapter = self.run / 'best_adapter'
        self.adapter.mkdir(parents=True)
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'fake test adapter')
        self.run_bindings = {'adapter_files_sha256': mismatch.original.model_binding(self.adapter)}
        source = self.root / 'frozen_source.txt'
        source.write_text('immutable control fixture', encoding='utf-8')
        self.bindings = {str(source): mismatch.sha256_file(source)}
        self.control = self.root / 'no_images'
        self.control.mkdir()
        self.report_path = self.control / 'ablation_report.json'
        self.predictions_path = self.control / 'no_image_predictions.jsonl'
        self.old = {
            'schema_version': 'vlm-no-image-ablation-v1', 'training_profile': self.profile, 'best_epoch': 2,
            'source_ids': self.reference['source_ids'], 'input_sha256': self.bindings,
            'adapter_files_sha256': self.run_bindings['adapter_files_sha256'],
            'panel_reproduction': {'passed': True}, 'zero_image_reproduction': {'passed': True},
            'intervention': {'inference_input_mode': 'text', 'all_images_removed': True,
                            'system_prompt_changed': False, 'word_content_changed': False, 'weights_changed': False},
            'official_val_or_test_read': False, 'optimizer_steps': 0, 'new_checkpoint_selection': False,
        }
        self.save_controls()
        panel = {r['source_id']: self.matched[r['source_id']] for r in self.groups['dev_panel']}
        self.common = (self.pilot, self.groups, panel, self.matched, self.reference, self.run_bindings, self.bindings)
        common = patch.object(mismatch.removal, 'load_reference', return_value=self.common)
        common.start()
        self.addCleanup(common.stop)
        self.args = SimpleNamespace(run=str(self.run), reference_evaluation=str(self.root / 'matched'),
            reference_audit=str(self.root / 'audit'), no_image_evaluation=str(self.control),
            model=str(self.root / 'base_model'), data=str(self.root / 'prepared' / 'manifest.json'),
            output=str(self.root / 'mismatch'), check_only=False)
        self.model, self.processor = object(), object()
        self.torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
            cuda=SimpleNamespace(reset_peak_memory_stats=Mock(), max_memory_allocated=lambda: 0))
        self.tqdm = ModuleType('tqdm')
        self.tqdm_auto = ModuleType('tqdm.auto')
        self.tqdm_auto.tqdm = lambda values, **kwargs: values

    def save_controls(self):
        self.old['comparisons'] = {}
        for role, rows in self.groups.items():
            left = mismatch.original.metrics_for(rows, self.matched, mismatch.original.SCOPES[role])
            right = mismatch.original.metrics_for(rows, self.no_images, mismatch.original.SCOPES[role])
            self.old['comparisons'][role] = {'matched_images': left, 'no_images': right,
                'matched_f1_minus_no_images': left['f1'] - right['f1'],
                'by_type_f1_delta': {label: left['by_type'][label]['f1'] - right['by_type'][label]['f1']
                                    for label in mismatch.LABELS}}
        mismatch.removal.write_json(self.report_path, self.old)
        self.predictions_path.write_text(''.join(json.dumps(r) + '\n' for r in self.no_images.values()), encoding='utf-8')

    @contextlib.contextmanager
    def runtime(self, generate=None, encoding=None):
        def fake_generate(model, processor, rows, mode, max_length, max_new_tokens, path):
            self.assertIs(model, self.model)
            self.assertEqual((mode, max_length, max_new_tokens), ('multi_image', 2048, 256))
            values = records(rows, bad_id=1 if path.name == 'mismatched_predictions.jsonl' else None)
            path.write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
        def fake_encode(processor, value, mode, training, budget):
            self.assertEqual((mode, training, budget), ('multi_image', False, 1792))
            self.assertEqual(len(value['images']), len(self.rows[value['source_id']]['images']))
            self.assertFalse({i['sha256'] for i in value['images']} &
                             {i['sha256'] for i in self.rows[value['source_id']]['images']})
            return {'input_ids': SimpleNamespace(shape=(1, 123))}
        with patch.dict(sys.modules, {'torch': self.torch, 'tqdm': self.tqdm, 'tqdm.auto': self.tqdm_auto}), \
                patch.object(mismatch.original, 'check_runtime', return_value=self.profile['versions']), \
                patch.object(mismatch.original, 'load_frozen_adapter', return_value=(self.model, self.processor)) as load, \
                patch.object(mismatch.original, 'generate_evaluation', side_effect=generate or fake_generate) as generated, \
                patch.object(mismatch, 'encode', side_effect=encoding or fake_encode) as encoded:
            yield load, generated, encoded

    def test_check_only_uses_no_heavy_dependencies_and_creates_no_output(self):
        self.args.check_only = True
        imported = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'PIL', 'numpy'}:
                raise AssertionError('Heavy dependency in check-only: ' + name)
            return imported(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guard):
            result = mismatch.evaluate(self.args)
        self.assertEqual(len(result['plan']), 4)
        self.assertEqual(len(result['skipped_buckets']), 2)
        self.assertFalse(Path(self.args.output).exists())

    def test_all_conditions_use_same_eligible_cohorts_and_only_mismatched_rows_are_generated(self):
        before = copy.deepcopy(self.rows)
        frozen = {p: p.read_bytes() for p in (self.report_path, self.predictions_path)}
        with self.runtime() as (load, generated, encoded):
            result = mismatch.evaluate(self.args)
        load.assert_called_once()
        self.assertEqual(generated.call_count, 2)
        self.assertEqual(encoded.call_count, 4)
        self.assertEqual([r['source_id'] for r in generated.call_args_list[0].args[2]], [0, 1])
        self.assertEqual([r['source_id'] for r in generated.call_args_list[1].args[2]], [0, 1, 2, 3])
        self.assertEqual(result['source_ids']['dev_nonpanel'], [2, 3])
        self.assertEqual(result['original_source_ids']['dev_nonpanel'], [2, 3, 4, 5])
        full = result['comparisons']['full_dev']
        for name in ('matched_images', 'no_images', 'mismatched_images'):
            self.assertEqual(full[name]['samples'], 4)
        self.assertEqual(full['mismatched_images']['malformed_responses'], 1)
        self.assertEqual(full['mismatched_images']['tp'], 3)
        self.assertAlmostEqual(full['matched_f1_minus_mismatched'], 1 - 6/7)
        self.assertEqual(result['maximum_mismatched_prompt_tokens'], 123)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertFalse(result['new_checkpoint_selection'])
        self.assertFalse(result['official_val_or_test_read'])
        self.assertEqual(self.rows, before)
        self.assertEqual(frozen, {p: p.read_bytes() for p in frozen})
        self.assertTrue((Path(self.args.output) / 'mismatch_report.json').is_file())

    def test_changed_no_image_report_bindings_flags_and_adapter_are_rejected(self):
        for name, value in (('input_sha256', {}), ('source_ids', {}), ('adapter_files_sha256', {}),
                            ('training_profile', {}), ('best_epoch', 1), ('optimizer_steps', 1),
                            ('panel_reproduction', {'passed': False}), ('zero_image_reproduction', {'passed': False}),
                            ('intervention', {}), ('official_val_or_test_read', True)):
            with self.subTest(name=name):
                mismatch.removal.write_json(self.report_path, {**self.old, name: value})
                with self.assertRaisesRegex(ValueError, 'identical frozen adapter'):
                    mismatch.load_controls(self.args)

    def test_no_image_raw_predictions_must_reproduce_metrics(self):
        values = records(self.rows)
        self.predictions_path.write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'ablation metrics'):
            mismatch.load_controls(self.args)

    def test_originally_image_free_control_must_still_agree(self):
        self.no_images[4] = records(self.rows, bad_id=4)[4]
        self.save_controls()
        with self.assertRaisesRegex(ValueError, 'image-free inputs disagree'):
            mismatch.load_controls(self.args)

    def test_panel_failure_stops_before_encoding_or_mismatched_generation(self):
        def bad_panel(model, processor, rows, mode, maximum, new_tokens, path):
            path.write_text(''.join(json.dumps(r) + '\n' for r in records(rows, bad_id=0)), encoding='utf-8')
        with self.runtime(generate=bad_panel) as (_, generated, encoded):
            with self.assertRaisesRegex(ValueError, 'mismatch generation stopped'):
                mismatch.evaluate(self.args)
        generated.assert_called_once()
        encoded.assert_not_called()
        self.assertTrue((Path(self.args.output) / 'failure.json').is_file())

    def test_budget_failure_stops_before_mismatched_generation(self):
        with self.runtime(encoding=Mock(side_effect=ValueError('Prompt exceeds fixed token budget'))) as (_, generated, _):
            with self.assertRaisesRegex(ValueError, 'fixed token budget'):
                mismatch.evaluate(self.args)
        generated.assert_called_once()
        self.assertFalse((Path(self.args.output) / 'mismatch_report.json').exists())

    def test_changed_input_and_changed_permutation_are_detected(self):
        def mutate_plan(processor, value, mode, training, budget):
            (Path(self.args.output) / 'image_permutation.json').write_text('changed', encoding='utf-8')
            return {'input_ids': SimpleNamespace(shape=(1, 123))}
        with self.runtime(encoding=mutate_plan):
            with self.assertRaisesRegex(ValueError, 'Frozen input or permutation changed'):
                mismatch.evaluate(self.args)

    def test_zero_eligible_rows_and_wrong_runtime_fail_before_output(self):
        with patch.object(mismatch, 'make_plan', return_value=([], [])):
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                mismatch.evaluate(self.args)
        with patch.object(mismatch.original, 'check_runtime', return_value={'wrong': 'runtime'}):
            with self.assertRaisesRegex(ValueError, 'exact runtime'):
                mismatch.evaluate(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_input_directories_and_existing_outputs_are_protected(self):
        for directory in (self.args.run, self.args.reference_evaluation, self.args.reference_audit,
                          self.args.no_image_evaluation, self.args.model, Path(self.args.data).parent):
            self.args.output = str(Path(directory) / 'nested')
            with self.assertRaisesRegex(ValueError, 'separate from frozen input'):
                mismatch.evaluate(self.args)
        existing = self.root / 'exists'
        existing.mkdir()
        marker = existing / 'keep.txt'
        marker.write_text('keep', encoding='utf-8')
        self.args.output = str(existing)
        with self.assertRaises(FileExistsError):
            mismatch.evaluate(self.args)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
