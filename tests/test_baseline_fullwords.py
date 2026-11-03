"""Full-word scoring and baseline provenance tests; no real model or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.vlm_ner import load_prepared, prepare_data, span_metrics
from scripts.create_context_internal_split import create_split
from scripts import evaluate_baseline_fullwords as baseline


LABELS = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-PER': 3, 'X': 4}


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class ProjectionTests(unittest.TestCase):
    def test_unobserved_gold_entities_remain_false_negatives(self):
        row = {'source_id': 8, 'tokens': ['Acme', 'Corp', 'Alice'], 'entities': [
            {'start': 0, 'end': 2, 'label': 'ORG'}, {'start': 2, 'end': 3, 'label': 'PER'}]}
        record, native_gold, native_prediction = baseline.project_path(
            row, [4, 1, 2, 4], [True, False, False, True], [0, 1], LABELS, 4)
        self.assertEqual(record['predicted'], [[0, 2, 'ORG']])
        self.assertEqual(span_metrics([native_gold], [native_prediction])['f1'], 1.)
        score = baseline.metrics_for([row], {8: record}, 'full original words')
        self.assertEqual((score['tp'], score['fp'], score['fn']), (1, 0, 1))

    def test_entity_crossing_window_boundary_is_not_shortened_in_gold(self):
        row = {'source_id': 1, 'tokens': ['Acme', 'Corp'],
               'entities': [{'start': 0, 'end': 2, 'label': 'ORG'}]}
        record, gold, predicted = baseline.project_path(
            row, [4, 1, 4], [True, False, True], [0], LABELS, 3)
        self.assertEqual(span_metrics([gold], [predicted])['f1'], 1.)
        score = baseline.metrics_for([row], {1: record}, 'full original words')
        self.assertEqual((score['tp'], score['fp'], score['fn']), (0, 1, 1))

    def test_empty_tokenization_gaps_do_not_join_nonadjacent_original_words(self):
        row = {'source_id': 2, 'tokens': ['Acme', '', 'Corp'],
               'entities': [{'start': 0, 'end': 3, 'label': 'ORG'}]}
        record, gold, predicted = baseline.project_path(
            row, [4, 1, 2, 4], [True, False, False, True], [0, 2], LABELS, 4)
        self.assertEqual(record['predicted'], [[0, 1, 'ORG'], [2, 3, 'ORG']])
        self.assertEqual(span_metrics([gold], [predicted])['f1'], 1.)
        score = baseline.metrics_for([row], {2: record}, 'full original words')
        self.assertEqual((score['tp'], score['fp'], score['fn']), (0, 2, 1))

    def test_wordpiece_continuations_special_tokens_and_x_predictions_are_filtered(self):
        row = {'source_id': 4, 'tokens': ['longword', 'Alice'], 'entities': []}
        record, _, native = baseline.project_path(
            row, [3, 4, 1, 3, 1], [True, False, True, False, True], [0, 1], LABELS, 5)
        self.assertEqual(record['predicted'], [[1, 2, 'PER']])
        self.assertEqual(native, [(1, 2, 'PER')])

    def test_invalid_paths_and_duplicate_original_indices_are_rejected(self):
        row = {'source_id': 1, 'tokens': ['A', 'B'], 'entities': []}
        with self.assertRaises(ValueError):
            baseline.project_path(row, [4, 1], [True, False, True], [0], LABELS, 3)
        with self.assertRaises(ValueError):
            baseline.project_path(row, [4, 1, 2, 4], [True, False, False, True], [0, 0], LABELS, 4)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        quiet = patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        source = self.root / 'original_train.txt'
        source.write_text(''.join(repr({'text': ['Acme', f'today{i}'],
            'label': ['B-ORG', 'O'], 'images': []}) + '\n' for i in range(20)), encoding='utf-8')
        internal = create_split(source, self.root / 'split', 'MI')
        self.data = prepare_data(internal, self.root / 'prepared', self.root, self.root, panel_size=2)
        self.manifest, rows = load_prepared(self.data)
        panel_ids = set(self.manifest['panel_source_ids'])
        self.groups = {'full_dev': rows['dev'], 'dev_panel': rows['dev_panel'],
                       'dev_nonpanel': [row for row in rows['dev'] if row['source_id'] not in panel_ids]}
        self.runs = self.root / 'outputs_context_internal'
        run = self.runs / 'none_seed1234' / 'mi_fixed_run'
        run.mkdir(parents=True)
        self.checkpoint = run / 'best_model.pt'
        self.checkpoint.write_bytes(b'fake baseline checkpoint')
        self.config = {'dataset': 'MI', 'seed': 1234, 'consistency_mode': 'none', 'skip_test': True,
            'visual_fusion': 'legacy', 'internal_manifest': str(internal), 'max_seq_length': 64,
            'data_path': str(internal.parent), 'batch_size': 4}
        save(run / 'config.json', self.config)
        (run / 'metrics.csv').write_text('epoch,dev_f1\n1,0.8\n2,1.0\n', encoding='utf-8')
        self.reference = self.runs / 'pilot_original.json'
        save(self.reference, {'schema_version': 'context-internal-pilot-v1', 'test_read': False,
            'dataset': 'MI', 'seed': 1234, 'native_dev_f1': {'none': 1.},
            'artifacts': {'none': {'checkpoint': str(self.checkpoint), 'config': self.config,
                                   'sha256': baseline.sha256_file(self.checkpoint)}}})
        self.records = [baseline.project_path(row, [4, 1, 0, 4], [True, False, False, True],
                         [0, 1], LABELS, 4)[0] for row in rows['dev']]
        self.vlm = self.root / 'vlm_eval'
        self.vlm.mkdir()
        vlm_records = copy.deepcopy(self.records)
        bad_id = self.groups['dev_nonpanel'][0]['source_id']
        for record in vlm_records:
            if record['source_id'] == bad_id:
                record.update(response='not JSON', predicted=[], error='invalid JSON')
        (self.vlm / 'dev_predictions.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in vlm_records), encoding='utf-8')
        indexed = baseline.checked_records(rows['dev'], vlm_records)
        save(self.vlm / 'evaluation_report.json', {
            'schema_version': 'vlm-ner-dev-evaluation-v1', 'official_val_or_test_read': False,
            'new_checkpoint_selection': False, 'optimizer_steps': 0,
            'profile': {'data_sha256': baseline.sha256_file(self.data), 'input_mode': 'multi_image'},
            'panel_reproduction': {'passed': True},
            'source_ids': {role: [row['source_id'] for row in values] for role, values in self.groups.items()},
            'metrics': {role: baseline.metrics_for(values, indexed, baseline.SCOPES[role])
                        for role, values in self.groups.items()},
        })
        self.cli = SimpleNamespace(runs_root=str(self.runs), checkpoint=None, reference_report=None,
            data=str(self.data), vlm_evaluation=str(self.vlm), output=str(self.root / 'comparison'),
            device='cpu', check_only=False)

    def run_comparison(self, native_f1=1.):
        with patch.object(baseline, 'check_baseline_runtime', return_value={'test': 'fake'}), \
                patch.object(baseline, 'collect_predictions',
                             return_value=(self.records, {'f1': native_f1}, {'test_only': True})) as collect:
            result = baseline.evaluate(self.cli)
        return result, collect

    def test_common_gold_comparison_and_input_files_preserved(self):
        protected = [self.checkpoint, self.checkpoint.parent / 'config.json',
                     self.reference, self.vlm / 'dev_predictions.jsonl']
        before = {str(path): path.read_bytes() for path in protected}
        result, collect = self.run_comparison()
        collect.assert_called_once()
        self.assertTrue(result['baseline_native_reproduction']['passed'])
        self.assertFalse(result['official_val_or_test_read'])
        self.assertEqual(result['optimizer_steps'], 0)
        full = result['comparisons']['full_dev']
        self.assertEqual(full['baseline']['samples'], 4)
        self.assertEqual(full['baseline']['f1'], 1.)
        self.assertAlmostEqual(full['vlm']['f1'], 6 / 7)
        self.assertAlmostEqual(full['vlm_f1_minus_baseline'], -1 / 7)
        self.assertEqual(result['baseline_word_coverage']['unobserved_words'], 0)
        self.assertEqual(before, {str(path): path.read_bytes() for path in protected})
        self.assertTrue((Path(self.cli.output) / 'comparison_report.json').is_file())

    def test_preflight_does_not_import_gpu_runtime_or_write_outputs(self):
        self.cli.check_only = True
        with patch.object(baseline, 'check_baseline_runtime') as runtime, \
                patch.object(baseline, 'collect_predictions') as collect:
            baseline.evaluate(self.cli)
        runtime.assert_not_called()
        collect.assert_not_called()
        self.assertFalse(Path(self.cli.output).exists())

    def test_native_reproduction_failure_blocks_final_comparison(self):
        with self.assertRaisesRegex(RuntimeError, 'does not reproduce'):
            self.run_comparison(native_f1=0.7)
        output = Path(self.cli.output)
        self.assertTrue((output / 'failure.json').is_file())
        self.assertFalse((output / 'comparison_report.json').exists())

    def test_full_train_and_auxiliary_checkpoints_are_rejected(self):
        plan = baseline.prepare(self.cli)
        split = baseline.verify_internal_manifest(Path(self.config['internal_manifest']), 'MI')
        for override in ({'data_path': str(self.root)}, {'consistency_mode': 'dropout'},
                         {'llm_token_loss_weight': 0.5}, {'skip_test': False}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                baseline.validate_baseline_config({**self.config, **override},
                    Path(self.config['internal_manifest']), split)
        self.assertEqual(plan['reference_f1'], 1.)

    def test_changed_checkpoint_is_rejected_by_historical_binding(self):
        self.checkpoint.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'original context-internal pilot report'):
            baseline.prepare(self.cli)

    def test_vlm_report_must_match_raw_predictions(self):
        report_path = self.vlm / 'evaluation_report.json'
        report = baseline.read_json(report_path)
        report['metrics']['full_dev']['f1'] = 0.99
        save(report_path, report)
        with self.assertRaisesRegex(ValueError, 'do not reproduce'):
            baseline.prepare(self.cli)

    def test_wrong_runtime_does_not_leave_an_output_directory(self):
        with patch.object(baseline, 'check_baseline_runtime', side_effect=RuntimeError('wrong environment')):
            with self.assertRaisesRegex(RuntimeError, 'wrong environment'):
                baseline.evaluate(self.cli)
        self.assertFalse(Path(self.cli.output).exists())

    def test_runtime_guard_rejects_mner_vlm_versions(self):
        with patch.dict(sys.modules, {'torch': SimpleNamespace(__version__='2.6.0+cu124'),
                                      'transformers': SimpleNamespace(__version__='4.51.3')}):
            with self.assertRaisesRegex(RuntimeError, 'original BERT/ViT environment'):
                baseline.check_baseline_runtime()

    def test_inference_loop_uses_gold_free_inputs_and_native_crf_paths_on_cpu(self):
        import torch
        plan = baseline.prepare(self.cli)
        rows = self.groups['full_dev']
        local_model = self.root / 'pretrained'
        local_model.mkdir()
        plan['config'] = {**plan['config'], 'bert_model': str(local_model), 'vit_model': str(local_model)}

        class Dataset:
            def __init__(self, args, name):
                self.examples = {'words': [row['tokens'] for row in rows], 'images': [[] for _ in rows]}
                self.processor = SimpleNamespace(get_label_mapping=lambda **_: LABELS)

            def __len__(self):
                return len(rows)

            def __getitem__(self, index):
                return (torch.tensor([101, 10, 11, 102]), torch.zeros(4, dtype=torch.long),
                        torch.ones(4, dtype=torch.long), torch.zeros(4, 3, 1, 1),
                        torch.zeros(4, dtype=torch.long), torch.tensor(index))

            def token_structure(self, index):
                return torch.tensor([True, False, False, True]), rows[index]['tokens'], [0, 1]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, input_ids, labels, **kwargs):
                if labels is not None or self.training or torch.is_grad_enabled():
                    raise AssertionError('Inference must be frozen, gold-free and without gradients')
                return [[4, 1, 0, 4] for _ in input_ids], None, {
                    'token_emissions': torch.zeros(len(input_ids), 4, len(LABELS))}

        model = Model()
        with patch.dict(sys.modules, {
            'models.agvs_mner': SimpleNamespace(AgvsMnerModel=lambda *args: model),
            'modules.dataset': SimpleNamespace(MultimodalInferenceDataset=Dataset),
            'modules.trainer': SimpleNamespace(load_compatible_model_checkpoint=lambda *args, **kwargs: []),
            'run': SimpleNamespace(set_seed=lambda _: None),
        }):
            records, native, _ = baseline.collect_predictions(plan, 'cpu')
        self.assertFalse(model.parameter.requires_grad)
        self.assertEqual(native['f1'], 1.)
        self.assertEqual(len(baseline.checked_records(rows, records)), len(rows))

    def test_image_hash_changes_are_rejected(self):
        image = self.root / 'source.jpg'
        image.write_bytes(b'original')
        dataset = SimpleNamespace(examples={'images': [['source.jpg']]}, _get_image_path=lambda _: image)
        audit = {'image_audit': {str(image): {'sha256': baseline.sha256_file(image)}}}
        self.assertEqual(baseline.audit_images(dataset, audit), 1)
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Source image changed'):
            baseline.audit_images(dataset, audit)


if __name__ == '__main__':
    unittest.main()
