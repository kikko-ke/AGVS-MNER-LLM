"""Offline error decomposition and provenance tests using stdlib fixtures only."""
import builtins
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.vlm_ner import load_prepared, prepare_data
from scripts.create_context_internal_split import create_split
from scripts import audit_baseline_qlora_errors as audit


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def prediction(row, entities, error=None):
    return {'source_id': row['source_id'], 'predicted': [list(entity) for entity in entities],
            'response': 'invalid JSON' if error else json.dumps({'entities': [
                dict(start=a, end=b, label=label) for a, b, label in entities]}),
            'error': error, 'original_words': len(row['tokens']),
            'retained_word_indices': list(range(len(row['tokens'])))}


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.row = {'source_id': 1, 'tokens': [str(index) for index in range(8)], 'images': [],
            'entities': [dict(start=a, end=a + 1, label=label)
                         for a, label in enumerate(('PER', 'ORG', 'MIS', 'LOC'))]}
        self.baseline = prediction(self.row, [(0, 1, 'PER'), (1, 2, 'ORG'), (4, 5, 'MIS'), (5, 6, 'ORG')])
        self.vlm = prediction(self.row, [(0, 1, 'PER'), (2, 3, 'MIS'), (4, 5, 'MIS'), (6, 7, 'LOC')])
        self.baseline['retained_word_indices'] = [0, 1, 4, 5, 6, 7]

    def analyse(self, rows=None, baseline=None, vlm=None):
        rows = rows or [self.row]
        baseline = baseline or {1: self.baseline}
        vlm = vlm or {1: self.vlm}
        left = audit.metrics_for(rows, baseline, 'fixture')
        right = audit.metrics_for(rows, vlm, 'fixture')
        return audit.analyse_group(rows, baseline, vlm, {
            'baseline': left, 'vlm': right, 'vlm_f1_minus_baseline': right['f1'] - left['f1']})

    def test_all_seven_entity_buckets_and_each_class_conserve_counts(self):
        result = self.analyse()
        self.assertEqual(result['entity_changes'], {name: 1 for name in audit.CHANGE_FIELDS})
        self.assertEqual(result['baseline']['f1'], 0.5)
        self.assertEqual(result['vlm']['f1'], 0.5)
        self.assertLess(result['by_type']['ORG']['vlm_f1_minus_baseline'], 0)
        self.assertGreater(result['by_type']['MIS']['vlm_f1_minus_baseline'], 0)
        self.assertEqual(result['recovered_gold_touching_unobserved_baseline_words'], 1)
        self.assertEqual(result['sample_outcomes']['both_wrong_different_entities'], 1)

    def test_micro_metrics_are_not_an_average_of_sample_f1(self):
        empty = {'source_id': 2, 'tokens': ['x'], 'entities': [], 'images': []}
        result = self.analyse([self.row, empty], {1: self.baseline, 2: prediction(empty, [])},
                              {1: self.vlm, 2: prediction(empty, [(0, 1, 'PER')])})
        self.assertEqual(result['baseline']['f1'], 0.5)
        self.assertAlmostEqual(result['vlm']['f1'], 4 / 9)
        self.assertEqual(result['entity_changes']['introduced_vlm_fp'], 2)
        self.assertEqual(sum(result['sample_outcomes'].values()), 2)

    def test_all_five_sample_outcomes(self):
        gold = {(0, 1, 'PER')}
        other = {(0, 1, 'ORG')}
        combinations = ((gold, gold), (gold, set()), (set(), gold), (set(), set()), (set(), other))
        self.assertEqual([audit.outcome(gold, left, right) for left, right in combinations], list(audit.OUTCOMES))

    def test_format_error_on_empty_gold_remains_explicit(self):
        row = {'source_id': 1, 'tokens': ['x'], 'entities': []}
        result = self.analyse([row], {1: prediction(row, [])}, {1: prediction(row, [], 'invalid JSON')})
        self.assertEqual(result['sample_outcomes']['both_exact_entities'], 1)
        self.assertEqual(result['malformed_vlm_samples'], 1)

    def test_review_selection_is_deterministic_bounded_and_nonpanel(self):
        cases = []
        for source_id in range(30):
            row = {**self.row, 'source_id': source_id}
            left, right = {source_id: self.baseline}, {source_id: self.vlm}
            cases.extend(audit.case_records([row], left, right, {0, 1, 2}))
        review = audit.select_review_cases(cases, 2)
        self.assertEqual(review, audit.select_review_cases(list(reversed(cases)), 2))
        self.assertLessEqual(len(review), 12)
        self.assertTrue(all(not case['in_selection_panel'] for case in review))
        for name in audit.CHANGE_FIELDS[1:]:
            self.assertEqual(sum(name in case['review_buckets'] for case in review), 2)

    def test_inconsistent_decomposition_is_rejected(self):
        result = self.analyse()
        changed = {**result['entity_changes'], 'lost_gold': 2}
        with self.assertRaisesRegex(ValueError, 'conserve'):
            audit.verify_counts(changed, result['baseline'], result['vlm'])


class OfflineAuditTests(unittest.TestCase):
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
        self.comparison_dir, self.vlm_dir = self.root / 'comparison', self.root / 'vlm'
        self.comparison_dir.mkdir()
        self.vlm_dir.mkdir()
        self.baseline_path = self.comparison_dir / 'baseline_predictions.jsonl'
        self.vlm_path = self.vlm_dir / 'dev_predictions.jsonl'
        self.baseline_records = [prediction(row, [(0, 1, 'ORG')]) for row in rows['dev']]
        vlm_records = copy.deepcopy(self.baseline_records)
        malformed_id = self.groups['dev_nonpanel'][0]['source_id']
        for record in vlm_records:
            if record['source_id'] == malformed_id:
                record.update(response='invalid JSON', predicted=[], error='bad JSON')
        for path, records in ((self.baseline_path, self.baseline_records), (self.vlm_path, vlm_records)):
            path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        left = audit.checked_records(rows['dev'], self.baseline_records)
        right = audit.checked_records(rows['dev'], vlm_records)
        metrics = {role: {'baseline': audit.metrics_for(values, left, role),
                          'vlm': audit.metrics_for(values, right, role)} for role, values in self.groups.items()}
        for value in metrics.values():
            value['vlm_f1_minus_baseline'] = value['vlm']['f1'] - value['baseline']['f1']
        ids = {role: [row['source_id'] for row in values] for role, values in self.groups.items()}
        self.vlm_report_path = self.vlm_dir / 'evaluation_report.json'
        save(self.vlm_report_path, {
            'schema_version': 'vlm-ner-dev-evaluation-v1', 'official_val_or_test_read': False,
            'optimizer_steps': 0, 'new_checkpoint_selection': False,
            'profile': {'data_sha256': audit.sha256_file(self.data), 'input_mode': 'multi_image'},
            'panel_reproduction': {'passed': True}, 'source_ids': ids,
            'metrics': {role: value['vlm'] for role, value in metrics.items()},
        })
        self.comparison_path = self.comparison_dir / 'comparison_report.json'
        self.comparison = {
            'schema_version': 'baseline-vlm-fullword-comparison-v1', 'official_val_or_test_read': False,
            'optimizer_steps': 0, 'new_checkpoint_selection': False,
            'baseline_native_reproduction': {'passed': True},
            'baseline_training_config': {'dataset': 'MI', 'seed': 1234, 'consistency_mode': 'none',
                                         'internal_manifest': str(internal)},
            'source_ids': ids, 'comparisons': metrics,
            'input_bindings': {str(path.resolve()): audit.sha256_file(path) for path in
                               (self.data, internal, self.vlm_report_path, self.vlm_path)},
            'evaluation_code_sha256': {str((audit.PROJECT / name).resolve()): audit.sha256_file(audit.PROJECT / name)
                for name in ('modules/vlm_ner.py', 'modules/internal_split.py', 'scripts/evaluate_vlm_ner.py')},
        }
        save(self.comparison_path, self.comparison)
        self.cli = SimpleNamespace(data=str(self.data), comparison=str(self.comparison_dir),
            vlm_evaluation=str(self.vlm_dir), output=str(self.root / 'audit'), review_per_bucket=2)

    def test_full_audit_never_imports_model_libraries_and_preserves_inputs(self):
        paths = [self.baseline_path, self.vlm_path, self.comparison_path, self.vlm_report_path, self.data]
        before = {str(path): path.read_bytes() for path in paths}
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'torchcrf', 'PIL', 'numpy'}:
                raise AssertionError('Offline audit imported a model/image library')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded_import):
            result = audit.audit(self.cli)
        full = result['analysis']['full_dev']
        self.assertEqual(full['entity_changes']['lost_gold'], 1)
        self.assertEqual(full['entity_changes']['recovered_gold'], 0)
        self.assertEqual(full['malformed_vlm_samples'], 1)
        self.assertAlmostEqual(full['vlm_f1_minus_baseline'], -1 / 7)
        self.assertEqual(sum(full['sample_outcomes'].values()), 4)
        self.assertEqual(result['model_inference_runs'], 0)
        self.assertTrue(result['gold_used_for_analysis'])
        self.assertFalse(result['deployment_selector_created'])
        self.assertEqual(before, {str(path): path.read_bytes() for path in paths})
        output = Path(self.cli.output)
        self.assertEqual(len(audit.read_records(output / 'error_cases.jsonl')), 4)
        with (output / 'by_type.csv').open(encoding='utf-8', newline='') as source:
            self.assertEqual(len(list(csv.DictReader(source))), 12)
        self.assertTrue((output / 'error_audit_report.json').is_file())

    def test_prediction_order_is_resolved_by_source_id(self):
        self.baseline_path.write_text(''.join(json.dumps(record) + '\n' for record in
                                             reversed(self.baseline_records)), encoding='utf-8')
        result = audit.audit(self.cli)
        self.assertEqual(result['analysis']['full_dev']['entity_changes']['lost_gold'], 1)

    def test_changed_bound_vlm_predictions_are_rejected_even_if_metrics_unchanged(self):
        self.vlm_path.write_text(self.vlm_path.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact'):
            audit.audit(self.cli)
        self.assertFalse(Path(self.cli.output).exists())

    def test_missing_duplicate_and_wrong_baseline_predictions_are_rejected(self):
        for records in (self.baseline_records[:-1], self.baseline_records + self.baseline_records[:1]):
            self.baseline_path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.load_inputs(self.cli)
        records = copy.deepcopy(self.baseline_records)
        records[0]['retained_word_indices'] = [1]
        self.baseline_path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'outside its observed window'):
            audit.load_inputs(self.cli)

    def test_saved_scores_and_source_ids_must_match_gold_and_predictions(self):
        self.comparison['comparisons']['full_dev']['baseline']['f1'] = 0.5
        save(self.comparison_path, self.comparison)
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            audit.load_inputs(self.cli)
        self.comparison['source_ids']['dev_nonpanel'] = []
        save(self.comparison_path, self.comparison)
        with self.assertRaisesRegex(ValueError, 'source IDs'):
            audit.load_inputs(self.cli)

    def test_existing_results_and_source_directories_are_protected(self):
        output = Path(self.cli.output)
        output.mkdir()
        marker = output / 'preserve.txt'
        marker.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            audit.audit(self.cli)
        self.assertEqual(marker.read_text(), 'keep')
        self.cli.output = str(self.comparison_dir / 'nested_audit')
        with self.assertRaisesRegex(ValueError, 'separate'):
            audit.audit(self.cli)


if __name__ == '__main__':
    unittest.main()
