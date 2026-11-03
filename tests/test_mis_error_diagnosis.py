"""Span-relation diagnostics and frozen-audit integration, without ML imports."""
import builtins
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import audit_mis_errors as mis


def record(row, entities, error=None):
    return {'source_id': row['source_id'], 'predicted': [list(entity) for entity in entities],
            'error': error, 'response': 'bad JSON' if error else json.dumps({'entities': [
                dict(start=a, end=b, label=label) for a, b, label in entities]}),
            'retained_word_indices': list(range(len(row['tokens']))), 'original_words': len(row['tokens'])}


class RelationTests(unittest.TestCase):
    def test_missing_exact_type_boundary_mixed_and_multiple_overlap_categories(self):
        anchor = (2, 5, 'MIS')
        cases = ((set(), 'no_overlap'), ({(2, 5, 'ORG')}, 'type_only'),
                 ({(1, 5, 'MIS')}, 'boundary_only'), ({(3, 6, 'PER')}, 'boundary_and_type'),
                 ({(2, 3, 'MIS'), (3, 5, 'MIS')}, 'multiple_overlaps'))
        for opposing, category in cases:
            with self.subTest(category=category):
                self.assertEqual(mis.classify(anchor, opposing, True)['category'], category)

    def test_type_confusion_direction_is_gold_to_prediction_in_both_views(self):
        fn = mis.classify((0, 1, 'MIS'), {(0, 1, 'ORG')}, True)
        fp = mis.classify((0, 1, 'MIS'), {(0, 1, 'PER')}, False)
        self.assertEqual(fn['type_confusion'], 'MIS->ORG')
        self.assertEqual(fp['type_confusion'], 'PER->MIS')

    def test_boundary_relation_uses_gold_and_prediction_roles(self):
        self.assertEqual(mis.classify((2, 5, 'MIS'), {(1, 6, 'MIS')}, True)['boundary_relation'],
                         'prediction_contains_gold')
        self.assertEqual(mis.classify((2, 5, 'MIS'), {(1, 6, 'MIS')}, False)['boundary_relation'],
                         'prediction_inside_gold')
        self.assertEqual(mis.classify((2, 5, 'MIS'), {(3, 6, 'MIS')}, True)['boundary_relation'], 'partial_overlap')

    def test_format_failure_is_separate_from_complete_omission(self):
        self.assertEqual(mis.classify((0, 1, 'MIS'), set(), True, True)['category'], 'malformed_response')
        with self.assertRaises(ValueError):
            mis.classify((0, 1, 'MIS'), set(), False, True)
        with self.assertRaises(ValueError):
            mis.classify((0, 1, 'MIS'), {(0, 1, 'MIS')}, True)

    def test_merge_is_ambiguous_and_fp_views_do_not_greedily_match_gold(self):
        diagnosis = mis.classify((0, 4, 'MIS'), {(0, 2, 'MIS'), (2, 4, 'ORG')}, False)
        self.assertEqual(diagnosis['category'], 'multiple_overlaps')
        self.assertEqual(len(diagnosis['overlaps']), 2)
        self.assertNotIn('type_confusion', diagnosis)


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        quiet = patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        # One selected-panel row, followed by a lost entity, a recovered entity,
        # a shared miss, a type confusion, and a malformed response.
        self.rows = [{'source_id': index, 'tokens': ['A', 'B', 'C', 'D'], 'images': [],
                      'entities': [{'start': 0, 'end': 2, 'label': 'MIS'}]} for index in range(6)]
        gold = [(0, 2, 'MIS')]
        self.baseline = {row['source_id']: record(row, gold) for row in self.rows}
        self.vlm = copy.deepcopy(self.baseline)
        self.vlm[1] = record(self.rows[1], [])
        self.baseline[2] = record(self.rows[2], [])
        self.baseline[3] = record(self.rows[3], [(0, 1, 'MIS')])
        self.vlm[3] = record(self.rows[3], [(0, 2, 'ORG')])
        self.vlm[4] = record(self.rows[4], [(0, 2, 'ORG'), (3, 4, 'MIS')])
        self.vlm[5] = record(self.rows[5], [], 'bad JSON')
        self.groups = {'full_dev': self.rows, 'dev_panel': self.rows[:1], 'dev_nonpanel': self.rows[1:]}
        self.metrics = {}
        for role, rows in self.groups.items():
            left = mis.previous.metrics_for(rows, self.baseline, mis.previous.SCOPES[role])
            right = mis.previous.metrics_for(rows, self.vlm, mis.previous.SCOPES[role])
            self.metrics[role] = {'baseline': left, 'vlm': right, 'vlm_f1_minus_baseline': right['f1'] - left['f1']}
        self.analysis = {role: mis.previous.analyse_group(rows, self.baseline, self.vlm, self.metrics[role])
                         for role, rows in self.groups.items()}
        self.audit_dir = self.root / 'old_audit'
        self.audit_dir.mkdir()
        frozen_input = self.root / 'predictions.json'
        frozen_input.write_text('frozen fixture', encoding='utf-8')
        self.bindings = {str(frozen_input.resolve()): mis.sha256_file(frozen_input)}
        self.old_path = self.audit_dir / 'error_audit_report.json'
        self.old = {'schema_version': 'baseline-qlora-error-audit-v1', 'diagnostic_only': True,
            'gold_used_for_analysis': True, 'official_val_or_test_read': False,
            'deployment_selector_created': False, 'model_inference_runs': 0, 'training_runs': 0,
            'input_sha256': self.bindings,
            'source_ids': {role: [row['source_id'] for row in rows] for role, rows in self.groups.items()},
            'analysis': self.analysis}
        self.old_path.write_text(json.dumps(self.old), encoding='utf-8')
        self.cli = SimpleNamespace(error_audit=str(self.audit_dir), comparison=str(self.root / 'comparison'),
            vlm_evaluation=str(self.root / 'vlm'), data=str(self.root / 'prepared' / 'manifest.json'),
            output=str(self.root / 'diagnosis'), review_per_category=2)
        self.inputs = (self.groups, self.baseline, self.vlm, self.metrics, self.bindings)

    def test_full_and_subset_counts_match_frozen_mis_fn_and_fp(self):
        records = mis.make_errors(self.rows, self.baseline, self.vlm, {0})
        full = mis.analyse_scope(records, set(range(6)), self.analysis['full_dev'])
        self.assertEqual(full['error_views']['lost_gold']['entities'], 3)
        self.assertEqual(full['error_views']['recovered_gold']['entities'], 1)
        self.assertEqual(full['error_views']['lost_gold']['categories']['no_overlap'], 1)
        self.assertEqual(full['error_views']['lost_gold']['categories']['type_only'], 1)
        self.assertEqual(full['error_views']['lost_gold']['categories']['malformed_response'], 1)
        self.assertEqual(full['error_views']['introduced_vlm_fp']['categories']['no_overlap'], 1)
        self.assertEqual(full['common_miss_category_pairs_baseline_to_vlm'], {'boundary_only->type_only': 1})
        panel = mis.analyse_scope(records, {0}, self.analysis['dev_panel'])
        self.assertEqual(panel['error_views']['vlm_fn']['entities'], 0)

    def test_shared_errors_are_counted_once_in_shared_views(self):
        records = mis.make_errors(self.rows, self.baseline, self.vlm, {0})
        full = mis.analyse_scope(records, set(range(6)), self.analysis['full_dev'])
        self.assertEqual(full['error_views']['both_missed_gold']['entities'], 1)
        self.assertEqual(len([record for record in records if record['relative_bucket'] == 'both_missed_gold'
                             and record['direction'] == 'fn']), 2)

    def test_missing_entity_diagnosis_is_not_silently_dropped(self):
        records = mis.make_errors(self.rows, self.baseline, self.vlm, {0})
        with self.assertRaisesRegex(ValueError, 'retain every error'):
            mis.analyse_scope(records[:-1], set(range(6)), self.analysis['full_dev'])

    def test_review_is_deterministic_and_excludes_panel(self):
        records = mis.make_errors(self.rows, self.baseline, self.vlm, {0})
        review = mis.select_review(records, 1)
        self.assertEqual(review, mis.select_review(list(reversed(records)), 1))
        self.assertTrue(all(not record['in_selection_panel'] for record in review))
        self.assertTrue(all(record['relative_bucket'] in mis.REVIEW_BUCKETS for record in review))
        self.assertIn('WORDS:', mis.render_example(review[0]))

    def test_complete_offline_entrypoint_never_imports_ml_libraries(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'PIL', 'numpy'}:
                raise AssertionError('MIS diagnosis imported an ML/image library')
            return original_import(name, *args, **kwargs)

        before = self.old_path.read_bytes()
        with patch.object(mis.previous, 'load_inputs', return_value=self.inputs), \
                patch('builtins.__import__', side_effect=guarded):
            result = mis.audit(self.cli)
        self.assertEqual(result['model_inference_runs'], 0)
        self.assertEqual(result['analysis']['full_dev']['error_views']['lost_gold']['entities'], 3)
        self.assertEqual(before, self.old_path.read_bytes())
        output = Path(self.cli.output)
        self.assertTrue((output / 'review_examples.txt').is_file())
        self.assertTrue((output / 'mis_diagnosis_report.json').is_file())

    def test_changed_snapshot_or_prior_counts_block_the_diagnosis(self):
        self.old['analysis']['full_dev']['entity_changes']['lost_gold'] += 1
        self.old_path.write_text(json.dumps(self.old), encoding='utf-8')
        with patch.object(mis.previous, 'load_inputs', return_value=self.inputs):
            with self.assertRaisesRegex(ValueError, 'do not reproduce'):
                mis.load_frozen(self.cli)
        self.old['input_sha256'] = {}
        self.old_path.write_text(json.dumps(self.old), encoding='utf-8')
        with patch.object(mis.previous, 'load_inputs', return_value=self.inputs):
            with self.assertRaisesRegex(ValueError, 'frozen inputs changed'):
                mis.load_frozen(self.cli)

    def test_existing_and_source_outputs_are_protected(self):
        output = Path(self.cli.output)
        output.mkdir()
        with self.assertRaises(FileExistsError):
            mis.audit(self.cli)
        self.cli.output = str(self.audit_dir / 'nested')
        with self.assertRaisesRegex(ValueError, 'separate'):
            mis.audit(self.cli)


if __name__ == '__main__':
    unittest.main()
