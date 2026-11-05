"""Regression checks for the BIO format and its existing encoder/scorer integration."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_vlm_bio as bio

try:
    import torch
except ImportError:
    torch = None


def row(source_id, tokens, entities=()):
    return {'source_id': source_id, 'tokens': tokens, 'images': [],
            'entities': [dict(start=start, end=end, label=label) for start, end, label in entities]}


class FormatTests(unittest.TestCase):
    def test_round_trip_keeps_adjacent_same_type_spans_and_all_entity_types(self):
        spans = [(0, 1, 'LOC'), (1, 3, 'LOC'), (4, 6, 'MIS'), (6, 8, 'ORG'), (9, 12, 'PER')]
        sample = row(1, list('abcdefghijkl'), spans)
        answer = bio.bio_target(sample)
        self.assertEqual(json.loads(answer)['tags'], '112034560788')
        self.assertEqual(bio.parse_bio(answer, 12), spans)
        self.assertEqual(bio.parse_bio('{"tags":"000"}', 3), [])

    def test_malformed_outputs_are_rejected_without_repair(self):
        for response in ('{"tags":"7"}', '{"tags":"790"}', '{"tags":"７８０"}', '{"tags":"800"}',
                         '{"tags":"760"}', '{"tags":"708"}', '{"tags":[7,8,0]}',
                         '{"tags":"780","extra":true}', '{"tags":"780"} trailing', None):
            with self.subTest(response=response):
                predicted, error = bio.decoded(response, 3)
                self.assertEqual(predicted, [])
                self.assertIsNotNone(error)

    def test_saved_spans_must_match_raw_bio_and_source_ids(self):
        sample = row(12, ['San', 'Jose'])
        record = {'source_id': 12, 'response': '{"tags":"12"}', 'predicted': [[0, 2, 'LOC']], 'error': None}
        self.assertEqual(bio.checked_records([sample], [record])[12], record)
        for records in ([], [record, record], [{**record, 'predicted': [[0, 2, 'ORG']]}]):
            with self.assertRaises(ValueError):
                bio.checked_records([sample], records)

    def test_evaluation_binds_combined_predictions_separately_from_training_inputs(self):
        first = row(1, ['San', 'Jose'], [(0, 2, 'LOC')])
        second = row(2, ['Ada'], [(0, 1, 'PER')])
        panel = [{'source_id': 1, 'response': '{"tags":"12"}', 'predicted': [[0, 2, 'LOC']], 'error': None}]
        nonpanel = [{'source_id': 2, 'response': '{"tags":"7"}', 'predicted': [[0, 1, 'PER']], 'error': None}]
        rows = {'dev_panel': [first], 'dev': [second, first]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            prior = output / 'prior_panel.jsonl'
            bio.write_records(prior, panel)
            adapter = output / 'best_adapter'
            inputs = {str(prior): bio.sha256_file(prior)}
            candidate = (2, adapter, {'adapter': 'fixture'}, {1: panel[0]}, inputs)
            with patch.object(bio.original, 'load_frozen_adapter', return_value=(object(), object())), \
                 patch.object(bio.backend, 'model_binding', return_value={'adapter': 'fixture'}), \
                 patch.object(bio, 'generate', side_effect=({1: panel[0]}, {2: nonpanel[0]})), \
                 patch('builtins.print'):
                name, report = bio.evaluate(SimpleNamespace(model='fixture'), {}, rows, [second], output, candidate)
            predictions = output / 'dev_predictions.jsonl'
            self.assertEqual(name, 'evaluation_report.json')
            self.assertEqual(bio.original.read_records(predictions), nonpanel + panel)
            self.assertEqual(report['dev_predictions_sha256'], bio.sha256_file(predictions))
            self.assertNotEqual(report['dev_predictions_sha256'], inputs[str(prior)])
            self.assertEqual(report['metrics']['full_dev']['f1'], 1.)


@unittest.skipIf(torch is None, 'CPU torch required for encoder integration checks')
class EncoderTests(unittest.TestCase):
    class Processor:
        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            prefix = messages[0]['content'] + '\n' + messages[1]['content'][-1]['text'] + '\nASSISTANT\n'
            if add_generation_prompt:
                return prefix
            return prefix + messages[2]['content'][0]['text'] + '<end>'

        def __call__(self, text, return_tensors, padding):
            return {'input_ids': torch.tensor([[ord(ch) for ch in text[0]]])}

    def setUp(self):
        self.profile = {'max_length': 2048, 'max_new_tokens': 256}
        self.processor = self.Processor()

    def test_training_masks_original_word_prompt_and_supervises_only_bio_answer(self):
        sample = row(1, ['San', 'Jose'], [(0, 2, 'LOC')])
        train = bio.encode(self.processor, sample, self.profile, True)
        infer = bio.encode(self.processor, sample, self.profile, False)
        labels = train['labels'][0]
        prompt = infer['input_ids'][0]
        self.assertTrue(torch.equal(train['input_ids'][0, :len(prompt)], prompt))
        self.assertTrue(torch.all(labels[:len(prompt)] == -100))
        self.assertEqual(''.join(chr(ch) for ch in labels[len(prompt):].tolist()), '{"tags":"12"}<end>')

    def test_inference_does_not_require_gold(self):
        inputs = {'source_id': 1, 'tokens': ['San', 'Jose'], 'images': []}
        self.assertIn('input_ids', bio.encode(self.processor, inputs, self.profile, False))

    def test_answer_budget_failure_prevents_truncation(self):
        sample = row(11, ['San', 'Jose'], [(0, 2, 'LOC')])
        with self.assertRaisesRegex(ValueError, 'BIO budgets exceeded for source_id=11'):
            bio.encode(self.processor, sample, {**self.profile, 'max_new_tokens': 8}, True)

    def test_generation_scores_invalid_bio_as_empty_without_changing_span_metric(self):
        first = row(1, ['San', 'Jose'], [(0, 2, 'LOC')])
        second = row(2, ['New', 'York'], [(0, 2, 'LOC')])
        responses = iter(['{"tags":"12"}', '{"tags":"2"}'])

        class Model:
            def eval(self):
                return self

            def generate(self, input_ids, **kwargs):
                suffix = torch.tensor([[ord(ch) for ch in next(responses)]])
                return torch.cat((input_ids, suffix), dim=1)

        class Tokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens):
                return ''.join(chr(ch) for ch in ids.tolist())

        self.processor.tokenizer = Tokenizer()
        with patch.object(torch, 'autocast', return_value=nullcontext()), \
             patch.object(bio.backend, 'move_to_cuda', side_effect=lambda batch: batch), \
             patch.object(bio, 'write_records') as writer:
            predictions = bio.generate(Model(), self.processor, [first, second], self.profile, Path('unused.jsonl'))
        writer.assert_called_once()
        metrics = bio.original.metrics_for([first, second], predictions, 'fixture')
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn'], metrics['malformed_responses']), (1, 0, 1, 1))
        self.assertAlmostEqual(metrics['f1'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
