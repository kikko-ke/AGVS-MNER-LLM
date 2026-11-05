"""Holdout boundary tests for linked texts and image identities."""
import unittest

from scripts.prepare_bio_joint_recheck import grouped_sources, split_plan


def row(source_id, word, images=()):
    return {'source_id': source_id, 'tokens': [word],
            'images': [{'sha256': image} for image in images]}


class GroupedRecheckTests(unittest.TestCase):
    def test_transitive_text_and_image_links_stay_together(self):
        rows = [row(0, 'Ada', ['first']), row(1, 'ADA', ['second']),
                row(2, 'Other', ['second']), row(3, 'Paris')]
        self.assertEqual(grouped_sources(rows), [[0, 1, 2], [3]])

    def test_dev_links_reserve_whole_groups_and_holdout_membership_is_repeatable(self):
        train = [row(0, 'Ada', ['first']), row(1, 'ADA', ['second'])]
        train.extend(row(index, str(index)) for index in range(2, 5020))
        dev = [row(6000, 'Other', ['second'])]
        result = split_plan(train, dev, [6000])
        reversed_result = split_plan(list(reversed(train)), dev, [6000])
        holdout = result['source_ids']['recheck_holdout']
        self.assertEqual(holdout, reversed_result['source_ids']['recheck_holdout'])
        self.assertEqual(result['train_samples_linked_to_old_dev'], 2)
        self.assertEqual(len(holdout), 1004)
        self.assertTrue(result['ready'])
        self.assertTrue({0, 1, 6000}.isdisjoint(holdout))
        self.assertEqual(set(result['source_ids']['retrain']) | set(holdout), set(range(5020)))

    def test_ineligible_or_oversize_groups_report_an_insufficient_holdout(self):
        train = [row(index, 'Repeated', ['same']) for index in range(5020)]
        result = split_plan(train, [row(6000, 'Dev')], [6000])
        self.assertEqual(result['holdout_samples'], 0)
        self.assertFalse(result['ready'])


if __name__ == '__main__':
    unittest.main()
