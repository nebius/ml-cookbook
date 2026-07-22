import unittest

from checkpoint_retention import plan_checkpoint_pruning


class CheckpointRetentionTests(unittest.TestCase):
    def test_keep_all_committed_still_removes_stale_partials(self):
        self.assertEqual(
            plan_checkpoint_pruning({10, 20, 30}, committed_step=20, keep_last=0),
            [(30, "stale partial")],
        )

    def test_retention_keeps_newest_committed_steps(self):
        self.assertEqual(
            plan_checkpoint_pruning({10, 20, 30}, committed_step=30, keep_last=2),
            [(10, "retention")],
        )

    def test_stale_partials_do_not_count_toward_retention(self):
        self.assertEqual(
            plan_checkpoint_pruning({10, 20, 30, 40}, committed_step=30, keep_last=2),
            [(40, "stale partial"), (10, "retention")],
        )

    def test_negative_retention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            plan_checkpoint_pruning({10}, committed_step=10, keep_last=-1)


if __name__ == "__main__":
    unittest.main()
