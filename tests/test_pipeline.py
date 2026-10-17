import contextlib
import io
import tempfile
import unittest

from loganomaly.run import run


class PipelineTests(unittest.TestCase):
    def _run(self, seed):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            return run(hours=24, clean_hours=12, seed=seed, out_dir=d, plot=False)

    def test_runs_end_to_end(self):
        table = self._run(seed=7)
        self.assertEqual(len(table), 4)
        for col in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
            self.assertTrue(((table[col] >= 0) & (table[col] <= 1)).all(), col)

    def test_hybrid_beats_chance(self):
        table = self._run(seed=7)
        self.assertGreater(table.loc["Isolation Forest + new-message rule", "roc_auc"], 0.6)

    def test_deterministic(self):
        a, b = self._run(seed=3), self._run(seed=3)
        self.assertTrue(a.round(6).equals(b.round(6)))


if __name__ == "__main__":
    unittest.main()
