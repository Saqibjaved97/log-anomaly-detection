import datetime as dt
import unittest

from loganomaly.features import to_frame, window_features, window_labels
from loganomaly.generate import Incident
from loganomaly.parse import parse_lines

LINES = [
    "2026-10-01T00:00:01.000Z INFO api GET /v1/payments/1 200 took 40ms",
    "2026-10-01T00:01:00.000Z INFO api GET /v1/payments/2 200 took 60ms",
    "2026-10-01T00:02:00.000Z ERROR gateway Transaction TX-1 declined by issuer code 51",
    "2026-10-01T00:05:30.000Z INFO api GET /v1/payments/3 200 took 50ms",
]


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = to_frame(parse_lines(LINES))

    def test_one_row_per_window(self):
        X, vocab = window_features(self.df, window="5min")
        self.assertEqual(len(X), 2)
        self.assertEqual(len(vocab), 2)

    def test_shares_and_error_rate(self):
        X, _ = window_features(self.df, window="5min")
        first = X.iloc[0]
        self.assertAlmostEqual(first["error_share"], 1 / 3)
        share_cols = [c for c in X.columns if c.startswith("share: ")]
        self.assertAlmostEqual(first[share_cols].sum(), 1.0)

    def test_unseen_templates_counted(self):
        _, vocab = window_features(self.df, window="5min")
        test = to_frame(parse_lines(["2026-10-01T01:00:00.000Z FATAL worker Worker W-3 terminated: OutOfMemoryError"]))
        X, _ = window_features(test, vocabulary=vocab, window="5min")
        self.assertEqual(X["unseen_templates"].iloc[0], 1)

    def test_labels_from_incidents(self):
        X, _ = window_features(self.df, window="5min")
        inc = [Incident(dt.datetime(2026, 10, 1, 0, 6), dt.datetime(2026, 10, 1, 0, 8), "db_timeout", 0.2)]
        self.assertEqual(list(window_labels(X.index, inc, "5min")), [0, 1])


if __name__ == "__main__":
    unittest.main()
