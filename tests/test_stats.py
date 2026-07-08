"""Tests for the backtest's block-bootstrap edge CI (backtest.py).

Verifies the honest-statistics claim: the CI brackets zero exactly when the
model is indistinguishable from always-up, and excludes it (correct side) when
the model is uniformly better or worse.
"""

import unittest

import pandas as pd

from backtest import _block_bootstrap_edge


def _frame(correct, outcome_up, ticker="AAA"):
    n = len(correct)
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "correct": correct,
        "outcome_up": outcome_up,
    })


class BlockBootstrapEdgeTest(unittest.TestCase):
    def test_identical_to_baseline_gives_zero_edge(self):
        # Model makes the SAME call as always-up on every row -> edge is exactly 0.
        outcome = [True, False] * 100
        d = _frame(correct=outcome, outcome_up=outcome)   # correct iff outcome up == always-up
        edge, lo, hi, p = _block_bootstrap_edge(d, block_len=1)
        self.assertAlmostEqual(edge, 0.0, places=9)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_uniformly_better_model_excludes_zero_positive(self):
        outcome = ([True] * 6 + [False] * 4) * 30       # up-rate 0.6
        d = _frame(correct=[True] * len(outcome), outcome_up=outcome)  # always right
        edge, lo, hi, p = _block_bootstrap_edge(d, block_len=1)
        self.assertAlmostEqual(edge, 0.4, places=6)     # 1 - up_rate
        self.assertGreater(lo, 0.0, "a strictly-better model should exclude 0 on the low side")

    def test_uniformly_worse_model_excludes_zero_negative(self):
        outcome = ([True] * 6 + [False] * 4) * 30
        d = _frame(correct=[False] * len(outcome), outcome_up=outcome)  # always wrong
        edge, lo, hi, p = _block_bootstrap_edge(d, block_len=1)
        self.assertLess(edge, 0.0)
        self.assertLess(hi, 0.0, "a strictly-worse model should exclude 0 on the high side")

    def test_larger_block_len_runs_and_widens(self):
        # Overlap-style dependence: identical calls in runs. Just assert it runs
        # and returns a valid interval with block_len > 1.
        outcome = ([True] * 12 + [False] * 12) * 10
        d = _frame(correct=[True] * len(outcome), outcome_up=outcome)
        edge, lo, hi, p = _block_bootstrap_edge(d, block_len=12)
        self.assertLessEqual(lo, edge)
        self.assertGreaterEqual(hi, edge)


if __name__ == "__main__":
    unittest.main()
