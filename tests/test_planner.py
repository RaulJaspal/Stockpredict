"""Tests for the trade planner's honesty invariants (app/analysis/planner.py).

The planner must never imply the levels create edge: on any history the
bracket's return, benchmarked against buy-and-hold over the same window, has a
level-edge of at most ~0 (stops/targets cap winners), and strictly negative
after costs.
"""

import unittest

import numpy as np
import pandas as pd

from app.analysis import planner
from app.analysis.technical import compute_indicators


def _ohlc(n, sigma=0.02, drift=0.0004, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    intraday = np.abs(rng.normal(0, sigma, n)) * close
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, sigma / 3, n)),
        "High": close + intraday,
        "Low": close - intraday,
        "Close": close,
        "Volume": rng.integers(1e6, 1e7, n).astype(float),
    }, index=pd.date_range("2018-01-01", periods=n, freq="B"))
    return compute_indicators(df)


class PlannerHonestyTest(unittest.TestCase):
    def test_costs_always_reduce_the_edge(self):
        # Guaranteed by construction: the net (after-cost) level edge is strictly
        # below the gross level edge, on any history.
        for drift, seed in [(0.0006, 1), (0.0, 2), (0.0003, 3)]:
            ind = _ohlc(700, drift=drift, seed=seed)
            plan = planner.trade_plan(ind, p_up=0.55, expected_range=None)
            self.assertIsNotNone(plan)
            self.assertLess(plan["net_level_edge_pct"], plan["level_edge_pct"])

    def test_levels_give_up_drift_in_an_up_market(self):
        # In a clearly up-drifting market the target caps winners, so buy-and-hold
        # beats the bracket -> the level edge is negative (the honest headline).
        ind = _ohlc(700, drift=0.0008, seed=5)
        plan = planner.trade_plan(ind, p_up=0.55, expected_range=None)
        self.assertGreater(plan["buy_hold_pct"], plan["gross_expectancy_pct"])
        self.assertLess(plan["level_edge_pct"], 0.0)

    def test_probabilities_sum_to_100(self):
        ind = _ohlc(700, seed=4)
        plan = planner.trade_plan(ind, p_up=0.55, expected_range=None)
        total = plan["target_first_pct"] + plan["stop_first_pct"] + plan["neither_pct"]
        self.assertAlmostEqual(total, 100.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
