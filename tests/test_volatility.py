"""Tests for the EWMA expected-range model (app/analysis/volatility.py).

Verifies the two properties the range promises: it reacts to volatility
changes (unlike a slow rolling window), and its band is calibrated to ~80%
coverage on a process with known volatility.
"""

import unittest

import numpy as np

from app.analysis import volatility


def _gbm(sigma_d, n, seed, drift=0.0):
    """A geometric-random-walk price path with daily log-vol `sigma_d`."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma_d, n)
    return 100.0 * np.exp(np.cumsum(rets))


class EwmaVolTest(unittest.TestCase):
    def test_none_without_enough_history(self):
        self.assertIsNone(volatility.ewma_daily_vol(np.zeros(volatility.EWMA_SEED - 1)))

    def test_recovers_known_vol(self):
        prices = _gbm(0.02, 4000, seed=1)
        sigma = volatility.ewma_daily_vol(np.diff(np.log(prices)))
        self.assertAlmostEqual(sigma, 0.02, delta=0.004)  # within ~20%

    def test_reacts_to_a_recent_vol_spike(self):
        # calm history, then a burst of high-vol days at the end.
        calm = np.diff(np.log(_gbm(0.01, 500, seed=2)))
        wild = np.diff(np.log(_gbm(0.05, 40, seed=3)))
        combined = np.concatenate([calm, wild])
        after_spike = volatility.ewma_daily_vol(combined)
        calm_only = volatility.ewma_daily_vol(calm)
        self.assertGreater(after_spike, 2 * calm_only,
                           "EWMA should jump after a run of high-volatility days")


class ExpectedRangeTest(unittest.TestCase):
    def test_shape_and_ordering(self):
        prices = _gbm(0.02, 800, seed=4)
        r = volatility.expected_range(prices)
        self.assertIsNotNone(r)
        self.assertLess(r["low"], prices[-1])
        self.assertGreater(r["high"], prices[-1])
        self.assertGreater(r["pct"], 0)
        self.assertGreater(r["sigma_annual_pct"], 0)

    def test_none_on_short_series(self):
        self.assertIsNone(volatility.expected_range(np.array([100.0, 101.0, 102.0])))

    def test_coverage_is_calibrated(self):
        # On a driftless known-vol process the shipped 80% band should cover
        # ~80% of realized 5-day moves out of sample (allow generous slack).
        prices = _gbm(0.02, 6000, seed=7)
        logp = np.log(prices)
        H = 5
        hits = total = 0
        for i in range(60, len(prices) - H):
            r = volatility.expected_range(prices[: i + 1])
            if r is None:
                continue
            fwd = prices[i + H]
            hits += int(r["low"] <= fwd <= r["high"])
            total += 1
        coverage = hits / total
        self.assertGreater(coverage, 0.73)
        self.assertLess(coverage, 0.87)


if __name__ == "__main__":
    unittest.main()
