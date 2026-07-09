"""Hermetic no-lookahead guarantee for the price model.

backtest.py runs a causality audit and a no-peek sentinel, but only against
live yfinance downloads — so the guarantee is never enforced in CI. This ports
the same two checks onto a synthetic OHLCV frame (no network), so every commit
proves the model cannot read the future:

  1. Causality audit  — features computed at row t are bit-identical when every
     row after t is deleted.
  2. No-peek sentinel — the prediction at t is bit-identical after every row
     after t is replaced with random-walk noise.
"""

import unittest

import numpy as np
import pandas as pd

from backtest import corrupt_future, predict_full
from app.analysis.predictor import _feature_frame
from app.analysis.technical import compute_indicators


def _synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    span = np.abs(rng.normal(0, 0.015, n)) * close
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.005, n)),
        "High": close + span,
        "Low": close - span,
        "Close": close,
        "Volume": rng.integers(1e6, 5e7, n).astype(float),
    }, index=pd.date_range("2016-01-01", periods=n, freq="B"))


class NoLookaheadTest(unittest.TestCase):
    def test_causality_audit(self):
        df = _synthetic(seed=1)
        i = len(df) // 2
        full = _feature_frame(compute_indicators(df)).iloc[i].to_numpy(float)
        trunc = _feature_frame(compute_indicators(df.iloc[: i + 1].copy())).iloc[-1].to_numpy(float)
        np.testing.assert_allclose(full, trunc, rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_no_peek_sentinel(self):
        rng = np.random.default_rng(2)
        df = _synthetic(seed=3)
        checks = 0
        for h in (1, 5, 21):
            i = len(df) - 1 - h - int(rng.integers(10, 200))
            p_real = predict_full(df, i, h)
            p_noise = predict_full(corrupt_future(df, i, rng), i, h)
            self.assertIsNotNone(p_real)
            self.assertLess(abs(p_real - p_noise), 1e-9,
                            f"prediction changed when the future was replaced with noise (h={h})")
            checks += 1
        self.assertEqual(checks, 3)


if __name__ == "__main__":
    unittest.main()
