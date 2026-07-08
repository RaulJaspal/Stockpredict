"""Safety-rail tests for the online learner (app/analysis/learner.py).

These replace the throwaway scratchpad tests the HANDOFF mentions: they live in
the repo now and run with the stdlib, no pytest needed:

    .venv/bin/python -m unittest discover -s tests

Each test redirects the learner's state file to a temp path so the real
model_state.json is never touched.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.analysis import learner


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _records(n, outcome_fn, seed=0, news_key="news_company", reversed_tail=None):
    """n synthetic resolved predictions. `outcome_fn(nc) -> P(up)` drives the
    label from the chosen news component; other components are neutral. If
    `reversed_tail` is set, the last `reversed_tail` fraction uses -outcome_fn
    (to exercise the shadow-adoption holdout)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="6h")
    recs = []
    for i in range(n):
        nc = float(rng.uniform(-1, 1))
        p = outcome_fn(nc)
        if reversed_tail and i >= int(n * (1 - reversed_tail)):
            p = 1.0 - p
        recs.append({
            "horizon_days": 5, "as_of": str(dates[i]),
            "base": 0.5, "tech": 0.0, "p_ml": None,
            "news_company": 0.0, "news_market": 0.0, "news_politics": 0.0,
            "outcome_up": bool(rng.random() < p),
            news_key: nc,
        })
    return recs


class LearnerRailsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig_path = learner.STATE_PATH
        learner.STATE_PATH = Path(self._tmp.name)
        learner._state = None                      # force reload from the temp file

    def tearDown(self):
        learner.STATE_PATH = self._orig_path
        learner._state = None
        Path(self._tmp.name).unlink(missing_ok=True)

    def test_gate_blocks_adaptation_below_min_resolved(self):
        recs = _records(learner.MIN_RESOLVED - 5, lambda nc: _sigmoid(1.5 * nc))
        out = learner.update_from_ledger(recs)
        self.assertEqual(out["source"], "prior")
        self.assertEqual(out["n_used"], len(recs))
        self.assertEqual(out["weights"], {k: round(v, 3) for k, v in learner.PRIOR_WEIGHTS.items()})

    def test_true_signal_is_kept(self):
        recs = _records(300, lambda nc: _sigmoid(1.5 * nc))
        out = learner.update_from_ledger(recs)
        self.assertEqual(out["source"], "adaptive")
        self.assertGreater(out["weights"]["news_company"], 0.60,
                           "a genuine news signal was wrongly shrunk")

    def test_genuine_signal_learns_higher_weight_than_noise(self):
        # The design deliberately uses a loose news prior (sigma 0.5) so weights
        # move slowly; an absolute noise threshold is therefore seed-sensitive
        # (the shadow test may restore the 0.45 prior or adopt a ~0 candidate).
        # The robust, meaningful claim is comparative: a real news->direction
        # signal must yield a substantially higher learned weight than pure noise.
        signal = learner.update_from_ledger(
            _records(300, lambda nc: _sigmoid(1.5 * nc), seed=1))["weights"]["news_company"]
        learner._state = None                      # fresh fit for the noise case
        noise = learner.update_from_ledger(
            _records(300, lambda nc: 0.5, seed=1))["weights"]["news_company"]
        self.assertGreater(signal - noise, 0.30,
                           f"signal weight {signal:.3f} not clearly above noise {noise:.3f}")
        self.assertLessEqual(noise, learner.PRIOR_WEIGHTS["news_company"] + 1e-6,
                             "pure noise must never inflate the news weight above its prior")

    def test_shadow_test_rejects_unstable_candidate(self):
        # Signal flips sign between the train (earliest 70%) and holdout (latest
        # 30%) segments -> the candidate must lose out-of-sample and be rejected.
        recs = _records(300, lambda nc: _sigmoid(2.0 * nc), reversed_tail=0.30)
        out = learner.update_from_ledger(recs)
        self.assertIn("rejected", out["source"])
        self.assertEqual(out["weights"], {k: round(v, 3) for k, v in learner.PRIOR_WEIGHTS.items()})

    def test_weights_always_within_bounds(self):
        recs = _records(300, lambda nc: _sigmoid(4.0 * nc))   # very strong signal
        out = learner.update_from_ledger(recs)
        for k, (lo, hi) in learner.BOUNDS.items():
            self.assertGreaterEqual(out["weights"][k], lo)
            self.assertLessEqual(out["weights"][k], hi)


if __name__ == "__main__":
    unittest.main()
