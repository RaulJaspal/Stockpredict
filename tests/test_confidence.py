"""Tests for the noise-aware confidence downgrade (predictor._confidence).

The per-ticker holdout grades 60 OVERLAPPING 5-day windows, so it carries only
~12 independent samples. The downgrade to 'low' must therefore fire only when
the holdout trails the always-up baseline by more than one standard error at
that effective sample size — not on every sub-baseline blip (which happens ~half
the time by noise).
"""

import unittest

from app.analysis.predictor import _confidence


def _holdout(hit, baseline, days=60):
    return {"hit_rate": hit, "baseline": baseline, "holdout_days": days,
            "effective_n": max(1, days // 5)}


class ConfidenceTest(unittest.TestCase):
    def test_no_downgrade_within_noise(self):
        # hit just below baseline but well within 1 SE (SE ~= 0.144 at eff_n=12).
        level, caveat = _confidence(0.62, _holdout(hit=0.55, baseline=0.60))
        self.assertIsNone(caveat)
        self.assertEqual(level, "high")   # |0.62-0.5| >= 0.10

    def test_downgrade_on_clear_shortfall(self):
        # hit far below baseline (beyond 1 SE) -> downgrade + caveat.
        level, caveat = _confidence(0.62, _holdout(hit=0.30, baseline=0.60))
        self.assertEqual(level, "low")
        self.assertIsNotNone(caveat)

    def test_no_holdout_no_caveat(self):
        level, caveat = _confidence(0.55, None)
        self.assertIsNone(caveat)
        self.assertEqual(level, "medium")


if __name__ == "__main__":
    unittest.main()
