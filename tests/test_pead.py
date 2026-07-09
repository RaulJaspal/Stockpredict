"""Tests for the post-earnings-drift tilt (predictor._pead_tilt).

The tilt must: never touch the weekly (default) horizon; fire only in the drift
window after a report; move the monthly call in the surprise's direction, scaled
and decaying; and stay bounded no matter how absurd the raw surprise is (Yahoo's
Surprise(%) can be hundreds of percent when the estimate is near zero).
"""

import unittest
from unittest import mock

from app.analysis import predictor
from app.config import HORIZON_DAYS, PEAD

MONTHLY = 21


def _earn(surprise, ago):
    return {"date": "2026-01-01", "surprise_pct": surprise, "sessions_ago": ago}


class PeadTiltTest(unittest.TestCase):
    def _tilt(self, earnings, h=MONTHLY):
        with mock.patch.object(predictor.market, "get_recent_earnings", return_value=earnings):
            return predictor._pead_tilt("X", h)

    def test_weekly_never_tilts(self):
        tilt, info = self._tilt(_earn(50.0, 1), h=HORIZON_DAYS)
        self.assertEqual(tilt, 0.0)
        self.assertIsNone(info)

    def test_no_earnings_no_tilt(self):
        tilt, info = self._tilt(None)
        self.assertEqual(tilt, 0.0)
        self.assertIsNone(info)

    def test_recent_beat_tilts_up(self):
        tilt, info = self._tilt(_earn(15.0, 1))
        self.assertGreater(tilt, 0)
        self.assertTrue(info["active"])
        self.assertEqual(info["direction"], "up")

    def test_recent_miss_tilts_down(self):
        tilt, _ = self._tilt(_earn(-15.0, 1))
        self.assertLess(tilt, 0)

    def test_stale_report_no_tilt(self):
        tilt, info = self._tilt(_earn(15.0, 30))
        self.assertEqual(tilt, 0.0)
        self.assertFalse(info["active"])

    def test_decay_reduces_tilt(self):
        fresh, _ = self._tilt(_earn(15.0, 1))
        older, _ = self._tilt(_earn(15.0, 8))
        self.assertGreater(fresh, older)

    def test_outlier_surprise_is_bounded(self):
        tilt, _ = self._tilt(_earn(5000.0, 1))     # +5000% must not blow up
        self.assertLessEqual(abs(tilt), PEAD["coef"] * PEAD["z_clip"])


if __name__ == "__main__":
    unittest.main()
