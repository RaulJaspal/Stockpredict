"""Tests for the GDELT client (app/data/gdelt.py) — parsing & caching, no network.

Each test seeds the on-disk cache so daily_tone() never makes a live call: it
must read, parse, and normalise the cached response deterministically.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.data import gdelt


class GdeltParseTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._orig = gdelt.CACHE_DIR
        gdelt.CACHE_DIR = self._dir

    def tearDown(self):
        gdelt.CACHE_DIR = self._orig

    def _seed(self, query, start, end, payload):
        path = gdelt._cache_path(query, start, end)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh)

    def test_parses_and_sorts_daily_tone(self):
        query, start, end = '"Acme"', "2020-01-01", "2020-01-04"
        # Deliberately out of order to prove daily_tone sorts by date.
        self._seed(query, start, end, {"timeline": [{"series": "Average Tone", "data": [
            {"date": "20200103T000000Z", "value": 1.5},
            {"date": "20200101T000000Z", "value": -2.0},
            {"date": "20200102T000000Z", "value": 0.25},
        ]}]})
        s = gdelt.daily_tone(query, start=start, end=end)
        self.assertEqual(len(s), 3)
        self.assertTrue(s.index.is_monotonic_increasing)
        self.assertIsNone(s.index.tz, "index should be tz-naive")
        self.assertEqual(s.index[0], pd.Timestamp("2020-01-01"))
        self.assertAlmostEqual(s.loc["2020-01-01"], -2.0)
        self.assertAlmostEqual(s.loc["2020-01-03"], 1.5)

    def test_empty_timeline_returns_empty_series(self):
        query, start, end = '"NothingHere"', "2020-01-01", "2020-02-01"
        self._seed(query, start, end, {"timeline": [{"series": "Average Tone", "data": []}]})
        s = gdelt.daily_tone(query, start=start, end=end)
        self.assertTrue(s.empty)

    def test_cache_path_is_deterministic_and_query_sensitive(self):
        a = gdelt._cache_path('"Acme"', "2020-01-01", "2020-02-01")
        b = gdelt._cache_path('"Acme"', "2020-01-01", "2020-02-01")
        c = gdelt._cache_path('"Other"', "2020-01-01", "2020-02-01")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
