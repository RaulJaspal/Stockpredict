"""Historical daily news-tone series from the GDELT 2.0 DOC API.

GDELT indexes worldwide news and exposes the average article *tone* per day for
any query (``mode=timelinetone``) — free, no API key, daily resolution back to
2017. We use it to reconstruct a historical company-news sentiment series so the
news tilt (untestable against a live headline archive) can finally be backtested
against outcomes. See ``news_backtest.py``.

Tone runs roughly [-100, +100] but in practice clusters in [-10, +10] and carries
a persistent negativity bias for business news, so callers standardise it per
query before use (a raw tone of -0.3 is a *positive* day for a company whose news
usually reads -1.5). Responses are cached on disk keyed by query+range — GDELT
asks for gentle, cached access (one request / 5 s) and a past range never changes.
"""

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_START = "2017-01-01"          # DOC 2.0 coverage begins here
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gdelt_cache")
MIN_INTERVAL = 8.0                  # seconds between live calls (GDELT asks 1 / 5 s;
                                    # a burst trips a longer IP cooldown, so stay clear)
THROTTLE_BACKOFF = 25.0             # flat wait after a rate-limit notice — hammering it
                                    # with fast retries only extends the penalty
_UA = "StockPredict-research/1.0 (educational backtest)"
_last_call = 0.0                    # module-level throttle across all queries


def _cache_path(query, start, end):
    key = hashlib.sha1(f"{query}|{start}|{end}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"tone_{key}.json")


def _throttle():
    global _last_call
    wait = MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _fetch(query, start, end, timeout, retries):
    """One live GDELT call. Returns the parsed JSON, or {'_error': ...} on
    failure (including the plain-text rate-limit notice GDELT returns as 200)."""
    params = {
        "query": query, "mode": "timelinetone", "format": "json",
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "000000",
    }
    url = GDELT_URL + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            txt = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
        except Exception as exc:                       # network / HTTP error
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(MIN_INTERVAL * (attempt + 1))
            continue
        if txt.strip().startswith("{"):
            return json.loads(txt)
        last = "throttled/non-json: " + txt.strip()[:100]
        time.sleep(THROTTLE_BACKOFF)                    # patient, flat cooldown on a rate-limit notice
    return {"_error": last}


def daily_tone(query, start=GDELT_START, end=None, timeout=60, retries=4, verbose=False):
    """Daily average article tone for ``query`` between ``start`` and ``end``
    (YYYY-MM-DD strings). Returns a tz-naive ``pd.Series`` indexed by day, or an
    empty Series if GDELT has no data. Cached on disk; a cache hit makes no call."""
    end = end or time.strftime("%Y-%m-%d")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(query, start, end)
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
    else:
        if verbose:
            print(f"  [gdelt] fetching {query!r} {start}..{end}")
        raw = _fetch(query, start, end, timeout, retries)
        if "_error" not in raw:                         # never cache a failure
            with open(path, "w") as fh:
                json.dump(raw, fh)
        elif verbose:
            print(f"  [gdelt] FAILED {query!r}: {raw['_error']}")
    data = ((raw.get("timeline") or [{}])[0]).get("data") or []
    if not data:
        return pd.Series(dtype=float, name="tone")
    idx = pd.to_datetime([d["date"] for d in data], utc=True).tz_localize(None).normalize()
    vals = [float(d["value"]) for d in data]
    return pd.Series(vals, index=idx, name="tone").sort_index()
