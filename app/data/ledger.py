"""Live prediction ledger: every prediction the app makes is appended to
predictions.jsonl (one record per ticker per trading day). As outcomes mature,
/api/track-record grades them against what prices actually did — a true
out-of-sample test that accumulates from the day the app started running,
and the only way the live news tilt can ever be validated.
"""

import json
import os
import threading
import time
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parents[2] / "predictions.jsonl"

# Read-only mode: when set, the app serves predictions but never writes to the
# ledger. Used by the local dashboard so the authoritative writer (the daily
# GitHub Actions "tick") is the ONLY thing appending — the git-tracked ledger
# then only ever changes via `git pull`, so it can never conflict or go dirty.
READONLY = os.environ.get("STOCKPREDICT_READONLY") == "1"

_lock = threading.Lock()
_seen = None


def _keys():
    global _seen
    if _seen is None:
        _seen = set()
        if LEDGER_PATH.exists():
            for line in LEDGER_PATH.read_text().splitlines():
                try:
                    r = json.loads(line)
                    _seen.add((r["ticker"], r["as_of"], r["horizon_days"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return _seen


def record(entry):
    """Append one prediction; deduped on (ticker, as_of, horizon).
    No-op in read-only mode (see READONLY) so only the daily job writes."""
    if READONLY:
        return
    key = (entry["ticker"], entry["as_of"], entry["horizon_days"])
    with _lock:
        seen = _keys()
        if key in seen:
            return
        seen.add(key)
        entry = {"logged_at": int(time.time()), **entry}
        with LEDGER_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def read_all():
    with _lock:
        if not LEDGER_PATH.exists():
            return []
        records = []
        for line in LEDGER_PATH.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def resolve_records(records):
    """Grade every record whose horizon has matured against the price that
    actually followed (one consistently adjusted frame per ticker). Returns
    {"resolved": [record + outcome fields], "pending": count}. Shared by
    /api/track-record and the online learner."""
    from .market import get_history  # local import keeps ledger dependency-light

    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)
    resolved, pending = [], 0
    for ticker, recs in by_ticker.items():
        try:
            df = get_history(ticker)
        except Exception:
            pending += len(recs)
            continue
        dates = df.index.strftime("%Y-%m-%d")
        pos = {s: i for i, s in enumerate(dates)}
        close = df["Close"].to_numpy(dtype=float)
        for r in recs:
            i = pos.get(r["as_of"])
            h = int(r["horizon_days"])
            if i is None or i + h >= len(close):
                pending += 1
                continue
            realized = float(close[i + h] / close[i] - 1)
            went_up = realized > 0
            end_price = float(close[i + h])
            lo, hi = r.get("range_low"), r.get("range_high")
            in_band = (bool(lo <= end_price <= hi)
                       if lo is not None and hi is not None else None)
            resolved.append({
                **r,
                "resolved_on": dates[i + h],
                "realized_pct": round(realized * 100, 2),
                "end_price": round(end_price, 4),
                "outcome_up": went_up,
                "correct": (r["direction"] == "up") == went_up,
                "in_band": in_band,
            })
    return {"resolved": resolved, "pending": pending}
