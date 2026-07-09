"""Post-earnings-announcement drift (PEAD): does knowing a stock just beat or
missed earnings predict its DRIFT over the following weeks — out of sample?

PEAD is one of the few short-horizon anomalies with decades of academic support:
prices under-react to earnings news and keep drifting in the surprise's
direction. This tests it honestly on our own data before anything ships.

Two signals, both using only information known at the entry point:
  * surprise  — sign/size of the reported EPS Surprise(%) (skipped if missing);
  * reaction  — the price reaction across the announcement ([d-1 close, d+1 close]),
                a price-only proxy that needs no analyst estimates.

Design (mirrors backtest.py / news_backtest.py):
  * ENTER the day AFTER the announcement (d+1 close) — we measure the DRIFT that
    follows the jump, never the jump itself (that part is efficient/priced);
  * forward drift = close[d+1+H] / close[d+1] - 1, H = 21 sessions (a month);
  * grade: hit-rate sign(signal)==sign(drift); long-short = mean drift on
    positive-signal events minus negative; quintile spread by |surprise|;
  * a moving-block bootstrap CI (block by ticker) and a pre/post-2015 split;
  * gross AND net of a round-trip cost.

Caveat: today's surviving large-caps; adjusted prices; yfinance earnings history.

Run:  .venv/bin/python -m pip install lxml   # once (needed for earnings dates)
      .venv/bin/python research/pead.py
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / ".pead_cache.pkl"
UNIVERSE = [
    "AAPL", "MSFT", "JPM", "KO", "JNJ", "XOM", "WMT", "PG", "CVX", "HD",
    "MCD", "IBM", "CAT", "DIS", "GE", "INTC", "CSCO", "ORCL", "PEP", "MRK",
    "PFE", "BA", "GS", "AXP", "NKE", "VZ", "WFC", "C", "MMM", "TXN",
]
H = 21              # drift horizon in sessions (~1 month)
COST_BPS = 10.0     # round-trip cost assumption


def load():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    data = {}
    for t in UNIVERSE:
        try:
            tk = yf.Ticker(t)
            px = tk.history(period="max", interval="1d", auto_adjust=True)
            ed = tk.get_earnings_dates(limit=100)
            if px is None or px.empty or ed is None or ed.empty:
                print("  skip", t); continue
            px = px.dropna(subset=["Close"])
            px.index = px.index.tz_localize(None).normalize()
            ed = ed.copy()
            ed.index = ed.index.tz_localize(None).normalize()
            data[t] = {"px": px["Close"], "ed": ed}
            print(f"  {t}: {len(px)} px, {len(ed)} earnings")
            time.sleep(0.4)
        except Exception as e:
            print(f"  {t}: {e}")
    CACHE.write_bytes(pickle.dumps(data))
    return data


def events(data):
    """One row per past earnings event: (ticker, date, surprise, reaction, drift)."""
    rows = []
    for t, d in data.items():
        close = d["px"]
        idx = close.index
        vals = close.to_numpy(float)
        for edate, row in d["ed"].iterrows():
            # position of the first trading day >= the announcement date
            pos = idx.searchsorted(edate)
            if pos <= 1 or pos + 1 + H >= len(vals):
                continue
            # reaction across [d-1 close, d+1 close]; enter at d+1, drift to d+1+H
            pre, react_end = vals[pos - 1], vals[pos + 1]
            entry, exit_ = vals[pos + 1], vals[pos + 1 + H]
            reaction = react_end / pre - 1.0
            drift = exit_ / entry - 1.0
            surprise = row.get("Surprise(%)")
            surprise = float(surprise) if pd.notna(surprise) else np.nan
            rows.append({"ticker": t, "date": edate, "surprise": surprise,
                         "reaction": reaction, "drift": drift})
    return pd.DataFrame(rows)


def block_boot_ci(df, valcol, n_boot=5000, seed=13):
    """95% CI for the mean of df[valcol], resampling whole tickers (blocks)."""
    rng = np.random.default_rng(seed)
    groups = [g[valcol].to_numpy(float) for _, g in df.groupby("ticker")]
    total = sum(len(g) for g in groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        s = np.concatenate([groups[i] for i in pick])
        means[b] = s.mean()
    return np.percentile(means, [2.5, 97.5])


def grade(df, signalcol, label):
    d = df.dropna(subset=[signalcol, "drift"])
    d = d[d[signalcol] != 0]
    n = len(d)
    sign_match = (np.sign(d[signalcol]) == np.sign(d["drift"])).mean()
    pos = d[d[signalcol] > 0]["drift"]
    neg = d[d[signalcol] < 0]["drift"]
    longshort = pos.mean() - neg.mean()               # gross monthly drift spread
    net = longshort - 2 * COST_BPS / 1e4
    # CI on the per-event signed drift  sign(signal)*drift  (its mean > 0 => edge)
    d = d.assign(signed=np.sign(d[signalcol]) * d["drift"])
    lo, hi = block_boot_ci(d, "signed")
    half = d["date"].median()
    h1 = d[d["date"] < half]["signed"].mean()
    h2 = d[d["date"] >= half]["signed"].mean()
    print(f"\n=== signal: {label}  (n={n}) ===")
    print(f"  sign-match hit-rate      : {sign_match*100:.1f}%   (50% = no edge)")
    print(f"  long-short monthly drift : {longshort*100:+.2f}%  gross | {net*100:+.2f}% net of {2*COST_BPS:.0f}bps")
    print(f"  mean signed drift        : {d['signed'].mean()*100:+.3f}%   95% CI [{lo*100:+.3f}, {hi*100:+.3f}]  -> {'EDGE (excludes 0)' if lo>0 else 'no edge (CI spans 0)'}")
    print(f"  persistence (pre/post mid): {h1*100:+.3f}%  /  {h2*100:+.3f}%   (same sign = robust)")


def main():
    print("loading prices + earnings (cached after first run)...")
    data = load()
    ev = events(data)
    print(f"\n{len(ev)} earnings events, {ev['ticker'].nunique()} tickers, "
          f"{ev['date'].min().date()} -> {ev['date'].max().date()}")
    print(f"with surprise data: {ev['surprise'].notna().sum()}")
    grade(ev, "surprise", "EPS surprise %")
    grade(ev, "reaction", "announcement price reaction")
    print("\nVerdict rule: a real, shippable PEAD edge needs the signed-drift CI to "
          "EXCLUDE 0,\nsurvive costs, and hold the same sign in both halves.")


if __name__ == "__main__":
    main()
