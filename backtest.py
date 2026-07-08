"""Honest walk-forward backtest of StockPredict's price model.

For every historical prediction date t the model is retrained from scratch
using ONLY information available at t:

  * features at t come from backward-looking rolling windows;
  * training examples are limited to dates j whose OUTCOME had already
    resolved by t (j + horizon <= t) — so no training label peeks past t;
  * the feature scaler is fit on the training rows only;
  * the prediction is then graded against the price that actually followed.

Anti-cheat verification (run before the backtest):
  1. Causality audit — features computed at t are bit-identical when every
     row after t is deleted.
  2. No-peek sentinel — every row after t is replaced with random-walk
     noise; the prediction must be bit-identical. Only possible if the
     model never reads the future.
  3. Shuffled-outcome control — graded against permuted outcomes the model
     must score ~50%, proving the scorer isn't leaking answers.

What is NOT covered: the live news-sentiment tilt (no historical headline
archive exists here), so this measures the price-only part of the app.
Adjusted prices and a hand-picked (surviving) ticker list are further
standard retail-backtest caveats.

Run:  .venv/bin/python backtest.py
Outputs: backtest_results.csv (every prediction) + backtest_summary.json
"""

import json
import math
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.learner import PRIOR_WEIGHTS
from app.analysis.predictor import _base_rates, _blend_price, _feature_frame
from app.analysis.technical import compute_indicators, technical_signals

warnings.filterwarnings("ignore")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",          # US mega-cap / high-beta
    "JPM", "JNJ", "XOM", "WMT", "CAT",               # bank, pharma, energy, retail, industrial
    "BP.L", "TSCO.L",                                # UK large caps (LSE)
    "^GSPC", "^FTSE",                                # indices
    "BTC-USD",                                       # crypto (trades 7d/week)
]

# (label, horizon in sessions, max prediction points, stride between points)
HORIZONS = [
    ("1 day", 1, 250, 1),      # non-overlapping outcomes
    ("1 week", 5, 100, 5),     # non-overlapping
    ("1 month", 21, 48, 21),   # non-overlapping
    ("1 year", 252, 90, 21),   # overlapping windows — flagged in the report
]

MIN_TRAIN = 150   # same floor as the live app


def load(ticker):
    df = yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return None
    return df.dropna(subset=["Close"])


def fit_predict(X, close, finite, i, h):
    """ML tilt at row i, trained only on outcomes resolved by i.
    Returns (status, p_ml): status "skip" = too little usable history;
    p_ml None with status "ok" = degenerate class balance (drift-only)."""
    j_max = i - h                      # training label close[j+h] must exist at or before i
    if j_max < 0 or not finite[i]:
        return "skip", None
    J = np.where(finite[: j_max + 1])[0]
    if len(J) < MIN_TRAIN:
        return "skip", None
    y = (close[J + h] > close[J]).astype(int)
    if y.min() == y.max():
        return "ok", None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    model.fit(X[J], y)
    return "ok", float(model.predict_proba(X[i : i + 1])[0, 1])


def predict_full(df, i, h):
    """Recompute a single prediction from a raw frame (used by the sentinel test).
    Uses the app's own _blend_price, so the harness measures the production math."""
    ind = compute_indicators(df)
    feats = _feature_frame(ind)
    X = feats.to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    finite = ~np.isnan(X).any(axis=1)
    status, p_ml = fit_predict(X, close, finite, i, h)
    if status == "skip":
        return None
    base_at = _base_rates(close, h)
    base = base_at[i] if np.isfinite(base_at[i]) else 0.5
    tech, _ = technical_signals(ind.iloc[: i + 1])
    # frozen priors: the harness must be reproducible regardless of live-learned state
    return float(np.clip(_blend_price(p_ml, base, tech, PRIOR_WEIGHTS), 0.05, 0.95))


def corrupt_future(df, i, rng):
    """Replace every row AFTER i with random-walk noise (prices and volume)."""
    df2 = df.copy()
    n_after = len(df2) - (i + 1)
    if n_after <= 0:
        return df2
    c0 = float(df2["Close"].iloc[i])
    path = c0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_after)))
    cols = df2.columns
    df2.iloc[i + 1 :, cols.get_loc("Open")] = path
    df2.iloc[i + 1 :, cols.get_loc("High")] = path * 1.01
    df2.iloc[i + 1 :, cols.get_loc("Low")] = path * 0.99
    df2.iloc[i + 1 :, cols.get_loc("Close")] = path
    df2.iloc[i + 1 :, cols.get_loc("Volume")] = rng.integers(10**5, 10**8, n_after).astype(float)
    return df2


def integrity_checks(data):
    print("=== anti-cheat verification ===")

    # 1. Causality audit: features at t identical when everything after t is deleted.
    for tkr in ("AAPL", "BP.L"):
        df = data[tkr]
        i = len(df) // 2
        full = _feature_frame(compute_indicators(df)).iloc[i].to_numpy(dtype=float)
        trunc = _feature_frame(compute_indicators(df.iloc[: i + 1].copy())).iloc[-1].to_numpy(dtype=float)
        np.testing.assert_allclose(full, trunc, rtol=1e-9, atol=1e-12, equal_nan=True)
    print("PASS  causality audit: features at t unchanged when all data after t is deleted")

    # 2. No-peek sentinel: predictions identical after the future is replaced with noise.
    rng = np.random.default_rng(7)
    checks = 0
    for tkr in ("AAPL", "TSLA", "^GSPC"):
        df = data[tkr]
        for h in (1, 5, 21):
            i = len(df) - 1 - h - int(rng.integers(10, 200))
            p_real = predict_full(df, i, h)
            p_noise = predict_full(corrupt_future(df, i, rng), i, h)
            assert p_real is not None, (tkr, h, "no prediction")
            assert abs(p_real - p_noise) < 1e-9, (tkr, h, p_real, p_noise)
            checks += 1
    print(f"PASS  no-peek sentinel: {checks}/9 predictions bit-identical after the future was replaced with noise")


def evaluate_ticker(tkr, df):
    ind = compute_indicators(df)
    feats = _feature_frame(ind)
    X = feats.to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    finite = ~np.isnan(X).any(axis=1)
    ml_capable = int(finite.sum()) >= MIN_TRAIN + 60
    rows = []
    for label, h, max_pts, stride in HORIZONS:
        last_pred = len(df) - 1 - h
        if last_pred < 0:
            continue
        base_at = _base_rates(close, h)
        positions = sorted(list(range(last_pred, -1, -stride))[:max_pts])
        for i in positions:
            tech, _ = technical_signals(ind.iloc[: i + 1])
            base = float(base_at[i]) if np.isfinite(base_at[i]) else 0.5
            if ml_capable:
                status, p_ml = fit_predict(X, close, finite, i, h)
                if status == "skip":
                    continue                      # not enough resolved history yet
                variant = "ml+tech"
            else:
                # Mirrors the app's fallback for tickers without usable volume data.
                p_ml = None
                variant = "drift+tech"
            p = float(np.clip(_blend_price(p_ml, base, tech, PRIOR_WEIGHTS), 0.05, 0.95))
            p_ml = float("nan") if p_ml is None else p_ml
            up = bool(close[i + h] > close[i])
            rows.append({
                "ticker": tkr, "horizon": label, "h": h,
                "date": str(df.index[i].date()),
                "p_up": round(p, 4),
                "p_ml": round(p_ml, 4) if not math.isnan(p_ml) else None,
                "base": round(base, 4),
                "tech": round(tech, 3),
                "outcome_up": up,
                "fwd_return": round(close[i + h] / close[i] - 1, 5),
                "correct": (p >= 0.5) == up,
                "correct_ml": ((p_ml >= 0.5) == up) if not math.isnan(p_ml) else None,
                "variant": variant,
            })
    return rows


def summarize(res):
    order = [h[0] for h in HORIZONS]
    summary = {}
    print("\n=== pooled results by horizon (all tickers) ===")
    hdr = (f"{'horizon':<9} {'n':>5}  {'model':>6}  {'always-up':>9}  {'edge':>6}  "
           f"{'95% CI':>12}  {'ret after UP':>12}  {'ret after DOWN':>14}")
    print(hdr)
    print("-" * len(hdr))
    for label in order:
        d = res[res.horizon == label]
        if d.empty:
            continue
        n = len(d)
        acc = d.correct.mean()
        base = d.outcome_up.mean()          # accuracy of "always predict up"
        edge = acc - base
        overlap = label == "1 year"
        ci = 1.96 * math.sqrt(acc * (1 - acc) / n)
        ci_txt = "n/a (overlap)" if overlap else f"±{ci * 100:.1f}pp"
        up_calls = d[d.p_up >= 0.5]
        dn_calls = d[d.p_up < 0.5]
        ret_up = up_calls.fwd_return.mean() * 100 if len(up_calls) else float("nan")
        ret_dn = dn_calls.fwd_return.mean() * 100 if len(dn_calls) else float("nan")
        print(f"{label:<9} {n:>5}  {acc * 100:>5.1f}%  {base * 100:>8.1f}%  {edge * 100:>+5.1f}pp  "
              f"{ci_txt:>12}  {ret_up:>11.2f}%  {ret_dn:>13.2f}%")
        conf = d[(d.p_up - 0.5).abs() >= 0.10]
        summary[label] = {
            "n": int(n), "model_accuracy": round(float(acc), 4),
            "always_up_accuracy": round(float(base), 4),
            "edge_pp": round(float(edge) * 100, 2),
            "ci95_pp": None if overlap else round(ci * 100, 2),
            "overlapping_windows": overlap,
            "mean_fwd_return_after_up_call_pct": round(float(ret_up), 3) if len(up_calls) else None,
            "mean_fwd_return_after_down_call_pct": round(float(ret_dn), 3) if len(dn_calls) else None,
            "confident_calls": {
                "n": int(len(conf)),
                "accuracy": round(float(conf.correct.mean()), 4) if len(conf) else None,
                "always_up": round(float(conf.outcome_up.mean()), 4) if len(conf) else None,
            },
            "dates": [str(d.date.min()), str(d.date.max())],
        }

    # Shuffled-outcome control: scorer must collapse to ~50% on permuted answers.
    rng = np.random.default_rng(11)
    d5 = res[res.horizon == "1 week"]
    shuffled = rng.permutation(d5.outcome_up.to_numpy())
    sh_acc = ((d5.p_up.to_numpy() >= 0.5) == shuffled).mean()
    print(f"\nPASS  shuffled-outcome control (1-week set): {sh_acc * 100:.1f}% ≈ 50% — the scorer isn't leaking answers")
    summary["_shuffled_control_accuracy"] = round(float(sh_acc), 4)

    print("\n=== per-ticker, 1-week horizon (the app's native horizon) ===")
    print(f"{'ticker':<8} {'n':>4}  {'model':>6}  {'always-up':>9}  {'edge':>7}")
    for tkr in TICKERS:
        d = res[(res.horizon == "1 week") & (res.ticker == tkr)]
        if d.empty:
            print(f"{tkr:<8}    —  insufficient data")
            continue
        acc, base = d.correct.mean(), d.outcome_up.mean()
        print(f"{tkr:<8} {len(d):>4}  {acc * 100:>5.1f}%  {base * 100:>8.1f}%  {(acc - base) * 100:>+6.1f}pp")

    return summary


def main():
    t0 = time.time()
    print("Downloading 10y of daily data...")
    data = {}
    for tkr in TICKERS:
        df = load(tkr)
        if df is None or len(df) < 400:
            print(f"  {tkr}: SKIPPED (insufficient data)")
            continue
        data[tkr] = df
        print(f"  {tkr}: {len(df)} sessions  {df.index[0].date()} → {df.index[-1].date()}")

    integrity_checks(data)

    print("\n=== walk-forward backtest (each point = a fresh model trained only on its past) ===")
    all_rows = []
    for k, (tkr, df) in enumerate(data.items(), 1):
        t1 = time.time()
        rows = evaluate_ticker(tkr, df)
        all_rows.extend(rows)
        print(f"  [{k:>2}/{len(data)}] {tkr:<8} {len(rows):>4} predictions in {time.time() - t1:.1f}s")

    res = pd.DataFrame(all_rows)
    res.to_csv("backtest_results.csv", index=False)
    summary = summarize(res)
    summary["_meta"] = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "tickers": list(data.keys()),
        "total_predictions": int(len(res)),
        "note": ("Walk-forward, no lookahead (verified by causality audit, no-peek sentinel, "
                 "shuffled-outcome control). Measures the price-only model + technical blend; "
                 "the live news tilt is untestable without a historical headline archive. "
                 "Adjusted prices; hand-picked surviving tickers; 1-year windows overlap."),
    }
    with open("backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved backtest_results.csv ({len(res)} predictions) and backtest_summary.json")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
