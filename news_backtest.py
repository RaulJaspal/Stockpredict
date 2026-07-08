"""Does company news tone actually predict direction? A walk-forward test.

The live app tilts its probability by news sentiment with a weight of 0.45
(company), 0.18 (market), 0.12 (politics) — numbers chosen by *judgment*, never
validated, because no historical headline archive was on hand. GDELT changes
that: it exposes a daily average-tone series per query back to 2017
(`app/data/gdelt.py`). This script reconstructs a historical company-news tone
feature and measures, out of sample, whether it improves the app's price-only
prediction at the 5-session horizon.

Method (mirrors the project's tune/validate discipline):
  * The price-only probability at each date is the *production* blend
    (`_blend_price` with the frozen priors), computed walk-forward exactly as in
    backtest.py — no lookahead.
  * The news feature is the company's GDELT tone, smoothed over 3 days and
    standardised against a trailing 365-day window (mean/std known at t only),
    squashed to ~[-1, 1] with tanh. Standardising removes business news's
    persistent negativity bias, so the feature means "unusually good/bad news
    for THIS company right now" — the thing that should move price.
  * Tickers are split tune (even index) / validate (odd). The incremental news
    weight is chosen on tune by minimising Brier score, then applied blind to
    validate, where we report Brier improvement (with a block-bootstrap CI),
    direction accuracy, and a model-free rank test (AUC of tone vs outcome).

Honest scope: this validates whether news *tone* carries directional signal and
its rough magnitude. It is NOT a drop-in replacement for the live weight — the
live feature is a VADER mean over reliable-outlet headlines, a different
construction from GDELT worldwide tone. What it can settle is the qualitative
question the 0.45 weight has never had an answer to: is there anything there?

Run:  .venv/bin/python news_backtest.py    (first run fetches + caches GDELT)
"""

import json
import math
import time
import warnings

import numpy as np
import pandas as pd

from app.analysis.learner import PRIOR_WEIGHTS
from app.analysis.predictor import _base_rates, _blend_price, _feature_frame, _logit, _sigmoid
from app.analysis.technical import compute_indicators, technical_signals
from app.data import gdelt
from backtest import MIN_TRAIN, fit_predict, load

warnings.filterwarnings("ignore")

HORIZON = 5          # the app's native horizon (5 sessions)
STRIDE = 5           # non-overlapping weekly outcomes
SMOOTH_DAYS = 3      # trailing tone smoothing (mirrors the 48h company half-life)
STD_WINDOW = 365     # calendar days for the trailing standardisation baseline
TANH_SCALE = 1.5     # z-score -> ~[-1, 1]

# Curated GDELT queries: a single tight company phrase each. Multi-term OR
# queries were tried but time out / get throttled server-side over a 9-year
# span, so we keep one precise phrase that avoids homonyms ("Apple Inc" not the
# fruit, "Amazon.com" not the river).
QUERIES = {
    "AAPL": '"Apple Inc"',
    "MSFT": '"Microsoft"',
    "NVDA": '"Nvidia"',
    "TSLA": '"Tesla Inc"',
    "AMZN": '"Amazon.com"',
    "JPM":  '"JPMorgan"',
    "JNJ":  '"Johnson & Johnson"',
    "XOM":  '"Exxon Mobil"',
    "WMT":  '"Walmart"',
    "CAT":  '"Caterpillar Inc"',
    "BP.L": '"BP plc"',
    "TSCO.L": '"Tesco"',
}
# Tune on even-indexed tickers, validate on odd — same split as the rest of the project.
TICKERS = list(QUERIES)
TUNE = TICKERS[0::2]
VALIDATE = TICKERS[1::2]

W_GRID = np.round(np.arange(-0.60, 0.601, 0.02), 3)   # candidate news weights (logit space)


def news_feature(tone, index_dates):
    """No-lookahead news feature aligned to `index_dates` (tz-naive, normalised).

    tone: daily GDELT tone Series. Returns a dict date -> feature in ~[-1, 1],
    NaN where < STD_WINDOW days of history exist. Everything at date t uses only
    tone known by t."""
    if tone.empty:
        return {}
    daily = tone.asfreq("D").ffill(limit=7)                      # fill short gaps only
    smooth = daily.rolling(f"{SMOOTH_DAYS}D").mean()
    roll_mean = smooth.rolling(f"{STD_WINDOW}D").mean()
    roll_std = smooth.rolling(f"{STD_WINDOW}D").std()
    z = (smooth - roll_mean) / roll_std.replace(0.0, np.nan)
    feat = np.tanh(z / TANH_SCALE)
    return {d: float(feat.get(d, np.nan)) for d in index_dates}


def collect(ticker):
    """Walk-forward rows for one ticker: price-only logit, news feature, outcome."""
    df = load(ticker)
    if df is None or len(df) < 400:
        return []
    tone = gdelt.daily_tone(QUERIES[ticker], verbose=True)
    if tone.empty:
        print(f"  {ticker}: no GDELT tone — skipped")
        return []

    ind = compute_indicators(df)
    X = _feature_frame(ind).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    finite = ~np.isnan(X).any(axis=1)
    ml_capable = int(finite.sum()) >= MIN_TRAIN + 60
    base_at = _base_rates(close, HORIZON)

    dates = [pd.Timestamp(t).tz_convert(None).normalize() if pd.Timestamp(t).tz
             else pd.Timestamp(t).normalize() for t in df.index]
    feat = news_feature(tone, set(dates))

    rows = []
    last_pred = len(df) - 1 - HORIZON
    for i in range(0, last_pred + 1, STRIDE):
        nf = feat.get(dates[i], np.nan)
        if not np.isfinite(nf):
            continue
        base = float(base_at[i]) if np.isfinite(base_at[i]) else 0.5
        if ml_capable:
            status, p_ml = fit_predict(X, close, finite, i, HORIZON)
            if status == "skip":
                continue
        else:
            p_ml = None
        tech, _ = technical_signals(ind.iloc[: i + 1])
        p_price = float(np.clip(_blend_price(p_ml, base, tech, PRIOR_WEIGHTS), 0.05, 0.95))
        rows.append({
            "ticker": ticker, "date": str(dates[i].date()),
            "logit_price": _logit(p_price),
            "news": float(nf),
            "outcome_up": bool(close[i + HORIZON] > close[i]),
        })
    print(f"  {ticker}: {len(rows)} usable weekly predictions "
          f"({rows[0]['date'] if rows else '—'} → {rows[-1]['date'] if rows else '—'})")
    return rows


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def accuracy(p, y):
    return float(np.mean((np.asarray(p) >= 0.5) == np.asarray(y)))


def auc(score, y):
    """Rank AUC of `score` predicting `y` (model-free) via the Mann-Whitney U."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = score.argsort()
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[y].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def apply_weight(df, w):
    return _vec_sigmoid(df["logit_price"].to_numpy() + w * df["news"].to_numpy())


def _vec_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def block_bootstrap_delta_brier(df, w, n_boot=3000, seed=17):
    """Bootstrap CI for Brier(price+news) - Brier(price-only) on `df`,
    resampling whole tickers (the natural independent block here). Negative =
    news helps."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("ticker", sort=False)]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        d = pd.concat([groups[k] for k in pick], ignore_index=True)
        y = d["outcome_up"].to_numpy()
        p0 = _vec_sigmoid(d["logit_price"].to_numpy())
        p1 = apply_weight(d, w)
        deltas[b] = brier(p1, y) - brier(p0, y)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(df, label, w):
    y = df["outcome_up"].to_numpy()
    p0 = _vec_sigmoid(df["logit_price"].to_numpy())
    p1 = apply_weight(df, w)
    return {
        "set": label, "n": int(len(df)), "weight": float(w),
        "brier_price_only": round(brier(p0, y), 5),
        "brier_price_news": round(brier(p1, y), 5),
        "brier_delta": round(brier(p1, y) - brier(p0, y), 5),
        "acc_price_only": round(accuracy(p0, y), 4),
        "acc_price_news": round(accuracy(p1, y), 4),
        "always_up": round(float(y.mean()), 4),
        "news_auc": round(auc(df["news"].to_numpy(), y), 4),
    }


def main():
    t0 = time.time()
    print("=== collecting walk-forward rows (price-only blend + GDELT news feature) ===")
    all_rows = []
    for tkr in TICKERS:
        all_rows.extend(collect(tkr))
    data = pd.DataFrame(all_rows)
    if data.empty:
        print("No rows collected (GDELT unreachable?). Aborting.")
        return
    data.to_csv("news_backtest_rows.csv", index=False)

    tune = data[data.ticker.isin(TUNE)].reset_index(drop=True)
    val = data[data.ticker.isin(VALIDATE)].reset_index(drop=True)

    # Choose the news weight on TUNE by minimising Brier, then apply blind to VALIDATE.
    y_t = tune["outcome_up"].to_numpy()
    briers = [(w, brier(apply_weight(tune, w), y_t)) for w in W_GRID]
    w_star = min(briers, key=lambda x: x[1])[0]
    w_star_val = min(((w, brier(apply_weight(val, w), val["outcome_up"].to_numpy())) for w in W_GRID),
                     key=lambda x: x[1])[0]        # best-on-validate, for reference only

    print(f"\n=== news weight chosen on TUNE ({'+'.join(TUNE)}) ===")
    print(f"  Brier-minimising weight on tune: w* = {w_star:+.2f}")
    print(f"  (for reference, best-on-validate would be {w_star_val:+.2f} — if these disagree in "
          f"sign, the signal is noise)")

    rows = [evaluate(tune, "tune", w_star), evaluate(val, "validate (blind w*)", w_star),
            evaluate(val, "validate (its own best w)", w_star_val)]
    print(f"\n{'set':<26} {'n':>5} {'w':>6} {'Brier p-only':>12} {'Brier +news':>12} "
          f"{'ΔBrier':>8} {'acc p-only':>11} {'acc +news':>10} {'news AUC':>9}")
    for r in rows:
        print(f"{r['set']:<26} {r['n']:>5} {r['weight']:>+6.2f} {r['brier_price_only']:>12.5f} "
              f"{r['brier_price_news']:>12.5f} {r['brier_delta']:>+8.5f} {r['acc_price_only']:>11.3f} "
              f"{r['acc_price_news']:>10.3f} {r['news_auc']:>9.3f}")

    lo, hi = block_bootstrap_delta_brier(val, w_star)
    helps = hi < 0                                  # Brier improvement CI entirely below 0
    print(f"\nValidate ΔBrier 95% CI (block-bootstrap over tickers): [{lo:+.5f}, {hi:+.5f}]")
    print(f"AUC of raw news tone vs outcome on validate: {rows[1]['news_auc']:.3f} "
          f"(0.50 = no rank signal)")
    if helps:
        print(f"\nVERDICT: company news tone carries out-of-sample directional signal "
              f"(w*={w_star:+.2f}). The news tilt is supported; magnitude on THIS feature is small.")
    else:
        print("\nVERDICT: no out-of-sample edge from company news tone at the 5-session horizon — "
              "the Brier-improvement CI includes zero and AUC ≈ 0.50.\n         The 0.45 company "
              "weight is not supported by this evidence; shrinking it toward 0 is the honest call\n"
              "         (the live learner's loose news prior already permits exactly that as "
              "outcomes accumulate).")

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "horizon_sessions": HORIZON,
        "tune_tickers": TUNE, "validate_tickers": VALIDATE,
        "n_rows_total": int(len(data)),
        "date_range": [str(data.date.min()), str(data.date.max())],
        "weight_star_tune": float(w_star),
        "weight_best_on_validate": float(w_star_val),
        "results": rows,
        "validate_delta_brier_ci95": [round(lo, 5), round(hi, 5)],
        "news_signal_supported": bool(helps),
        "feature": {"source": "GDELT 2.0 timelinetone", "smooth_days": SMOOTH_DAYS,
                    "std_window_days": STD_WINDOW, "tanh_scale": TANH_SCALE},
        "note": ("Validates whether company news TONE predicts 5-session direction, tune/validate "
                 "split by ticker parity. Not a drop-in live weight: GDELT worldwide tone differs "
                 "from the app's VADER-over-reliable-outlets feature. News weight in logit space."),
    }
    with open("news_backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved news_backtest_rows.csv ({len(data)} rows) and news_backtest_summary.json")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
