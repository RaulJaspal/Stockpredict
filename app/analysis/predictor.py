"""The prediction engine.

Pipeline for one ticker (validated by the walk-forward harness in backtest.py):

  1. Two years of daily prices -> technical indicators (technical.py).
  2. The probability is anchored on the ticker's own historical up-rate over
     the horizon (the "drift anchor") — the strongest signal the backtest
     found in price data.
  3. A logistic-regression model trained on that ticker's history, and the
     composite technical read, tilt the anchor by small validated weights
     (config.MODEL). The most recent 60 sessions are held out to grade this
     exact blended output against an always-up baseline, and that hit-rate
     is reported with every prediction.
  4. Live news sentiment (company, business, politics/world) adds modest
     tilts in logit space (config.BLEND) — modest because news cannot be
     backtested without a historical headline archive.
  5. Recent volatility gives an expected 5-day price range (~80% band).

This is a statistical estimate of short-horizon odds, not a crystal ball;
every response carries a disclaimer and its own backtest numbers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (BLEND, CONFIDENCE, DEFAULT_HORIZON, HALF_LIFE_COMPANY,
                      HALF_LIFE_MACRO, HOLDOUT_DAYS, HORIZON_DAYS, HORIZONS,
                      MIN_ROWS_FOR_ML, MODEL, MODEL_VERSION)
from ..data import ledger, market, news
from . import learner, planner, sentiment, technical, volatility

FEATURES = [
    "ret1", "ret5", "ret10", "ret21", "rsi14", "macd_hist_n",
    "dist_sma20", "dist_sma50", "bb_pctb", "bb_bw",
    "vol21", "vol_ratio", "stoch_k", "atr_n",
]

DISCLAIMER = ("Statistical estimate for research purposes only — not financial advice. "
              "Markets are inherently uncertain and past patterns may not repeat.")


def _logit(p):
    p = float(np.clip(p, 0.03, 0.97))
    return float(np.log(p / (1 - p)))


def _sigmoid(x):
    return float(1 / (1 + np.exp(-x)))


def _feature_frame(ind):
    close = ind["Close"]
    f = pd.DataFrame(index=ind.index)
    f["ret1"] = close.pct_change()
    f["ret5"] = close.pct_change(5)
    f["ret10"] = close.pct_change(10)
    f["ret21"] = close.pct_change(21)
    f["rsi14"] = ind["rsi14"] / 100.0
    f["macd_hist_n"] = ind["macd_hist"] / close
    f["dist_sma20"] = close / ind["sma20"] - 1
    f["dist_sma50"] = close / ind["sma50"] - 1
    f["bb_pctb"] = ind["bb_pctb"]
    f["bb_bw"] = ind["bb_bw"]
    f["vol21"] = ind["vol21"]
    f["vol_ratio"] = ind["vol_ratio"]
    f["stoch_k"] = ind["stoch_k"] / 100.0
    f["atr_n"] = ind["atr14"] / close
    return f[FEATURES]


def _base_rates(close_values, h):
    """base_at[i] = share of samples j (j + h <= i) whose h-session outcome was
    up — the drift anchor, using only information resolved by row i."""
    n = len(close_values)
    base_at = np.full(n, np.nan)
    if n <= h:
        return base_at
    up = (close_values[h:] > close_values[:-h]).astype(float)
    csum = np.cumsum(up)
    for i in range(h + 29, n):          # require >= 30 resolved samples
        m = i - h + 1
        base_at[i] = csum[m - 1] / m
    return base_at


def _blend_price(p_ml, base, tech_score, weights=None):
    """The validated price-only probability: drift anchor + small tilts.
    `weights` defaults to the learner's current (adaptive) weights; the
    backtest harness passes the frozen priors for reproducibility."""
    w = weights or learner.current_weights()
    x = _logit(base)
    if p_ml is not None:
        x += w["k_ml"] * (_logit(p_ml) - _logit(base))
    x += w["tech"] * float(tech_score)
    return _sigmoid(x)


def _price_model(ind, h=HORIZON_DAYS):
    """Drift anchor + ML tilt + an honest holdout of the blended output, for an
    h-session horizon.

    Returns {base, p_ml, holdout} — p_ml/holdout are None when there is not
    enough clean feature history (the app then runs drift + technicals only).
    """
    close_v = ind["Close"].to_numpy(dtype=float)
    n = len(close_v)
    base_at = _base_rates(close_v, h)
    base = float(base_at[-1]) if np.isfinite(base_at[-1]) else 0.5
    out = {"base": round(base, 3), "p_ml": None, "holdout": None}

    features = _feature_frame(ind)
    X = features.to_numpy(dtype=float)
    finite = ~np.isnan(X).any(axis=1)
    labeled = [i for i in range(n - h) if finite[i]]
    if len(labeled) < MIN_ROWS_FOR_ML + 20 or not finite[n - 1]:
        return out

    y = (close_v[np.array(labeled) + h] > close_v[np.array(labeled)]).astype(int)
    # Scale the holdout with the horizon so longer horizons still get a usable
    # number of (near-)independent windows (~12 * h sessions targets ~12 of
    # them), capped at a third of the data so enough training rows remain.
    holdout_n = min(max(HOLDOUT_DAYS, 12 * h), len(labeled) // 3)
    train_pos, test_pos = labeled[:-holdout_n], labeled[-holdout_n:]
    y_train, y_test = y[:-holdout_n], y[-holdout_n:]

    def fit(rows, targets):
        if targets.min() == targets.max():
            return None                              # degenerate: fall back to drift
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        model.fit(X[rows], targets)
        return model

    hold_model = fit(train_pos, y_train)

    # Grade the exact blended output (anchor + ML tilt + technical tilt) on the
    # held-out sessions, against the always-up baseline. NOTE the windows overlap
    # (consecutive days, each a 5-session outcome), so the *effective* number of
    # independent tests is ~ holdout_days / horizon, not holdout_days — the
    # comparison to baseline is treated with that much noise (see _confidence).
    correct = []
    for pos, actual in zip(test_pos, y_test):
        p_ml_i = float(hold_model.predict_proba(X[pos:pos + 1])[0, 1]) if hold_model else None
        base_i = base_at[pos] if np.isfinite(base_at[pos]) else 0.5
        tech_i, _ = technical.technical_signals(ind.iloc[:pos + 1])
        p_i = _blend_price(p_ml_i, base_i, tech_i)
        correct.append(int((p_i >= 0.5) == bool(actual)))
    out["holdout"] = {
        "hit_rate": round(float(np.mean(correct)), 3),
        "baseline": round(float(np.mean(y_test)), 3),   # always-up accuracy
        "holdout_days": int(holdout_n),
        "effective_n": max(1, int(holdout_n // h)),      # non-overlapping equiv.
        "train_rows": int(len(train_pos)),
    }

    live_model = fit(labeled, y)
    if live_model is not None:
        out["p_ml"] = round(float(live_model.predict_proba(X[n - 1:n])[0, 1]), 4)
    return out


def _expected_range(ind, h=HORIZON_DAYS):
    """~80% price band over the h-session horizon from an EWMA volatility
    forecast and empirically-calibrated fat-tailed quantiles (volatility.py).
    Replaces the old 21-day-rolling × 1.34 heuristic: same 80% coverage, but
    sharper and far better calibrated across calm/stressed regimes (validated
    walk-forward, LOO coverage 0.800; see research/vol_backtest.py)."""
    return volatility.expected_range(ind["Close"].to_numpy(dtype=float), h)


def _confidence(p_up, holdout):
    edge = abs(p_up - 0.5)
    level = ("high" if edge >= CONFIDENCE["high"]
             else "medium" if edge >= CONFIDENCE["medium"] else "low")
    caveat = None
    # Only downgrade when the holdout MEANINGFULLY loses to always-up — i.e. the
    # shortfall exceeds one standard error of a hit-rate at the *effective*
    # (non-overlapping) sample size. Without this, the plain hit_rate < baseline
    # test fires on pure noise roughly half the time, because the 60 overlapping
    # 5-day windows carry only ~12 independent samples.
    if holdout:
        eff_n = holdout.get("effective_n") or max(1, holdout["holdout_days"] // HORIZON_DAYS)
        se = (0.25 / eff_n) ** 0.5
        if holdout["hit_rate"] < holdout["baseline"] - se:
            level = "low"
            caveat = ("On the last {} sessions this blend clearly trailed the always-up "
                      "baseline for this ticker (beyond sampling noise) — treat the call "
                      "with extra caution.").format(holdout["holdout_days"])
    return level, caveat


def _assess(ticker, h=HORIZON_DAYS):
    """Shared core: everything needed for both the full analysis and a
    screener snapshot, computed once for an h-session horizon."""
    df = market.get_history(ticker)
    quote = market.get_quote(ticker)
    ind = technical.compute_indicators(df)
    tech_score, signals = technical.technical_signals(ind)
    pm = _price_model(ind, h)

    company_news = sentiment.annotate(news.get_ticker_news(ticker, quote["name"]))
    macro_news = sentiment.annotate(news.get_macro_news())
    business = [a for a in macro_news if a["category"] == "business"]
    politics = [a for a in macro_news if a["category"] in ("politics", "world")]

    s_company = sentiment.aggregate(company_news, HALF_LIFE_COMPANY)
    s_market = sentiment.aggregate(business, HALF_LIFE_MACRO)
    s_politics = sentiment.aggregate(politics, HALF_LIFE_MACRO)

    W = learner.current_weights()
    p_price = _blend_price(pm["p_ml"], pm["base"], tech_score, W)
    logit = _logit(p_price)
    logit += W["news_company"] * s_company["score"]
    logit += W["news_market"] * s_market["score"]
    logit += W["news_politics"] * s_politics["score"]
    p_up = float(np.clip(_sigmoid(logit), 0.05, 0.95))

    confidence, caveat = _confidence(p_up, pm["holdout"])
    expected_range = _expected_range(ind, h)

    # Log to the live ledger (deduped per ticker/day) so /api/track-record can
    # grade this prediction against reality once the horizon matures — both the
    # direction call AND the expected-range band. Only the DEFAULT horizon is
    # logged, so the live track record stays a single clean weekly test; longer
    # horizons are a view (they carry their own on-page holdout backtest).
    if h == HORIZON_DAYS:
        try:
            ledger.record({
                "model_version": MODEL_VERSION,
                "ticker": quote["ticker"],
                "as_of": quote["as_of"],
                "horizon_days": h,
                "price": quote["price"],
                "p_up": round(p_up, 4),
                "direction": "up" if p_up >= 0.5 else "down",
                "confidence": confidence,
                "base": pm["base"],
                "p_ml": pm["p_ml"],
                "tech": round(tech_score, 3),
                "news_company": s_company["score"],
                "news_market": s_market["score"],
                "news_politics": s_politics["score"],
                "range_low": expected_range["low"] if expected_range else None,
                "range_high": expected_range["high"] if expected_range else None,
                "weights": {k: round(v, 3) for k, v in W.items()},
            })
        except OSError:
            pass  # a read-only disk must never break predictions

    return {
        "quote": quote,
        "ind": ind,
        "signals": signals,
        "tech_score": tech_score,
        "pm": pm,
        "p_up": p_up,
        "horizon_days": h,
        "confidence": confidence,
        "caveat": caveat,
        "expected_range": expected_range,
        "trade_plan": planner.trade_plan(ind, p_up, expected_range),
        "news_lists": {"company": company_news, "market": business, "politics": politics},
        "sentiments": {"company": s_company, "market": s_market, "politics": s_politics},
    }


def _prediction_payload(a):
    return {
        "horizon_days": a["horizon_days"],
        "direction": "up" if a["p_up"] >= 0.5 else "down",
        "prob_up": round(a["p_up"], 3),
        "confidence": a["confidence"],
        "caveat": a["caveat"],
        "expected_range": a["expected_range"],
        "as_of": a["quote"]["as_of"],
    }


def _components_payload(a):
    pm = a["pm"]
    return {
        "drift": {"base": pm["base"],
                  "note": f"historical {a['horizon_days']}-session up-rate (the anchor)"},
        "ml_model": ({"prob_up": pm["p_ml"], "tilt": round(pm["p_ml"] - pm["base"], 3)}
                     if pm["p_ml"] is not None else None),
        "technical": {"score": round(a["tech_score"], 3),
                      "label": sentiment.label(a["tech_score"])},
        "news_company": a["sentiments"]["company"],
        "news_market": a["sentiments"]["market"],
        "news_politics": a["sentiments"]["politics"],
        "weights": {**learner.current_weights(),
                    "learning": learner.state_summary(),
                    "note": "k_ml/tech tilt the drift anchor; news tilts the result in logit space; "
                            "weights adapt online from resolved live outcomes, anchored to backtested priors"},
    }


def analyze(ticker, horizon=DEFAULT_HORIZON):
    """Full payload for the analysis page, for the chosen horizon key
    ('1w'/'1m'). Unknown keys fall back to the default."""
    h = HORIZONS.get(horizon, HORIZONS[DEFAULT_HORIZON])["days"]
    a = _assess(ticker, h)
    holdout = a["pm"]["holdout"]
    return {
        "quote": a["quote"],
        "horizon": {"key": horizon if horizon in HORIZONS else DEFAULT_HORIZON,
                    "days": h,
                    "options": [{"key": k, "label": v["label"]} for k, v in HORIZONS.items()]},
        "prediction": _prediction_payload(a),
        "components": _components_payload(a),
        "trade_plan": a["trade_plan"],
        "signals": a["signals"],
        "backtest": ({
            **holdout,
            "note": ("Hit-rate of this exact blend on the most recent {} sessions it never "
                     "trained on, vs. always predicting up. The {}-day windows overlap, so "
                     "this is only ~{} independent tests — indicative, not decisive."
                     ).format(holdout["holdout_days"], h, holdout["effective_n"]),
        } if holdout else None),
        "news": {k: v[:12] for k, v in a["news_lists"].items()},
        "earnings_date": market.get_next_earnings(ticker),
        "disclaimer": DISCLAIMER,
    }


def snapshot(ticker):
    """Compact card for the signal screener — same numbers as the full page."""
    a = _assess(ticker)
    q = a["quote"]
    return {
        "ticker": q["ticker"],
        "name": q["name"],
        "price": q["price"],
        "currency": q["currency"],
        "change_pct": q["change_pct"],
        "direction": "up" if a["p_up"] >= 0.5 else "down",
        "prob_up": round(a["p_up"], 3),
        "confidence": a["confidence"],
        "caveat_flag": bool(a["caveat"]),
        "drift": a["pm"]["base"],
        "tech_score": round(a["tech_score"], 3),
        "news_score": a["sentiments"]["company"]["score"],
        "plan": ({
            "mode": a["trade_plan"]["mode"],
            "buy_at": a["trade_plan"]["entry_pullback"],
            "sell_at": a["trade_plan"]["target"],
            "stop_at": a["trade_plan"]["stop"],
            "sessions": a["trade_plan"]["median_sessions_to_target"],
            "sell_odds_pct": a["trade_plan"]["target_first_pct"],
        } if a["trade_plan"] else None),
    }


def history_payload(ticker, days=260):
    """OHLCV + overlays for charting, most recent `days` sessions."""
    df = market.get_history(ticker)
    ind = technical.compute_indicators(df).iloc[-days:]

    def num(value):
        value = float(value)
        return None if np.isnan(value) or np.isinf(value) else round(value, 4)

    candles = []
    for ts, row in ind.iterrows():
        candles.append({
            "t": ts.strftime("%Y-%m-%d"),
            "o": num(row["Open"]), "h": num(row["High"]),
            "l": num(row["Low"]), "c": num(row["Close"]),
            "v": num(row["Volume"]),
            "sma20": num(row["sma20"]), "sma50": num(row["sma50"]),
            "bb_up": num(row["bb_up"]), "bb_lo": num(row["bb_lo"]),
            "rsi": num(row["rsi14"]),
        })
    return {"ticker": ticker.upper(), "candles": candles}
