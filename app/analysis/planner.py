"""Trade planner: entry/exit levels with empirically measured odds and timing.

Levels are ATR-based (ATR14 = the ticker's typical daily range):
  buy-the-dip limit   = close − 0.6·ATR
  take-profit target  = entry + 1.5·ATR
  protective stop     = entry − 1.0·ATR

The odds are NOT theory: for every day in the ticker's own history we simulate
that exact bracket forward (using intraday High/Low) and report how often the
target was touched before the stop, how often the stop hit first, how often
neither did within 20 sessions, and the median sessions-to-target when it got
there. When target and stop are both touched in the same session the stop is
counted first (conservative).

These are volatility mechanics, not an edge — a 1.5R/1R bracket on a driftless
walk hits the target first ~40% of the time by construction. The numbers exist
so a user can see what a given order placement has actually done on this
ticker, and roughly how long it takes. Not financial advice.
"""

import numpy as np

SCAN_SESSIONS = 20      # how far forward each simulated bracket looks
FILL_SESSIONS = 5       # window for the limit-buy fill statistic
PULLBACK_ATR = 0.6
TARGET_ATR = 1.5
STOP_ATR = 1.0
HOLDER_TARGET_ATR = 1.0  # sell-into-strength level for the holder framing
MIN_SAMPLES = 60


def _bracket_stats(ind, target_mult, stop_mult, scan=SCAN_SESSIONS):
    close = ind["Close"].to_numpy(dtype=float)
    high = ind["High"].to_numpy(dtype=float)
    low = ind["Low"].to_numpy(dtype=float)
    atr = ind["atr14"].to_numpy(dtype=float)
    n = len(close)
    target_first = stop_first = neither = 0
    times = []
    for i in range(n - scan - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        target = close[i] + target_mult * atr[i]
        stop = close[i] - stop_mult * atr[i]
        outcome = 0
        for j in range(1, scan + 1):
            hit_stop = low[i + j] <= stop
            hit_target = high[i + j] >= target
            if hit_stop:                    # same-session double hit -> stop (conservative)
                outcome = -1
                break
            if hit_target:
                outcome = 1
                times.append(j)
                break
        if outcome == 1:
            target_first += 1
        elif outcome == -1:
            stop_first += 1
        else:
            neither += 1
    total = target_first + stop_first + neither
    if total < MIN_SAMPLES:
        return None
    within_10 = sum(1 for t in times if t <= 10)
    return {
        "sample_n": int(total),
        "target_first_pct": round(100 * target_first / total, 1),
        "stop_first_pct": round(100 * stop_first / total, 1),
        "neither_pct": round(100 * neither / total, 1),
        "median_sessions_to_target": int(np.median(times)) if times else None,
        "target_within_10_pct": round(100 * within_10 / total, 1),
    }


def _fill_stats(ind, pullback_mult=PULLBACK_ATR, within=FILL_SESSIONS):
    """How often a limit-buy 0.6·ATR below the close filled within 5 sessions."""
    close = ind["Close"].to_numpy(dtype=float)
    low = ind["Low"].to_numpy(dtype=float)
    atr = ind["atr14"].to_numpy(dtype=float)
    n = len(close)
    filled = total = 0
    for i in range(n - within - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        level = close[i] - pullback_mult * atr[i]
        total += 1
        if low[i + 1:i + 1 + within].min() <= level:
            filled += 1
    return round(100 * filled / total, 1) if total >= MIN_SAMPLES else None


def _sessions_to_days(sessions):
    """Trading sessions -> rough calendar days (5 sessions/week)."""
    return int(round(sessions * 7 / 5)) if sessions else None


def trade_plan(ind, p_up, expected_range):
    """Statistical order-placement scenario for the current price. Returns None
    when there is too little history to measure the bracket honestly."""
    price = float(ind["Close"].iloc[-1])
    atr = float(ind["atr14"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None

    mode = "buy" if p_up >= 0.5 else "hold"
    target_mult = TARGET_ATR if mode == "buy" else HOLDER_TARGET_ATR
    bracket = _bracket_stats(ind, target_mult, STOP_ATR)
    if bracket is None:
        return None

    median = bracket["median_sessions_to_target"]
    plan = {
        "mode": mode,
        "price": round(price, 2),
        "atr": round(atr, 2),
        "entry_market": round(price, 2),
        "entry_pullback": round(price - PULLBACK_ATR * atr, 2),
        "entry_fill_pct": _fill_stats(ind),
        "target": round(price + target_mult * atr, 2),
        "stop": round(price - STOP_ATR * atr, 2),
        "stretch_target": (expected_range or {}).get("high"),
        "risk_pct": round(100 * STOP_ATR * atr / price, 2),
        "reward_pct": round(100 * target_mult * atr / price, 2),
        "risk_reward": round(target_mult / STOP_ATR, 2),
        "median_sessions_to_target": median,
        "median_calendar_days": _sessions_to_days(median),
        **{k: bracket[k] for k in ("sample_n", "target_first_pct", "stop_first_pct",
                                    "neither_pct", "target_within_10_pct")},
        "note": ("Volatility mechanics measured on this ticker's own history — not an edge, "
                 "and not financial advice. Odds assume fills exactly at the levels."),
    }
    return plan
