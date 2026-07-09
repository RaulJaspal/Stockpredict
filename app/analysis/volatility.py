"""EWMA volatility forecasting and a calibrated expected-range interval.

The 5-day price band shipped with every prediction used to be
``+/- 1.34 * sigma21 * sqrt(5)``, where sigma21 is a 21-day equal-weighted
rolling std and 1.34 a multiplier fit for 80% coverage. That is well-calibrated
*on average*, but mis-calibrated *across regimes*: an equal-weighted 21-day
window under-covers in calm markets (band too tight) and over-covers in
stressed ones, because it keeps a volatility spike in view for a full month
after it has passed.

An EWMA forecast (RiskMetrics, lambda=0.97) reacts to volatility changes
immediately. Paired with empirically-calibrated *asymmetric* quantiles of
standardized 5-day log returns, it delivers the same 80% coverage with ~6%
sharper bands overall (~10% sharper in stressed markets) and roughly half the
calibration drift from calm to stressed regimes. The asymmetry also captures
the small positive 5-day drift and the fatter left tail for free.

Validated walk-forward with no lookahead on 10y of daily data for 15 tickers,
quantiles fit by tune/validate ticker split and confirmed leave-one-ticker-out
(LOO coverage: mean 0.800, range 0.775-0.828). See ``research/vol_backtest.py``.
"""

import numpy as np

EWMA_LAMBDA = 0.97      # RiskMetrics-style decay; ~33-session effective memory
EWMA_SEED = 21          # observations used to seed the variance recursion

# Empirically-calibrated quantiles of the standardized h-day log return
#   z = log(P_{t+h} / P_t) / (sigma_daily * sqrt(h)),
# pooled across 15 tickers, 10y. The 10th/90th percentiles bound an 80% central
# interval; the asymmetry (|hi| > |lo|) encodes the positive drift, which grows
# with the horizon (a month compounds more upside than a week). Calibrated
# per-horizon — the 5-day numbers do NOT simply sqrt-scale to a month.
# See research/vol_backtest.py.
_QUANTILES = {
    5:  (-1.1372, 1.2937),    # weekly
    21: (-1.1089, 1.4462),    # monthly
}
_CAL_HORIZON = 5
# Back-compat aliases (the 5-day constants other modules/tests may import).
Z_LO_5D, Z_HI_5D = _QUANTILES[5]


def _quantiles_for(horizon_days):
    """Calibrated (lo, hi) for a horizon; falls back to sqrt-scaling the nearest
    calibrated horizon for anything not directly calibrated."""
    if horizon_days in _QUANTILES:
        return _QUANTILES[horizon_days]
    base_h = min(_QUANTILES, key=lambda h: abs(h - horizon_days))
    lo, hi = _QUANTILES[base_h]
    scale = np.sqrt(horizon_days / base_h)
    return lo * scale, hi * scale


def ewma_daily_vol(log_returns):
    """RiskMetrics EWMA daily-volatility forecast from a 1-D array of daily
    *log* returns (most recent last). Every finite return is used, so the
    estimate reflects the latest close. Returns None with too little history."""
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < EWMA_SEED:
        return None
    var = float(np.var(r[:EWMA_SEED], ddof=1))
    for x in r[EWMA_SEED:]:
        var = EWMA_LAMBDA * var + (1.0 - EWMA_LAMBDA) * x * x
    return float(np.sqrt(var)) if var > 0 else None


def expected_range(close, horizon_days=_CAL_HORIZON):
    """~80% price band over the horizon from an EWMA vol forecast and the
    calibrated fat-tailed quantiles.

    `close` is a 1-D sequence of adjusted closes (most recent last). Returns
    {low, high, pct, sigma_annual_pct} or None when history is too short.

    Quantiles are calibrated per horizon (weekly and monthly directly; other
    horizons sqrt-scale the nearest calibrated one — see _quantiles_for).
    """
    prices = np.asarray(close, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < EWMA_SEED + 2:
        return None
    spot = float(prices[-1])
    sigma_d = ewma_daily_vol(np.diff(np.log(prices)))
    if sigma_d is None:
        return None

    q_lo, q_hi = _quantiles_for(horizon_days)
    s = sigma_d * np.sqrt(horizon_days)
    low = spot * np.exp(q_lo * s)
    high = spot * np.exp(q_hi * s)
    return {
        "low": round(float(low), 2),
        "high": round(float(high), 2),
        # mean half-width as a % of spot, so the UI's "+/- X%" stays meaningful
        "pct": round(float((high - low) / 2 / spot * 100), 2),
        "sigma_annual_pct": round(float(sigma_d) * np.sqrt(252) * 100, 1),
    }
