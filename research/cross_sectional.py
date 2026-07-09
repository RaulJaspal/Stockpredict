"""Cross-sectional signal research: is there a directional edge in RANKING
liquid large-caps against each other, where absolute-direction prediction found
none? Tests the two anomalies with real academic support at short horizons:

  * short-term reversal (STR): buy last week's losers, short its winners;
  * cross-sectional momentum (12-1): buy 12-month winners (skip last month).

Method (honest, matches the repo):
  * long-short = top quintile minus bottom quintile, equal-weight;
  * NON-OVERLAPPING holding periods (stride = hold) so weekly spreads are ~iid;
  * every signal at t uses only prices <= t; forward return is t -> t+hold;
  * report gross AND net of a round-trip cost; annualized Sharpe; hit rate;
    a moving-block bootstrap CI on the spread; and first-half/second-half
    stability (a real anomaly should persist out of its discovery sample).

Survivorship: today's liquid names, adjusted prices — a real, stated caveat.

VERDICT (2026-07, 40 names, 2016-2026): no edge that clears the bar at the
app's 5-day horizon. Short-term reversal is dead on liquid mega-caps (hit <=
0.49, CIs straddle 0). Cross-sectional 12-1 momentum is the one signal with a
consistent positive sign in BOTH halves (gross Sharpe ~0.45), but its 95% CI
still straddles zero on this sample and after realistic costs the net Sharpe is
<= 0.16 (negative at weekly rebalancing, where turnover eats it). Momentum is
strongest at the 21-day hold and is the one signal worth revisiting with a
larger universe / at a monthly horizon — see the multi-horizon roadmap item.

Run:  .venv/bin/python research/cross_sectional.py
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / ".xs_cache.pkl"

# ~40 liquid US large-caps across sectors (diversified cross-section).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "ORCL",
    "JPM", "BAC", "WFC", "GS", "V", "MA", "AXP",
    "JNJ", "LLY", "PFE", "MRK", "ABBV", "UNH",
    "XOM", "CVX", "COP",
    "WMT", "COST", "HD", "MCD", "NKE", "KO", "PEP", "PG",
    "DIS", "CAT", "BA", "GE", "HON", "IBM",
]


def load_matrix():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    frames = {}
    for t in UNIVERSE:
        df = yf.Ticker(t).history(period="10y", interval="1d", auto_adjust=True)
        if df is not None and not df.empty and len(df) > 500:
            frames[t] = df["Close"]
            print(f"  {t} {len(df)}")
        time.sleep(0.25)
    mat = pd.DataFrame(frames).sort_index()
    CACHE.write_bytes(pickle.dumps(mat))
    return mat


def block_boot_ci(x, block=1, n_boot=5000, seed=13):
    """Two-sided 95% CI for the mean of series x via moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, nb * block)[:, :n] % n
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def signal(logret, i, kind):
    """Signal value per ticker at row i, from log-price array; None if unusable."""
    if kind == "str1w":
        if i < 5: return None
        return -(logret[i] - logret[i - 5])            # reverse last-week return
    if kind == "str1m":
        if i < 21: return None
        return -(logret[i] - logret[i - 21])
    if kind == "mom12_1":
        if i < 252: return None
        return logret[i - 21] - logret[i - 252]        # 12m return skipping last month
    raise ValueError(kind)


def run(mat, kind, hold, cost_bps=10.0):
    tickers = list(mat.columns)
    dates = mat.index
    logp = {t: np.log(mat[t].to_numpy(dtype=float)) for t in tickers}
    n = len(dates)
    spreads, longs, shorts = [], [], []
    for i in range(252, n - hold, hold):                # non-overlapping, need 12m history
        sigs = {}
        for t in tickers:
            lp = logp[t]
            if not np.isfinite(lp[i]) or not np.isfinite(lp[i + hold]):
                continue
            s = signal(lp, i, kind)
            if s is not None and np.isfinite(s):
                sigs[t] = s
        if len(sigs) < 10:
            continue
        names = sorted(sigs, key=sigs.get)              # ascending signal
        q = max(1, len(names) // 5)
        bottom, top = names[:q], names[-q:]             # low signal / high signal
        fwd = {t: logp[t][i + hold] - logp[t][i] for t in names}
        long_r = np.mean([fwd[t] for t in top])
        short_r = np.mean([fwd[t] for t in bottom])
        spreads.append(long_r - short_r)
        longs.append(long_r); shorts.append(short_r)
    spreads = np.array(spreads)
    if len(spreads) < 20:
        return None
    # net of cost: full turnover both legs each rebalance -> 2 * 2 * cost per period
    net = spreads - 4 * cost_bps / 1e4
    periods_per_year = 252 / hold
    def stats(x):
        mu = x.mean(); sd = x.std(ddof=1)
        sharpe = mu / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
        return mu, sharpe
    mu_g, sh_g = stats(spreads)
    mu_n, sh_n = stats(net)
    lo, hi = block_boot_ci(spreads, block=1)
    half = len(spreads) // 2
    mu1, _ = stats(spreads[:half]); mu2, _ = stats(spreads[half:])
    return {
        "kind": kind, "hold": hold, "n_rebal": len(spreads),
        "gross_mean_pct": mu_g * 100, "gross_sharpe": sh_g,
        "net_mean_pct": mu_n * 100, "net_sharpe": sh_n,
        "hit_rate": float((spreads > 0).mean()),
        "ci95_pct": (lo * 100, hi * 100),
        "half1_pct": mu1 * 100, "half2_pct": mu2 * 100,
        "mean_long_pct": np.mean(longs) * 100, "mean_short_pct": np.mean(shorts) * 100,
    }


def main():
    print("loading universe...")
    mat = load_matrix()
    print(f"{mat.shape[1]} names, {mat.shape[0]} sessions, "
          f"{mat.index[0].date()} -> {mat.index[-1].date()}\n")
    hdr = (f"{'signal':<9} {'hold':>4} {'n':>4} {'gross%':>7} {'grSh':>5} "
           f"{'net%':>7} {'netSh':>6} {'hit':>5} {'95% CI (pp)':>16} "
           f"{'H1%':>6} {'H2%':>6}")
    print(hdr); print("-" * len(hdr))
    for kind in ("str1w", "str1m", "mom12_1"):
        for hold in (5, 21):
            r = run(mat, kind, hold)
            if not r:
                continue
            ci = f"[{r['ci95_pct'][0]:+.2f},{r['ci95_pct'][1]:+.2f}]"
            print(f"{kind:<9} {hold:>4} {r['n_rebal']:>4} {r['gross_mean_pct']:>+7.3f} "
                  f"{r['gross_sharpe']:>5.2f} {r['net_mean_pct']:>+7.3f} {r['net_sharpe']:>+6.2f} "
                  f"{r['hit_rate']:>5.2f} {ci:>16} {r['half1_pct']:>+6.2f} {r['half2_pct']:>+6.2f}")
    print("\ngross% = mean long-short spread per rebalance (before cost)")
    print("net%   = after 10bps/side round-trip cost on full turnover")
    print("H1/H2  = first-half / second-half mean spread (persistence check)")
    print("A real, tradeable edge needs: CI excluding 0, net Sharpe > ~0.4, and "
          "the SAME sign in both halves.")


if __name__ == "__main__":
    main()
