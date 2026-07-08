"""Standard technical indicators plus a transparent composite signal read."""

import numpy as np
import pandas as pd


def sma(series, n):
    return series.rolling(n).mean()


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close, fast=12, slow=26, signal_n=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, signal_n)
    return line, signal, line - signal


def bollinger(close, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, pct_b, bandwidth


def atr(df, n=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False).mean()


def stochastic_k(df, n=14):
    low = df["Low"].rolling(n).min()
    high = df["High"].rolling(n).max()
    return 100 * (df["Close"] - low) / (high - low)


def compute_indicators(df):
    out = df.copy()
    close = out["Close"]
    out["ret1"] = close.pct_change()
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["rsi14"] = rsi(close)
    out["macd_line"], out["macd_signal"], out["macd_hist"] = macd(close)
    (out["bb_mid"], out["bb_up"], out["bb_lo"],
     out["bb_pctb"], out["bb_bw"]) = bollinger(close)
    out["atr14"] = atr(out)
    out["stoch_k"] = stochastic_k(out)
    out["vol21"] = out["ret1"].rolling(21).std()
    out["vol_ratio"] = out["Volume"].rolling(5).mean() / out["Volume"].rolling(21).mean()
    return out


def technical_signals(ind):
    """Human-readable signal breakdown and a weighted composite in [-1, 1]."""
    row = ind.iloc[-1]
    close = float(row["Close"])
    signals = []

    def add(name, score, text, weight):
        signals.append({
            "name": name,
            "score": round(float(np.clip(score, -1, 1)), 2),
            "text": text,
            "weight": weight,
        })

    if not np.isnan(row["sma50"]):
        score = 0.5 if close > row["sma50"] else -0.5
        parts = [f"price {'above' if close > row['sma50'] else 'below'} its 50-day average"]
        if not np.isnan(row["sma200"]):
            golden = row["sma50"] > row["sma200"]
            score += 0.5 if golden else -0.5
            parts.append("50-day above 200-day (long-term uptrend)" if golden
                         else "50-day below 200-day (long-term downtrend)")
        add("Trend", score, "; ".join(parts), 1.2)

    if len(ind) > 22:
        r21 = close / float(ind["Close"].iloc[-22]) - 1
        add("Momentum (1 month)", np.clip(r21 * 8, -1, 1),
            f"{r21 * 100:+.1f}% over the last 21 sessions", 1.0)

    r = row["rsi14"]
    if not np.isnan(r):
        if r <= 30:
            add("RSI (14)", 0.7, f"RSI {r:.0f} — oversold; rebounds often follow", 1.0)
        elif r >= 70:
            add("RSI (14)", -0.7, f"RSI {r:.0f} — overbought; pullback risk", 1.0)
        else:
            add("RSI (14)", (r - 50) / 50 * 0.3, f"RSI {r:.0f} — neutral zone", 1.0)

    hist = row["macd_hist"]
    if not np.isnan(hist) and len(ind) > 1:
        prev = ind["macd_hist"].iloc[-2]
        rising = not np.isnan(prev) and hist > prev
        if hist > 0:
            score, text = (0.6, "bullish and strengthening") if rising else (0.2, "bullish but fading")
        else:
            score, text = (-0.2, "bearish but improving") if rising else (-0.6, "bearish and weakening")
        add("MACD", score, f"histogram {text}", 1.0)

    pct_b = row["bb_pctb"]
    if not np.isnan(pct_b):
        if pct_b > 1:
            add("Bollinger bands", -0.5, "closed above the upper band — stretched", 0.7)
        elif pct_b < 0:
            add("Bollinger bands", 0.5, "closed below the lower band — stretched down", 0.7)
        else:
            add("Bollinger bands", 0.0, f"inside the bands (%B {pct_b:.2f})", 0.7)

    k = row["stoch_k"]
    if not np.isnan(k):
        if k <= 20:
            add("Stochastic", 0.5, f"%K {k:.0f} — oversold", 0.6)
        elif k >= 80:
            add("Stochastic", -0.5, f"%K {k:.0f} — overbought", 0.6)
        else:
            add("Stochastic", 0.0, f"%K {k:.0f} — neutral", 0.6)

    vol_ratio = row["vol_ratio"]
    if not np.isnan(vol_ratio) and not np.isnan(row["ret1"]):
        if vol_ratio >= 1.4:
            up_day = row["ret1"] > 0
            add("Volume", 0.4 if up_day else -0.4,
                f"{vol_ratio:.1f}× average volume on {'an up' if up_day else 'a down'} day "
                f"({'accumulation' if up_day else 'distribution'})", 0.6)
        else:
            add("Volume", 0.0, f"volume near its recent average ({vol_ratio:.1f}×)", 0.6)

    if not signals:
        return 0.0, signals
    total_weight = sum(s["weight"] for s in signals)
    composite = sum(s["score"] * s["weight"] for s in signals) / total_weight
    return float(np.clip(composite, -1, 1)), signals
