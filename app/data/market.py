"""Market data via Yahoo Finance (yfinance) with light caching."""

import bisect
import datetime as _dt

import pandas as pd
import requests
import yfinance as yf

from ..config import CACHE_TTL, HISTORY_PERIOD
from .cache import cached

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def get_history(ticker, period=HISTORY_PERIOD):
    """Daily OHLCV history as a DataFrame. Raises LookupError for unknown tickers."""
    key = ("hist", ticker.upper(), period)

    def fetch():
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df if not df.empty else None

    df = cached(key, CACHE_TTL["history"], fetch)
    if df is None:
        raise LookupError(f"No price data found for '{ticker}'. Check the ticker symbol.")
    return df


def get_info(ticker):
    def fetch():
        try:
            return yf.Ticker(ticker).get_info() or {}
        except Exception:
            return {}

    return cached(("info", ticker.upper()), CACHE_TTL["info"], fetch) or {}


def get_quote(ticker):
    df = get_history(ticker)
    info = get_info(ticker)
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName") or ticker.upper(),
        "currency": info.get("currency") or "",
        "exchange": info.get("fullExchangeName") or info.get("exchange") or "",
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "price": round(last, 4),
        "change": round(last - prev, 4),
        "change_pct": round((last / prev - 1) * 100, 3) if prev else 0.0,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
    }


def get_mini_quote(symbol):
    """Lightweight quote (price + day change) for the market-overview strip."""
    def fetch():
        df = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        return {
            "symbol": symbol,
            "price": round(last, 2),
            "change_pct": round((last / prev - 1) * 100, 2) if prev else 0.0,
        }

    quote = cached(("mini", symbol), CACHE_TTL["mini"], fetch)
    if quote is None:
        raise LookupError(symbol)
    return quote


def get_next_earnings(ticker):
    """Next scheduled earnings date (ISO string) or None — an event-risk flag."""
    def fetch():
        try:
            cal = yf.Ticker(ticker).calendar
        except Exception:
            return ""
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return ""
        try:
            today = _dt.date.today()
            future = sorted(d for d in dates if isinstance(d, _dt.date) and d >= today)
            pick = future[0] if future else dates[0]
            return pick.isoformat()
        except Exception:
            return ""

    return cached(("earnings", ticker.upper()), CACHE_TTL["info"], fetch) or None


def get_recent_earnings(ticker):
    """Most recent PAST earnings report with its EPS surprise and how many
    trading sessions ago it landed — the input to the PEAD tilt (predictor).
    Returns {date, surprise_pct, sessions_ago} or None. Cached (earnings history
    changes only quarterly). Needs lxml (yfinance parses the earnings table)."""
    def fetch():
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=8)
        except Exception:
            return None                          # transient/parse failure -> retry later
        if ed is None or ed.empty:
            return {}                            # cache "checked, none" (don't retry)
        try:
            idx_dates = list(get_history(ticker).index.strftime("%Y-%m-%d"))
        except LookupError:
            return {}
        if not idx_dates:
            return {}
        last_date = idx_dates[-1]
        best = None
        for edate, row in ed.iterrows():
            ds = edate.strftime("%Y-%m-%d")
            if ds > last_date:                   # future / not-yet-reported (ISO strings sort chronologically)
                continue
            surprise = row.get("Surprise(%)")
            if pd.isna(surprise):
                continue
            pos = bisect.bisect_left(idx_dates, ds)   # first trading day >= report date
            if pos >= len(idx_dates):
                continue
            sessions_ago = len(idx_dates) - 1 - pos
            if best is None or sessions_ago < best["sessions_ago"]:
                best = {"date": ds, "surprise_pct": round(float(surprise), 2),
                        "sessions_ago": int(sessions_ago)}
        return best or {}

    result = cached(("recent_earn", ticker.upper()), CACHE_TTL["info"], fetch)
    return result or None


def search_symbols(query):
    """Ticker lookup via Yahoo Finance's public search endpoint."""
    def fetch():
        quotes = []
        try:
            r = requests.get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": 8, "newsCount": 0},
                headers={"User-Agent": UA},
                timeout=8,
            )
            r.raise_for_status()
            quotes = r.json().get("quotes", [])
        except Exception:
            try:
                quotes = yf.Search(query, max_results=8).quotes
            except Exception:
                return []
        results = []
        for q in quotes:
            if not q.get("symbol"):
                continue
            results.append({
                "symbol": q["symbol"],
                "name": q.get("shortname") or q.get("longname") or "",
                "exchange": q.get("exchDisp") or q.get("exchange") or "",
                "type": q.get("quoteTypeDisp") or q.get("quoteType") or "",
            })
        return results

    return cached(("search", query.strip().lower()), CACHE_TTL["search"], fetch) or []
