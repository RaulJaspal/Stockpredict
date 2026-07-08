"""FastAPI server: JSON API + the dashboard front-end.

Run with:  uvicorn app.server:app --port 8000
"""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analysis import learner, predictor, sentiment
from .config import HALF_LIFE_MACRO, INDICES, SCREENER_TICKERS
from .data import ledger, market, news
from .data.cache import cached

LEARN_INTERVAL_S = 6 * 3600

app = FastAPI(title="StockPredict", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _clean(obj):
    """Replace NaN/inf with null so every response is strictly valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@app.get("/api/predict/{ticker}")
def predict(ticker: str):
    try:
        return _clean(predictor.analyze(ticker.strip()))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/history/{ticker}")
def history(ticker: str, days: int = Query(260, ge=30, le=500)):
    try:
        return _clean(predictor.history_payload(ticker.strip(), days))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/news/{ticker}")
def ticker_news(ticker: str):
    try:
        quote = market.get_quote(ticker.strip())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    articles = sentiment.annotate(news.get_ticker_news(quote["ticker"], quote["name"]))
    return _clean({"ticker": quote["ticker"], "articles": articles})


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=40)):
    return {"results": market.search_symbols(q)}


@app.get("/api/screener")
def screener():
    """Signal snapshots for the watchlist, computed concurrently and cached."""
    def build():
        def safe(ticker):
            try:
                return predictor.snapshot(ticker)
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            cards = [c for c in pool.map(safe, SCREENER_TICKERS) if c]
        return {"generated_at": int(time.time()), "cards": cards}

    return _clean(cached(("screener",), 600, build))


@app.get("/api/track-record")
def track_record():
    """Grade every logged prediction whose horizon has matured against the
    price that actually followed. Outcomes come from a single consistently
    adjusted price frame per ticker, never from the logged price."""
    def build():
        records = ledger.read_all()
        if not records:
            return {"n_logged": 0, "n_resolved": 0, "n_pending": 0, "stats": None,
                    "resolved": [], "since": None, "learning": learner.state_summary()}
        graded = ledger.resolve_records(records)
        pending = graded["pending"]
        resolved = [{
            "ticker": r["ticker"],
            "as_of": r["as_of"],
            "resolved_on": r["resolved_on"],
            "direction": r["direction"],
            "p_up": r["p_up"],
            "confidence": r.get("confidence"),
            "base": r.get("base"),
            "realized_pct": r["realized_pct"],
            "correct": r["correct"],
            "outcome_up": r["outcome_up"],
        } for r in graded["resolved"]]
        stats = None
        if resolved:
            n = len(resolved)
            hit = sum(r["correct"] for r in resolved) / n
            always_up = sum(r["outcome_up"] for r in resolved) / n
            brier = sum((r["p_up"] - r["outcome_up"]) ** 2 for r in resolved) / n
            with_base = [r for r in resolved if r.get("base") is not None]
            brier_drift = (sum((r["base"] - r["outcome_up"]) ** 2 for r in with_base) / len(with_base)
                           if with_base else None)
            by_conf = {}
            for tier in ("high", "medium", "low"):
                sub = [r for r in resolved if r.get("confidence") == tier]
                if sub:
                    by_conf[tier] = {"n": len(sub),
                                     "hit_rate": round(sum(r["correct"] for r in sub) / len(sub), 3)}
            stats = {"hit_rate": round(hit, 3), "always_up": round(always_up, 3),
                     "brier": round(brier, 4),
                     "brier_drift": round(brier_drift, 4) if brier_drift is not None else None,
                     "by_confidence": by_conf}
        resolved.sort(key=lambda r: r["resolved_on"], reverse=True)
        return {
            "n_logged": len(records),
            "n_resolved": len(resolved),
            "n_pending": pending,
            "since": min(r["as_of"] for r in records),
            "stats": stats,
            "resolved": resolved[:30],
            "learning": learner.state_summary(),
        }

    return _clean(cached(("track",), 600, build))


@app.get("/api/market/overview")
def market_overview():
    tiles = []
    for index in INDICES:
        try:
            tiles.append({**market.get_mini_quote(index["symbol"]), "label": index["label"]})
        except Exception:
            continue
    macro = sentiment.annotate(news.get_macro_news())
    business = [a for a in macro if a["category"] == "business"]
    politics = [a for a in macro if a["category"] in ("politics", "world")]
    return _clean({
        "indices": tiles,
        "sentiment": {
            "market": sentiment.aggregate(business, HALF_LIFE_MACRO),
            "politics": sentiment.aggregate(politics, HALF_LIFE_MACRO),
        },
        "headlines": macro[:14],
    })


def _learning_loop():
    """Autonomous loop while the server runs: log fresh predictions for the
    whole watchlist, resolve matured ones, and update the adaptive weights.
    The model keeps learning even when nobody opens the page."""
    time.sleep(90)  # let the server warm up first
    while True:
        try:
            for ticker in SCREENER_TICKERS:
                try:
                    predictor.snapshot(ticker)   # logs to the ledger (deduped per day)
                except Exception:
                    continue
            summary = learner.update_from_ledger()
            print(f"[learn] {time.strftime('%H:%M')} weights={summary['weights']} "
                  f"source={summary['source']} n={summary['n_used']}", flush=True)
        except Exception as exc:
            print(f"[learn] cycle failed: {exc}", flush=True)
        time.sleep(LEARN_INTERVAL_S)


@app.on_event("startup")
def _start_learning():
    threading.Thread(target=_learning_loop, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000)
