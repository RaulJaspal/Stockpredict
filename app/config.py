"""Central configuration: data sources, model parameters, cache lifetimes.

Every data source here is a free, official endpoint of an established provider:
Yahoo Finance for market data, and the public RSS feeds of BBC News, Sky News,
The Guardian, CNBC and MarketWatch for news.
"""

# ---------------------------------------------------------------------------
# News sources (official RSS endpoints of reliable outlets)
# ---------------------------------------------------------------------------
MACRO_FEEDS = [
    {"source": "BBC News",    "category": "business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    {"source": "BBC News",    "category": "politics", "url": "https://feeds.bbci.co.uk/news/politics/rss.xml"},
    {"source": "Sky News",    "category": "business", "url": "https://feeds.skynews.com/feeds/rss/business.xml"},
    {"source": "Sky News",    "category": "politics", "url": "https://feeds.skynews.com/feeds/rss/politics.xml"},
    {"source": "Sky News",    "category": "world",    "url": "https://feeds.skynews.com/feeds/rss/world.xml"},
    {"source": "The Guardian","category": "business", "url": "https://www.theguardian.com/uk/business/rss"},
    {"source": "CNBC",        "category": "business", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"source": "MarketWatch", "category": "business", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
]

# ---------------------------------------------------------------------------
# Market overview tiles shown on the dashboard
# ---------------------------------------------------------------------------
INDICES = [
    {"symbol": "^GSPC",   "label": "S&P 500"},
    {"symbol": "^IXIC",   "label": "Nasdaq"},
    {"symbol": "^DJI",    "label": "Dow Jones"},
    {"symbol": "^FTSE",   "label": "FTSE 100"},
    {"symbol": "GC=F",    "label": "Gold"},
    {"symbol": "CL=F",    "label": "Oil (WTI)"},
    {"symbol": "BTC-USD", "label": "Bitcoin"},
]

# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------
HORIZON_DAYS = 5          # DEFAULT horizon (weekly); the tracked/learned one
HOLDOUT_DAYS = 60         # honest out-of-sample window for the backtest
MIN_ROWS_FOR_ML = 150     # below this, fall back to drift + technicals only
HISTORY_PERIOD = "2y"

# Selectable prediction horizons. The default (weekly) is the one the live
# ledger logs and the online learner adapts on. Longer horizons are offered as
# a VIEW: they carry a stronger drift anchor (higher raw hit-rate — ~58% at a
# month vs ~57% at a week in the walk-forward backtest) and their own calibrated
# range, but are not auto-logged, so the track record stays a clean weekly test.
HORIZONS = {
    "1w": {"days": 5,  "label": "1 week"},
    "1m": {"days": 21, "label": "1 month"},
}
DEFAULT_HORIZON = "1w"

# Drift-anchored blend, validated by walk-forward backtest (see backtest.py).
# The probability is built around the ticker's historical up-rate ("drift
# anchor"); the ML model and the technical read are small tilts. A 2016-2026
# walk-forward test (7,278 predictions, 15 tickers, tuned on half the tickers
# and validated on the other half) showed larger weights only subtract value.
MODEL = {
    "shrink_ml": 0.15,    # k: logit(base) + k * (logit(p_ml) - logit(base))
    "tech_weight": 0.10,  # w: + w * technical_score
}

# News tilts (logit space). Now with evidence: news_backtest.py reconstructs a
# historical company-news tone series from GDELT (2017-2026) and grades it
# out-of-sample. Across 4,145 walk-forward weekly predictions the company-tone
# tilt showed NO directional edge (validate ΔBrier CI [-0.00044, +0.00036]
# straddles 0; AUC 0.51; Brier-minimising weight only +0.06/+0.08 on a
# [-1,1] feature). So the old judgment weights (0.45/0.18/0.12) were far too
# large. They are cut to small, evidence-consistent values — not zero, because
# the live VADER-over-reliable-outlets feature differs from GDELT worldwide tone
# and the online learner can still raise them if live outcomes justify it.
# (Only company news was directly tested; market/politics are shrunk by the same
# logic — diffuse macro tone at a 5-day horizon is even less likely to predict.)
BLEND = {
    "news_company":   0.15,   # company-specific news sentiment in [-1, 1]
    "news_market":    0.08,   # broad business-news sentiment
    "news_politics":  0.05,   # politics & world-affairs sentiment
}

# Confidence tiers, calibrated on the walk-forward backtest: historically,
# calls with |p - 0.5| >= 0.10 were right ~67% of the time, ~55% below 0.05.
CONFIDENCE = {"medium": 0.05, "high": 0.10}

# Stamped on every ledger record so results can be segmented when the model
# changes. Bump on any change to MODEL, BLEND or the feature set.
# v2.1: blend weights become adaptive — learned online from resolved ledger
# outcomes (see analysis/learner.py), anchored to these priors.
# v2.2: news priors cut to evidence-consistent values after news_backtest.py
# found no out-of-sample edge from company news tone (GDELT 2017-2026).
MODEL_VERSION = "2.2-newsval"

# Symbols scanned by the dashboard's signal screener.
SCREENER_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
    "JPM", "V", "JNJ", "LLY", "XOM", "WMT", "KO", "DIS",
    "BP.L", "HSBA.L", "^GSPC", "BTC-USD",
]

# Recency half-lives for news sentiment (hours)
HALF_LIFE_COMPANY = 48.0
HALF_LIFE_MACRO = 24.0

CACHE_TTL = {
    "history": 900,     # 15 min
    "info": 86400,      # 24 h
    "feed": 600,        # 10 min
    "search": 3600,     # 1 h
    "mini": 300,        # 5 min
}
