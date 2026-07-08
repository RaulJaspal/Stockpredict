# StockPredict — Market Signal Engine

A local web app that produces an **honest, explainable short-horizon stock prediction**,
built and weighted by walk-forward backtesting, from reliable sources only:

1. **Drift anchor** — the probability starts from the ticker's own historical 5-session
   up-rate. A 10-year walk-forward test (7,278 predictions, 15 tickers, strict
   no-lookahead — see `backtest.py`) found this drift is the strongest signal in daily
   price data, so everything else is a small tilt around it.
2. **Price tilts** — a per-ticker logistic-regression model and a composite technical
   read (trend, momentum, RSI, MACD, Bollinger, volume) nudge the anchor with
   backtest-validated weights (0.15 / 0.10; larger weights measurably subtracted
   accuracy). The most recent 60 sessions are held out to grade this exact blend
   against an always-up baseline, reported with every prediction.
3. **Live news & politics sentiment** — headlines from the official feeds of **BBC News,
   Sky News, The Guardian, CNBC and MarketWatch** (business, politics and world desks),
   plus the company's own news wire, scored with a finance-tuned VADER sentiment model
   and weighted by recency. News cannot be backtested without a historical archive, so
   it deliberately gets modest weight.

The result is a probability, a calibrated confidence tier, and an expected 5-day price
range from realised volatility.

## Quick start

```bash
./run.sh
```

Then open **http://127.0.0.1:8000**. First run creates a virtualenv and installs
dependencies (Python 3.10+ recommended; built on 3.14).

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.server:app --port 8000
```

## What you get

- **Signal screener** at the top: 20 watchlist symbols scanned concurrently, each with
  direction, probability, confidence and a one-line trade plan (buy level · sell level ·
  typical sessions to get there), filterable by All / High / Medium / Low confidence
  toggles — click any card for the full analysis
- **Trade planner** per ticker: buy at market or a limit-buy on the dip (with the
  historical fill rate), a take-profit sell order and a protective stop sized off ATR,
  each with empirically measured odds — how often that exact bracket hit the sell level
  before the stop on this ticker's own history, the median sessions (and calendar days)
  it took, and the bracket's historical expectancy (≈ zero, stated plainly: levels
  manage risk, they don't create edge)
- Search any Yahoo Finance symbol (US stocks, LSE `.L`, indices `^GSPC`, crypto `BTC-USD`…)
- Market overview: S&P 500, Nasdaq, Dow, FTSE 100, gold, oil, Bitcoin + live business and
  politics sentiment gauges
- Per ticker: prediction card (direction, probability, confidence, expected range, next
  earnings date, backtest vs always-up baseline), evidence breakdown (drift anchor, model
  tilt, technicals, news), seven technical signals in plain English, candlestick chart
  with SMA 20/50 + Bollinger bands + volume + projected range, an accessible data table,
  and tabbed Company / Market / Politics news with per-headline sentiment
- Light and dark mode; responsive down to phone width

## Backtesting (`backtest.py`)

`.venv/bin/python backtest.py` reruns the full walk-forward evaluation: 7,278
predictions across 15 symbols and four horizons (1 day / 1 week / 1 month / 1 year),
where every prediction comes from a model trained only on data that existed at that
moment. Anti-cheat checks run first: a causality audit (features at t unchanged when
all later data is deleted), a no-peek sentinel (predictions bit-identical after the
future is replaced with random noise), and a shuffled-outcome control.

Result for the current model (direction hit-rate vs always-predict-up baseline):

| horizon | n | model | always-up | edge |
|---|---|---|---|---|
| 1 day | 3,750 | 53.6% | 53.5% | +0.1pp |
| 1 week | 1,500 | 54.9% | 55.3% | −0.3pp |
| 1 month | 720 | 59.9% | 60.6% | −0.7pp |
| 1 year* | 1,308 | 69.6% | 74.1% | −4.5pp |

\* overlapping windows. The honest reading: no price-only strategy tested here beats
simply knowing markets drift up — the original unshrunk model *lost* 2–10pp to that
baseline; the current drift-anchored blend matches it while adding calibration, an
expected range, per-headline news context and per-ticker honesty checks. Confidence
tiers are calibrated on the same test (pooled: ~67% hit-rate for "high" vs ~54% for
"low", driven mostly by month/year drift). Caveats: the news tilt is untestable
historically, tickers are today's survivors, prices are retroactively adjusted.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/predict/{ticker}` | full analysis: prediction, components, signals, backtest, news |
| `GET /api/screener` | watchlist snapshots with direction, probability and confidence |
| `GET /api/track-record` | logged predictions graded against realized prices |
| `GET /api/history/{ticker}?days=260` | OHLCV + indicator overlays for charting |
| `GET /api/news/{ticker}` | company headlines with sentiment |
| `GET /api/market/overview` | index tiles + macro sentiment |
| `GET /api/search?q=` | symbol lookup |

## Honesty & limitations

Short-horizon stock movements are mostly noise; nothing can reliably predict them, and a
well-calibrated tool will often — correctly — sit near 50%. This app is built to be honest
about that: every prediction ships with its own out-of-sample backtest next to the naive
baseline, confidence is downgraded when the model isn't beating that baseline, and
sentiment reflects only the last few days of headlines, not fundamentals.

**Nothing this app produces is financial advice.** It is an educational research tool.

## Sources

- **Market data:** Yahoo Finance (via `yfinance`)
- **News:** official RSS feeds of BBC News, Sky News, The Guardian, CNBC, MarketWatch;
  per-ticker wire via Yahoo Finance news
- **Sentiment:** VADER (`vaderSentiment`) extended with a finance/politics lexicon
- **Model:** scikit-learn logistic regression; FastAPI + Lightweight Charts (TradingView) UI

## Adaptive learning (v2.1)

The blend weights are no longer fixed: while the server runs, it logs the whole
watchlist every 6 hours and, as outcomes resolve, re-estimates the weights by MAP
logistic regression on the live results (`app/analysis/learner.py`, state in
`model_state.json`). The backtest-validated priors act as a Gaussian anchor whose
tightness reflects the evidence behind each weight — price weights (validated on
7,278 backtest predictions) move slowly; news weights (never backtestable) are
loose enough for live data to raise them if news genuinely predicts, or shrink
them toward zero if it doesn't.

Safety rails, verified by synthetic tests: no adaptation below 40 resolved
outcomes; every weight bounded; and a shadow test — candidate weights are fit on
the earliest 70% of outcomes and adopted only if they don't lose to the frozen
priors on the most recent 30% (Brier score). Fail → priors restored. The model
can get better with evidence; it cannot quietly get worse. `backtest.py` always
uses the frozen priors so published backtests stay reproducible.

## Live track record

Every prediction the app makes (screener and analysis pages alike) is appended to
`predictions.jsonl` — deduped per ticker per trading day, stamped with the model
version. `GET /api/track-record` grades matured calls against the prices that
actually followed and the dashboard shows the running hit rate, Brier score vs
drift-alone, and per-confidence-tier results. This is the one test no backtest can
fake, and the only way the news tilt gets validated: predictions graded on data
that did not exist when they were made.

For work-in-progress state and next steps across sessions, see **HANDOFF.md**.

## Layout

```
app/
  config.py           feeds, indices, model parameters, blend weights
  server.py           FastAPI app + static hosting
  data/market.py      prices, quotes, earnings dates, symbol search (cached)
  data/news.py        concurrent RSS ingestion + company matching (cached)
  analysis/technical.py   indicators + plain-English signal readings
  analysis/sentiment.py   finance-tuned VADER scoring + recency weighting
  analysis/predictor.py   ML model, backtest, evidence blend, expected range
  static/             dashboard (index.html, style.css, app.js)
```
