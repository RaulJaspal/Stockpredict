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
   and weighted by recency. A historical GDELT backtest (see below) found no directional
   edge from company news tone, so v2.2 gives it a small, evidence-consistent weight.
4. **Post-earnings drift (PEAD)** — the app's **one backtest-validated directional edge**.
   When a company recently reported, the sign of its earnings surprise predicts the
   following month's drift (57.6% hit-rate over ~2,960 events, 95% CI on the effect
   excludes zero — see below). It's an event-driven tilt on the **monthly** view, active
   only in the ~10 sessions after a report and fading as the drift is spent.

The result is a probability, a calibrated confidence tier, and an expected price range,
over a selectable **1-week or 1-month** horizon. The monthly view carries a stronger
drift anchor, so it is higher-accuracy in raw hit-rate (~58% vs ~57% weekly in the
backtest) — though the *edge over the drift baseline* stays ≈zero at every horizon; it is
more accurate, not more skillful. The range is the app's most *predictable* output —
direction at short horizons is near-random, but volatility is strongly autocorrelated —
so it is forecast with an EWMA (RiskMetrics λ=0.97) volatility model and
empirically-calibrated fat-tailed quantiles (calibrated per horizon), and its live
coverage is tracked in the track record (see below). Only the weekly call is logged to
the live ledger, so the track record stays a single clean test; the monthly view carries
its own on-page holdout backtest.

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

Result for the current model (direction hit-rate vs always-predict-up baseline).
The **edge 95% CI** is a moving-block bootstrap (contiguous per-ticker blocks sized
to the outcome-window overlap) — a plain binomial CI would assume the predictions are
independent, which they are not:

| horizon | n | model | always-up | edge | edge 95% CI | verdict |
|---|---|---|---|---|---|---|
| 1 day | 3,750 | 53.6% | 53.5% | +0.1pp | [−0.1, +0.3] | ≈ zero |
| 1 week | 1,500 | 56.6% | 56.7% | −0.1pp | [−0.5, +0.2] | ≈ zero |
| 1 month | 720 | 58.1% | 58.6% | −0.6pp | [−1.4, +0.3] | ≈ zero |
| 1 year* | 1,308 | 69.4% | 73.8% | −4.4pp | [−8.2, −0.8] | loses to baseline |

\* overlapping windows. The honest reading, now with intervals: at every short horizon
the edge's 95% CI **straddles zero** — the model is statistically indistinguishable from
simply knowing markets drift up. It does not beat the baseline; it matches it (the
original unshrunk model *lost* 2–10pp to it). The only edge distinguishable from zero is
the 1-year one, and it is **negative** — drift below 0.5 after a crash predicts down years
that then recover. The blend earns its keep not by beating the baseline but by adding
calibration, an expected range, per-headline news context and per-ticker honesty checks.
Confidence tiers are calibrated on the same test (pooled: ~67% hit-rate for "high" vs
~54% for "low", driven mostly by month/year drift). Caveats: tickers are today's
survivors and prices are retroactively adjusted. (Exact figures drift slightly
run-to-run because the 10-year window ends on the day you run it.)

## News validation (`news_backtest.py`, v2.2)

The news tilt used to be the app's biggest unvalidated assumption — weighted 0.45
(company) purely by judgment, because no historical headline archive was on hand.
[GDELT 2.0](https://www.gdeltproject.org/) removes that excuse: it publishes a daily
average-article-tone series for any query back to 2017. `news_backtest.py`
reconstructs a per-company tone feature (smoothed, then standardised against a
trailing 365-day window so it means "unusually good/bad news for *this* company",
no lookahead) and grades it out of sample at the 5-session horizon, tune/validate
split by ticker.

**Result across 4,145 walk-forward weekly predictions (9 companies, 2017–2026): no
directional edge from company news tone.** The Brier-minimising weight was only
+0.06 to +0.08 on a [−1, 1] feature; the validate-set Brier improvement CI is
[−0.00044, +0.00036] (straddles zero) and the tone-vs-outcome rank AUC is 0.51
(0.50 = nothing). So the 0.45 weight was ~6× larger than even the statistically-zero
fitted effect. **v2.2 cuts the news priors to small, evidence-consistent values
(company 0.45→0.15, market 0.18→0.08, politics 0.12→0.05)** — not to zero, because
the live feature (VADER over reliable-outlet headlines) is a different construction
from GDELT worldwide tone, and the online learner can still raise them if live
outcomes ever justify it. Only company news was directly tested; market and politics
are shrunk by the same logic (diffuse macro tone at 5 days is even less likely to
predict). Run: `.venv/bin/python news_backtest.py` (first run fetches + caches GDELT,
paced under its rate limit).

## Volatility / expected-range validation (`research/vol_backtest.py`)

Because short-horizon *direction* has no edge, the expected-range band is the app's most
useful output — and it is now forecast honestly. The old band was `±1.34·σ₂₁·√5` (a
21-day equal-weighted rolling std with a fudge multiplier). That is well-calibrated on
average but mis-calibrated *across regimes*: an equal-weighted window under-covers in
calm markets and over-covers in stressed ones (it keeps a passed volatility spike in view
for a full month). Replaced with an **EWMA (RiskMetrics λ=0.97) forecast + empirically-
calibrated asymmetric quantiles** of standardized 5-day log returns. Walk-forward,
tune/validate split by ticker, over 10y and 15 tickers:

| metric (validate tickers) | old `rolling21×1.34` | new `ewma97` |
|---|---|---|
| overall coverage (target 0.80) | 0.805 | 0.795 |
| overall band width | 10.35% | **9.68%** (~6% sharper) |
| width in stressed regime | 17.50% | **15.87%** (~10% sharper) |
| calm→stressed coverage drift | 0.11 | **0.06** (~2× flatter) |

Leave-one-ticker-out coverage of the frozen quantiles: mean 0.800, range 0.775–0.828 —
they generalize, not overfit. Shipped in `app/analysis/volatility.py`. The asymmetry also
captures the small positive 5-day drift for free. **Every band is now graded live**: the
ledger logs `range_low`/`range_high`, and `/api/track-record` reports the realized 80%
coverage — the honest check that the volatility model stays calibrated out of sample.

## Post-earnings drift — the one real edge (`research/pead.py`)

Everything else here matches the drift baseline but doesn't beat it. PEAD does. On
**2,957 earnings events** (30 large-caps, 2001–2026, walk-forward, enter the day *after*
the report so the announcement jump itself is never captured), the sign of the reported
EPS surprise predicts the **next month's drift**:

| metric | result |
|---|---|
| directional hit-rate | **57.6%** (50% = no edge) |
| long-short monthly drift | +1.07% gross · **+0.87% net** of 20bps |
| signed-drift 95% CI (block-bootstrap by ticker) | **[+0.85%, +1.36%] — excludes zero** |
| breadth | **28 of 30 tickers** individually > 50% |
| horizon profile | 52.5% @ 1wk → 57.6% @ 1mo → 58.7% @ 2mo (drift *accrues*) |
| persistence | +0.82% first half / +1.38% second half (same sign) |
| dose-response | worst-surprise quintile +0.35% vs best +1.98% |

This matches 40 years of academic PEAD literature and has **zero fitted parameters**
(sign of surprise → direction), so the whole sample is already out of sample. A logistic
fit of drift-direction on the standardized surprise gives the shipped tilt (logit +0.085
per σ, `config.PEAD`), applied to the monthly prediction only (it's weak at a week) and
decaying over the sessions after the report. Adding it improves the monthly Brier out of
sample. Caveats: today's surviving large-caps; adjusted prices; Yahoo earnings history.
Run: `.venv/bin/python research/pead.py` (needs `lxml` for the earnings table).

## Cross-sectional signal research (`research/cross_sectional.py`)

Absolute direction has no edge, but the robust anomalies (short-term reversal, momentum)
live in *relative* ranking, so this tests a long-short (top-minus-bottom-quintile)
backtest across 40 liquid large-caps, 2016–2026, non-overlapping holds, gross **and** net
of costs, with a block-bootstrap CI and a first-/second-half persistence check.
**Verdict: no edge that clears the bar at the 5-day horizon.** Short-term reversal is dead
on liquid mega-caps (hit-rate ≤ 0.49, CIs straddle 0). Cross-sectional 12-1 momentum is
the one signal with a consistent positive sign in both halves (gross Sharpe ~0.45), but
its 95% CI still straddles zero on this sample and its net Sharpe is ≤ 0.16 after costs
(negative at weekly rebalancing). Documented, not shipped — momentum is strongest at a
*monthly* hold and is the one signal worth revisiting with the multi-horizon roadmap.

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
- **Historical news tone:** GDELT 2.0 DOC API (`news_backtest.py`, for validation only)
- **Model:** scikit-learn logistic regression; FastAPI + Lightweight Charts (TradingView) UI

## Adaptive learning (v2.1)

The blend weights are no longer fixed: while the server runs, it logs the whole
watchlist every 6 hours and, as outcomes resolve, re-estimates the weights by MAP
logistic regression on the live results (`app/analysis/learner.py`, state in
`model_state.json`). The priors act as a Gaussian anchor whose tightness reflects
the evidence behind each weight — price weights (validated on 7,278 backtest
predictions) move slowly; the news priors were cut in v2.2 after the GDELT
backtest found no edge, but stay loose enough for live data to raise them if news
genuinely predicts, or shrink them further toward zero if it doesn't.

Safety rails, verified by the tests in `tests/`: no adaptation below 40 resolved
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
drift-alone, per-confidence-tier results, and the **realized coverage of the expected-
range band** (should sit near 80% — a live calibration check on the volatility model).
This is the one test no backtest can fake: predictions graded on data that did not exist
when they were made — and, with the GDELT news backtest now in place, a second independent
check on the news tilt.

For work-in-progress state and next steps across sessions, see **HANDOFF.md**.

## Layout

```
app/
  config.py           feeds, indices, model parameters, blend weights
  server.py           FastAPI app + static hosting
  data/market.py      prices, quotes, earnings dates, symbol search (cached)
  data/news.py        concurrent RSS ingestion + company matching (cached)
  data/gdelt.py       cached GDELT daily news-tone client (for news_backtest.py)
  analysis/technical.py   indicators + plain-English signal readings
  analysis/sentiment.py   finance-tuned VADER scoring + recency weighting
  analysis/predictor.py   ML model, backtest, evidence blend, expected range
  analysis/volatility.py  EWMA vol forecast + calibrated expected-range band
  analysis/learner.py     online MAP weight learning + shadow adoption test
  analysis/planner.py     ATR trade brackets with empirical odds + honest level edge
  static/             dashboard (index.html, style.css, app.js)
backtest.py           walk-forward price backtest + block-bootstrap edge CIs
news_backtest.py      walk-forward GDELT news-tone validation (v2.2)
research/             validated experiments: vol_backtest.py, cross_sectional.py
tests/                stdlib unittest suite — `python -m unittest discover -s tests`
.github/workflows/    CI: runs the hermetic test suite on every push/PR
```
