# HANDOFF — session state & next steps

*Last updated: 2026-07-06 evening. Read this first when resuming work.*

## What this project is

StockPredict: a local FastAPI web app (`http://127.0.0.1:8000`, start with `./run.sh`)
that predicts 5-session stock direction from a **drift anchor** (the ticker's
historical up-rate) plus small validated tilts (per-ticker logistic regression,
technicals) and live news sentiment (BBC/Sky/Guardian/CNBC/MarketWatch + Yahoo
company wire). Everything is graded honestly: per-ticker holdout vs always-up
baseline in the UI, a full walk-forward harness in `backtest.py`, and a live
prediction ledger (`predictions.jsonl` → `/api/track-record`).

## State right now

- **Model version `2.1-adaptive`** (see `app/config.py: MODEL_VERSION`).
  Blend: `logit(base) + k_ml·(logit(p_ml) − logit(base)) + w_tech·tech + w_c·news_co + w_m·news_mkt + w_p·news_pol`, clipped [0.05, 0.95].
  Weights are now **adaptive** (`app/analysis/learner.py`, state in
  `model_state.json`): MAP logistic regression on resolved ledger outcomes,
  Gaussian-anchored to the backtested priors (0.15/0.10/0.45/0.18/0.12).
  Prior widths encode evidence: k_ml/tech tight (σ 0.08, backtest-validated),
  news loose (σ 0.5/0.35/0.3, never backtestable — live data can raise OR
  zero them). Rails: 40-outcome gate; per-weight bounds; shadow adoption test
  (candidate fit on earliest 70% of outcomes must not lose to priors on the
  latest 30%, Brier, margin 0.0005 — else priors restored). Synthetic tests
  (scratchpad test_learner.py, recreate if needed): gate holds; planted
  news signal → weight kept high vs zeroed without signal; noisy candidate
  correctly rejected. NOTE: binary 5-day outcomes are information-poor —
  expect months of logging before news weights move materially; that is by
  design, not a bug. The server runs a learning cycle 90 s after startup and
  every 6 h (logs `[learn]` lines): snapshots the watchlist (logs predictions)
  then updates weights. `backtest.py` passes frozen PRIOR_WEIGHTS so published
  backtests stay reproducible regardless of live-learned state.
- Expected-range multiplier is **1.34** (not Gaussian 1.28) — fit for 80% coverage
  on tune tickers, verified 80.0% on held-out tickers.
- Confidence tiers: |p−0.5| ≥ 0.10 high, ≥ 0.05 medium, else low; capped to low
  when the ticker's 60-session holdout loses to always-up.
- UI: signal screener (20 symbols, All/High/Medium/Low toggles, per-card plan
  line) + live track record section + full per-ticker analysis page with a
  **Trade planner card** (`app/analysis/planner.py`): ATR-sized buy/limit-buy/
  take-profit/stop levels with empirical bracket odds (target-first %, stop-first %,
  median sessions-to-target, fill rate of the dip order) simulated on the
  ticker's own 2y of High/Low data; same-session double-hits count as stop
  (conservative). Verified with Playwright (both themes, mobile, zero console
  errors).
- **Ledger**: 20 predictions logged 2026-07-06 (the screener watchlist), all
  pending. First outcomes resolve ~2026-07-13. Resolution path was tested with
  a synthetic aged record and works (grades from a consistently-adjusted price
  frame, never the logged price).

## Evidence so far (all in `backtest_results.csv` / `backtest_summary.json`)

Walk-forward, 7,278 predictions, 15 tickers, 2016–2026, strict no-lookahead
(causality audit + no-peek noise sentinel + shuffled-outcome control all PASS).
Edge = direction accuracy minus always-up baseline. The edge now ships with a
**moving-block bootstrap 95% CI** (per-ticker blocks sized to the outcome-window
overlap = ceil(h/stride); see `backtest.py::_block_bootstrap_edge`) — the honest
CI, because consecutive predictions are autocorrelated:

| horizon | v1 (unshrunk) | v2 edge | edge 95% CI (block-bootstrap) |
|---|---|---|---|
| 1 day (n=3750) | −1.7pp | **+0.1pp** | [−0.1, +0.3] — ≈ zero |
| 1 week (n=1500) | −3.9pp | **−0.1pp** | [−0.5, +0.2] — ≈ zero |
| 1 month (n=720) | −4.4pp | **−0.6pp** | [−1.4, +0.3] — ≈ zero |
| 1 year (n=1308, overlapping) | −9.9pp | **−4.4pp** | [−8.2, −0.8] — loses |

Read this the honest way: **only the 1-year edge is distinguishable from zero, and
it's negative.** At 1d/1w/1m the CI straddles zero — indistinguishable from always-up.
The paired-delta bootstrap CIs are *tighter* than the old naive Wald ±1.6pp because
the model and always-up make identical calls on nearly every row, so the edge has
little variance — it is precisely, confidently ~0. (Also fixed: the shuffled-outcome
control now compares against the correct null — the up-rate, not 50% — because a
drift-anchored model that calls UP almost everywhere scores the up-rate on random
labels, not a coin flip.)

Key findings, so we don't re-litigate them:
1. **No price-only edge exists in these features.** Per-ticker ML at full
   strength loses 2–10pp to drift; a pooled cross-ticker model (34k rows,
   monthly walk-forward refits — `scratchpad/pooled.py`, cached OHLCV in
   scratchpad) converges to *exactly* the drift call: +0.0pp standalone edge at
   1d/1w/1m on validation tickers. Pooling is exhausted; don't revisit without
   new features.
2. **Calibration is good in the working range** (bin mean p 0.638 → realized
   0.641) but the long-horizon tails are overconfident: p<0.45 bucket realized
   0.81 up-rate (post-crash recovery years), p>0.70 realized 0.73. If a
   multi-horizon UI ships, clamp long-horizon probabilities harder.
3. **Confidence tiers**: pooled across horizons low 54.1% / medium 55.5% /
   high 67.2% accuracy; at the 5-session horizon alone all tiers are mid-50s
   (high = 53.1%, n=239 — noise). UI copy already says this honestly.
4. 1-year edge is negative because drift < 0.5 after crashes predicts DOWN
   years that recover. A "recovery prior" would fix it but would be fit on
   this same data — don't, unless validated on pre-2016 data.
5. Tune/validate discipline: parameters were chosen on even-indexed tickers,
   checked on odd-indexed. Keep doing that for any new knob.

## Prioritized next steps

1. **Let the ledger accumulate** (zero work now that the server self-logs every
   6 h): ~20 predictions/day. The 40-outcome adaptation gate opens ~2 trading
   days after logging starts (first outcomes resolve 5 sessions out, so ~mid-July
   2026); meaningful news-weight movement needs months. Watch `[learn]` lines in
   the server log and the "learning" block in `/api/track-record`. The server
   must be RUNNING to learn — which makes deployment (step 4) the real unlock.
2. **Historical news backtest via GDELT** (the big one): GDELT 2.0 DOC API is
   free and reaches back years — query daily article tone for a company name,
   build a historical sentiment series, and extend `backtest.py` with a news
   feature to finally validate/reweight the news tilts (currently 0.45/0.18/0.12,
   chosen by judgment not evidence). Watch rate limits; cache aggressively;
   match on cleaned company names like `news.py::_clean_company` does.
3. **Multi-horizon UI**: the harness already evaluates 1d/21d; the predictor
   only ships 5d. Add a horizon selector (1w default) reusing `_base_rates` +
   per-horizon holdout, with the tail-clamping from finding 2 and per-horizon
   drift honesty. Bump MODEL_VERSION and keep ledger records per horizon.
4. **Deployment to real HTTPS** (user asked about "https kind of thing"
   originally): Render/Fly free tier; needs the ledger file on a persistent
   volume, and a cron/scheduler to hit `/api/screener` daily so the track
   record builds without anyone opening the page.
5. **Screener conveniences**: direction filter (up/down), sort options,
   user-editable watchlist (persist to localStorage or a config endpoint).
6. Nice-to-haves: RSI pane on the chart, earnings-proximity warning in the
   screener cards, sector grouping.

## Gotchas (learned the hard way)

- **yfinance 1.5.1**: `Ticker.news` uses the nested `content` schema —
  `news.py::_yahoo_ticker_news` handles both shapes. The old per-ticker RSS
  feed (`feeds.finance.yahoo.com`) is dead; don't resurrect it.
- **^FTSE / some indices have zero Volume** → `vol_ratio` all-NaN → ML skipped;
  the drift+tech fallback covers it (`_price_model` returns `p_ml=None`).
- Company-name matching must be case-sensitive word-boundary on the *title*
  ("Apple" once matched a candy-store story via summary text).
- **lightweight-charts is pinned to 4.2.0** on unpkg — v5 changed the API
  (`addSeries(...)` instead of `addCandlestickSeries`). The chart must be
  created *after* its container is unhidden or it renders at width 0.
- macOS has no `timeout` command; Firefox headless `--screenshot` silently
  fails — use Playwright (installed in the session scratchpad, not the repo).
- Server runs via `./run.sh`; kill with `lsof -ti:8000 | xargs kill`. uvicorn
  here runs without `--reload`, so **restart after backend edits**.
- The in-memory TTL cache means a restart refetches everything; screener cold
  load is ~5s, warm ~instant. `/api/track-record` caches 600s.
- `backtest.py` imports `_base_rates`/`_blend_price`/`_feature_frame` from the
  app so harness ≡ production math. If you change the blend, the harness
  follows automatically — rerun it and update README + this file.

## File map

```
run.sh                    one-command launcher (venv + uvicorn :8000)
backtest.py               walk-forward harness + anti-cheat checks (~36s)
backtest_results.csv      7,278 graded predictions (current model)
backtest_summary.json     pooled stats per horizon
predictions.jsonl         live ledger (append-only, deduped ticker/day)
model_state.json          adaptive weights + update history (learner)
HANDOFF.md                this file — keep it updated at session end
app/config.py             prior weights, tiers, watchlist, MODEL_VERSION
app/analysis/predictor.py drift anchor, blend, holdout, analyze/snapshot
app/analysis/learner.py   online MAP weight learning + shadow adoption test
app/analysis/planner.py   ATR trade plans with empirical bracket odds
app/analysis/technical.py indicators + plain-English signals
app/analysis/sentiment.py finance-tuned VADER + recency weighting
app/data/market.py        yfinance wrapper (history/quote/earnings/search)
app/data/news.py          RSS + Yahoo company wire, concurrent + cached
app/data/ledger.py        prediction ledger (JSONL)
app/server.py             FastAPI: predict/history/news/search/screener/
                          track-record/market-overview + static hosting
app/static/               index.html, style.css, app.js (screener, chart,
                          track record, news tabs — dataviz-skill palette)
```
