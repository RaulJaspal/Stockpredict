# HANDOFF — session state & next steps

*Last updated: 2026-07-09. Read this first when resuming work.*

## Session 2026-07-09 changes (newest first)

Theme: the price *direction* model has no edge (proven), so this session doubled
down on what IS predictable and on honest evaluation — without touching the
direction blend (published backtest numbers are unchanged, verified: same 14
features, same priors, same `_blend_price`).

- **Expected-range model upgraded to EWMA + calibrated quantiles** (`app/analysis/
  volatility.py`, new). Replaces the old `±1.34·σ₂₁·√5`. EWMA(λ=0.97) vol forecast +
  empirically-calibrated asymmetric log-return quantiles `Z_LO_5D=-1.1372 /
  Z_HI_5D=+1.2937`. Validated in `research/vol_backtest.py`: same ~80% coverage, ~6%
  sharper overall / ~10% sharper in stressed regimes, ~2× flatter calm→stressed
  calibration, LOO coverage mean 0.800 (0.775–0.828). Wired into `predictor._expected_range`.
- **Expected-range band is now graded live.** Ledger logs `range_low`/`range_high`
  (`predictor._assess`); `ledger.resolve_records` computes `in_band`; `/api/track-record`
  reports realized coverage (target 0.80); the dashboard shows a "Range coverage" tile.
  Populates as bands mature (none resolved yet as of this session).
- **Cross-sectional signal research** (`research/cross_sectional.py`, new). Long-short
  quintile backtest, 40 large-caps, 2016–2026, net of costs, block-bootstrap CI,
  half-sample persistence. VERDICT: **no edge clearing the bar at 5 days.** STR dead on
  mega-caps; 12-1 momentum has consistent positive sign (gross Sharpe ~0.45) but CI
  straddles 0 and net Sharpe ≤ 0.16. Documented, NOT shipped (GDELT precedent). Momentum
  is the one thing worth revisiting at a *monthly* horizon — ties into next-step #3.
- **Evaluation-honesty fixes:**
  - The per-ticker UI holdout graded 60 OVERLAPPING 5-day windows as if independent
    (~12 effective). Added `effective_n`; `_confidence` now only downgrades when the
    holdout trails baseline by > 1 SE at the effective n (was firing on noise ~half the
    time). UI note states the effective sample.
  - Planner expectancy was called "≈ 0" but never subtracted costs. Now computes empirical
    gross expectancy, the buy-and-hold benchmark over the same window, and the **level edge**
    (bracket − buy-and-hold), net of `COST_BPS_PER_SIDE=5`. The honest headline: the
    bracket's return is mostly drift; the levels' own edge is ≤ 0 (they cap winners),
    negative after costs. Frontend updated to show this.
- **Infra:** `.github/workflows/ci.yml` runs the unittest suite on push/PR. Whole suite is
  now **hermetic** (proven with DNS blocked). New hermetic `tests/test_no_lookahead.py`
  ports the causality audit + no-peek sentinel from `backtest.py` onto synthetic data, so
  the anti-cheat guarantee runs in CI. Tests: 12 → 26. `.gitignore` now excludes research
  `*.pkl` caches.
- **PEAD shipped — the app's FIRST validated directional edge.** `research/pead.py`:
  on 2,957 earnings events (30 large-caps, 2001-2026, enter day *after* the report), the
  EPS-surprise sign predicts the next month's drift — 57.6% hit, +0.87% net long-short,
  signed-drift 95% CI [+0.85%,+1.36%] EXCLUDES 0, 28/30 tickers, monotonic, robust across
  halves, horizon profile 52.5%@1wk→58.7%@2mo (matches the literature). Zero fitted
  params → whole sample is OOS. Shipped as an event-driven MONTHLY-only tilt
  (`config.PEAD` coef 0.085/σ; `predictor._pead_tilt`; `market.get_recent_earnings`, needs
  **lxml** — added to requirements). Active only ≤10 sessions post-report, decaying;
  clips the raw Surprise(%) (wild outliers, e.g. NKE +466%) before standardizing. Weekly
  is untouched (no earnings fetch → still bit-identical/fast). UI shows a "Post-earnings
  drift" driver row + plain-English note when active. Improves monthly Brier OOS. NOT yet
  live-tracked (monthly isn't logged) — the strong backtest is the validation; live
  tracking of PEAD-active monthly calls is a possible follow-up. Tests +7 (test_pead.py).
- **Multi-horizon (weekly + monthly) prediction view.** `config.HORIZONS` = {1w:5,
  1m:21}, `DEFAULT_HORIZON="1w"`. The predictor is now horizon-parameterized
  (`_price_model(ind, h)`, `_expected_range(ind, h)`, `_assess(ticker, h)`,
  `analyze(ticker, horizon)`); `/api/predict/{t}?horizon=1w|1m`; UI has a 1w/1m toggle
  in the prediction card. Monthly carries a stronger drift anchor → higher raw hit-rate
  (~58% vs ~57% weekly in the backtest), and its OWN calibrated range quantiles
  (`volatility._QUANTILES` = {5:(-1.1372,1.2937), 21:(-1.1089,1.4462)} — a month
  compounds more upside; do NOT sqrt-scale the 5-day). Holdout window scales with the
  horizon (`min(max(60, 12*h), len//3)`) so monthly gets ~6 effective windows not ~2.
  **Only the DEFAULT (weekly) horizon is logged to the ledger** (`_assess` guards on
  `h == HORIZON_DAYS`), so the live track record stays a single clean weekly test;
  monthly is a view with its own on-page holdout backtest. Weekly predictions are
  bit-identical to before (holdout_n doesn't affect p_up/base) → MODEL_VERSION unchanged,
  track-record continuity preserved. Edge is still ≈zero at every horizon — monthly is
  more *accurate* (drift), not skillful.
- **Always-on WITHOUT a server: GitHub Actions is now the persistence + learning layer.**
  The Mac isn't always on, so a scheduled job replaces the always-on server for logging/
  learning (viewing is still local). Architecture:
  - `scripts/tick.py` — the headless equivalent of `_learning_loop`, once: snapshot the
    watchlist (log), resolve matured, update weights. Per-ticker retries; exits non-zero
    only if nothing logged.
  - `.github/workflows/tick.yml` — cron `30 21 * * 1-5` (weekdays 21:30 UTC, ~1h after US
    close) + `workflow_dispatch`. Runs the tick and **commits `predictions.jsonl` +
    `model_state.json` back to the repo** (git history = the persistent store). Free:
    ~3 min/run × ~22 days ≈ 66 min/mo (private free tier = 2000/mo).
  - **Single-writer rule enforced by a read-only mode.** `STOCKPREDICT_READONLY=1` makes
    `ledger.record` a no-op and disables the server learning loop (`ledger.READONLY`,
    `server._start_learning`). The **LaunchAgent plist now sets this env var**, and
    `sync.sh` sets it too — so the daily GitHub job is the ONLY writer and the tracked
    ledger changes ONLY via `git pull`. No conflicts, no dirty tree locally. This RESOLVES
    the earlier open decision: keep the state files tracked (they ARE the persistence).
  - `./sync.sh` — `git pull` + serve the read-only dashboard at :8000 and open the browser.
  - To re-arm/inspect: `gh workflow run daily-tick` (manual run); GitHub emails on failure.
  - **To revert to Mac-only always-on:** remove `EnvironmentVariables` from the plist and
    disable the workflow. Local server would resume writing (and can then diverge from any
    GitHub writes — pick one writer).

## Session 2026-07-08 changes (newest first)

- **Repo is now under git** (was not before). `.gitignore` excludes `.venv/`,
  `__pycache__/`, `gdelt_cache/`, scratch. Commit as you go from here.
- **Backtest edge now ships a moving-block bootstrap 95% CI** (`backtest.py::
  _block_bootstrap_edge`) — the honest CI given overlapping windows. Finding
  sharpened: at 1d/1w/1m the edge CI straddles zero (indistinguishable from
  always-up); only 1-year is distinguishable and it's negative. Also fixed the
  shuffled-outcome control (it compared to 50%, wrong for an always-up-ish model;
  now compares to the up-rate).
- **GDELT news backtest DONE** (was next-step #2). `app/data/gdelt.py` +
  `news_backtest.py`. Verdict: **no out-of-sample edge from company news tone**
  at 5 sessions (4,145 walk-forward preds, 9 companies, 2017-2026; validate
  ΔBrier CI [-0.00044,+0.00036], AUC 0.51, best weight +0.06/+0.08 on a [-1,1]
  feature). Acted on it: **v2.2 cut news priors** company 0.45→0.15, market
  0.18→0.08, politics 0.12→0.05 (`config.py BLEND`); `MODEL_VERSION` →
  `2.2-newsval`; `model_state.json` regenerated to the new priors (it was still
  on `source:"prior"`, n_used 0). Not zeroed — GDELT tone ≠ live VADER feature,
  and the learner can still raise them. Only company news was directly tested.
- **Tests moved into the repo** (`tests/`, stdlib unittest, no pytest dep):
  learner rails, block-bootstrap CI, GDELT parse/cache. Run with
  `.venv/bin/python -m unittest discover -s tests`. Replaces the scratchpad ones.
- GDELT gotcha: it enforces ~1 request / 5 s and a *burst* trips a multi-minute
  IP cooldown returning HTTP 429 / a plain-text notice. The client paces at 8 s
  and backs off 25 s on throttle; a heavy 9-year query still 429s occasionally
  (AAPL/XOM/WMT were skipped this run — rerun `news_backtest.py` to fill them,
  their cache is empty so they'll refetch). Single-phrase queries only; multi-term
  OR queries time out server-side.

## What this project is

StockPredict: a local FastAPI web app (`http://127.0.0.1:8000`, start with `./run.sh`)
that predicts 5-session stock direction from a **drift anchor** (the ticker's
historical up-rate) plus small validated tilts (per-ticker logistic regression,
technicals) and live news sentiment (BBC/Sky/Guardian/CNBC/MarketWatch + Yahoo
company wire). Everything is graded honestly: per-ticker holdout vs always-up
baseline in the UI, a full walk-forward harness in `backtest.py`, and a live
prediction ledger (`predictions.jsonl` → `/api/track-record`).

## State right now

- **Model version `2.2-newsval`** (see `app/config.py: MODEL_VERSION`).
  Blend: `logit(base) + k_ml·(logit(p_ml) − logit(base)) + w_tech·tech + w_c·news_co + w_m·news_mkt + w_p·news_pol`, clipped [0.05, 0.95].
  Weights are **adaptive** (`app/analysis/learner.py`, state in
  `model_state.json`): MAP logistic regression on resolved ledger outcomes,
  Gaussian-anchored to the priors (now **0.15/0.10/0.15/0.08/0.05** — news cut
  in v2.2 after the GDELT backtest found no edge; see the session-change note).
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
- Expected range is an **EWMA(λ=0.97) vol forecast + calibrated asymmetric quantiles**
  (`app/analysis/volatility.py`, quantiles `-1.1372 / +1.2937`); replaced the old
  `1.34·σ₂₁·√5` heuristic this session (see the 2026-07-09 change note). Graded live.
- Confidence tiers: |p−0.5| ≥ 0.10 high, ≥ 0.05 medium, else low; capped to low only
  when the ticker's 60-session holdout trails always-up by **> 1 SE** at the effective
  (~12) non-overlapping sample size (was: any shortfall, which fired on noise).
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
6. **Company news tone has no 5-day directional edge** (`news_backtest.py`,
   GDELT 2017-2026, 4,145 walk-forward preds): validate ΔBrier CI straddles 0,
   AUC 0.51. Drove the v2.2 news-prior cut. Don't restore the old 0.45 weight
   without new evidence. Market/politics tilts remain judgment-only (untested).

## Prioritized next steps

1. **Let the ledger accumulate** (zero work now that the server self-logs every
   6 h): ~20 predictions/day. The 40-outcome adaptation gate opens ~2 trading
   days after logging starts (first outcomes resolve 5 sessions out, so ~mid-July
   2026); meaningful news-weight movement needs months. Watch `[learn]` lines in
   the server log and the "learning" block in `/api/track-record`. The server
   must be RUNNING to learn — which makes deployment (step 4) the real unlock.

1b. **Verify the macOS auto-start → learn pipeline after a reboot.** A LaunchAgent
   (`~/Library/LaunchAgents/com.stockpredict.server.plist`, `RunAtLoad` +
   `KeepAlive`) starts `.venv/bin/python -m uvicorn app.server:app` on :8000 at
   login; the server then runs a learn cycle 90 s after startup and every 6 h
   (`server.py::_learning_loop`). Verified working 2026-07-08 (log shows scheduled
   `[learn]` lines, now stamped with the v2.2 weights). To re-confirm after a
   reboot / if learning ever seems stalled:
   - `launchctl print gui/$(id -u)/com.stockpredict.server` → state should be
     `running` (or `launchctl kickstart -k gui/$(id -u)/com.stockpredict.server`
     to force a restart without rebooting);
   - `curl -s 127.0.0.1:8000/api/track-record | python3 -m json.tool | grep -A6 learning`;
   - `grep '\[learn\]' ~/Library/Logs/StockPredict.log | tail` → a fresh line
     within ~2 min of startup, `source` flips `prior → adaptive` once n ≥ 40.
   Gotchas: the agent assumes deps are already installed (it doesn't `pip install`
   — run `./run.sh` once after any dependency change) and that `.venv` exists at
   the path in the plist; if `.venv` is deleted, run `./run.sh` to rebuild it.
2. ~~**Historical news backtest via GDELT**~~ **DONE 2026-07-08** — see the
   session-change note up top. Verdict: no company-news-tone edge; news priors
   cut in v2.2. Follow-ups if revisited: (a) rerun `news_backtest.py` to fill the
   3 tickers 429-skipped this run (AAPL/XOM/WMT); (b) extend it to the *market*
   and *politics* tilts with index/macro GDELT queries (only company was tested);
   (c) the fitted +0.06/+0.08 sign was consistent across tune/validate — a hair
   of signal, worth a second look with a better feature (event counts, tone
   *change* vs level) before concluding it's exactly zero.
3. **Multi-horizon UI + monthly momentum** (now the most promising direction lead):
   the harness already evaluates 1d/21d; the predictor only ships 5d. Add a horizon
   selector (1w default) reusing `_base_rates` + per-horizon holdout, with the
   tail-clamping from finding 2 and per-horizon drift honesty. AND: at the *monthly*
   horizon, revisit cross-sectional 12-1 momentum (`research/cross_sectional.py` found
   consistent positive sign in both halves, gross Sharpe ~0.45, but CI straddled 0 on 40
   names / 5-day). A bigger universe + monthly hold is where it's most likely to clear the
   bar — validate long-short with block-bootstrap CI before shipping any momentum tilt.
   For the range at other horizons, `volatility.expected_range(close, horizon_days=h)`
   already sqrt-scales the 5-day quantiles (mild approx; recalibrate per-horizon quantiles
   in `research/vol_backtest.py` if you ship non-5-day ranges). Bump MODEL_VERSION and keep
   ledger records per horizon.
3b. **CI already runs the hermetic suite** (`.github/workflows/ci.yml`, added 2026-07-09).
   If you add network-touching tests, mock them — the suite is currently proven hermetic.
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
backtest.py               walk-forward harness + anti-cheat + block-bootstrap edge CIs (~36s)
backtest_results.csv      7,278 graded predictions (current model)
backtest_summary.json     pooled stats per horizon (+ edge CIs, verdicts)
news_backtest.py          walk-forward GDELT news-tone validation (v2.2)
news_backtest_summary.json / _rows.csv   its results
research/vol_backtest.py       expected-range vol validation (EWMA vs rolling, LOO)
research/cross_sectional.py    long-short reversal/momentum study (no 5-day edge)
research/pead.py               post-earnings-drift study (THE validated edge; monthly tilt)
.github/workflows/ci.yml       runs the hermetic unittest suite on push/PR
predictions.jsonl         live ledger (append-only, deduped ticker/day; now logs range band)
model_state.json          adaptive weights + update history (learner)
HANDOFF.md                this file — keep it updated at session end
tests/                    stdlib unittest suite (learner, stats, gdelt, volatility,
                          planner, confidence, no-lookahead) — hermetic, 26 tests
app/config.py             prior weights, tiers, watchlist, MODEL_VERSION
app/analysis/predictor.py drift anchor, blend, holdout, analyze/snapshot
app/analysis/volatility.py EWMA vol forecast + calibrated expected-range band
app/analysis/learner.py   online MAP weight learning + shadow adoption test
app/analysis/planner.py   ATR trade plans + empirical odds + honest level edge
app/analysis/technical.py indicators + plain-English signals
app/analysis/sentiment.py finance-tuned VADER + recency weighting
app/data/market.py        yfinance wrapper (history/quote/earnings/search)
app/data/news.py          RSS + Yahoo company wire, concurrent + cached
app/data/gdelt.py         cached GDELT daily news-tone client (rate-limit paced)
app/data/ledger.py        prediction ledger (JSONL)
app/server.py             FastAPI: predict/history/news/search/screener/
                          track-record/market-overview + static hosting
app/static/               index.html, style.css, app.js (screener, chart,
                          track record, news tabs — dataviz-skill palette)
```
