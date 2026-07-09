"""One daily 'tick' — the headless equivalent of the server's learning loop,
built to run once in CI (GitHub Actions) and exit.

It logs a fresh prediction for every watchlist ticker (deduped per ticker/day),
resolves any predictions whose 5-session horizon has matured, and updates the
adaptive blend weights. The workflow then commits predictions.jsonl and
model_state.json back to the repo, so the git history IS the persistent store —
no always-on server or database required.

Per-ticker errors are isolated and retried (yfinance is occasionally flaky), so
one bad symbol never sinks the run. Exits non-zero only if it could not log a
single ticker, which surfaces as a red run + an email from GitHub.

Run locally:  .venv/bin/python -m scripts.tick
"""

import sys
import time

from app.analysis import learner, predictor
from app.config import SCREENER_TICKERS


def _snapshot_with_retry(ticker, attempts=3):
    for k in range(attempts):
        try:
            predictor.snapshot(ticker)          # logs to the ledger (deduped)
            return True
        except Exception as exc:                # noqa: BLE001 — isolate flaky data
            print(f"  {ticker}: attempt {k + 1}/{attempts} failed: {exc}", flush=True)
            time.sleep(3 * (k + 1))
    return False


def main():
    t0 = time.time()
    ok = sum(_snapshot_with_retry(t) for t in SCREENER_TICKERS)
    print(f"[tick] logged/updated {ok}/{len(SCREENER_TICKERS)} tickers "
          f"in {time.time() - t0:.0f}s", flush=True)

    summary = learner.update_from_ledger()      # resolve matured + re-fit weights
    print(f"[tick] weights={summary['weights']} source={summary['source']} "
          f"n_resolved={summary['n_used']}", flush=True)

    if ok == 0:
        print("[tick] ERROR: could not log any ticker — failing the run", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
