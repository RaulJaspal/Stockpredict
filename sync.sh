#!/usr/bin/env bash
# View the dashboard with the freshest data. Pulls the latest ledger + weights
# that the daily GitHub job committed, then serves the app locally in READ-ONLY
# mode — so opening it never writes to the ledger and never conflicts with
# GitHub (the daily "tick" is the single source of truth).
#
#   ./sync.sh
#
set -e
cd "$(dirname "$0")"

echo "Pulling the latest ledger + weights from GitHub..."
git pull --ff-only || {
  echo "git pull failed (local changes?). Fix, then re-run ./sync.sh" >&2
  exit 1
}

# Open the browser shortly after the server comes up (best-effort, macOS/Linux).
( sleep 3; (open http://127.0.0.1:8000 || xdg-open http://127.0.0.1:8000) >/dev/null 2>&1 || true ) &

echo "Serving read-only dashboard at http://127.0.0.1:8000  (Ctrl-C to stop)"
STOCKPREDICT_READONLY=1 exec ./run.sh
