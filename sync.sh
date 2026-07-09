#!/usr/bin/env bash
# Refresh + view the dashboard with the freshest data. Pulls the ledger + weights
# the daily GitHub job committed, then shows the app at http://127.0.0.1:8000 in
# READ-ONLY mode (never writes, never conflicts — the daily "tick" is the single
# source of truth).
#
#   ./sync.sh
#
set -e
cd "$(dirname "$0")"

echo "Pulling the latest ledger + weights from GitHub..."
git pull --ff-only || {
  echo "git pull failed — you may have local commits/edits. Resolve, then re-run." >&2
  exit 1
}

open_browser() {
  ( sleep 3; (open http://127.0.0.1:8000 || xdg-open http://127.0.0.1:8000) >/dev/null 2>&1 || true ) &
}

if lsof -ti:8000 >/dev/null 2>&1; then
  # The always-on read-only server (the login LaunchAgent) is already up. Reload
  # it so it serves the freshly-pulled data right away, then open the browser.
  # No second server is started, so there's no port clash.
  echo "Reloading the always-on dashboard so it shows the new data..."
  launchctl kickstart -k "gui/$(id -u)/com.stockpredict.server" >/dev/null 2>&1 || true
  open_browser
  echo "Open:  http://127.0.0.1:8000"
else
  # Nothing on :8000 (auto-start disabled or Mac just booted) — start a read-only
  # server in the foreground; Ctrl-C to stop it.
  open_browser
  echo "Serving read-only dashboard at http://127.0.0.1:8000  (Ctrl-C to stop)"
  STOCKPREDICT_READONLY=1 exec ./run.sh
fi
