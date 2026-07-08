#!/usr/bin/env bash
# One-command launcher: creates the virtualenv on first run, then serves the app.
set -e
cd "$(dirname "$0")"

# Recreate the venv if it's missing OR was built at a different path — pip's
# console scripts (uvicorn, pip) hardcode an absolute interpreter path, so a
# moved/copied .venv leaves them broken. We invoke everything through the venv's
# python (`-m`), which only relies on the still-valid python symlink, and rebuild
# when even that is broken.
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c '' 2>/dev/null; then
  echo "Creating virtual environment..."
  rm -rf .venv
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt

echo
echo "  StockPredict running at  http://127.0.0.1:8000"
echo
exec .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
