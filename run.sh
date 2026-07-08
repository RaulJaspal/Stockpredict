#!/usr/bin/env bash
# One-command launcher: creates the virtualenv on first run, then serves the app.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run — creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo
echo "  StockPredict running at  http://127.0.0.1:8000"
echo
exec uvicorn app.server:app --host 127.0.0.1 --port 8000
