#!/usr/bin/env bash
# Double-click this file to open StockPredict.
# (The app normally starts itself at login — this starts it only if needed.)
cd "$(dirname "$0")"
if curl -s -o /dev/null --max-time 2 http://127.0.0.1:8000/; then
  echo "StockPredict is already running — opening it in your browser."
  open http://127.0.0.1:8000
  exit 0
fi
echo "Starting StockPredict… your browser will open in a few seconds."
echo "Keep this window open (minimised is fine) — closing it stops the app."
echo
(sleep 5; open http://127.0.0.1:8000) &
./run.sh
