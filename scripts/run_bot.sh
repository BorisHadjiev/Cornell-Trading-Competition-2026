#!/bin/bash
# Run the live bot, retrying while the testnet is between seasons ("No competition configured"
# makes the start-up host check fail) or after a crash. Stops retrying after a clean 12-hour run.
#   bash scripts/run_bot.sh [max-seconds]
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE="$(./venv/bin/python -c 'import certifi;print(certifi.where())')"
MAX="${1:-43200}"
for attempt in $(seq 1 120); do
  started=$(date +%s)
  ./venv/bin/python -m model.live --max-seconds "$MAX" 2>&1 | tee -a overnight.log
  ran=$(( $(date +%s) - started ))
  if [ "$ran" -ge $(( MAX - 60 )) ]; then
    echo "run complete after ${ran}s"; break
  fi
  echo "bot exited after ${ran}s (attempt $attempt); retrying in 15s" | tee -a overnight.log
  sleep 15
done
