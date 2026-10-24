#!/bin/bash
# Relaunch the live bot with the code now on disk, but only in a round pause.
#   bash scripts/relaunch_in_pause.sh [max-wait-seconds]
# Waits until the exchange clock is `ready` or returns 403/404, then stops the running bot
# (it cancels its own rests on exit) and lets scripts/run_bot.sh start it again within 15 s.
# If no run loop is alive, this starts one. Exits non-zero if no pause arrives in time.
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE="$(./venv/bin/python -c 'import certifi;print(certifi.where())')"
WAIT="${1:-900}"
./venv/bin/python - "$WAIT" <<'EOF' || { echo "no pause within the wait; nothing restarted"; exit 1; }
import sys, time
sys.path.insert(0, ".")
from model import live
k, u = live.api_credentials(); ex = live.Exchange(live.resolve_host(u, k), k)
deadline = time.time() + float(sys.argv[1]); st = None
while time.time() < deadline:
    try:
        c = ex.get("/api/v1/competitions/time"); st = c.get("state")
        if st == "ready":
            print("pause: ready at", c.get("fictional_time"), "elapsed", c.get("elapsed_active_seconds")); sys.exit(0)
    except live.ExchangeError as exc:
        if exc.code in (403, 404):
            print("pause: exchange returned", exc.code); sys.exit(0)
        st = f"http {exc.code}"
    except Exception as exc:
        st = f"err {str(exc)[:30]}"
    time.sleep(3)
print("last state", st); sys.exit(1)
EOF
pkill -INT -f "model.live --max-seconds"
sleep 3
if ! pgrep -f run_bot.sh >/dev/null; then
  echo "no run loop; starting one"
  nohup bash scripts/run_bot.sh 43200 >/dev/null 2>&1 &
fi
for i in $(seq 1 12); do
  sleep 5
  if pgrep -f "model.live --max-seconds" >/dev/null && grep -q '"kind": "connected"' <(tail -5 overnight.log); then
    echo "relaunched and connected"; tail -3 overnight.log | cut -c1-160; exit 0
  fi
done
echo "bot process: $(pgrep -f 'model.live --max-seconds' | wc -l | tr -d ' ') (check the terminal tab)"; tail -3 overnight.log | cut -c1-160
