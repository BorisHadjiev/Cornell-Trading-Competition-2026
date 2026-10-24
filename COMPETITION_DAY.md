# Competition day

## What runs

Branch `review/live-fixes`. The research book at half size with an edge floor:

| Setting | Value | Why |
|---|---|---|
| `LIVE_BOOK` | full | the book scored on the holdout |
| `LIVE_SIZE_FRACTION` | 0.5 | halves the swing of a wrong-way season |
| `LIVE_EDGE_FLOOR_DOLLARS` / `_SPREADS` | 1.0 / 2.0 | open or reverse only on an edge above the live noise |
| `KINK_PROXIMITY` | 1.0 | insurer sleeve stays on across the whole layer |
| `LOCATION_GATE` | all | path value for confident stall and eyewall storms too |
| `ENSEMBLE_SIZE` | 48 | less decision-to-decision noise in the target |
| `SEASON_STOP_LOSS` | 3,000 | flatten and hand to the backup; the only stop |
| `TEAM_ACTION_BUDGET` | 120 / min | leaves room for the browser and anyone else on the key |
| `LIVE_CROSS_AT_TOUCH_DEPTH` | on | a crossing reduction takes only the shown size |

Evidence for each is in `overnight_report.md` and `model/reports/live_rules.md`.

## Before the open (one person)

1. Mac plugged in, lid open, Claude app or a terminal open. `caffeinate -i -s -t 43200` in a tab.
2. **One bot on the team key.** Nobody else runs a bot. Nobody clicks orders in the browser while it
   runs (the 180 actions a minute are shared; 55 of our orders were rejected overnight when they were not).
3. Start it 5 to 10 minutes before the first open:

```bash
cd ~/Cornell-Trading-Competition-2026 && bash scripts/run_bot.sh 43200
```

   It waits through "No competition configured", connects, prints `connected`, and at the round-1
   open replaces the packet anchor with the first quotes (`"kind": "open", "source": "first quotes"`).
4. Health check any time (read-only):

```bash
cd ~/Cornell-Trading-Competition-2026 && SSL_CERT_FILE="$(./venv/bin/python -c 'import certifi;print(certifi.where())')" ./venv/bin/python scripts/health_check.py
```

## During the day

- Leave it alone. It carries positions across rounds, survives the pauses, reconnects on a drop,
  and restores its state from `state/<competition_id>.jsonl` if it has to be restarted.
- If it must be restarted: `Ctrl-C` in its tab and run the same command. Do it in the pause between
  rounds if you can.
- Announcements: the bot stands down 45 s on the named stocks and then adopts the house's repricing.
  This code has never seen a real announcement, so if one lands, watch the next decision line.
- The season stop at −3,000 flattens everything and switches to the backup rule for the rest of
  the season. That is by design; do not restart it to get the main book back.

## What not to touch on the day

`fitted.json`, the sleeves, the size, the floor. Every one of those has a replay behind it; a
change on the day has none.
