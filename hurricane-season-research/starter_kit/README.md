# Hurricane Season starter kit

This kit contains four small tools. Use them to work with the research data and to test your bot
on the testnet.

You need Python 3.10 or later. `history.py` and `check_recording.py` use only the standard
library. `record_feed.py` and `example_bot.py` also need the `websockets` library:

```bash
pip install "websockets>=14"
```

| File | Function |
|---|---|
| `history.py` | Loads the research data: seasons, storms, the feed, the house prices and the settlement values. |
| `record_feed.py` | Records the market-data stream of the exchange to a file. The stream contains the storm feed. It can also contain quotes and trades. |
| `check_recording.py` | Compares a testnet recording with the research data, message by message. |
| `example_bot.py` | A simple bot that shows the full loop: case data, clock, feed, quotes and limit orders. |

## Load the research data

Run this code from the `starter_kit` folder.

```python
from history import load

data = load("../history")                   # the folder with manifest.json
for season in data.seasons:
    for storm in season.storms:             # name, true pattern, damage, insured losses
        ...
season = data.seasons[0]
season.messages(round=1, type="advisory", source="official")
season.track(1)                             # [(age, x, y, radius), ...]
season.prices["HAVEN"], season.times        # house consensus price at each 2 s sample
season.settlement["HAVEN"]                  # settlement value
```

## Test your bot on the testnet

The testnet replays the research seasons in sequence, exactly as they are in the data:

- The messages are the same, with the same `seq`.
- The house quotes around the prices of that season.
- The symbols settle at the settlement values of that season.

The fictional date tells you which season plays.

1. Sign in to the testnet. Create a team or join a team. Make a bot key on the Team page.
2. Start the recorder while your bot runs:
   `python record_feed.py --url wss://hurricane.cornelltradingcompetition.org --key BOT_KEY --out recording.jsonl`
3. After a season, check the recording:
   `python check_recording.py recording.jsonl ../history/`

If the check passes, it prints `recording matches the published feed`. If the check fails, it
shows one or more of these problems:

- **Missing messages**: your connection stopped. The exchange does not send old messages again.
  A `snapshot` comes every 30 s and summarizes the round until that time.
- **Different or unexpected messages**: your code has a problem when it reads or stores the
  messages.

You can start to record during a season. The check starts at the first message that you received.

After each season, the testnet prepares the next season. Your keys and your WebSocket connection
stay the same. Your positions go back to zero, and the competition gets a new id.

## The example bot

`example_bot.py` connects with a bot key and reads `case.json` from the exchange. When an
official advisory gives an island a probability of hurricane conditions of 80% or more, the bot
sells a small quantity of each island business that has a site on that island.

The bot shows how the parts connect. It is not a strategy that makes money. Replace its `decide`
function with your model.

Run the bot with `--dry-run` first. The bot then shows the orders that it would send. The bot uses
limit orders at the bid. It does not use market orders.
