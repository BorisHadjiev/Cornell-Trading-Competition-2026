# Hurricane Season research data

This folder contains past hurricane seasons of the Archipelago. Use them to make and test your
strategy before the competition.

The same simulator made these seasons and the live season. They use the same rules, map,
companies, ETFs, feed and timing. Each season has three rounds. Each round is one market day with
one storm. The dates of season N are approximately N years before the live season. No storm in
these seasons is a storm of the live season.

The testnet replays these seasons in sequence, exactly as they are here. Use `starter_kit/` to
load the data and to check a testnet recording against it.

## Contents

| Path | Contents |
|---|---|
| `case.json` | The public case data: map, islands, company sites, ETF baskets, position limits, storm outlooks and timing |
| `history/manifest.json` | All seasons and their storms, with the true pattern, the damage and the insured losses |
| `history/season-NNN/feed.jsonl` | The feed of the season, one message on each line, as the exchange published it |
| `history/season-NNN/market.csv` | The house prices during the season |
| `history/season-NNN/settlement.json` | The settlement value of each symbol at the end of the season |
| `starter_kit/` | A data loader, a feed recorder, a recording check for the testnet and an example bot |

## Conventions

- **Time**: fictional US/Eastern time. A round is one market day, from 09:30 to 16:00. One round
  takes 780 real seconds (13 minutes). The fictional clock is 30 times faster than real time. The
  `age` of a storm is the number of real seconds since the storm formed.
- **Map**: 1000 by 700 units. The origin is at the top left, and `y` increases to the south.
  Storms form south of the map and move north.
- **Money**: prices are in dollars for each share. Insured losses are in millions of dollars
  (`_musd`).

## `history/manifest.json`

`seasons` lists all seasons in sequence. Each season has these fields:

| Field | Meaning |
|---|---|
| `season` | The season number |
| `dir` | The folder of the season in `history/` |
| `season_loss_musd` | The insured losses of the Archipelago for the full season, in $M |
| `storms` | One entry for each storm, in round order (see below) |

Each storm has these fields:

| Field | Meaning |
|---|---|
| `round` | The round, 1 to 3 |
| `storm` | The name of the storm |
| `pattern` | The true steering pattern of the storm when it formed. The keys are the same as in `outlooks` in `case.json`. |
| `turned_erratic_at` | The storm age in seconds when the storm became erratic, or `null` if it did not become erratic |
| `open` | The opening time of the round |
| `damage_pct` | The damage that this storm added to each island business, as a percentage of its site-weighted asset value. This value comes from the final damage report. |
| `industry_loss_musd` | The insured losses from this storm, in $M. This value comes from the final damage report. |

The manifest gives the true events. The feed gives only the information that traders had at that
time.

## `feed.jsonl`

This file contains all the messages that the feed published during the season, in `seq` order.
Each line is one JSON object. The messages are storm observations, tracks, official advisories,
wire rumors, island impacts and damage reports. The appendix of the case packet gives the message
types and their fields.

The file does not contain snapshots, because a snapshot only summarizes the messages before it.

## `market.csv`

This file gives the house consensus price of each symbol every 2 real seconds. It is the price
that a trader saw on the screen. The price is a few seconds behind the news, and it has small quote
noise. It is not a traded price.

The file has no order book. Use the testnet to see the spreads, the depth and the house quotes.

| Column | Meaning |
|---|---|
| `time` | The fictional time of the sample |
| `round` | The round of the sample |
| One column for each symbol | The house price of the symbol, in dollars |

## `settlement.json`

This file gives the settlement value of each symbol at the close of the last round of the season.
Open positions settle at these values.

## What the data does not contain

- **News and announcements**: the live season can have company or government announcements. An
  announcement changes the economics of a company from that time. These seasons have no news and
  no announcements.
- **The numbers of the model**: the data does not give how easily a storm damages each site, the
  cost of an outage, the repair capacity of the rebuilders, or where each insurance layer starts
  and stops. Use this data to estimate these numbers.
