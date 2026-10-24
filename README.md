# Cornell Trading Competition 2026

Systematic equities. One hurricane season, three storms. Positions, damage, and insured losses carry. At the close of round 3, stocks settle at fundamental value and ETFs at their basket. Score is season PnL.

This file is the current strategy. Update it in the same pass as any change to the model or the live rules.

## Architecture

Fit on seasons 1–500. Score seasons 501–600. Do not refit on the holdout.

1. **Feed.** Storm position, track, official advisories, wire, impacts, preliminary and final damage. Advisories are evidence, not the label. A storm is erratic at age `t` only when `turned_erratic_at` is set and `t` has reached it.
2. **Pattern filter.** Probability over direct strike, island chain, recurve, stall, eyewall replacement, and erratic, plus a hazard of flipping to erratic. Before the first advisory the filter is worse than the base rate, so the book uses the training prior until an official advisory exists.
3. **Paths.** Sample remaining paths from storms with the same age-label. Keep intensity from the storm observations. Track points do not have intensity. Average damage across paths. Do not damage the average path.
4. **Damage.** A gaussian wind kernel on the 17 sites in `case.json`. Company damage is the weighted sum of its sites. Outage time is minutes of wind above a fitted threshold. Impact is the start of that clock.
5. **Value.**
   - Island names lose value from permanent damage and from time shut. GRID and FERRY are on every island. PORT is one Cascadia site.
   - RECON repairs port, grid, and ferry. MAT supplies all repairs. Both curves are concave and stateful across the season.
   - GEN is paid for grid-outage hours. It has no damage term.
   - HAVEN pays insured losses above an estimated retention up to an estimated attachment. REIN pays above that attachment. Trade them only near a kink. Far from it, the target is zero. Bounds are estimated, not taken from the packet.
   - ARCH holds 1 PORT, 1 ISLE, 2 AQUA, 1 GRID, 2 FERRY, 1 HAVEN. It is not an island hedge. REBLD holds 2 RECON, 2 MAT, 1 GEN.
6. **Portfolio.** One probability-weighted target per name. Deadband. PORT, GRID, FERRY, and ARCH share one Cascadia risk cap. Size stays fractional until the track turns. Fills in research are the next house mark. That is optimistic. There is no spread and no other team in `market.csv`.

A round-1 price against season-end settlement is not a house mistake. Later storms have not happened. At the round close the house mark was closer to settlement than our model, so we do not add a carry sleeve.

## Strategies

| Strategy | What it is | State |
|---|---|---|
| Location | Trade the path once direct, chain, and recurve separate | On. The live book is this sleeve, on the physical names, from the first order |
| Insurer kink | HAVEN vs REIN from season-to-date insured loss, only near a boundary | Off for the whole season |
| Composition | Long the rebuilder who gets that hit, short the one who does not | Off for the whole season |
| Severity | Radius and intensity, without betting the landfall | Off |
| Revision | Preliminary damage vs the final report | Off for the whole season |
| Known damage | Trade assessed damage after the final report | Off for the whole season |
| ETF gap | House mark vs the basket | Off. No create or redeem |
| Limit reserve | Half of PORT, ARCH, and REIN held back from the first advisory | On for the location book |
| Pair over location | Shrink the one-name location trade when a composition or insurer pair is already on | Off. Those pairs are not on |

The live process sends that table. The holdout numbers below are the research book, not this season's orders.

Held-out season profit after the 35% cut, in case dollars, 100 seasons:

| Fill | Mean | Median | 10th percentile |
|---|---:|---:|---:|
| Next mark | 7374 | 7137 | 1330 |
| 2 marks later | 4046 | 3655 | −1772 |
| 5 marks later | 2891 | 2597 | −3402 |

The one-mark row is the ceiling if we are filled immediately in an empty book. It is not a setting we can choose. Other teams make the fill later and share the house’s mistake. A serious competitor is closer to the five-mark row than to 7,000. The starter bot (sell an island at 10% of the limit when the advisory is at least 80%) made about 350 in a typical season.

If the obvious island short fills 5 marks later and the pair trades still fill on the next mark, the book before the reserve was mean 6,702, median 6,189, 10th percentile 1,267. With the reserve and the smaller location add, that crowded book is mean 5,676, median 5,060, 10th percentile 354. Both rules beat an all-late book in all 4 blocks of 25 seasons. The empty-book next-mark result falls from 7,374 / 7,137 / 1,330 to 6,336 / 5,824 / 1,207. That drop is the size we held back.

## Other traders

1. Do not chase the first advisory. If the price is already at our value, add nothing. If it has gone past us and the book is one-sided, fade a little. If it has not reached us, join, and do not pay through.
2. Keep half of PORT, ARCH, and REIN for the turn in the track and for the insurers. The first advisory cannot use all of it.
3. The live book does not trade a pair. It is the location sleeve, then the backup.
4. Cross the spread only when the gap is wider than the spread. Otherwise rest. Reductions cross. Resting orders are cancelled at the house half-life. The whole book is flattened when the round ends. After a name opens, that round only reduces it. Do not cancel and replace every two seconds.
5. For about 45 seconds after an announcement, do not fade the mark. Research seasons have no announcements. This is a live rule, not a fitted lag.
6. Do not create or redeem.
7. Obvious island trades are treated as slow fills. Pair trades may still be fast. A change stays only if the late-fill season does not get worse, and the sign agrees in at least 3 of 4 blocks of 25 holdout seasons.

## Run on a laptop

The contest bot runs on your machine, not in the cloud. Cloud runs are only for connection tests.

```bash
pip install -r requirements.txt
export CTC_API_KEY=your-bot-key
python -m model.live
```

Do not commit the key. `model/fitted.json` is already in the repo, so a live session does not need the research zip. The research zip is only for a refit. The process stays up for 4 hours by default, including the ready gaps between rounds, and appends `book-log.jsonl` every second.

## Prototype

The live book is the location sleeve on PORT, ISLE, AQUA, GRID, and FERRY from the first order. Severity, the insurer kink, composition, revision, known damage, and ETF orders stay off for the whole season. The reserve on PORT still applies. This is not a refit.

If the exchange season mark (realized plus unrealized) reaches −400, cancel, flatten, and only the backup trades for the rest of the season. At −800, flatten and send no more risk. A backup name at −150 is flattened and not reopened. Those three bounds are pre-set. They were not fit on seasons 1–500 or 501–600, and the table below was not used to move them.

The backup sells about 10% of the limit, at most once per name per storm, when an official advisory puts one of that company's islands at 80%. No overnight carry and no add after the name is open.

Backup on seasons 1–500, case dollars. Seasons 501–600 were not scored.

| Fill | Mean | Median | 10th percentile |
|---|---:|---:|---:|
| Next mark | 544.93 | 550.85 | −16.19 |
| 2 marks later | 482.20 | 517.30 | −60.24 |
| 5 marks later | 486.54 | 521.95 | −46.30 |
| Crowded island short | 485.92 | 519.05 | −46.57 |

An order is target minus the exchange position minus working leaves. A cancel that returns already filled is a fill, and the account is reread before the next order. A rest uses the live touch. A reduction crosses. An announcement cancels resting orders. The book is flat when the round leaves `running`.

Each second of `book-log.jsonl` has, for every name, bid, ask, house mark, target, exchange position, and working quantity. The house mark is the quote's mark, or last or price if that is the field the feed sends. Working quantity is signed, and a sell is negative. The log does not place an order.

## What we will not add

No neural net. No hardcoded insurance bounds. No class prior that depends on the round or the previous storm (those tests failed). No momentum sleeve (the mark’s first move does not predict the next 40 seconds). No new parameter fit on seasons 501–600.
