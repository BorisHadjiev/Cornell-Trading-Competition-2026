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
| Location | Trade the path once direct, chain, and recurve separate | On. Still pays if the fill is a few marks late |
| Insurer kink | HAVEN vs REIN from season-to-date insured loss, only near a boundary | On. Knots agreed across two training halves |
| Composition | Long the rebuilder who gets that hit, short the one who does not | On. Still pays if the fill is late |
| Severity | Radius and intensity, without betting the landfall | Small. Do not size it up |
| Revision | Preliminary damage vs the final report | 35% size. Loses money if the fill is late |
| Known damage | Trade assessed damage after the final report | 35% size. Loses money if the fill is late |
| ETF gap | House mark vs the basket | Off until the live book shows a gap after the spread |

Held-out season profit after the 35% cut, in case dollars, 100 seasons:

| Fill | Mean | Median | 10th percentile |
|---|---:|---:|---:|
| Next mark | 7374 | 7137 | 1330 |
| 2 marks later | 4046 | 3655 | −1772 |
| 5 marks later | 2891 | 2597 | −3402 |

The one-mark row is the ceiling if we are filled immediately in an empty book. It is not a setting we can choose. Other teams make the fill later and share the house’s mistake. A serious competitor is closer to the five-mark row than to 7,000. The starter bot (sell an island at 10% of the limit when the advisory is at least 80%) made about 350 in a typical season.

## Other traders

1. Do not chase the first advisory. If the price is already at our value, add nothing. If it has gone past us and the book is one-sided, fade a little. If it has not reached us, join, and do not pay through.
2. Keep half of PORT, ARCH, and REIN for the turn in the track and for the insurers. The first advisory cannot use all of it.
3. Prefer a pair (composition, or HAVEN vs REIN) to a one-name island short.
4. Cross the spread only when the gap is wider than the spread. Otherwise rest. Do not cancel and replace every two seconds.
5. For about 45 seconds after an announcement, do not fade the mark. Research seasons have no announcements. This is a live rule, not a fitted lag.
6. Create or redeem only when both books still disagree after the spread.
7. Obvious island trades are treated as slow fills. Pair trades may still be fast. A change stays only if the late-fill season does not get worse, and the sign agrees in at least 3 of 4 blocks of 25 holdout seasons.

## What we will not add

No neural net. No hardcoded insurance bounds. No class prior that depends on the round or the previous storm (those tests failed). No momentum sleeve (the mark’s first move does not predict the next 40 seconds). No new parameter fit on seasons 501–600.
