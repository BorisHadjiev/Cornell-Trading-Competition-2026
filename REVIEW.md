# Review of the Hurricane Season bot (24 Oct 2026)

Branch `review/live-fixes`, based on PR 3 (`cursor/live-backup-bot-9da3`), which is `main` plus the
loss-cap runner. Every number below is from the 100 held-out seasons 501–600 with the committed
`fitted.json`; nothing was refit. The replay scripts are described in `model/reports/live_rules.md`.

## 1. The case in one page

**The world.** Four islands, ten stocks, two ETFs. Five island businesses own 17 sites (a port,
resorts, fish farms, substations, ferry terminals). Three rebuilders earn from repair work.
Two insurers pay the season's insured losses in layers: HAVEN pays above its retention up to its
limit, REIN pays above that limit.

**The season.** Three rounds, each a 13-real-minute market day with one hurricane. Positions, damage
and the running insured-loss total carry across rounds. At the close of round 3 every stock settles
at its fundamental value, every ETF at its basket, and the score is total PnL. So the game is: know
each company's end-of-season value before the house does, and trade against the house price.

**What moves value.** Island businesses lose a permanent percentage of value per damage point and
lose revenue while a site is shut. Rebuilders gain with diminishing returns (RECON on port, grid and
ferry damage; MAT on all damage; GEN on grid-outage hours). Insurers lose by the part of the season
total in their layer. Research data: mean season loss is $1,009M, 95% of seasons breach the fitted
HAVEN retention ($413M), 17% reach the REIN attachment ($1,325M). PORT and HAVEN move the most
(mean |settlement − open| of $16.5 and $14.3); ARCH, which holds both, moves $31.5.

**What you see.** Storm position, velocity, radius and intensity every 2 s; the track every 20 s;
an official advisory every 60 s with an outlook (direct / chain / recurve / stall / eyewall /
erratic) and per-island probabilities; wire rumours about every 45 s; impact messages; a
preliminary damage report 30 s after the storm, a final one 40 s before the close. Official
advisories name the true pattern 70% of the time, wire 40%. Nothing in the feed says how damage
maps to dollars; that is what the 600 research seasons are for.

**The market.** The house quotes every symbol around a slow, imperfect consensus (about 45 s to
digest an announcement) and trades noise and wire rumours. Other teams trade on the same books.
Position limits include resting orders. 180 order actions per minute per team, 300 per bot key.
ETF create/redeem is immediate, free, booked at the house marks, and counts as one action.

## 2. What the bot does, and why it is the right shape

Fit on seasons 1–500, scored on 501–600. From the public feed it builds a physical state and prices
it:

1. **Pattern filter.** A softmax over the six outlooks from the track, advisories, wire and cone,
   plus a hazard of turning erratic. Before the first advisory it is worse than the base rate, so
   the training prior is used until one exists.
2. **Paths.** Sample remaining paths from historical storms with the same age-label, scale by the
   live storm's intensity, and average the *damage* across paths (not the damage of the average
   path).
3. **Damage.** A Gaussian wind kernel over the 17 sites, integrated along the path; outage hours
   are time above a fitted wind threshold. Held-out damage MAE is roughly half the mean-only
   baseline for every company.
4. **Value.** Linear in damage and outage for island names, concave in work for rebuilders, a
   two-knot layer for the insurers (knots agreed across both training halves), ETFs from their
   baskets. Handed the true inputs, this prices settlement to $1.24 against $8.42 for the open.
5. **Portfolio.** One probability-weighted target per name (tanh of edge over $3, times limit),
   a deadband, scenario loss caps, a shared Cascadia budget, a half-line reserve on PORT/ARCH/REIN
   until the track turns, and smaller size for the two sleeves (revision, known damage) that lose
   money when the fill is late.

This is the correct design for this case: the simulator is physical, the research pack is large,
the house is slow, and settlement is at fundamental value, so a calibrated fundamental model that
trades the gap to the house mark is the whole edge. Held out, next-mark fills, no spread:
**mean $6,336, median $5,824, 10th percentile $1,207, 6 losing seasons of 100**, against $256 for
the starter bot. With fills five marks late it is still $2,651.

## 3. What was wrong

Ordered by how much it would have cost on the day.

### 3.1 The live book was not the book that was tested

Both live runners took the research targets apart before sending them.

- `main` zeroed the "severity" sleeve. The sleeve names are an attribution, not independent bets:
  `severity` is the whole position implied by the unconditional storm mean, and `location` is only
  the residual on top of it (`s_loc − s_sev`). Dropping `severity` leaves the location view
  *minus a fade of the house*: after an advisory has moved the mark, that residual is usually
  larger than the real target and often in the wrong direction. Holdout: **$6,336 → $3,440**.
- PR 3 went further and sent only that residual on the five island names: **$1,881**.
- PR 3's stack (location residual + one-entry rule + round-end flatten + 150/400/800 caps):
  **$506, 28 losing seasons**. This is the bot that lost on the testnet (competition 90602797), and
  the loss was taken as evidence the model was wrong. The model was fine; the surgery on it was
  not.

Fix: `config.LIVE_BOOK = "full"` sends the research book. A `no_severity` mode (flat until a path,
preliminary or final view exists) is kept for comparison: $5,196. The `location` mode is kept so
the comparison is reproducible.

### 3.2 "One entry per name per round, then only reduce"

Added after one HAVEN round trip on the testnet. It also blocks the add the PORT/ARCH reserve
exists for (half the line held back *so that it can be added after the turn*), and the
final-report trade. Holdout: **$6,336 → $883, 26 losing seasons**. Off by default
(`LIVE_ONE_ENTRY_PER_ROUND = False`).

### 3.3 Flatten everything at the round close

Justified by "HAVEN carried short from 34 at the round-1 close and covered at 43 the next
morning". In the data for that season (239) HAVEN closed round 1 at 33.20 and opened round 2 at
33.22. The move to 43 happened inside round 2 on the first advisory for Omar, which a flat book
would have re-entered into anyway. Across all 600 seasons the close-to-open gap has mean ≈ 0 and a
standard deviation of $0.02–0.08 per stock. Flattening buys no protection and pays two spreads per
round (the research book turns over about $2.1M a season; at 10 bps a side that is already $2,100
of the $6,336). Off by default; the round close now cancels resting orders and carries.

### 3.4 Loss caps that fire on noise

A $150 per-name cap on a book where PORT alone moves $0.87 on an average advisory: fired 363 times
in 100 seasons, **$6,336 → $3,349, 33 losing seasons**. Even a $2,500 book switch fired in 11
seasons and turned the 10th percentile from +$1,207 to −$2,517, because the edge is paid at
settlement and mid-season drawdowns revert. Defaults now: no per-name cap, no switch, a $6,000
season stop as a circuit breaker for a broken model (worst holdout season: −$3,845).

### 3.5 The runner forgot the season

Three independent gaps that compound on the competition day:

- **Snapshots were ignored.** The exchange never replays old news; a bot that connects late or
  reconnects after a gap only gets the 30-second snapshot. Without it, a round-2 start values
  every name as if round 1 did no damage. `apply_snapshot` now consumes damage reports from every
  round, the latest advisory, impacts, the latest observation and the track.
- **No state across a restart.** Every news frame, the locked open and any anchor shift are now
  appended to `state/<competition_id>.jsonl` and replayed on start. A restart in round 3 knows
  rounds 1–2.
- **Wrong anchor after a restart.** The value model is anchored on the *season* open (seasons
  differ: PORT opens between 66.8 and 76.3 across research seasons, and the round-2 open differs
  from the season open by $8.6 on PORT and $18.5 on ARCH). The old code took "the first full set of
  quotes" as the open, whatever the round. Now: the matched research season on the testnet; the
  first quotes only at a round-1 open before the storm forms; otherwise the case packet's starting
  prices, which are by definition the live season's open.
- **New season, old data.** On the testnet the next season starts under a new competition id while
  the socket stays up. PR 3 reset the caps but kept the old season's observations and damage
  reports. `_new_season` now rebuilds the feed state, the anchor, the path library and the store.

### 3.6 Announcements would have been faded for the rest of the season

The live season has announcements that "change the economics of a company"; the research seasons
have none, so the value model cannot know about them. The runner stood down for 45 s and then
resumed trading its unchanged fair value against a house that had correctly repriced, at full size,
for the rest of the season. Now, after the stand-down, the anchor of each named stock is moved by
the part of the house's move the storm model did not produce (`anchor_shift_for`), so the house's
view of the announcement is adopted and the storm model keeps working on top of it. An
announcement with no symbols shifts all ten stocks. Verified in the offline smoke test.

### 3.7 A cancel-and-replace loop that would have hit the rate limit

Resting orders were cancelled at the fitted house half-life of the event that justified them. For
a damage report that is 4 seconds. Ten names, cancel plus replace every 4 s, is 300 actions a
minute against a team cap of 180; the budget guard then dropped placements and left names with no
order. Now a rest lives at least 20 s (`MIN_REST_SECONDS`), and one still sitting where a fresh
rest would go is refreshed rather than cancelled.

### 3.8 Smaller

- Opening rests now sit one tick inside the house touch (capped at a quarter spread) so they are
  first in the queue for house flow instead of behind the house at its own price.
- The exposure integral assumed a 2-second row everywhere. Research data is exactly that, so the
  fit is unchanged, but a live gap (reconnect, 4-second track catch-up) under-counted wind. Rows
  now carry their own age gap, and `track` messages fill observation gaps.
- `requirements.txt` pinned `numpy==2.2.6` / `pandas==2.2.3`, which do not install on the Python
  3.14 on this machine. Loosened to `>=`.
- `run_fit.py` wrote its report to a path that exists only on one cloud machine and would crash a
  laptop refit after rewriting `fitted.json`. Now `model/reports/model-results.md`.
- `--max-seconds` on `main` was 50 minutes; a season with pauses is about 48. PR 3 already made it
  4 hours and kept the socket up through `ready`.

## 4. One model change, passed by the team's own rule

The location sleeve switched on only when the top two of *direct, chain, recurve* separated by
0.15. A storm the filter is confident is a stall or eyewall replacement (29% of storms) never got
the path-conditional value and sat on the unconditional mean. Switching on when the top two of
*all six* classes separate: **$6,336 → $7,337, better in 4 of 4 blocks of 25** (`LOCATION_GATE =
"all"`), and with fills five marks late $2,651 → $3,546, so it passes the late-fill test too. Not a
refit; `fitted.json` is untouched.

## 5. What was checked and found fine

- No look-ahead: manifest labels are poisoned during replay, a future damage report cannot change
  the current state, and the path library excludes the live season on the testnet.
- Feed handling: `seq` dedup, ordering, late-join `age` labelling, the erratic flip rule.
- Momentum / fade: after an official advisory or a wire rumour, the correlation between the first
  20 s of house move and the next 100 s is +0.04 (none). Fading loses. The README's "no momentum
  sleeve" is right.
- Between-round gaps: none (above).
- Backtest reproduces `reports/seasons.csv` exactly.

## 6. Where the alpha is, and what to do next

The edge in this case is **a better and faster fundamental value than the house**, realised at
settlement. Typical approaches at a competition like this, roughly in order of how much they
earn here:

1. **Path before the advisory calls it.** The house follows advisories; the track plus the
   analog-path ensemble separates direct/chain/recurve earlier. This is the location sleeve, the
   largest attributed earner.
2. **Physics before the damage report.** The exposure integral along the observed track predicts
   the final damage to about half the error of the mean. The revision and known-damage sleeves
   trade the two reports, but only at 35% because they die with a late fill.
3. **The insurer layer.** HAVEN is almost always in-layer, so it is a near-linear function of the
   running season loss; REIN only matters in the 17% of seasons near $1,325M. Trade them when the
   total is near a knot, which the bot does.
4. **Rebuilder composition.** Who gets the work depends on *which* sites are hit; long the one who
   does, short the one who does not.
5. **ETF vs basket.** The research ETF-gap sleeve earned about $930 a season on the house marks
   but was never sent. The rules make it better than that: create/redeem is booked *at the marks*,
   so it is a free, mid-price way to put on a basket-shaped position. The direct-hit book is
   literally short ARCH / long REBLD. Creating 50 ARCH gets short 50 PORT, 50 ISLE, 100 AQUA,
   50 GRID, 100 FERRY, 50 HAVEN at mid for one ARCH spread instead of six stock spreads. Test on
   the testnet with quantity 1 and read the booked prices before trusting it.

What matters most now is **execution cost**, which the research never saw: the book turns over
about $2.1M per season, so at 10 bps a side the holdout mean falls from $6,336 to $4,173 and at 25
bps to $928. Widening the deadband does not help (turnover barely moves), because the whole target
vector is re-sent whenever any one name leaves the band. The two things to do on the testnet before
the competition:

1. Record spreads and fill rates for a season (`book-log.jsonl` now has bid, ask, mark, target,
   position and working quantity every second) and put the measured half-spread into the replay.
2. The live runner now holds any name whose target moved less than 15% of its limit instead of
   re-sending the whole vector (`LIVE_PER_NAME_DEADBAND`). Holdout: turnover $2.04M → $1.67M,
   mean $7,337 → $7,170 at zero cost, $5,292 → $5,502 at 10 bps a side, $3,546 → $3,553 five
   marks late. Research is unchanged. A larger path ensemble is in `live_rules.md` section 3.

Also before the day: confirm the clock states the exchange actually sends (`running`, `ready`,
`draft`, `ended` are the ones the code knows), confirm the snapshot field names against one
recorded frame, and agree who on the team is allowed to click in the browser while the bot runs,
since the 180 actions a minute are shared.

## 7. Run

```bash
pip install -r requirements.txt
export CTC_API_KEY=your-bot-key
python -m model.live            # stays up 4 hours, survives ready gaps, replays state/ on restart
python -m unittest discover -s model -p "test_*.py"
```
