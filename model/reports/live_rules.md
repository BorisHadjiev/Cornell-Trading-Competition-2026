# Holdout replay of the live-only rules

Seasons 501–600, the committed `fitted.json`, the research backtest's next-house-mark fill unless a
row says otherwise. Nothing here is a refit. "blocks" counts the four blocks of 25 seasons in which
the row beats the research book (the team's 3-of-4 rule for keeping a change). Turnover is the
season's traded notional; "cost" charges that many basis points of it per side.

The replay harness rebuilds `Models` from `fitted.json` exactly as `model.live` does and
reproduces `reports/seasons.csv` to the cent before any rule is applied.

## 1. The books the live runners sent, and the live-only rules

| Row | Mean | Median | 10th pct | Losing seasons | vs research | Blocks |
|---|---:|---:|---:|---:|---:|---:|
| Research book (`LIVE_BOOK = "full"`, old gate) | 6,336 | 5,824 | 1,207 | 6 | | |
| `main` live: severity sleeve zeroed | 3,440 | 3,361 | 910 | 3 | −2,896 | 0/4 |
| `no_severity`: flat until a path / report view exists | 5,196 | 5,000 | −59 | 11 | −1,140 | 0/4 |
| PR 3 live: location residual on island names only | 1,881 | 1,588 | −220 | 14 | −4,455 | 0/4 |
| Research + flatten at every round close (no spread charged) | 6,334 | 5,812 | 1,169 | 7 | −2 | 1/4 |
| Research + one entry per name per round, then reduce only | 883 | 861 | −880 | 26 | −5,453 | 0/4 |
| Research + both rules | 943 | 834 | −936 | 25 | −5,393 | 0/4 |
| PR 3 stack: location residual + both rules + caps 150/400/800 | 506 | 272 | −360 | 28 | −5,830 | 0/4 |
| Research + caps 150 / 400 / 800 | 3,349 | 2,517 | −665 | 33 | −2,987 | 0/4 |
| Research + caps none / 2,500 / 5,000 | 5,689 | 5,697 | −2,517 | 14 | −647 | 1/4 |
| Research, fill 2 marks late | 3,628 | 3,157 | −2,403 | 16 | | |
| Research, fill 5 marks late | 2,651 | 2,263 | −3,822 | 31 | | |
| Research, 10 bps per side | 4,173 | 3,747 | −1,258 | 17 | −2,163 | |
| Research, 25 bps per side | 928 | 416 | −4,367 | 45 | −5,408 | |
| Research, 5 marks late and 10 bps | 831 | 460 | −5,406 | 46 | | |

Caps: the 150-dollar per-name cap fired 363 times in 100 seasons and the 400-dollar switch in 31;
the 2,500-dollar switch fired in 11 seasons and never recovered (the switch is modelled as flatten
and stop; the backup would add a few hundred). The book's edge is paid at settlement, so a
mid-season drawdown on a fundamental position usually reverts; a mark-to-market stop converts it
into a realised loss.

Why zeroing "severity" is not turning a sleeve off: `severity` is the position implied by the
unconditional storm mean, `location` is only the residual `s_loc − s_sev` on top of it, and the
book is their sum. Without `severity`, after an advisory has moved the mark the residual is the
location view minus a fade of the house.

Season 239 (the HAVEN anecdote behind the round-end flatten): HAVEN closed round 1 at 33.20 and
opened round 2 at 33.22; the move to 43 came inside round 2 on Omar's first advisory. Across the
600 seasons the close-to-open house gap has mean ≈ 0 and standard deviation 0.02–0.08 per stock
(0.24 on ARCH).

## 2. Turnover, cadence, and the location gate

| Row | Mean | Median | 10th pct | Losing | Turnover | vs research | Blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Research book, whole-vector deadband 0.08, 60 s timer | 6,336 | 5,824 | 1,207 | 6 | 2.16M | | |
| Deadband 0.15 | 6,236 | 5,708 | 878 | 8 | 2.10M | −100 | 0/4 |
| Deadband 0.25 | 6,211 | 5,742 | 823 | 7 | 2.02M | −125 | 0/4 |
| Decisions only at feed events (no 60 s timer) | 5,971 | 5,555 | −121 | 11 | 1.74M | −365 | 0/4 |
| Events only, deadband 0.15 | 5,846 | 5,510 | −219 | 11 | 1.68M | −490 | 0/4 |
| **Location gate on all six classes** (`LOCATION_GATE = "all"`) | **7,337** | **7,066** | **1,626** | 6 | 2.04M | **+1,001** | **4/4** |
| Location gate on all classes, fill 2 marks late | 4,656 | 5,026 | −904 | 16 | | +1,028 vs the 2-mark research row | |
| Location gate on all classes, fill 5 marks late | 3,546 | 3,337 | −1,473 | 24 | 1.67M | +895 vs the 5-mark research row | |

The deadband barely moves turnover because the whole target vector is re-sent whenever any one
name leaves the band. The location gate is kept: it is better at the next mark in all four blocks
and better in the late-fill case, and `fitted.json` is untouched.

## 3. Per-name deadband and ensemble size (on the all-class-gate book)

| Row | Mean | Median | 10th pct | Losing | Turnover | vs gate=all book |
|---|---:|---:|---:|---:|---:|---:|
| Gate=all book, whole-vector deadband 0.08 | 7,337 | 7,066 | 1,626 | 6 | 2.04M | |
| Gate=all, 10 bps per side | 5,292 | 5,122 | −117 | 11 | 2.04M | |
| Gate=all, fill 5 marks late | 3,546 | 3,337 | −1,473 | 24 | 1.67M | |
| Per-name deadband 0.08 | 7,228 | 7,040 | 1,396 | 6 | 1.88M | −109 |
| Per-name deadband 0.08, 10 bps | 5,351 | 5,232 | −118 | 11 | 1.88M | +59 vs the 10 bps row |
| Per-name deadband 0.08, 5 marks late | 3,609 | 3,523 | −1,511 | 24 | 1.53M | +63 vs the 5-mark row |
| **Per-name deadband 0.15** (`LIVE_PER_NAME_DEADBAND`) | 7,170 | 7,129 | 1,624 | 6 | 1.67M | −167 |
| Per-name deadband 0.15, 10 bps | 5,502 | 5,379 | 125 | 9 | 1.67M | +210 vs the 10 bps row |
| Per-name deadband 0.15, 5 marks late | 3,553 | 3,373 | −1,383 | 25 | 1.38M | +7 vs the 5-mark row |
| Ensemble 48 paths instead of 16 | 7,498 | 7,406 | 2,066 | 6 | 1.95M | +161 |
| Ensemble 48, per-name 0.15, 10 bps | 5,679 | 5,209 | 133 | 10 | 1.59M | +177 vs the per-name 10 bps row |

The per-name deadband is a live-execution rule, not a research change: at zero cost it gives up a
little, with any spread it earns it back, and it trades 18% less. It is on in `model.live` only.
The larger path ensemble reduces target noise (less churn, better tail) at three times the model
time per decision, which the live loop can afford. Block check: better in only 2 of 4 blocks at
the next mark (+161) and 3 of 4 five marks late (+210), so under the team's 3-of-4 rule
`ENSEMBLE_SIZE` stays at 16. It is the first candidate to retest once testnet spreads are known.

## 4. Insurer gate width (after live season 246)

| Row | Mean | Median | 10th pct | Losing | vs research | Blocks |
|---|---:|---:|---:|---:|---:|---:|
| Research book, KINK_PROXIMITY 0.35, next mark | 7,337 | 7,066 | 1,626 | 6 | | |
| KINK_PROXIMITY 0.6, next mark | 7,886 | 8,257 | 1,229 | 8 | +549 | 4/4 |
| **KINK_PROXIMITY 1.0, next mark** | 7,928 | 8,257 | 1,229 | 8 | +592 | 4/4 |
| Research book, 5 marks late | 3,546 | 3,337 | −1,473 | 24 | | |
| KINK_PROXIMITY 0.6, 5 marks late | 3,769 | 3,676 | −2,745 | 25 | +224 | 3/4 |
| **KINK_PROXIMITY 1.0, 5 marks late** | 3,810 | 3,671 | −2,745 | 26 | +264 | 3/4 |
| Ensemble 48 + kink 1.0, next mark | 8,092 | 8,084 | 1,042 | 7 | +755 | 4/4 |
| Ensemble 48 + kink 1.0, 5 marks late | 3,926 | 4,395 | −2,562 | 25 | +380 | 4/4 |

Applied: `KINK_PROXIMITY = 1.0` (one change for one losing season). The ensemble-48 combination
passes too and is the next candidate once a further season is in.
