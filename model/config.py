"""Policy knobs. Estimated economics live in model/fitted.json, not here.

Baseline odds and clip are the example bot's published defaults, not a fit.
"""

SEED = 2026
HOLDOUT_START = 501  # fit seasons 1..500, report 501..600

ENSEMBLE_SIZE = 16
DEADBAND_FRACTION = 0.08
PRE_TURN_FRACTION = 0.35
MIN_ETF_GAP = 1.0
EDGE_DOLLARS_FOR_FULL_SIZE = 3.0
LOCATION_GAP = 0.15
# The location sleeve switches on when the filter separates the path. "lookalike" is the original
# gate: the top two of direct, chain and recurve must differ by LOCATION_GAP. "all" also switches
# on when the top two of all six classes differ by LOCATION_GAP, so a storm the filter is sure is
# a stall or an eyewall replacement (29% of storms) gets the path-conditional value instead of the
# unconditional mean. Holdout replay, next-mark fill: 6,336 -> 7,337, better in 4 of 4 blocks of 25.
LOCATION_GATE = "all"
SCENARIO_CAP_DOLLARS = 4000.0
# Insurer sleeve gate: HAVEN/REIN are traded when the estimated season loss is within this
# fraction of the layer width of a knot. 0.35 left a dead zone in the middle of HAVEN's layer
# (732-1006M) where the sleeve was switched off; live season 246 (24 Oct) lost its HAVEN short
# there right before the payoff. 1.0 keeps the sleeve on across the whole layer. Holdout replay:
# next-mark mean 7,337 -> 7,928 (4/4 blocks), 5-mark 3,546 -> 3,810 (3/4 blocks); the 10th
# percentile is worse (1,626 -> 1,229), so this buys expectation with some tail.
KINK_PROXIMITY = 1.0
TURN_DEGREES = 25.0
OUTAGE_TILT = 0.15
CONE_WEIGHT = 0.25
DISAGREE_SCALE = 0.5
DECISION_EVERY_REAL_SECONDS = 60
HOUSE_HORIZON_REAL_SECONDS = 90
QUOTE_NOISE_DOLLARS = 0.05

BASELINE_ODDS = 0.8
BASELINE_CLIP = 0.1

# Live only. The research backtest does not read these, and they are not sleeves.
# Announcements are absent from the research seasons: stand down rather than invent a trade.
# Spread crossing needs a book. market.csv has no spread, so research fills stay on the house mark.
LIVE_ANNOUNCEMENT_STAND_DOWN = True
LIVE_SPREAD_CROSSING = False
# Create/redeem stays off. There is no live ETF exception.
LIVE_ETF_ORDERS = False

# Which book the live process sends. Holdout replay in model/reports/live_rules.md.
#   "full"         the research book exactly: 6,336 a season with the old gate, 7,337 with LOCATION_GATE "all"
#   "no_severity"  the research book, but flat until a path, preliminary or final view exists: 5,196
#   "location"     the PR 3 experiment, the location residual on the island names only: 1,881
# The old live change "zero the severity sleeve" is not a mode: it left the location residual minus
# a fade of the house and scored 3,440.
LIVE_BOOK = "full"

# Round handling. The research backtest carries positions across rounds, and the house mark
# moves by well under a tenth of a dollar between a round close and the next open (600 seasons,
# std 0.02-0.08 per stock). Flattening at the bell and re-entering at the open pays two spreads
# for no protection, and the one-entry rule blocks the post-turn add that the PORT/ARCH reserve
# exists for. Both stay available as flags for a testnet comparison.
LIVE_FLATTEN_AT_ROUND_END = False
LIVE_ONE_ENTRY_PER_ROUND = False

# Loss bounds, in case dollars of exchange mark (realized plus unrealized). None disables one.
# Holdout replay (model/reports/live_rules.md): the book's edge is paid at settlement, so a
# mid-season drawdown usually reverts. A 150-dollar per-name cap fires on ordinary advisory
# noise (PORT moves about 0.9 dollars per advisory) and cuts the mean from 6,336 to 3,349; a
# 2,500-dollar switch to the backup fired in 11 of 100 seasons and locked in losses the book
# would have recovered (10th percentile 1,207 to -2,517). The one stop that stays is a circuit
# breaker for a broken model: the worst holdout season was -3,845.
NAME_LOSS_CAP = None
BOOK_SWITCH_LOSS = None
SEASON_STOP_LOSS = 6000.0

# Live only: a name whose target is within this fraction of its limit of the current position is
# held instead of re-sent. The research deadband is all-or-nothing on the whole vector, so one
# name leaving the band re-sends every name. Holdout replay: turnover 2.04M -> 1.67M a season,
# mean 7,337 -> 7,170 at zero cost and 5,292 -> 5,502 at 10 bps a side. None turns it off.
LIVE_PER_NAME_DEADBAND = 0.15

# Live only: a crossing reduction takes at most the size shown at the touch and rests the
# remainder one tick inside our side. The house shows 14-60 shares at the touch and 600 about
# four dollars away, so a 200-share cross walks the book. Live seasons 243-247: the gap between
# the season mark and position-times-mid was -300 to -1,000 a season, most of it from crosses.
LIVE_CROSS_AT_TOUCH_DEPTH = True

# Live only: every target is scaled by this before the deadband and the edge floor. The research
# sizing (tanh of edge over 3 dollars, times the limit) reaches the position limits on a 5-dollar
# edge, and live season 250 (24 Oct) rode a wrong-way book at the limits to -4,548. Half size
# halves both the expectation and the swing; it is a variance control, not a fit.
LIVE_SIZE_FRACTION = 0.5

# Live only: an edge floor. A name is opened, or reversed through flat, only when the model's
# edge against the live mid is at least the larger of LIVE_EDGE_FLOOR_DOLLARS and
# LIVE_EDGE_FLOOR_SPREADS times the quoted spread. Reductions and same-side adds are not gated.
# Overnight 24 Oct, 8 testnet seasons: live mids sit about $1.3 from the research house marks on
# PORT and HAVEN (90th percentile $3-5), the size of a typical edge, and the book reversed 10-17
# times a season trading that noise; the model view on the live target path earned about 900 a
# season against 8,000 on the research marks. The research replay has no other participants and
# cannot score this rule; it is judged on live seasons. None turns it off.
LIVE_EDGE_FLOOR_DOLLARS = 1.0
LIVE_EDGE_FLOOR_SPREADS = 2.0

# Resting orders. A rest is cancelled when the quote has moved away from it, but never sooner
# than MIN_REST_SECONDS: the fitted house half-life after a damage report is 4 s, and a 4-second
# cancel-and-replace loop on ten names would exhaust the team's 180 actions per minute.
MIN_REST_SECONDS = 20.0
# Rest one tick inside the house touch so our order is first in the queue for house flow.
# Capped at a quarter of the spread so a tight book is not crossed by accident.
REST_IMPROVE = 0.01

# Case packet, section 3: the house price at the first open of the live season. The value model
# is anchored on the season open. On the testnet the open comes from the matched research season
# or the first quotes; in the live competition, or after a restart in round 2 or 3 with no saved
# open, this table is the anchor.
PACKET_OPEN = {
    "PORT": 72.48, "ISLE": 50.26, "AQUA": 27.33, "GRID": 36.41, "FERRY": 17.91,
    "RECON": 52.38, "MAT": 35.81, "GEN": 40.30, "HAVEN": 39.55, "REIN": 89.36,
    "ARCH": 289.20, "REBLD": 216.68,
}

# Structural groups from the case text and the architecture, not estimated.
CLASSES = ("direct", "chain", "recurve", "stall", "eyewall", "erratic")
PHYSICAL = ("PORT", "ISLE", "AQUA", "GRID", "FERRY")
REBUILDERS = ("RECON", "MAT", "GEN")
INSURERS = ("HAVEN", "REIN")
NAMES = PHYSICAL + REBUILDERS + INSURERS + ("ARCH", "REBLD")
CASCADIA_BUDGET = ("PORT", "GRID", "FERRY", "ARCH")
LOOKALIKE = ("direct", "chain", "recurve")
ARCH_BASKET = {"PORT": 1, "ISLE": 1, "AQUA": 2, "GRID": 1, "FERRY": 2, "HAVEN": 1}
REBLD_BASKET = {"RECON": 2, "MAT": 2, "GEN": 1}

SLEEVES = (
    "severity",
    "location",
    "insurer_kink",
    "composition",
    "revision",
    "known_damage",
    "etf_gap",
)
