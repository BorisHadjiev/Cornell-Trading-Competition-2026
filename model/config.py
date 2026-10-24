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
SCENARIO_CAP_DOLLARS = 4000.0
KINK_PROXIMITY = 0.35
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

# Pre-set loss bounds. Not estimated, not fit on seasons 1–500 or 501–600,
# and not chosen by looking at a backtest table.
NAME_LOSS_CAP = 150.0
BOOK_SWITCH_LOSS = 400.0
SEASON_STOP_LOSS = 800.0

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
