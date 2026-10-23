"""Season replay. A decision at a bar fills on the next house mark.

Positions carry across the three rounds. Settlement is applied only at the
final bar, as a scoring price. Manifest labels are not inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import baseline, portfolio
from model.announce import record
from model.config import (
    BASELINE_CLIP,
    BASELINE_ODDS,
    DEADBAND_FRACTION,
    DECISION_EVERY_REAL_SECONDS,
    LOCATION_GAP,
    LOOKALIKE,
    NAMES,
    OUTAGE_TILT,
    PHYSICAL,
    PRE_TURN_FRACTION,
    SEED,
    SLEEVES,
)
from model.damage import integrate_exposure, outage_hours, site_exposure
from model.filter import CLASSES, features_from, first_turn_age
from model.revision import apply_revision

IDX = {name: i for i, name in enumerate(NAMES)}


@dataclass
class Models:
    value: object
    damage: object
    sites: list
    revision_intercept: np.ndarray
    revision_slope: np.ndarray
    loss_intercept: float
    loss_slope: float
    loss_coef: np.ndarray
    filt: object
    library: object
    limits: np.ndarray
    mean_damage: np.ndarray
    mean_outage: np.ndarray
    mean_loss: float
    median_ri: float
    by_class_damage: dict
    by_class_outage: dict
    by_class_loss: dict
    storm_seconds: float
    dissipate_seconds: float
    sites_by_company: dict


def _severity(radius: float, intensity: float, median_ri: float) -> float:
    if median_ri <= 1e-8:
        return 0.0
    return float(np.clip(radius * intensity / median_ri - 1.0, -0.8, 1.5))


def _location_on(probs: np.ndarray) -> bool:
    look = [float(probs[CLASSES.index(name)]) for name in LOOKALIKE]
    ordered = sorted(look)
    return ordered[-1] - ordered[-2] >= LOCATION_GAP


def _path_damage(models: Models, paths: list[np.ndarray], our_intensity: float) -> tuple[np.ndarray, np.ndarray, float]:
    damages = []
    outages = []
    losses = []
    fit = models.damage
    for path in paths:
        if len(path) <= 1:
            continue
        analog_intensity = max(float(path[0, 6]), 0.15)
        scale = our_intensity / analog_intensity
        rest = path[1:]
        # Slope only. The storm-level intercept is added once by the caller.
        damages.append(fit.slope * integrate_exposure(rest, models.sites, fit.form, fit.dt) * scale)
        outages.append(outage_hours(rest, models.sites, fit.form, fit.dt, fit.tau) * scale)
        exposure = site_exposure(rest, models.sites, fit.form, fit.dt)
        losses.append(scale * float(models.loss_coef @ exposure))
    if not damages:
        return np.zeros(5), np.zeros(5), 0.0
    # Average across paths, not the damage of the average path.
    return np.mean(damages, axis=0), np.mean(outages, axis=0), float(np.mean(losses))


def _state_at(season, bar: int, models: Models, rng: np.random.Generator) -> dict:
    """Physical state from the feed and the marks up to this bar. No manifest fields."""
    t = season.times[bar]
    mark = season.prices[bar]
    rnd = int(season.rounds[bar])
    events = [event for event in season.events if event["time"] <= t]
    visible = {}
    for storm in season.storms:
        take = storm.times <= t
        visible[storm.round] = storm.obs[take]

    finals = {}
    prelims = {}
    for event in events:
        if event["type"] != "damage_report":
            continue
        if event["stage"] == "final":
            finals[int(event["round"])] = event
        elif event["stage"] == "preliminary":
            prelims[int(event["round"])] = event

    obs = visible.get(rnd, np.zeros((0, 7)))
    age = float(obs[-1, 0]) if len(obs) else None
    sev = 0.0 if age is None else _severity(float(obs[-1, 5]), float(obs[-1, 6]), models.median_ri)
    official = _latest_official(events)
    # Before the first official advisory the track has not separated the lookalike
    # classes, and the filter is worse than the training frequency. Use the prior.
    if len(obs) and official is not None:
        feats = features_from(obs[-1], events, obs_times_last(season, rnd, t), models.storm_seconds)
        probs = models.filt.predict_proba(feats)
        hazard = models.filt.hazard(feats)
    else:
        probs = models.filt.prior.copy()
        hazard = float(models.filt.prior[CLASSES.index("erratic")])

    location = _location_on(probs) and len(obs) > 0
    mean_d = models.mean_damage * (1.0 + sev)
    mean_o = models.mean_outage * (1.0 + sev)
    mean_l = models.mean_loss * (1.0 + sev)

    loc_d, loc_o, loc_l = mean_d, mean_o, mean_l
    if location:
        paths = models.library.sample_mixture(probs, age,  models_ensemble(), rng, obs[-1, 1:3])
        pulled = [_pull(path, _latest_official(events), models) for path in paths]
        dmg, shut, loss = _path_damage(models, pulled, float(obs[-1, 6]))
        realized_d = models.damage.slope * integrate_exposure(obs, models.sites, models.damage.form, models.damage.dt)
        realized_o = outage_hours(obs, models.sites, models.damage.form, models.damage.dt, models.damage.tau)
        realized_l = float(models.loss_coef @ site_exposure(obs, models.sites, models.damage.form, models.damage.dt))
        loc_d = np.maximum(models.damage.intercept + realized_d + dmg, 0.0)
        loc_o = realized_o + shut
        loc_l = max(realized_l + loss, 0.0)

    rev_d, rev_o, rev_l = loc_d, loc_o, loc_l
    prelim = prelims.get(rnd)
    final = finals.get(rnd)
    if prelim is not None and final is None:
        raw = np.array([float(prelim["damage_pct"].get(name, 0.0)) for name in PHYSICAL])
        rev_d = np.maximum(apply_revision(models.revision_intercept, models.revision_slope, raw), 0.0)
        rev_l = max(models.loss_intercept + models.loss_slope * float(prelim["industry_loss_musd"]), 0.0)

    known_d, known_o, known_l = rev_d, rev_o, rev_l
    if final is not None:
        known_d = np.array([float(final["damage_pct"].get(name, 0.0)) for name in PHYSICAL])
        known_l = float(final["industry_loss_musd"])
        if len(obs):
            known_o = outage_hours(obs, models.sites, models.damage.form, models.damage.dt, models.damage.tau)

    def accumulate(current_d, current_o, current_l):
        damage = np.zeros(5)
        outage = np.zeros(5)
        loss = 0.0
        for round_id in (1, 2, 3):
            if round_id < rnd and round_id in finals:
                event = finals[round_id]
                damage += np.array([float(event["damage_pct"].get(name, 0.0)) for name in PHYSICAL])
                loss += float(event["industry_loss_musd"])
                past = visible.get(round_id, np.zeros((0, 7)))
                if len(past):
                    outage += outage_hours(past, models.sites, models.damage.form, models.damage.dt, models.damage.tau)
                else:
                    outage += models.mean_outage
            elif round_id == rnd:
                damage += current_d
                outage += current_o
                loss += current_l
            else:
                damage += models.mean_damage
                outage += models.mean_outage
                loss += models.mean_loss
        return damage, outage, loss

    turned = first_turn_age(obs, models_turn()) is not None if len(obs) else False
    full_age = models.storm_seconds - models.dissipate_seconds
    fraction = 1.0 if (final is not None or turned or (age is not None and age >= full_age)) else PRE_TURN_FRACTION
    # Born-erratic has no special detector. A high erratic probability is just the filter.
    if hazard > 0.5 and final is None:
        fraction = min(fraction, PRE_TURN_FRACTION + (1.0 - PRE_TURN_FRACTION) * (1.0 - hazard))

    near = _near_kink(accumulate(known_d, known_o, known_l)[2], models)
    return {
        "mark": mark,
        "open": season.open_prices,
        "probs": probs,
        "hazard": hazard,
        "fraction": fraction,
        "near": near,
        "final": final is not None,
        "prelim": prelim is not None and final is None,
        "location": location,
        "sev": accumulate(mean_d, mean_o, mean_l),
        "loc": accumulate(loc_d, loc_o, loc_l),
        "rev": accumulate(rev_d, rev_o, rev_l),
        "known": accumulate(known_d, known_o, known_l),
        "round": rnd,
        "age": age,
    }


def obs_times_last(season, rnd: int, t):
    for storm in season.storms:
        if storm.round == rnd:
            take = storm.times <= t
            if take.any():
                return storm.times[take][-1]
    return t


def _latest_official(events: list[dict]) -> dict | None:
    found = None
    for event in events:
        if event["type"] == "advisory" and event.get("source") == "official":
            found = event
    return found


def _pull(path: np.ndarray, advisory: dict | None, models: Models) -> np.ndarray:
    from model.config import CONE_WEIGHT
    if advisory is None or CONE_WEIGHT <= 0:
        return path
    cone = advisory.get("cone") or []
    if not cone or len(path) == 0:
        return path
    out = path.copy()
    for row in out:
        age = float(row[0])
        point = cone[-1]
        for candidate in cone:
            if float(candidate["age"]) >= age:
                point = candidate
                break
        row[1] = (1.0 - CONE_WEIGHT) * row[1] + CONE_WEIGHT * float(point["x"])
        row[2] = (1.0 - CONE_WEIGHT) * row[2] + CONE_WEIGHT * float(point["y"])
    return out


def models_ensemble() -> int:
    from model.config import ENSEMBLE_SIZE
    return ENSEMBLE_SIZE


def models_turn() -> float:
    from model.config import TURN_DEGREES
    return TURN_DEGREES


def _near_kink(season_loss: float, models: Models) -> bool:
    from model.config import KINK_PROXIMITY
    retention = models.value.retention
    limit = retention + models.value.width
    dist = min(abs(season_loss - retention), abs(season_loss - limit))
    return dist <= KINK_PROXIMITY * max(models.value.width, 1.0)


def _value_shares(models: Models, open_prices, mark, damage, outage, loss, fraction) -> np.ndarray:
    fair = models.value.predict(open_prices, damage, outage, loss)
    return portfolio.shares_from_edge(fair - mark, models.limits, fraction)


def _scenario_value(models: Models, open_prices, rnd: int, finals: dict, class_name: str) -> np.ndarray:
    damage = np.zeros(5)
    outage = np.zeros(5)
    loss = 0.0
    for round_id in (1, 2, 3):
        if round_id < rnd and round_id in finals:
            event = finals[round_id]
            damage += np.array([float(event["damage_pct"].get(name, 0.0)) for name in PHYSICAL])
            loss += float(event["industry_loss_musd"])
            outage += models.mean_outage
        elif round_id == rnd:
            damage += models.by_class_damage[class_name]
            outage += models.by_class_outage[class_name]
            loss += models.by_class_loss[class_name]
        else:
            damage += models.mean_damage
            outage += models.mean_outage
            loss += models.mean_loss
    return models.value.predict(open_prices, damage, outage, loss)


def targets_from_state(state: dict, models: Models, season, bar: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Share targets and the sleeve decomposition. `season` is used only for scenario finals already in the feed."""
    t = season.times[bar]
    finals = {}
    for event in season.events:
        if event["time"] <= t and event["type"] == "damage_report" and event["stage"] == "final":
            finals[int(event["round"])] = event
    fraction = state["fraction"]
    mark = state["mark"]
    opened = state["open"]
    s_sev = _value_shares(models, opened, mark, *state["sev"], fraction)
    s_loc_full = _value_shares(models, opened, mark, *state["loc"], fraction)
    s_rev_full = _value_shares(models, opened, mark, *state["rev"], fraction)
    s_known_full = _value_shares(models, opened, mark, *state["known"], fraction)
    sleeves = {name: np.zeros(len(NAMES)) for name in SLEEVES}
    sleeves["severity"] = s_sev
    sleeves["location"] = s_loc_full - s_sev
    if state["prelim"]:
        sleeves["revision"] = s_rev_full - s_loc_full
    if state["final"]:
        sleeves["known_damage"] = s_known_full - s_loc_full
        sleeves["revision"] = np.zeros(len(NAMES))
    probs = state["probs"]
    duration = float(probs[CLASSES.index("stall")] + probs[CLASSES.index("eyewall")])
    if duration >= LOCATION_GAP:
        tilt = duration * OUTAGE_TILT * fraction
        sleeves["composition"][IDX["GEN"]] += tilt * models.limits[IDX["GEN"]]
        sleeves["composition"][IDX["RECON"]] -= 0.5 * tilt * models.limits[IDX["RECON"]]
    if not state["near"]:
        for shares in sleeves.values():
            shares[IDX["HAVEN"]] = 0.0
            shares[IDX["REIN"]] = 0.0
    fair = models.value.predict(opened, *state["known"])
    stat = models.value.statistical(opened, *state["known"])
    fundamental = portfolio.disagree_scale(sum(sleeves.values()), fair, stat, mark)
    # Put the disagree cut back into the sleeves in proportion.
    sleeves = portfolio._scale_to_match(sleeves, fundamental)
    etf = portfolio.etf_mark_sleeve(mark, models.limits, fraction)
    etf = portfolio.fit_inside_limits(fundamental, etf, models.limits)
    sleeves["etf_gap"] = etf
    scenario_values = {
        name: _scenario_value(models, opened, state["round"], finals, name)
        for name in ("direct", "chain", "recurve", "erratic")
    }
    edges = fair - mark
    edges[IDX["ARCH"]] = mark[IDX["ARCH"]] - sum(q * mark[IDX[n]] for n, q in portfolio.ARCH_BASKET.items())
    edges[IDX["REBLD"]] = mark[IDX["REBLD"]] - sum(q * mark[IDX[n]] for n, q in portfolio.REBLD_BASKET.items())
    sleeves = portfolio.apply_scenario_caps(
        sleeves, mark, scenario_values, scenario_values["direct"], models.limits, edges,
    )
    total = sum(sleeves.values())
    total = np.clip(total, -models.limits, models.limits)
    sleeves = portfolio._scale_to_match(sleeves, total)
    _rebucket(sleeves)
    # Fill-delay check: revision and known damage collapse by a 5-mark fill.
    # Size those two sleeves at the fractional knob. Do not scale severity up.
    # Do not touch location or insurer_kink. Applied after rebucket so the kink
    # book is not reduced. Spread-crossing is not a sleeve here; the fill is the
    # house mark. A live book would be required before crossing a spread.
    sleeves["revision"] *= PRE_TURN_FRACTION
    sleeves["known_damage"] *= PRE_TURN_FRACTION
    total = sum(sleeves.values())
    return total, sleeves


def _rebucket(sleeves: dict[str, np.ndarray]) -> None:
    """Report rebuilders as composition and insurers as the kink, without changing the book."""
    donors = ("severity", "location", "revision", "known_damage")
    for name in ("HAVEN", "REIN"):
        i = IDX[name]
        for src in donors:
            sleeves["insurer_kink"][i] += sleeves[src][i]
            sleeves[src][i] = 0.0
    for name in ("RECON", "MAT", "GEN"):
        i = IDX[name]
        for src in donors:
            sleeves["composition"][i] += sleeves[src][i]
            sleeves[src][i] = 0.0


def decision_bars(season) -> list[int]:
    bars = {0}
    step = max(1, int(DECISION_EVERY_REAL_SECONDS / 2))
    n = len(season.times)
    for storm in season.storms:
        if len(storm.times) == 0:
            continue
        start = int(np.searchsorted(season.times, storm.times[0], side="left"))
        end = int(np.searchsorted(season.times, storm.times[-1], side="right"))
        for bar in range(start, min(end, n - 1), step):
            bars.add(bar)
    for event in season.events:
        if event["type"] not in ("advisory", "impact", "damage_report", "announcement"):
            continue
        bar = int(np.searchsorted(season.times, event["time"], side="left"))
        if bar < n - 1:
            bars.add(bar)
    return sorted(bars)


def run_season(season, models: Models, rng: np.random.Generator) -> dict:
    n = len(season.times)
    position = np.zeros(len(NAMES))
    cash = 0.0
    sleeves = {name: np.zeros(len(NAMES)) for name in SLEEVES}
    held = np.zeros((n, len(SLEEVES), len(NAMES)))
    sleeve_names = list(SLEEVES)
    pending = None
    decisions = set(decision_bars(season))
    base_pos = np.zeros(len(NAMES))
    base_cash = 0.0
    base_pending = None
    sold: set[tuple[int, str]] = set()
    seen_base: set[int] = set()
    announcements: list = []
    limits = {name: float(models.limits[IDX[name]]) for name in NAMES}

    # Causality guard: manifest damage is not an input. Poison it and keep going.
    poisoned = []
    for storm in season.storms:
        poisoned.append((storm.final_damage, storm.pattern, storm.flip))
        storm.final_damage = {name: 1e9 for name in PHYSICAL}
        storm.pattern = "poison"
        storm.flip = 0.0

    try:
        for bar in range(n):
            if pending is not None and pending["bar"] == bar:
                trade = pending["target"] - position
                cash -= float(trade @ season.prices[bar])
                position = pending["target"]
                sleeves = pending["sleeves"]
                pending = None
            if base_pending is not None and base_pending["bar"] == bar:
                trade = base_pending["target"] - base_pos
                base_cash -= float(trade @ season.prices[bar])
                base_pos = base_pending["target"]
                base_pending = None
            for s_index, name in enumerate(sleeve_names):
                held[bar, s_index] = sleeves[name]

            if bar not in decisions or bar + 1 >= n:
                continue
            for event in season.events:
                if event["time"] == season.times[bar] and event["type"] == "announcement":
                    # Live stand-down: record only. No research sleeve is added.
                    record(event, announcements)
            state = _state_at(season, bar, models, rng)
            target, sleeve_target = targets_from_state(state, models, season, bar)
            if not portfolio.deadband_holds(position, target, models.limits, DEADBAND_FRACTION):
                pending = {"bar": bar + 1, "target": target, "sleeves": sleeve_target}

            rnd = int(season.rounds[bar])
            for event_index, event in enumerate(season.events):
                if event["time"] > season.times[bar] or event_index in seen_base:
                    continue
                if event["type"] != "advisory" or event.get("round") != rnd:
                    continue
                seen_base.add(event_index)
                names = baseline.symbols_to_sell(event, models.sites_by_company, {s for r, s in sold if r == rnd}, BASELINE_ODDS)
                if not names:
                    continue
                new_base = base_pos.copy() if base_pending is None else base_pending["target"].copy()
                for symbol in names:
                    sold.add((rnd, symbol))
                    new_base[IDX[symbol]] = baseline.clip_shares(symbol, limits, BASELINE_CLIP)
                base_pending = {"bar": bar + 1, "target": new_base}
    finally:
        for storm, (damage, pattern, flip) in zip(season.storms, poisoned):
            storm.final_damage = damage
            storm.pattern = pattern
            storm.flip = flip

    settlement = season.settlement
    prices = season.prices
    our_pnl = cash + float(position @ settlement)
    base_pnl = base_cash + float(base_pos @ settlement)
    diff = np.diff(prices, axis=0)
    attributed = {}
    for s_index, name in enumerate(sleeve_names):
        mark_to_market = float(np.sum(held[:-1, s_index] * diff))
        tail = float(held[-1, s_index] @ (settlement - prices[-1]))
        attributed[name] = mark_to_market + tail
    return {
        "pnl": our_pnl,
        "baseline_pnl": base_pnl,
        "attributed": attributed,
        "announcements": announcements,
        "position": position,
    }
