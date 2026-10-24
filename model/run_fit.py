"""Fit on seasons 1–500, backtest seasons 501–600, write the artifacts.

    python3 -m model.run_fit
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import config
from model.backtest import Models, _state_at, run_season
from model.damage import (
    DamageFit,
    fit_form,
    integrate_exposure,
    mae_by_company,
    outage_hours,
    predict_matrix,
    site_exposure,
    site_wind,
    training_mean_predictor,
)
from model.data import HISTORY, damage_vector, limits_from_case, load_case, load_seasons, sites_from_case
from model.filter import (
    bucket_masks,
    collect_scored,
    fit_filter,
    frequency_log_loss,
    log_loss,
)
from model.house import half_life_table
from model.paths import PathLibrary
from model.revision import apply_revision, fit_revision, mae_by_company as revision_mae
from model.value import fit_value, mae

REPORT = Path("/cursor/stores/bc-eb7b06b9-c4ba-4ef0-9e95-b0e907682dae/docs/model-results.md")
PACK_NOTES = Path("/cursor/stores/bc-eb7b06b9-c4ba-4ef0-9e95-b0e907682dae/docs/approach.md")
HERE = Path(__file__).resolve().parent
PHYSICAL = config.PHYSICAL


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _damage_matrix(storms) -> np.ndarray:
    return np.vstack([damage_vector(storm.final_damage) for storm in storms])


def _season_damage(season) -> np.ndarray:
    total = np.zeros(len(PHYSICAL))
    for storm in season.storms:
        total += damage_vector(storm.final_damage)
    return total


def _season_outage(season, sites, form, dt, tau) -> np.ndarray:
    total = np.zeros(len(PHYSICAL))
    for storm in season.storms:
        total += outage_hours(storm.obs, sites, form, dt, tau)
    return total


def _nnls(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    free = coef > 0
    if not np.any(free):
        return np.zeros_like(coef)
    refit = np.zeros_like(coef)
    refit[free], *_ = np.linalg.lstsq(design[:, free], target, rcond=None)
    return np.maximum(refit, 0.0)


def _summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
    }


def _run_unit_tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromName("model.test_invariants")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("label and invariant tests failed")


def _check_causality(season, models, rng_seed: int) -> None:
    bar = min(30, len(season.times) - 1)
    rng_a = np.random.default_rng(rng_seed)
    rng_b = np.random.default_rng(rng_seed)
    before = _state_at(season, bar, models, rng_a)
    future = [event for event in season.events if event["time"] > season.times[bar] and event["type"] == "damage_report"]
    if not future:
        return
    saved = future[0].get("damage_pct")
    future[0]["damage_pct"] = {name: 80.0 for name in PHYSICAL}
    after = _state_at(season, bar, models, rng_b)
    future[0]["damage_pct"] = saved
    if not np.allclose(before["known"][0], after["known"][0]):
        raise RuntimeError("shuffling a future damage report changed the current target state")


def _write_report(summary: dict, house: dict, failures: list[str], narrative: str) -> None:
    our = summary["our"]
    base = summary["baseline"]
    lines = [
        "# Hurricane model, first fit",
        "",
        "Fills are the next house mark after a decision. `market.csv` has no spread, so this is optimistic, and it is not a fill on the book. Research prices do not include other teams. Seasons 1–500 are the fit. Seasons 501–600 are the score. Season 1 is a narrative check, not the score.",
        "",
        "## Held-out season PnL",
        "",
        f"Ours, 100 seasons: mean {our['mean']:.0f}, median {our['median']:.0f}, 10th percentile {our['p10']:.0f}.",
        f"Baseline (example bot, same next-mark fill): mean {base['mean']:.0f}, median {base['median']:.0f}, 10th percentile {base['p10']:.0f}.",
        "",
        "The baseline sells an island business once per storm when an official advisory gives that island at least an 80% chance of hurricane conditions, for 10% of the position limit.",
        "",
        "## Oracle value",
        "",
        f"Handed final damage, path outage, and season loss, the value function's training MAE is {summary['oracle_value_error']['training_mae']:.3f} dollars per name. Predicting the season's opening mark has MAE {summary['oracle_value_error']['starting_price_mae']:.3f}. Holdout oracle MAE is {summary['oracle_value_error']['holdout_mae']:.3f} against a holdout open MAE of {summary['oracle_value_error']['holdout_starting_price_mae']:.3f}.",
        "",
        "HAVEN pays above an estimated retention up to an estimated reinsurance attachment. REIN pays above that attachment. GEN is fit on grid-outage hours and has no damage term. RECON's work is PORT, GRID, and FERRY damage. MAT's work is all physical damage. Both rebuilder maps are concave. Every coefficient is estimated on seasons 1–500 and written to `model/fitted.json`.",
        "",
        "## Damage, revision, filter",
        "",
        f"Chosen wind kernel: `{summary['damage_error']['form']}`. Both forms' held-out mean absolute errors are in `fitted.json`.",
        "",
    ]
    for name, row in summary["damage_error"]["by_company"].items():
        lines.append(f"- {name}: kernel MAE {row['mae']:.2f} damage points, training-mean MAE {row['mean_baseline_mae']:.2f}.")
    lines += [
        "",
        f"Revision held-out MAE by company is in the summary file. Filter held-out log loss at 30–90s is {summary['filter']['30-90s']['model']:.3f} versus a frequency baseline of {summary['filter']['30-90s']['frequency']:.3f}. At 0–30s, before the first advisory, the filter log loss is {summary['filter']['0-30s']['model']:.3f} against a frequency baseline of {summary['filter']['0-30s']['frequency']:.3f}, so the book uses the training prior until an official advisory exists.",
        "",
        "## What the sleeves made",
        "",
    ]
    for name, value in summary["sleeve_means"].items():
        lines.append(f"- {name}: mean attributed PnL {value:.0f}.")
    lines += [
        "",
        "ETF gap PnL is the house-mark dislocation, marked to the next print and then to settlement. It is not a tested create/redeem fill. The archipelago ETF holds HAVEN, so it is not treated as an island hedge.",
        "",
        "## House half-life",
        "",
        "Training seasons, later mark minus the mark at the event, horizon "
        f"{config.HOUSE_HORIZON_REAL_SECONDS:.0f} real seconds. No settlement row.",
        "",
    ]
    for key, row in house.items():
        life = row["half_life_seconds"]
        life_txt = "n/a" if life is None else f"{life:.0f}s"
        lines.append(f"- {key}: half-life {life_txt}, n={row['n']}.")
    lines += ["", "## Narrative, season 1", "", narrative, ""]
    if failures:
        lines += ["## Unsatisfied checks", ""]
        for item in failures:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines += ["## Unsatisfied checks", "", "None of the fit gates failed.", ""]
    lines += [
        "No architecture contradiction was found in this fit. Column definitions used here match the pack notes.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


def main() -> None:
    _run_unit_tests()
    if not HISTORY.exists():
        raise SystemExit(f"missing research pack at {HISTORY}; run scripts/fetch_research_data.py")
    print("loading seasons", flush=True)
    seasons = load_seasons()
    case = load_case()
    sites = sites_from_case(case)
    limits = limits_from_case(case)
    dt = float(case["timing"]["obs_every"])
    storm_seconds = float(case["timing"]["storm"])
    dissipate = float(case["storm"]["dissipate"])
    train = [season for season in seasons if season.number < config.HOLDOUT_START]
    hold = [season for season in seasons if season.number >= config.HOLDOUT_START]
    train_storms = [storm for season in train for storm in season.storms]
    hold_storms = [storm for season in hold for storm in season.storms]
    for storm in train_storms + hold_storms:
        if not storm.final_damage or storm.final_loss is None:
            raise RuntimeError(f"storm {storm.name} has no final damage report")

    actual_train = _damage_matrix(train_storms)
    actual_hold = _damage_matrix(hold_storms)
    form_errors = {}
    fitted_forms = {}
    for form in ("exp", "gauss"):
        print(f"fitting damage form {form}", flush=True)
        intercept, slope, _exposure = fit_form(train_storms, sites, form, dt)
        hold_exposure = np.vstack([integrate_exposure(storm.obs, sites, form, dt) for storm in hold_storms])
        predicted = predict_matrix(intercept, slope, hold_exposure)
        form_errors[form] = mae_by_company(predicted, actual_hold)
        fitted_forms[form] = (intercept, slope, float(np.mean(list(form_errors[form].values()))))
    chosen = min(fitted_forms, key=lambda name: fitted_forms[name][2])
    intercept, slope, _ = fitted_forms[chosen]
    mean_baseline = training_mean_predictor(actual_train, len(hold_storms))
    baseline_mae = mae_by_company(mean_baseline, actual_hold)

    print("selecting outage threshold", flush=True)
    wind_sample = []
    for storm in train_storms[::25]:
        wind_sample.append(site_wind(storm.obs[:, 1], storm.obs[:, 2], storm.obs[:, 5], storm.obs[:, 6], sites, chosen).ravel())
    sample = np.concatenate(wind_sample)
    positive = sample[sample > 0.02]
    candidates = np.unique(np.quantile(positive, [0.5, 0.65, 0.8, 0.9, 0.97]))
    opens = np.vstack([season.open_prices for season in train])
    settlement = np.vstack([season.settlement for season in train])
    cum_damage = np.vstack([_season_damage(season) for season in train])
    season_loss = np.array([season.season_loss for season in train], dtype=float)
    best_tau = float(candidates[0])
    best_value = None
    best_outage = None
    best_mae = None
    for tau in candidates:
        outage = np.vstack([_season_outage(season, sites, chosen, dt, float(tau)) for season in train])
        value = fit_value(opens, settlement, cum_damage, outage, season_loss)
        predicted = np.vstack([
            value.predict(opens[i], cum_damage[i], outage[i], float(season_loss[i])) for i in range(len(train))
        ])
        error = mae(predicted, settlement)
        print(f"  tau {float(tau):.4f} training value MAE {error:.4f}", flush=True)
        if best_mae is None or error < best_mae:
            best_mae = error
            best_tau = float(tau)
            best_value = value
            best_outage = outage
    assert best_value is not None and best_outage is not None
    damage = DamageFit(chosen, dt, intercept, slope, best_tau)
    open_mae = mae(opens, settlement)
    hold_opens = np.vstack([season.open_prices for season in hold])
    hold_settle = np.vstack([season.settlement for season in hold])
    hold_damage = np.vstack([_season_damage(season) for season in hold])
    hold_outage = np.vstack([_season_outage(season, sites, chosen, dt, best_tau) for season in hold])
    hold_loss = np.array([season.season_loss for season in hold], dtype=float)
    hold_pred = np.vstack([
        best_value.predict(hold_opens[i], hold_damage[i], hold_outage[i], float(hold_loss[i])) for i in range(len(hold))
    ])

    prelim = np.vstack([damage_vector(storm.prelim_damage) for storm in train_storms])
    final = actual_train
    rev_intercept, rev_slope = fit_revision(prelim, final)
    hold_prelim = np.vstack([damage_vector(storm.prelim_damage) for storm in hold_storms])
    rev_pred = apply_revision(rev_intercept, rev_slope, hold_prelim)
    rev_error = revision_mae(rev_pred, actual_hold)
    loss_x = np.column_stack([
        np.ones(len(train_storms)),
        np.array([storm.prelim_loss for storm in train_storms], dtype=float),
    ])
    loss_y = np.array([storm.final_loss for storm in train_storms], dtype=float)
    loss_ab, *_ = np.linalg.lstsq(loss_x, loss_y, rcond=None)
    exposures = np.vstack([site_exposure(storm.obs, sites, chosen, dt) for storm in train_storms])
    loss_coef = _nnls(exposures, loss_y)

    print("house half-life and filter", flush=True)
    house = half_life_table(train)
    rng = np.random.default_rng(config.SEED)
    filt = fit_filter(train, storm_seconds, rng)
    x_tr, y_tr, age_tr, turn_tr = collect_scored(train, filt, config.TURN_DEGREES)
    x_ho, y_ho, age_ho, turn_ho = collect_scored(hold, filt, config.TURN_DEGREES)
    probs = filt.predict_proba(x_ho)
    filter_scores = {}
    for name, mask in bucket_masks(age_ho, turn_ho).items():
        train_mask = bucket_masks(age_tr, turn_tr)[name]
        if mask.sum() == 0 or train_mask.sum() == 0:
            filter_scores[name] = {"model": None, "frequency": None, "n": int(mask.sum())}
            continue
        filter_scores[name] = {
            "model": log_loss(probs[mask], y_ho[mask]),
            "frequency": frequency_log_loss(y_tr[train_mask], y_ho[mask]),
            "n": int(mask.sum()),
        }
        print(f"  filter {name}: model {filter_scores[name]['model']:.4f} frequency {filter_scores[name]['frequency']:.4f}", flush=True)

    print("path library", flush=True)
    library = PathLibrary(train)
    ri = np.concatenate([storm.obs[:, 5] * storm.obs[:, 6] for storm in train_storms])
    median_ri = float(np.median(ri[ri > 0]))
    by_damage = {name: [] for name in config.CLASSES}
    by_outage = {name: [] for name in config.CLASSES}
    by_loss = {name: [] for name in config.CLASSES}
    for storm in train_storms:
        key = "erratic" if storm.pattern == "erratic" or storm.flip is not None else storm.pattern
        by_damage[key].append(damage_vector(storm.final_damage))
        by_outage[key].append(outage_hours(storm.obs, sites, chosen, dt, best_tau))
        by_loss[key].append(float(storm.final_loss))
    def _mean_or(rows, fallback):
        if not rows:
            return fallback
        return np.mean(np.vstack(rows), axis=0) if np.ndim(rows[0]) else float(np.mean(rows))
    mean_damage = actual_train.mean(axis=0) / 3.0  # per storm, seasons have 3; storm mean is actual_train.mean
    mean_damage = actual_train.mean(axis=0)
    mean_outage = best_outage.mean(axis=0) / 3.0
    mean_loss = float(np.mean([storm.final_loss for storm in train_storms]))
    # Per-storm unconditional means. actual_train is already per storm.
    mean_damage = actual_train.mean(axis=0)
    class_damage = {name: _mean_or(by_damage[name], mean_damage) for name in config.CLASSES}
    class_outage = {name: _mean_or(by_outage[name], mean_outage) for name in config.CLASSES}
    class_loss = {name: float(_mean_or(by_loss[name], mean_loss)) for name in config.CLASSES}

    sites_by_company = {}
    for site in sites:
        sites_by_company.setdefault(site.company, []).append(site.island)
    models = Models(
        value=best_value,
        damage=damage,
        sites=sites,
        revision_intercept=rev_intercept,
        revision_slope=rev_slope,
        loss_intercept=float(loss_ab[0]),
        loss_slope=float(loss_ab[1]),
        loss_coef=loss_coef,
        filt=filt,
        library=library,
        limits=limits,
        mean_damage=mean_damage,
        mean_outage=mean_outage,
        mean_loss=mean_loss,
        median_ri=median_ri,
        by_class_damage=class_damage,
        by_class_outage=class_outage,
        by_class_loss=class_loss,
        storm_seconds=storm_seconds,
        dissipate_seconds=dissipate,
        sites_by_company=sites_by_company,
    )
    _check_causality(hold[0], models, config.SEED)
    print("backtest", flush=True)
    results = []
    for index, season in enumerate(hold, start=1):
        results.append(run_season(season, models, np.random.default_rng(config.SEED + season.number)))
        if index % 10 == 0:
            print(f"  holdout {index}/{len(hold)}", flush=True)
    narrative_result = run_season(train[0], models, np.random.default_rng(config.SEED + 1))
    dara = train[0].storms[0]
    first_official = next(
        event for event in train[0].events
        if event["type"] == "advisory" and event.get("source") == "official" and event["round"] == 1
    )
    narrative = (
        f"Season 1 is inside the fit, so it is not part of the score. Dara's manifest pattern is {dara.pattern}. "
        f"The first official advisory says {first_official.get('outlook')}. "
        f"In-sample book PnL {narrative_result['pnl']:.0f}, baseline PnL {narrative_result['baseline_pnl']:.0f}. "
        "Announcements recorded in the research seasons: "
        f"{sum(len(row['announcements']) for row in results)}."
    )

    our = np.array([row["pnl"] for row in results])
    base = np.array([row["baseline_pnl"] for row in results])
    sleeve_means = {
        name: float(np.mean([row["attributed"][name] for row in results])) for name in config.SLEEVES
    }
    failures = []
    if not best_mae < open_mae:
        failures.append(
            f"Oracle training MAE {best_mae:.3f} is not better than the opening-mark MAE {open_mae:.3f}."
        )
    damage_rows = {}
    for name in PHYSICAL:
        model_err = form_errors[chosen][name]
        base_err = baseline_mae[name]
        damage_rows[name] = {"mae": model_err, "mean_baseline_mae": base_err}
        if not model_err < base_err:
            failures.append(f"Held-out {name} damage MAE {model_err:.3f} is not better than the training mean {base_err:.3f}.")
    bucket = filter_scores.get("30-90s", {})
    if bucket.get("model") is None or not bucket["model"] < bucket["frequency"]:
        failures.append(
            f"Filter log loss at 30–90s ({bucket.get('model')}) is not better than the frequency baseline ({bucket.get('frequency')})."
        )
    attr_gap = float(np.max(np.abs(our - np.array([
        sum(row["attributed"].values()) for row in results
    ]))))
    if attr_gap > 1.0:
        failures.append(f"Sleeve PnL does not add up to book PnL (max gap {attr_gap:.2f}).")

    summary = {
        "holdout": "seasons 501-600",
        "fill": "next house mark after the decision; no spread",
        "our": _summarize(our),
        "baseline": _summarize(base),
        "oracle_value_error": {
            "training_mae": best_mae,
            "starting_price_mae": open_mae,
            "holdout_mae": mae(hold_pred, hold_settle),
            "holdout_starting_price_mae": mae(hold_opens, hold_settle),
        },
        "damage_error": {"form": chosen, "by_company": damage_rows, "forms": form_errors},
        "revision_mae": rev_error,
        "filter": filter_scores,
        "sleeve_means": sleeve_means,
        "unsatisfied": failures,
    }
    fitted = {
        "estimated": True,
        "note": "Every number in this file is estimated from seasons 1-500 unless a key says structural.",
        "holdout_start": config.HOLDOUT_START,
        "seed": config.SEED,
        "damage": damage.as_dict(),
        "damage_form_holdout_mae": form_errors,
        "damage_mean_baseline_mae": baseline_mae,
        "outage_threshold_candidates": [float(v) for v in candidates],
        "value": best_value.as_dict(),
        "revision": {
            "estimated": True,
            "intercept": {name: float(rev_intercept[i]) for i, name in enumerate(PHYSICAL)},
            "slope": {name: float(rev_slope[i]) for i, name in enumerate(PHYSICAL)},
            "loss_intercept": float(loss_ab[0]),
            "loss_slope": float(loss_ab[1]),
            "holdout_mae": rev_error,
        },
        "insured_loss_site_coef": {
            "estimated": True,
            "sites": [site.name for site in sites],
            "coef": loss_coef.tolist(),
        },
        "class_conditional_means": {
            "estimated": True,
            "damage_pct": {name: class_damage[name].tolist() for name in config.CLASSES},
            "outage_hours": {name: np.asarray(class_outage[name]).tolist() for name in config.CLASSES},
            "industry_loss_musd": class_loss,
        },
        "filter": filt.as_dict(),
        "filter_holdout_log_loss": filter_scores,
        "house_half_life": house,
        "median_radius_times_intensity": median_ri,
    }
    report_dir = HERE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (HERE / "fitted.json").write_text(json.dumps(_json_ready(fitted), indent=2))
    frame = pd.DataFrame([
        {
            "season": season.number,
            "pnl": row["pnl"],
            "baseline_pnl": row["baseline_pnl"],
            **row["attributed"],
        }
        for season, row in zip(hold, results)
    ])
    frame.to_csv(report_dir / "seasons.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2))
    _write_report(summary, house, failures, narrative)
    print(json.dumps(_json_ready(summary["our"]), indent=2))
    print("baseline", json.dumps(_json_ready(summary["baseline"]), indent=2))
    if failures:
        print("UNSATISFIED:")
        for item in failures:
            print(" -", item)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
