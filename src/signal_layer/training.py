"""Purged train -> probability calibration -> policy tuning -> future evaluation."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from . import __version__
from .config import Config
from .models import fit_model
from .policy import PolicyParameters, PolicyReplay
from .provenance import code_fingerprint
from .targets import mature_rows


def temporal_partitions(dataset, cutoff, config):
    cutoff = pd.Timestamp(cutoff).normalize()
    tune_start = cutoff - pd.Timedelta(days=config.model.tuning_days - 1)
    cal_start = tune_start - pd.Timedelta(days=config.model.calibration_days)
    train_start = cal_start - pd.Timedelta(days=config.model.train_window_days)
    train = mature_rows(
        dataset.loc[(dataset.date >= train_start) & (dataset.date < cal_start)],
        cal_start - pd.Timedelta(days=1),
    )
    calibration = mature_rows(
        dataset.loc[(dataset.date >= cal_start) & (dataset.date < tune_start)],
        tune_start - pd.Timedelta(days=1),
    )
    tuning = mature_rows(
        dataset.loc[(dataset.date >= tune_start) & (dataset.date <= cutoff)], cutoff
    )
    for name, part, minimum in [
        ("train", train, config.model.min_train_rows),
        ("calibration", calibration, config.model.min_calibration_rows),
        ("tuning", tuning, 20),
    ]:
        if len(part) < minimum:
            raise ValueError(
                f"{name}: {len(part)} mature rows, need {minimum}; move cutoff later or increase history"
            )
        missing = set(config.data.corridors) - set(part.iso)
        if missing:
            raise ValueError(f"{name}: no mature data for {sorted(missing)}")
    return train, calibration, tuning


def calibrate_policy(method, predictions, tuning, fitted, config):
    selected, trials = {}, []
    for iso in config.data.corridors:
        p = predictions.loc[predictions.iso == iso].copy()
        truth = tuning.loc[tuning.iso == iso]
        weeks = max(1, (p.date.max() - p.date.min()).days / 7)
        is_rule = method.startswith("rule_") or method == "random_policy"
        grid = itertools.product(
            config.policy.level_thresholds,
            [0.0] if is_rule else config.policy.probability_thresholds,
            [0.0] if is_rule else config.policy.contact_costs_bps,
        )
        best, best_score = None, -float("inf")
        for level, probability, cost in grid:
            params = PolicyParameters(level, probability, cost)
            replay = PolicyReplay(config, method, fitted.risk_seed)
            decisions = replay.run(p, {iso: params}, truth)
            sent = decisions.loc[decisions.decision == "send", ["date", "iso"]].merge(
                truth[["date", "iso", "y_gain_bps", "y_regret_bps", "y_stale_bps"]],
                on=["date", "iso"],
            )
            n = len(sent)
            utility = (
                sent.y_gain_bps
                - config.risk.regret_penalty * sent.y_regret_bps
                - config.risk.stale_penalty * sent.y_stale_bps
                - cost
            )
            score = float(utility.sum() / weeks)
            # A deficit is reported, never repaired by overriding safety filters.
            feasible = n >= config.policy.min_tuning_signals
            trials.append(
                dict(
                    method=method,
                    iso=iso,
                    level=level,
                    probability=probability,
                    contact_cost_bps=cost,
                    signals=n,
                    frequency=n / weeks,
                    utility_per_week=score,
                    eligible_for_selection=feasible,
                )
            )
            if feasible and score > best_score:
                best_score, best = score, params
        if best is None:
            best = PolicyParameters(enabled=False, status="insufficient_tuning_signals")
        else:
            best.status = "calibrated"
        # Random policy is an evaluation control, not a claimed valuable signal.
        if method == "random_policy":
            best = PolicyParameters(probability=0, status="random_control")
        selected[iso] = best
    return selected, pd.DataFrame(trials)


def fit_snapshot(features, targets, columns, config: Config, cutoff, out, progress=print):
    cutoff = pd.Timestamp(cutoff).normalize()
    dataset = features.loc[features.eligible & (features.date <= cutoff)].merge(
        targets, on=["date", "iso"], validate="one_to_one"
    )
    train, calibration, tuning = temporal_partitions(dataset, cutoff, config)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    models, parameters, all_trials, importance = {}, {}, [], []
    for method in config.model.methods:
        progress(
            f"  fit {method}: train={len(train)}, calibration={len(calibration)}, tune={len(tuning)}"
        )
        fitted = fit_model(method, train, calibration, columns, config)
        models[method] = fitted
        p = fitted.predict(tuning[features.columns].reset_index(drop=True))
        params, trials = calibrate_policy(method, p, tuning, fitted, config)
        parameters[method], all_trials = params, all_trials + [trials]
        importance.extend(dict(method=method, **v) for v in fitted.predictor.importance())
        if method == "catboost" and config.backtest.ablations:
            for suffix in ("no_wait", "no_uncertainty"):
                name = f"catboost_{suffix}"
                models[name] = fitted
                par, tr = calibrate_policy(name, p, tuning, fitted, config)
                parameters[name], all_trials = par, all_trials + [tr]
            name = "catboost_no_regime"
            reduced = [
                c for c in columns if c not in ("regime", "shock_z", "vol_ratio", "trend_strength")
            ]
            fitted_reduced = fit_model("catboost", train, calibration, reduced, config)
            for seed_row in fitted_reduced.risk_seed:
                seed_row["regime"] = "range"
            models[name] = fitted_reduced
            pred = fitted_reduced.predict(tuning[features.columns].reset_index(drop=True))
            par, tr = calibrate_policy(name, pred, tuning, fitted_reduced, config)
            parameters[name], all_trials = par, all_trials + [tr]
    metadata = {
        "version": __version__,
        "code_sha256": code_fingerprint(),
        "trained_as_of": str(cutoff.date()),
        "feature_columns": list(columns),
        "partitions": {},
    }
    for name, part in [("train", train), ("calibration", calibration), ("tuning", tuning)]:
        metadata["partitions"][name] = {
            "rows": len(part),
            "start": str(part.date.min().date()),
            "end": str(part.date.max().date()),
            "latest_label_known_on": str(part.label_known_on.max().date()),
        }
    snapshot = {
        "metadata": metadata,
        "config": config.to_dict(),
        "models": models,
        "parameters": parameters,
    }
    temporary = directory / "model.joblib.tmp"
    joblib.dump(snapshot, temporary, compress=3)
    temporary.replace(directory / "model.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "policy.json").write_text(
        json.dumps(
            {m: {i: asdict(p) for i, p in v.items()} for m, v in parameters.items()}, indent=2
        ),
        encoding="utf-8",
    )
    pd.concat(all_trials, ignore_index=True).to_csv(directory / "policy_trials.csv", index=False)
    pd.DataFrame(importance, columns=["method", "head", "feature", "importance"]).to_csv(
        directory / "feature_importance.csv", index=False
    )
    return snapshot
