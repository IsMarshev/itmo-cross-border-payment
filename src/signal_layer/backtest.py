"""Rolling-origin orchestration with saved fold models and an auditable run manifest."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .engine import environment_manifest, snapshot_data
from .metrics import (
    prediction_diagnostics,
    random_day_draws,
    random_policy_draws,
    summarize,
    waiting_episodes,
)
from .policy import PolicyReplay
from .training import fit_snapshot


def run_backtest(engine, out, resume=False, dashboard=True, progress=print):
    config = engine.config
    directory = Path(out)
    manifest_path = directory / "manifest.json"
    environment = environment_manifest(config)
    if directory.exists() and any(directory.iterdir()):
        if not resume or not manifest_path.exists():
            raise FileExistsError(
                f"{directory} is not empty; use --resume for an identical run or choose a new --out"
            )
        old = json.loads(manifest_path.read_text())
        if old.get("environment") != environment or old.get("configuration") != config.to_dict():
            raise ValueError(
                "Cannot resume: code, dependencies, data or configuration changed; choose a new --out"
            )
    directory.mkdir(parents=True, exist_ok=True)
    progress("Preparing event-time features and calendar-time targets...")
    panel, features, targets, columns = engine.prepare()
    common_end = panel.loc[panel.iso.isin(config.data.corridors)].groupby("iso").date.max().min()
    end = common_end - pd.Timedelta(days=max(config.targets.horizons) + config.policy.max_wait_days)
    if config.backtest.end:
        end = min(end, pd.Timestamp(config.backtest.end))
    start = pd.Timestamp(config.backtest.start)
    if start > end:
        raise ValueError(
            f"No evaluable interval: requested {start.date()}, mature history ends {end.date()}"
        )
    holdout_start = max(start, end - pd.Timedelta(days=config.backtest.holdout_days - 1))
    manifest = {
        "kind": "backtest",
        "status": "running",
        "environment": environment,
        "configuration": config.to_dict(),
        "eval_start": str(start.date()),
        "eval_end": str(end.date()),
        "holdout_start": str(holdout_start.date()),
        "evaluation_contract": "h=calendar days; all choices past-only; CBR reference, not execution; holdout uses frozen configuration with scheduled rolling refits",
        "folds": [],
    }
    config.save(directory / "config.yaml")
    snapshot_data(config, directory)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    states, decisions_list, fold_metrics = {}, [], []
    fold_start, fold_number = start, 0
    while fold_start <= end:
        fold_end = min(end, fold_start + pd.Timedelta(days=config.backtest.fold_days - 1))
        # A boundary makes development and reserved evaluation explicitly separable.
        if fold_start < holdout_start <= fold_end:
            fold_end = holdout_start - pd.Timedelta(days=1)
        cutoff = fold_start - pd.Timedelta(days=1)
        name = f"{fold_number:03d}_{fold_start.date()}"
        fold_dir = directory / "folds" / name
        progress(
            f"Fold {fold_number + 1}: {fold_start.date()} .. {fold_end.date()} (fit cutoff {cutoff.date()})"
        )
        model_path = fold_dir / "model.joblib"
        if resume and model_path.exists():
            snapshot = joblib.load(model_path)
            if snapshot["metadata"]["trained_as_of"] != str(cutoff.date()):
                raise ValueError("Cached fold cutoff mismatch")
            progress("  reusing saved models; replaying decisions")
        else:
            snapshot = fit_snapshot(features, targets, columns, config, cutoff, fold_dir, progress)
        f = features.loc[
            features.eligible & features.date.between(fold_start, fold_end)
        ].reset_index(drop=True)
        if f.empty:
            raise ValueError(f"No eligible source updates in fold {name}")
        prediction_cache, fold_decisions = {}, []
        for method, fitted in snapshot["models"].items():
            if id(fitted) not in prediction_cache:
                prediction_cache[id(fitted)] = fitted.predict(f)
            prediction = prediction_cache[id(fitted)].copy()
            replay = PolicyReplay(config, method, fitted.risk_seed, states.get(method))
            d = replay.run(prediction, snapshot["parameters"][method], targets)
            d["fold"] = fold_number
            d["phase"] = "holdout" if fold_start >= holdout_start else "development"
            states[method] = replay.policy
            fold_decisions.append(d)
        fold_frame = pd.concat(fold_decisions, ignore_index=True)
        fold_frame.to_csv(fold_dir / "decisions.csv.gz", index=False)
        decisions_list.append(fold_frame)
        fm = summarize(fold_frame, targets, config, fold_start, fold_end)
        fm["fold"], fm["phase"] = (
            fold_number,
            "holdout" if fold_start >= holdout_start else "development",
        )
        fold_metrics.append(fm)
        manifest["folds"].append(
            {
                "fold": fold_number,
                "start": str(fold_start.date()),
                "end": str(fold_end.date()),
                "model": str(model_path.relative_to(directory)),
                "phase": fm.phase.iloc[0],
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        fold_start, fold_number = fold_end + pd.Timedelta(days=1), fold_number + 1
    decisions = pd.concat(decisions_list, ignore_index=True)
    progress("Scoring signals, uncertainty, random controls and waiting episodes...")
    summary = summarize(decisions, targets, config, start, end)
    diagnostics, calibration = prediction_diagnostics(decisions, targets)
    waiting = waiting_episodes(decisions, targets, config)
    random_days = random_day_draws(decisions, targets, config)
    reference = decisions.loc[decisions.method == decisions.method.iloc[0]].drop_duplicates(
        ["date", "iso"]
    )
    random_policy = random_policy_draws(reference, targets, config, start, end)
    decisions.to_csv(directory / "decisions.csv.gz", index=False)
    decisions.loc[decisions.decision == "send"].to_csv(directory / "signals.csv", index=False)
    targets.loc[targets.date.between(start, end)].to_csv(directory / "outcomes.csv.gz", index=False)
    summary.to_csv(directory / "summary.csv", index=False)
    pd.concat(fold_metrics, ignore_index=True).to_csv(directory / "fold_metrics.csv", index=False)
    diagnostics.to_csv(directory / "diagnostics.csv", index=False)
    calibration.to_csv(directory / "calibration.csv", index=False)
    waiting.to_csv(directory / "waiting_episodes.csv", index=False)
    random_days.to_csv(directory / "random_day_draws.csv", index=False)
    random_policy.to_csv(directory / "random_policy_draws.csv", index=False)
    manifest["status"], manifest["n_decisions"], manifest["n_signals"] = (
        "complete",
        len(decisions),
        int((decisions.decision == "send").sum()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if dashboard:
        engine.dashboard(directory)
    progress(f"Complete: {directory.resolve()}")
    return directory
