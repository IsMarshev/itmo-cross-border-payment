"""Public API: prepare, train, backtest, historical signals and dashboard."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import joblib
import pandas as pd

from . import __version__
from .config import Config, config_from_dict, load_config
from .data import data_manifest, load_rates
from .features import build_features
from .policy import PolicyReplay
from .provenance import code_fingerprint
from .targets import build_targets
from .training import fit_snapshot


def environment_manifest(config):
    return {
        "version": __version__,
        "python": platform.python_version(),
        "code_sha256": code_fingerprint(),
        "data": data_manifest(config),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "catboost", "plotly"]
        },
    }


def snapshot_data(config, directory):
    out = Path(directory) / "data"
    out.mkdir(parents=True, exist_ok=True)
    for iso in sorted(set(config.data.corridors + config.data.context)):
        source = Path(config.data.directory) / f"rates_{iso}.csv"
        target = out / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    if config.data.holidays_file:
        source = Path(config.data.holidays_file)
        if source.resolve() != (out / "holidays.csv").resolve():
            shutil.copy2(source, out / "holidays.csv")


def portable_config(config, directory):
    cfg = config_from_dict(config.to_dict())
    data = Path(directory) / "data"
    if data.is_dir():
        cfg.data.directory = str(data.resolve())
        if cfg.data.holidays_file:
            cfg.data.holidays_file = str((data / "holidays.csv").resolve())
    return cfg


class SignalEngine:
    def __init__(self, config: Config | str | Path | None = None):
        self.config = config.validate() if isinstance(config, Config) else load_config(config)
        self.snapshot, self.run_directory = None, None

    def prepare(self, as_of=None):
        panel = load_rates(self.config, as_of=as_of)
        features, columns = build_features(panel, self.config)
        targets = build_targets(panel, features, self.config)
        return panel, features, targets, columns

    def train(self, out="artifacts/model", as_of=None, progress=print):
        directory = Path(out)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} is not empty; choose a new model directory")
        panel, features, targets, columns = self.prepare(as_of)
        cutoff = (
            pd.Timestamp(as_of)
            if as_of
            else panel.loc[panel.iso.isin(self.config.data.corridors)]
            .groupby("iso")
            .date.max()
            .min()
        )
        if cutoff > panel.date.max():
            raise ValueError("Training cutoff is beyond the available data")
        directory.mkdir(parents=True, exist_ok=True)
        self.snapshot = fit_snapshot(
            features, targets, columns, self.config, cutoff, directory, progress
        )
        snapshot_data(self.config, directory)
        self.config.save(directory / "config.yaml")
        manifest = environment_manifest(self.config)
        manifest.update(kind="training", trained_as_of=str(cutoff.date()))
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.run_directory = directory
        return directory / "model.joblib"

    @classmethod
    def load(cls, path):
        """Load only trusted, locally produced joblib artifacts (pickle format)."""
        path = Path(path)
        directory = path.parent if path.is_file() else path
        model_path = path if path.is_file() else directory / "model.joblib"
        if model_path.exists():
            snapshot = joblib.load(model_path)
            if snapshot["metadata"]["version"] != __version__:
                raise ValueError("Artifact version mismatch; retrain with the installed version")
            if snapshot["metadata"].get("code_sha256") != code_fingerprint():
                raise ValueError(
                    "Artifact code differs from installed code; restore the original revision or retrain"
                )
            source_root = next(
                (p for p in [directory, *list(directory.parents)[:2]] if (p / "data").is_dir()),
                directory,
            )
            engine = cls(portable_config(config_from_dict(snapshot["config"]), source_root))
            engine.snapshot, engine.run_directory = snapshot, directory
            return engine
        if not (directory / "manifest.json").exists():
            raise FileNotFoundError(f"No model or backtest manifest in {directory}")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("environment", {}).get("code_sha256") != code_fingerprint():
            raise ValueError(
                "Backtest code differs from installed code; restore the original revision or rerun"
            )
        engine = cls(portable_config(load_config(directory / "config.yaml"), directory))
        engine.run_directory = directory
        return engine

    def signals(self, as_of, method="catboost", corridor=None):
        """Replay budgets, episodes and delayed risk updates up to an exact date cut."""
        cutoff = pd.Timestamp(as_of).normalize()
        panel, features, targets, _ = self.prepare(cutoff)
        states, all_decisions = {}, []
        if self.snapshot is not None:
            trained = pd.Timestamp(self.snapshot["metadata"]["trained_as_of"])
            if cutoff < trained:
                raise ValueError(
                    f"Artifact trained as of {trained.date()}; cannot apply it retrospectively to {cutoff.date()}"
                )
            snapshots = [(self.snapshot, trained, cutoff)]
        elif self.run_directory is not None:
            manifest = json.loads((self.run_directory / "manifest.json").read_text())
            if manifest.get("kind") != "backtest":
                raise ValueError("Use a fitted model or a completed backtest directory")
            if cutoff < pd.Timestamp(manifest["eval_start"]) or cutoff > pd.Timestamp(
                manifest["eval_end"]
            ):
                raise ValueError(
                    "Date is outside the archived backtest period; train a dated deployment artifact"
                )
            snapshots = []
            for fold in manifest["folds"]:
                if pd.Timestamp(fold["start"]) > cutoff:
                    break
                snap = joblib.load(self.run_directory / fold["model"])
                snapshots.append(
                    (snap, pd.Timestamp(fold["start"]), min(cutoff, pd.Timestamp(fold["end"])))
                )
        else:
            raise ValueError("Train or load an artifact before requesting signals")
        for snapshot, start, end in snapshots:
            if method not in snapshot["models"]:
                raise ValueError(
                    f"Method {method!r} unavailable; choose {list(snapshot['models'])}"
                )
            f = features.loc[features.eligible & features.date.between(start, end)].copy()
            if f.empty:
                continue
            prediction = snapshot["models"][method].predict(f.reset_index(drop=True))
            replay = PolicyReplay(
                self.config, method, snapshot["models"][method].risk_seed, states.get(method)
            )
            decisions = replay.run(prediction, snapshot["parameters"][method], targets)
            states[method] = replay.policy
            all_decisions.append(decisions)
        complete = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
        current = (
            complete.loc[complete.date == cutoff].copy() if not complete.empty else pd.DataFrame()
        )
        present = set(current.iso) if not current.empty else set()
        for iso in self.config.data.corridors:
            if iso in present:
                continue
            rates = panel.loc[panel.iso == iso]
            last = rates.iloc[-1] if len(rates) else None
            fresh = last is not None and last.date == cutoff
            row = {
                "date": cutoff,
                "iso": iso,
                "method": method,
                "decision": "abstain",
                "reason": "insufficient_history" if fresh else "no_new_observation",
                "source_date": None if last is None else last.date,
                "rub_per_unit": None if last is None else last.rub_per_unit,
            }
            current = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
        if corridor:
            if corridor not in self.config.data.corridors:
                raise ValueError(f"Unknown corridor: {corridor}")
            current = current.loc[current.iso == corridor]
        return current.reset_index(drop=True)

    def backtest(self, out="reports/backtest", resume=False, dashboard=True, progress=print):
        from .backtest import run_backtest

        return run_backtest(self, out, resume=resume, dashboard=dashboard, progress=progress)

    @staticmethod
    def dashboard(run, out=None):
        from .dashboard import generate_dashboard

        return generate_dashboard(run, out)
