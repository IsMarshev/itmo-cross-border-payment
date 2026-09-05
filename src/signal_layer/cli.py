"""One CLI for data, fitting, evaluation, reporting and point-in-time serving."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import fetch_rates
from .engine import SignalEngine


def parser():
    p = argparse.ArgumentParser(
        prog="fx-signals", description="FX signal training and honest walk-forward evaluation"
    )
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "backtest", "run"):
        s = commands.add_parser(name)
        s.add_argument("--config", default="configs/default.yaml")
        s.add_argument("--data-dir")
        s.add_argument("--corridors", nargs="+")
        s.add_argument("--methods", nargs="+")
        default_out = {
            "prepare": "artifacts/prepared",
            "train": "artifacts/model",
            "backtest": "reports/backtest",
            "run": "reports/full",
        }[name]
        s.add_argument("--out", default=default_out)
        if name in ("prepare", "train"):
            s.add_argument("--as-of")
        if name in ("backtest", "run"):
            s.add_argument("--resume", action="store_true")
            s.add_argument("--no-dashboard", action="store_true")
    s = commands.add_parser("signals")
    s.add_argument(
        "--artifact", required=True, help="Model directory/model.joblib or backtest directory"
    )
    s.add_argument("--as-of", required=True)
    s.add_argument("--method", default="catboost")
    s.add_argument("--corridor")
    s.add_argument("--data-dir", help="Override bundled rates with updated source CSVs")
    s.add_argument("--out", help="Optional .csv or .json output")
    s = commands.add_parser("dashboard")
    s.add_argument("--run", required=True)
    s.add_argument("--out")
    s = commands.add_parser("fetch")
    s.add_argument("--out", default="currency_data")
    s.add_argument(
        "--currencies", nargs="+", default=["TJS", "UZS", "KGS", "AMD", "KZT", "USD", "EUR", "CNY"]
    )
    s.add_argument("--start", default="2011-01-01")
    s.add_argument("--end", default=str(pd.Timestamp.today().date()))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch":
            fetch_rates(args.out, args.currencies, args.start, args.end)
            print(f"Saved rates to {Path(args.out).resolve()}")
            return 0
        if args.command == "dashboard":
            print(SignalEngine.dashboard(args.run, args.out))
            return 0
        if args.command == "signals":
            engine = SignalEngine.load(args.artifact)
            if args.data_dir:
                engine.config.data.directory = args.data_dir
                engine.config.data.end = None
            frame = engine.signals(args.as_of, method=args.method, corridor=args.corridor)
            payload = frame.to_json(
                orient="records", date_format="iso", force_ascii=False, indent=2
            )
            if args.out:
                out = Path(args.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(out, index=False) if out.suffix == ".csv" else out.write_text(
                    payload, encoding="utf-8"
                )
            print(payload)
            return 0
        config = load_config(args.config)
        if args.data_dir:
            config.data.directory = args.data_dir
        if args.corridors:
            config.data.corridors = args.corridors
        if args.methods:
            config.model.methods = args.methods
        config.validate()
        engine = SignalEngine(config)
        if args.command == "prepare":
            panel, features, targets, columns = engine.prepare(args.as_of)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            panel.to_csv(out / "rates.csv.gz", index=False)
            features.to_csv(out / "features.csv.gz", index=False)
            targets.to_csv(out / "targets.csv.gz", index=False)
            (out / "feature_columns.json").write_text(json.dumps(columns, indent=2))
            print(
                f"{len(panel)} rate updates; {features.eligible.sum()} eligible feature rows -> {out.resolve()}"
            )
        elif args.command == "train":
            print(engine.train(args.out, args.as_of, progress=partial(print, flush=True)))
        else:
            engine.backtest(
                args.out,
                resume=args.resume,
                dashboard=not args.no_dashboard,
                progress=partial(print, flush=True),
            )
            if args.command == "run":
                final = Path(args.out) / "final_model"
                if args.resume and (final / "model.joblib").exists():
                    print(f"Reusing final model: {final}")
                else:
                    print(engine.train(final, progress=partial(print, flush=True)))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"fx-signals: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
