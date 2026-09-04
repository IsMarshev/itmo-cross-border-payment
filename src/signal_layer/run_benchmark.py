"""CLI for CBSB-1.

    uv run python -m signal_layer.run_benchmark --out reports/benchmark

Writes the full evidence set as CSV plus a markdown scorecard, and prints the
leaderboard. Everything is reproducible from the seed in the spec.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark.dashboard import build_dashboard
from .benchmark.report import render_console_summary, render_scorecard
from .benchmark.runner import run_benchmark
from .benchmark.spec import BenchmarkSpec
from .benchmark.strategies import DEFAULT_STRATEGY_NAMES
from .config import Settings
from .data.normalization import read_rate_directory
from .utility_risk import UtilityRiskConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CBSB-1 signal benchmark")
    parser.add_argument("--corridors", nargs="+", default=["TJS", "UZS", "KGS", "AMD", "KZT"])
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGY_NAMES))
    parser.add_argument("--data-dir", default=None, help="directory of rates_<ISO>.csv files")
    parser.add_argument("--out", default="reports/benchmark")
    parser.add_argument("--eval-start", default="2021-09-01")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fold-months", type=int, default=6)
    parser.add_argument("--bad-push-bps", type=float, default=30.0)
    parser.add_argument("--lam", type=float, default=2.0, help="price of error for the MVP")
    parser.add_argument("--random-trials", type=int, default=500)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=21)
    parser.add_argument("--min-train", type=int, default=750)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else Settings.from_environment().data_dir

    spec = BenchmarkSpec(
        corridors=tuple(c.upper() for c in args.corridors),
        horizon=args.horizon,
        eval_start=pd.Timestamp(args.eval_start),
        fold_months=args.fold_months,
        bad_push_bps=args.bad_push_bps,
        random_trials=args.random_trials,
        bootstrap_trials=args.bootstrap_trials,
        seed=args.seed,
    )
    model_config = UtilityRiskConfig(
        horizon=spec.horizon,
        execution_offset=spec.execution_offset,
        bad_push_bps=spec.bad_push_bps,
        local_min_tolerance_bps=spec.local_min_tolerance_bps,
        lam=args.lam,
        min_train=args.min_train,
        refit_every=args.refit_every,
    )

    panel = read_rate_directory(data_dir, currencies=spec.all_currencies)
    result = run_benchmark(
        panel,
        spec,
        tuple(args.strategies),
        model_config=model_config,
        run_audit=not args.no_audit,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("leaderboard", result.leaderboard),
        ("per_corridor", result.per_corridor),
        ("per_fold", result.per_fold),
        ("gates", result.gates),
        ("horizons", result.horizon_table),
        ("lambda_sweep", result.lambda_sweep),
        ("audit", result.audit),
        ("signals", result.signals),
        ("coefficients", result.coefficients),
    ):
        if len(frame):
            frame.to_csv(out / f"{name}.csv", index=False)

    (out / "scorecard.md").write_text(render_scorecard(result, spec), encoding="utf-8")
    dashboard = build_dashboard(out, panel, spec)
    print(render_console_summary(result))
    print(f"\n  Отчёт:   {out / 'scorecard.md'}")
    print(f"  Дашборд: {dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
