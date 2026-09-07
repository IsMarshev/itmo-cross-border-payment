"""CLI for the canonical Stage-4 backtest and its audit artifacts."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from signal_layer.config import Settings
from signal_layer.services import BacktestService, RateService

DEFAULT_CORRIDORS = ("TJS", "UZS", "KGS", "AMD", "KZT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a leakage-safe signal-policy backtest")
    parser.add_argument("--data-dir", type=Path, default=Path("currency_data"))
    parser.add_argument("--corridors", nargs="+", default=list(DEFAULT_CORRIDORS))
    parser.add_argument("--score-source", choices=("baseline", "ridge"), default="baseline")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epsilon-bps", type=float, default=30.0)
    parser.add_argument("--window", choices=("week", "month"), default="week")
    parser.add_argument("--max-signals-per-window", type=int, default=2)
    parser.add_argument("--cooldown-observations", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--bootstrap-trials", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/backtest"))
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Do not generate dashboard.html after writing backtest artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rate_service = RateService(Settings(data_dir=args.data_dir.resolve()).data_dir)
    result = BacktestService(rate_service).run(
        args.corridors,
        score_source=args.score_source,
        as_of=args.as_of,
        horizon=args.horizon,
        epsilon_bps=args.epsilon_bps,
        window=args.window,
        max_signals_per_window=args.max_signals_per_window,
        cooldown_observations=args.cooldown_observations,
        min_train=args.min_train,
        random_trials=args.random_trials,
        bootstrap_trials=args.bootstrap_trials,
        seed=args.seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    result.decision_log.to_json(
        args.out / "decision_log.jsonl",
        orient="records",
        lines=True,
        date_format="iso",
    )
    result.random_log.to_json(
        args.out / "random_baseline.jsonl",
        orient="records",
        lines=True,
        date_format="iso",
    )
    result.report.to_json(
        args.out / "summary.json",
        orient="records",
        indent=2,
    )
    print(result.report.round(3).to_string(index=False))
    print(f"\nArtifacts written to {args.out}")
    # The dashboard for this CLI was lost in the m0->backend merge and is not
    # rebuilt here: CBSB-1 supersedes it and renders a richer page from its own
    # artifacts. See `python -m signal_layer.run_benchmark`.


if __name__ == "__main__":
    main()
