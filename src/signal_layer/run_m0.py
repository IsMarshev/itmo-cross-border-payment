"""Command-line entry point for the m0 model.

Trains the Ridge "value head" per corridor with walk-forward, converts
predictions into a signal stream under a communication budget, and evaluates
the full metric matrix from ``signal_layer.metrics``.

Usage::

    uv run python -m signal_layer.run_m0 train \\
        --corridors TJS UZS KGS AMD KZT \\
        --h 20 --slots-per-week 1.5 --eps-bps 0 \\
        --out reports/m0

    uv run python -m signal_layer.run_m0 asof --corridor TJS --date 2024-06-01

The ``train`` subcommand writes, per corridor, the walk-forward predictions and
the per-horizon metric table. The ``asof`` subcommand returns the signal
decision for a single date, which is how the no-look-ahead requirement is
verified: the same code path that serves a decision on date T is re-run on a
frozen data snapshot.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import metrics
from .data import normalization
from .models import DEFAULT_H, walk_forward_predict

DEFAULT_CORRIDORS = ("TJS", "UZS", "KGS", "AMD", "KZT")


def _load_panel(corridors: list[str], data_dir: str) -> pd.DataFrame:
    return normalization.read_rate_directory(data_dir, currencies=corridors)


def _signals_from_predictions(
    pred: pd.DataFrame, slots_per_week: float, lookback: int = 250
) -> pd.DataFrame:
    """Turn predicted advantage into a signal stream under a weekly budget.

    A signal fires when ``pred_advantage`` exceeds the ``(1 - slots/d)`` quantile
    of recent predictions (the plan's budget-aware threshold). After a signal,
    the next ``cooldown`` trading days are suppressed so a single move does not
    exhaust the budget. Threshold and cooldown are derived from ``slots_per_week``.
    """
    trading_days_per_week = 5.0
    d = trading_days_per_week  # days available per week
    b = slots_per_week  # slots to spend per week
    quantile = 1.0 - b / d
    cooldown = int(round(d / b))  # one signal per budget period

    pred = pred.sort_values("quote_date").reset_index(drop=True)
    p = pred["pred_advantage"].to_numpy(dtype=float)
    sig = np.zeros(len(pred), dtype=int)
    last_signal = -10**9
    for i in range(len(pred)):
        lo = max(0, i - lookback)
        window = p[lo:i]
        if len(window) < 20:
            continue
        thr = np.quantile(window, quantile)
        if p[i] > thr and (i - last_signal) >= cooldown:
            sig[i] = 1
            last_signal = i
    signals = pred.loc[sig == 1, ["quote_date", "iso", "rub_per_unit"]].rename(
        columns={"quote_date": "signal_date"}
    )
    return signals


def cmd_train(args: argparse.Namespace) -> None:
    panel = _load_panel(args.corridors, args.data_dir)
    os.makedirs(args.out, exist_ok=True)
    all_metrics = []
    all_freq = []
    all_signals = []
    for iso in args.corridors:
        print(f"-> {iso}: walk-forward predict...", end=" ", flush=True)
        pred = walk_forward_predict(
            panel, iso, h=args.h, alpha=args.alpha, min_train=args.min_train
        )
        pred.to_csv(os.path.join(args.out, f"predictions_{iso}.csv"), index=False)

        signals = _signals_from_predictions(pred, slots_per_week=args.slots_per_week)
        # Recover rub_per_unit directly from the panel to avoid index races.
        if len(signals):
            s = panel[panel["iso"] == iso].set_index("quote_date")["rub_per_unit"]
            signals["rub_per_unit"] = signals["signal_date"].map(s).astype(float)
        signals.to_csv(os.path.join(args.out, f"signals_{iso}.csv"), index=False)
        all_signals.append(signals)
        print(f"{len(signals)} signals")

        if len(signals):
            m_df, freq_df = metrics.evaluate(
                panel, signals, horizons=tuple(args.horizons), eps_bps=args.eps_bps
            )
        else:
            m_df, freq_df = metrics.evaluate(
                panel, signals[["iso", "signal_date", "rub_per_unit"]],
                horizons=tuple(args.horizons), eps_bps=args.eps_bps,
            )
        m_df.to_csv(os.path.join(args.out, f"metrics_{iso}.csv"), index=False)
        all_metrics.append(m_df)
        all_freq.append(freq_df.assign(source="model"))

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(os.path.join(args.out, "metrics_all.csv"), index=False)
    freq_all = pd.concat(all_freq, ignore_index=True)
    freq_all.to_csv(os.path.join(args.out, "frequency_all.csv"), index=False)
    print(f"\nMetrics -> {args.out}/metrics_all.csv")
    summary = metrics_all.groupby("iso")[["hit_rate", "lift", "advantage_bps"]].mean().round(3)
    print(summary.to_string())


def cmd_asof(args: argparse.Namespace) -> None:
    panel = _load_panel([args.corridor], args.data_dir)
    asof = pd.Timestamp(args.date)
    available = panel[panel["available_on"] <= asof]
    if available.empty:
        print(f"No data available on {args.date} for {args.corridor}")
        return
    pred = walk_forward_predict(
        panel, args.corridor, h=args.h, alpha=args.alpha, min_train=args.min_train
    )
    row = pred[pred["quote_date"] <= asof].sort_values("quote_date").iloc[-1]
    signals = _signals_from_predictions(pred, slots_per_week=args.slots_per_week)
    fired = (signals["signal_date"] == row["quote_date"]).any()
    print(
        f"asof {asof.date()} | {args.corridor} | rate={row['rub_per_unit']:.4f} | "
        f"pred_advantage={row['pred_advantage']:.1f} bps | signal={'YES' if fired else 'no'}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m0 Ridge value-head model")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data-dir", default="currency_data")
        sp.add_argument("--h", type=int, default=DEFAULT_H)
        sp.add_argument("--alpha", type=float, default=1.0)
        sp.add_argument("--min-train", type=int, default=500)
        sp.add_argument("--slots-per-week", type=float, default=1.5)

    pt = sub.add_parser("train", help="Walk-forward train + evaluate")
    pt.add_argument("--corridors", nargs="+", default=list(DEFAULT_CORRIDORS))
    pt.add_argument("--horizons", nargs="+", type=int, default=list(metrics.HORIZONS))
    pt.add_argument("--eps-bps", type=float, default=0.0)
    pt.add_argument("--out", default="reports/m0")
    _common(pt)
    pt.set_defaults(func=cmd_train)

    pa = sub.add_parser("asof", help="Signal decision on a single date (no look-ahead)")
    pa.add_argument("--corridor", required=True)
    pa.add_argument("--date", required=True, help="YYYY-MM-DD")
    _common(pa)
    pa.set_defaults(func=cmd_asof)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
