"""Command-line entry point for the indicator matrix.

Runs every rule-based indicator (plus the Ridge model as a baseline) across all
corridors with per-corridor walk-forward calibration, evaluates each on the
trailing year, and writes the ``indicator × corridor × metrics`` matrix required
by the case brief (Stage 2, point 5: backtest with a direct verdict on which
indicators are informative and which to drop).

Usage::

    uv run python -m signal_layer.run_indicators \\
        --corridors USD TJS UZS KGS AMD KZT \\
        --monthly-budget 50000 --cadence-days 5 \\
        --out reports/indicators

The matrix CSV has one row per (indicator, corridor) with: uplift vs DCA and vs
random (currency bought on the same budget), lift / hit-rate at h=5 and h=15,
signal frequency (per week) and clustering (series_share), the calibrated
parameters, and a verdict (keep / marginal / drop) with the reason. A separate
``waiting_cost.csv`` reports the price of waiting for slow confirmation.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import indicators as ind
from .run_m0 import DEFAULT_CORRIDORS, _load_panel

# USD is included so the matrix covers the sanity-check corridor too.
MATRIX_CORRIDORS = ("USD", *DEFAULT_CORRIDORS)

# Communication-policy band: 1–2 pushes per week. Outside it an indicator is
# impractical regardless of accuracy (brief: "30 signals/month is unusable").
FREQ_MIN = 1.0
FREQ_MAX = 2.0
# Lift target from the brief: stable lift >= 1.3; floor (useless) = 1.0.
LIFT_TARGET = 1.3
LIFT_FLOOR = 1.0


def _classify(row: dict) -> tuple[str, str]:
    """Verdict (keep/marginal/drop) + reason, from the brief's criteria.

    * **drop**   — lift < 1.0 (no better than a random day) OR loses to DCA.
    * **keep**   — beats DCA, lift >= target, and frequency within policy band.
    * **marginal** — positive but not meeting all of the above.
    """
    lift = row.get("lift_h5", np.nan)
    up_dca = row.get("uplift_vs_dca", np.nan)
    per_week = row.get("per_week", np.nan)
    reasons = []
    if np.isfinite(lift) and lift < LIFT_FLOOR:
        reasons.append(f"lift {lift:.2f}<1.0 (≈random)")
    if np.isfinite(up_dca) and up_dca <= 0:
        reasons.append(f"vs DCA {up_dca:+.2f}% (loses to cadence)")
    if reasons:
        return "drop", "; ".join(reasons)
    if np.isfinite(lift) and lift >= LIFT_TARGET and np.isfinite(up_dca) and up_dca > 0:
        if np.isfinite(per_week) and (per_week < FREQ_MIN or per_week > FREQ_MAX):
            reasons.append(f"freq {per_week:.1f}/wk outside {FREQ_MIN}-{FREQ_MAX}")
            return "marginal", "; ".join(reasons)
        return "keep", f"lift {lift:.2f}, vs DCA {up_dca:+.2f}%, freq {per_week:.1f}/wk"
    return "marginal", f"lift {lift:.2f}, vs DCA {up_dca:+.2f}%"


def _run_matrix(
    panel: pd.DataFrame,
    corridors: list[str],
    *,
    monthly_budget: float,
    cadence_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the indicator×corridor matrix + waiting-cost table (trailing year)."""
    last = panel["quote_date"].max()
    start = last - pd.DateOffset(years=1)

    rows = []
    calibrated_signals: dict[tuple[str, str], pd.DataFrame] = {}
    for iso in corridors:
        print(f"\n== {iso} ==")
        for spec in ind.INDICATORS:
            grid = ind.grid_for(spec)
            cal = ind.calibrate(
                panel, iso, spec, grid,
                monthly_budget=monthly_budget, cadence_days=cadence_days,
            )
            best = cal["best_params"]
            ev = ind.evaluate_indicator(
                panel, iso, spec, best,
                monthly_budget=monthly_budget, cadence_days=cadence_days,
                start=start, end=last,
            )
            verdict, reason = _classify(ev)
            row = {
                **ev,
                "best_params": best if best else "{}",
                "verdict": verdict,
                "reason": reason,
            }
            rows.append(row)
            calibrated_signals[(iso, spec.name)] = spec.fn(panel, iso, **best)
            params_str = str(best) if best else "{}"
            print(
                f"  {spec.name:18s} params={params_str:<22} "
                f"vs DCA={ev['uplift_vs_dca']:+6.2f}% "
                f"lift={ev['lift_h5']:.2f} "
                f"(other {ev['lift_h5_other']:.2f}) "
                f"freq={ev['per_week']:.1f}/wk "
                f"n={ev['n_signals']:4d} -> {verdict}"
            )

    matrix = pd.DataFrame(rows)

    # Waiting cost: fast (value) vs slow (reversal_from_min) per corridor.
    wc_rows = []
    for iso in corridors:
        fast = calibrated_signals.get((iso, "value"))
        slow = calibrated_signals.get((iso, "reversal_from_min"))
        if fast is None or slow is None:
            continue
        cost = ind.waiting_cost(panel, iso, fast, slow)
        wc_rows.append({"iso": iso, "fast": "value", "slow": "reversal_from_min",
                        "waiting_cost_bps": cost})
    waiting = pd.DataFrame(wc_rows)
    return matrix, waiting


def _print_verdicts(matrix: pd.DataFrame) -> None:
    """Direct output: which indicators are informative, which to drop and why."""
    print("\n" + "=" * 70)
    print("ВЕРДИКТ ПО ИНДИКАТОРАМ (по коридорам)")
    print("=" * 70)
    for name in matrix["indicator"].unique():
        sub = matrix[matrix["indicator"] == name]
        keeps = sub[sub["verdict"] == "keep"]
        drops = sub[sub["verdict"] == "drop"]
        speed = sub["speed"].iloc[0]
        print(f"\n● {name} ({speed}):")
        if len(keeps):
            print(f"  информативен на: {', '.join(keeps['iso'])}")
            for _, r in keeps.iterrows():
                print(f"    {r['iso']}: {r['reason']}")
        if len(drops):
            print(f"  исключить на: {', '.join(drops['iso'])}")
            for _, r in drops.iterrows():
                print(f"    {r['iso']}: {r['reason']}")
        marg = sub[sub["verdict"] == "marginal"]
        if len(marg):
            print(f"  на границе: {', '.join(marg['iso'])}")
            for _, r in marg.iterrows():
                print(f"    {r['iso']}: {r['reason']}")


def cmd_run(args: argparse.Namespace) -> None:
    panel = _load_panel(args.corridors, args.data_dir)
    os.makedirs(args.out, exist_ok=True)
    matrix, waiting = _run_matrix(
        panel, args.corridors,
        monthly_budget=args.monthly_budget, cadence_days=args.cadence_days,
    )
    matrix.to_csv(os.path.join(args.out, "indicators_matrix.csv"), index=False)
    waiting.to_csv(os.path.join(args.out, "waiting_cost.csv"), index=False)
    print(f"\nМатрица -> {args.out}/indicators_matrix.csv")
    print(f"Цена ожидания -> {args.out}/waiting_cost.csv")
    _print_verdicts(matrix)
    if not waiting.empty:
        print("\nЦена ожидания (быстрый value → медленный reversal_from_min):")
        print(waiting.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Indicator × corridor metrics matrix")
    p.add_argument("--corridors", nargs="+", default=list(MATRIX_CORRIDORS))
    p.add_argument("--data-dir", default="currency_data")
    p.add_argument(
        "--monthly-budget", type=float, default=50_000.0,
        help="RUB per calendar month per corridor (total spent = this x months)",
    )
    p.add_argument("--cadence-days", type=int, default=5,
                   help="DCA buy every N trading days (5≈weekly)")
    p.add_argument("--out", default="reports/indicators")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
