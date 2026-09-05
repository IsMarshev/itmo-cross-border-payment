"""Export the live signal layer's own output for the demo stand.

Every number the stand shows comes from here: the rate panel, the days the
communication policy actually selected, the fact each push may state, and the
no-look-ahead audit. Run from the repository root:

    uv run python demo/export_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from signal_layer.data import read_rate_directory
from signal_layer.signals import signal_table, signals_asof

DATA = Path("currency_data")
CORRIDORS = ["TJS", "UZS", "KGS", "AMD", "KZT"]
NAMES = {
    "TJS": ("сомони", "Таджикистан", "TJS"),
    "UZS": ("сум", "Узбекистан", "UZS"),
    "KGS": ("сом", "Кыргызстан", "KGS"),
    "AMD": ("драм", "Армения", "AMD"),
    "KZT": ("тенге", "Казахстан", "KZT"),
}
START = pd.Timestamp("2025-09-01")

panel = read_rate_directory(DATA)
table = signal_table(panel, CORRIDORS)

out = {
    "meta": {
        "source": "ЦБ РФ, дневные официальные курсы",
        "panel_from": panel["quote_date"].min().strftime("%Y-%m-%d"),
        "panel_to": panel["quote_date"].max().strftime("%Y-%m-%d"),
        "observations": int(len(panel)),
        "window_from": START.strftime("%Y-%m-%d"),
    },
    "corridors": {},
    "asof_audit": [],
}

for iso in CORRIDORS:
    corridor = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    values = corridor.set_index("quote_date")["rub_per_unit"].astype(float)
    dates = list(values.index)

    view = corridor[corridor["quote_date"] >= START]
    # EWMA over the *whole* history, then sliced: seeding it at the start of the
    # displayed window would draw a warm-up artefact the layer never saw. Span 10
    # is the window the walk-forward tuner selected on every signal in this period.
    trend = corridor["rub_per_unit"].astype(float).ewm(span=10, adjust=False).mean()
    trend.index = corridor["quote_date"]
    series = [
        {
            "d": d.strftime("%Y-%m-%d"),
            "r": round(float(v), 6),
            "e": round(float(trend.loc[d]), 6),
        }
        for d, v in zip(view["quote_date"], view["rub_per_unit"], strict=True)
    ]

    # Median absolute daily move over the displayed window: the size of an
    # ordinary day, used to set the disclosure threshold in the prototype.
    moves = view["rub_per_unit"].astype(float).pct_change().dropna().abs() * 10000
    median_move = float(np.median(moves)) if len(moves) else float("nan")
    p90_move = float(np.quantile(moves, 0.9)) if len(moves) else float("nan")

    sig = table[(table["iso"] == iso) & (table["signal_date"] >= START)]
    signals = []
    for _, r in sig.iterrows():
        d = r["signal_date"]
        i = dates.index(d)
        here = float(values.iloc[i])
        follow = []
        for step in range(1, 6):
            if i + step < len(dates):
                nxt = float(values.iloc[i + step])
                follow.append(
                    {
                        "d": dates[i + step].strftime("%Y-%m-%d"),
                        "r": round(nxt, 6),
                        "bps": round((nxt / here - 1) * 10000, 1),
                    }
                )
        fwd = values.iloc[i + 1 : i + 11]
        signals.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "rate": round(here, 6),
                "follow": follow,
                "fwd10_bps": None if fwd.empty else round((float(fwd.mean()) / here - 1) * 10000, 1),
                "deviation_pct": round(float(r["deviation_pct"]), 3),
                "window": str(r["window"]),
                "span": int(str(r["window"]).split("=")[1]),
                "speed": str(r["speed"]),
                "strength_pct": None if not np.isfinite(float(r["strength_pct"])) else round(float(r["strength_pct"]), 4),
                "level_percentile": None if not np.isfinite(float(r["level_percentile"])) else round(float(r["level_percentile"]), 1),
                "scenario": str(r["scenario"]),
                "direction": str(r["direction"]),
                "message": str(r["message"]),
            }
        )

    weeks = (view["quote_date"].max() - view["quote_date"].min()).days / 7
    out["corridors"][iso] = {
        "iso": iso,
        "name": NAMES[iso][0],
        "country": NAMES[iso][1],
        "series": series,
        "signals": signals,
        "stats": {
            "n_signals": len(signals),
            "per_week": round(len(signals) / weeks, 2),
            "median_move_bps": round(median_move, 1),
            "p90_move_bps": round(p90_move, 1),
            "last_rate": round(float(values.iloc[-1]), 6),
            "last_date": dates[-1].strftime("%Y-%m-%d"),
        },
    }

# No-look-ahead audit: recompute with the future physically removed.
full = signal_table(panel, ["TJS"])
for asof in ["2026-05-19", "2026-06-10", "2026-07-10", "2026-08-21"]:
    t = pd.Timestamp(asof)
    cut = signals_asof(panel, ["TJS"], t)
    a = sorted(cut[cut["signal_date"] >= START]["signal_date"].dt.strftime("%Y-%m-%d"))
    b = sorted(full[(full["signal_date"] >= START) & (full["signal_date"] <= t)]["signal_date"].dt.strftime("%Y-%m-%d"))
    out["asof_audit"].append(
        {"asof": asof, "n": len(a), "identical": a == b, "fired_today": asof in a}
    )

payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
dst = Path("demo/data.js")
dst.parent.mkdir(exist_ok=True)
dst.write_text(
    "/* Generated by scripts/export_demo.py from currency_data/ — real signal-layer output. */\n"
    f"window.DEMO_DATA = {payload};\n",
    encoding="utf-8",
)
print("wrote", dst, dst.stat().st_size)
for iso, c in out["corridors"].items():
    print(iso, c["stats"])
print(out["asof_audit"])
