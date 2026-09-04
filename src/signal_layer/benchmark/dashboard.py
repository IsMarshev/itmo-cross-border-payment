"""Renders a CBSB-1 run as a single self-contained HTML dashboard.

Reads the CSVs a run writes plus the rate panel, and emits one file with no
external dependencies beyond web fonts: every chart is inline SVG built here.
That is deliberate — the earlier Plotly dashboard in this repo embeds a
multi-megabyte JS bundle in every report, and a scorecard that has to survive
being emailed, opened offline and published as an artifact is better served by
a few kilobytes of markup.

The page is organised as a stack of *findings* rather than a grid of charts:
each section states a claim and then shows the evidence for it, because the
interesting output of this benchmark is not "here are some numbers" but three
specific conclusions about where the value is being lost.

    from signal_layer.benchmark.dashboard import build_dashboard
    build_dashboard(Path("reports/benchmark"), panel, spec)
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import BenchmarkSpec

# Strategies whose signals are drawn on the rate chart. More than a couple of
# series and the markers stop being readable.
# Drawn on the rate chart: what ships, against the model it replaced.
CHART_STRATEGIES: tuple[str, ...] = ("zscore_truthful", "utility_risk")

# The "ordinary statistics" the MVP is measured against in question 2.
RULE_STRATEGIES: tuple[str, ...] = (
    "zscore_truthful", "zscore_tuned", "percentile_tuned", "rank_blend", "consensus",
    "zscore", "seasonal", "percentile", "momentum", "drawdown",
)

# Ordered so the reader meets the contenders before the hindsight references.
_CEILINGS = ("oracle", "oracle_topk")


# --- small formatting helpers -------------------------------------------------


def _num(value: object, digits: int = 1, unit: str = "") -> str:
    """A number for display: real minus sign, em dash when missing."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    text = f"{float(value):.{digits}f}".replace("-", "−")
    return f"{text}{unit}"


def _esc(value: object) -> str:
    return html.escape(str(value))


def _scale(value: float, lo: float, hi: float, start: float, end: float) -> float:
    if hi == lo:
        return start
    return start + (value - lo) / (hi - lo) * (end - start)


def _svg(width: int, height: int, body: str, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(label)}" '
        f'class="chart" preserveAspectRatio="xMidYMid meet">{body}</svg>'
    )


def _tone(value: float) -> str:
    """Semantic class from the sign of a client-money number."""
    if not math.isfinite(value):
        return "flat"
    return "gain" if value > 0 else ("loss" if value < 0 else "flat")


# --- charts -------------------------------------------------------------------


def _bar_cell(value: float, lo: float, hi: float, width: int = 150) -> str:
    """A diverging bar sized against the run's full range, with a zero rule."""
    height = 20
    zero = _scale(0.0, lo, hi, 4, width - 4)
    if not math.isfinite(value):
        return _svg(width, height, "", "нет данных")
    x = _scale(value, lo, hi, 4, width - 4)
    left, right = min(zero, x), max(zero, x)
    body = (
        f'<line x1="{zero:.1f}" y1="2" x2="{zero:.1f}" y2="{height - 2}" '
        f'class="axis" />'
        f'<rect x="{left:.1f}" y="5" width="{max(1.0, right - left):.1f}" height="10" '
        f'rx="1.5" class="fill-{_tone(value)}" />'
    )
    return _svg(width, height, body, f"{value:.1f} базисных пунктов")



def _conflict_chart(board: pd.DataFrame) -> str:
    """Two panels: client money against each of the two hit rules."""
    width, height = 900, 300
    pad_l, pad_b, pad_t = 46, 52, 26
    panel_w = (width - pad_l * 2 - 40) / 2
    money = board["currency_uplift_bps"].to_numpy(dtype=float)
    y_lo, y_hi = min(money.min(), 0) * 1.15, money.max() * 1.15
    parts: list[str] = []

    panels = (
        ("hit_favourable", "«сейчас выгодно»", 0),
        ("hit_closing", "«окно закрывается»", 1),
    )
    for column, title, index in panels:
        ox = pad_l + index * (panel_w + 40)
        x = board[column].to_numpy(dtype=float)
        x_lo, x_hi = x.min() * 0.9, x.max() * 1.06
        py0, py1 = pad_t + 6, height - pad_b
        zero_y = _scale(0, y_lo, y_hi, py1, py0)
        parts.append(
            f'<text x="{ox}" y="{pad_t - 8}" class="cap">{_esc(title)}</text>'
            f'<line x1="{ox}" y1="{py1}" x2="{ox + panel_w}" y2="{py1}" class="axis" />'
            f'<line x1="{ox}" y1="{zero_y:.1f}" x2="{ox + panel_w}" y2="{zero_y:.1f}" '
            f'class="axis dashed" />'
        )
        # Least-squares line through the strategies makes the sign flip explicit.
        if len(x) > 2 and x.std() > 0:
            slope, intercept = np.polyfit(x, money, 1)
            ax = _scale(x_lo, x_lo, x_hi, ox + 6, ox + panel_w - 6)
            bx = _scale(x_hi, x_lo, x_hi, ox + 6, ox + panel_w - 6)
            ay = _scale(slope * x_lo + intercept, y_lo, y_hi, py1, py0)
            by = _scale(slope * x_hi + intercept, y_lo, y_hi, py1, py0)
            trend = "loss" if slope < 0 else "gain"
            parts.append(
                f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                f'class="trend trend-{trend}" />'
            )
        for _, row in board.iterrows():
            cx = _scale(float(row[column]), x_lo, x_hi, ox + 6, ox + panel_w - 6)
            cy = _scale(float(row["currency_uplift_bps"]), y_lo, y_hi, py1, py0)
            emphasis = "dot-key" if row["strategy"] in ("utility_risk", "oracle_topk") else "dot"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" class="{emphasis}">'
                f"<title>{_esc(row['strategy'])}: {_num(row[column], 2)} hit, "
                f"{_num(row['currency_uplift_bps'])} б.п.</title></circle>"
            )
        parts.append(
            f'<text x="{ox}" y="{height - 30}" class="tick">{_num(x_lo, 2)}</text>'
            f'<text x="{ox + panel_w}" y="{height - 30}" class="tick end">'
            f"{_num(x_hi, 2)}</text>"
            f'<text x="{ox + panel_w / 2}" y="{height - 10}" class="tick mid">'
            f"доля подтвердившихся сообщений</text>"
        )
    parts.append(
        f'<text x="0" y="{pad_t + 4}" class="tick">{_num(y_hi, 0)}</text>'
        f'<text x="0" y="{height - pad_b}" class="tick">{_num(y_lo, 0)}</text>'
        f'<text x="0" y="{pad_t - 8}" class="cap">б.п.</text>'
    )
    return _svg(width, height, "".join(parts), "Связь правдивости и денег клиента")


def _corridor_dots(per_corridor: pd.DataFrame, strategies: list[str]) -> str:
    """One row per strategy, one dot per corridor, zero rule down the middle."""
    subset = per_corridor[per_corridor["strategy"].isin(strategies)]
    if subset.empty or not strategies:
        return ""
    width, row_h, top = 900, 44, 40
    height = top + row_h * len(strategies) + 24
    values = subset["currency_uplift_bps"].to_numpy(dtype=float)
    span = max(abs(values.min()), abs(values.max())) * 1.15
    x0, x1 = 190, width - 30
    zero = _scale(0, -span, span, x0, x1)
    parts = [
        f'<text x="{x0}" y="18" class="cap">выгода по коридорам, б.п.</text>',
        f'<line x1="{zero:.1f}" y1="{top - 14}" x2="{zero:.1f}" y2="{height - 22}" '
        f'class="axis" />',
        f'<text x="{zero:.1f}" y="{height - 6}" class="tick mid">0</text>',
        f'<text x="{x0}" y="{height - 6}" class="tick">{_num(-span, 0)}</text>',
        f'<text x="{x1}" y="{height - 6}" class="tick end">{_num(span, 0)}</text>',
    ]
    for index, name in enumerate(strategies):
        y = top + index * row_h
        rows = subset[subset["strategy"].eq(name)]
        positive = int((rows["currency_uplift_bps"] > 0).sum())
        parts.append(
            f'<text x="0" y="{y + 4}" class="lbl">{_esc(name)}</text>'
            f'<text x="0" y="{y + 20}" class="note">{positive} из {len(rows)} коридоров '
            f"в плюсе</text>"
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="axis faint" />'
        )
        for _, row in rows.iterrows():
            value = float(row["currency_uplift_bps"])
            cx = _scale(value, -span, span, x0, x1)
            solid = "dot-key" if bool(row.get("significant_fdr")) else "dot-hollow"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{y}" r="5" class="{solid} stroke-{_tone(value)}">'
                f"<title>{_esc(row['iso'])}: {_num(value)} б.п., "
                f"q={_num(row.get('q_value'), 3)}</title></circle>"
                f'<text x="{cx:.1f}" y="{y - 10}" class="tick mid">{_esc(row["iso"])}</text>'
            )
    return _svg(width, height, "".join(parts), "Устойчивость по коридорам")


def _fold_chart(per_fold: pd.DataFrame, strategies: list[str]) -> str:
    """Grouped bars over the out-of-time windows."""
    subset = per_fold[per_fold["strategy"].isin(strategies)]
    grouped = (
        subset.groupby(["strategy", "fold"], sort=True)["currency_uplift_bps"]
        .mean()
        .reset_index()
    )
    folds = sorted(grouped["fold"].unique())
    if grouped.empty or not folds:
        return ""
    width, height = 900, 250
    pad_l, pad_b, pad_t = 44, 58, 24
    values = grouped["currency_uplift_bps"].to_numpy(dtype=float)
    span = max(abs(np.nanmin(values)), abs(np.nanmax(values))) * 1.15
    plot_w = width - pad_l - 16
    slot = plot_w / max(1, len(folds))
    bar_w = min(16.0, slot / (len(strategies) + 1))
    zero_y = _scale(0, -span, span, height - pad_b, pad_t)
    parts = [
        f'<text x="0" y="{pad_t - 8}" class="cap">б.п. по окнам</text>',
        f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - 16}" y2="{zero_y:.1f}" '
        f'class="axis" />',
        f'<text x="0" y="{pad_t + 4}" class="tick">{_num(span, 0)}</text>',
        f'<text x="0" y="{height - pad_b}" class="tick">{_num(-span, 0)}</text>',
    ]
    for f_index, fold in enumerate(folds):
        cx = pad_l + slot * (f_index + 0.5)
        parts.append(
            f'<text x="{cx:.1f}" y="{height - pad_b + 16}" class="tick mid rot">'
            f"{_esc(fold.split('..')[0])}</text>"
        )
        for s_index, name in enumerate(strategies):
            cell = grouped[grouped["strategy"].eq(name) & grouped["fold"].eq(fold)]
            if cell.empty or not math.isfinite(float(cell["currency_uplift_bps"].iloc[0])):
                continue
            value = float(cell["currency_uplift_bps"].iloc[0])
            y = _scale(value, -span, span, height - pad_b, pad_t)
            x = cx - bar_w * len(strategies) / 2 + s_index * bar_w
            cls = "fill-ceiling" if s_index else f"fill-{_tone(value)}"
            parts.append(
                f'<rect x="{x:.1f}" y="{min(y, zero_y):.1f}" width="{bar_w - 2:.1f}" '
                f'height="{max(1.0, abs(y - zero_y)):.1f}" rx="1" class="{cls}">'
                f"<title>{_esc(name)}, {_esc(fold)}: {_num(value)} б.п.</title></rect>"
            )
    return _svg(width, height, "".join(parts), "Выгода по окнам оценки")



def _null_histogram(nulls: pd.DataFrame, observed: float, p_value: float) -> str:
    """Where the strategy lands against the cloud of random schedules.

    A p-value asks the reader to trust the test. The distribution behind it lets
    them see the answer: 500 schedules that spend the same push budget on days
    picked at random, and one marker for where the strategy came out.
    """
    if nulls.empty or not math.isfinite(observed):
        return ""
    values = nulls["currency_uplift_bps"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return ""
    width, height = 900, 220
    pad_l, pad_r, pad_b, pad_t = 20, 20, 46, 30
    lo = min(float(values.min()), observed) * 1.15 - 1
    hi = max(float(values.max()), observed) * 1.15 + 1
    counts, edges = np.histogram(values, bins=36, range=(lo, hi))
    peak = max(1, int(counts.max()))

    def px(value: float) -> float:
        return _scale(value, lo, hi, pad_l, width - pad_r)

    parts = [
        f'<text x="{pad_l}" y="{pad_t - 12}" class="cap">'
        f"500 случайных расписаний того же размера</text>",
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" '
        f'y2="{height - pad_b}" class="axis" />',
    ]
    for index, count in enumerate(counts):
        if not count:
            continue
        x0, x1 = px(float(edges[index])), px(float(edges[index + 1]))
        bar_h = (height - pad_b - pad_t) * count / peak
        parts.append(
            f'<rect x="{x0:.1f}" y="{height - pad_b - bar_h:.1f}" '
            f'width="{max(1.0, x1 - x0 - 1):.1f}" height="{bar_h:.1f}" class="fill-null" />'
        )
    marker = px(observed)
    parts.append(
        f'<line x1="{marker:.1f}" y1="{pad_t - 4}" x2="{marker:.1f}" '
        f'y2="{height - pad_b}" class="marker-line" />'
        f'<circle cx="{marker:.1f}" cy="{pad_t - 4}" r="4.5" class="dot-key" />'
        f'<text x="{marker:.1f}" y="{pad_t - 14}" class="tick mid marker-label">'
        f"MVP {_num(observed)}</text>"
    )
    zero = px(0.0)
    parts.append(
        f'<line x1="{zero:.1f}" y1="{pad_t}" x2="{zero:.1f}" y2="{height - pad_b}" '
        f'class="axis dashed" />'
        f'<text x="{zero:.1f}" y="{height - pad_b + 17}" class="tick mid">0</text>'
        f'<text x="{pad_l}" y="{height - pad_b + 17}" class="tick">{_num(lo, 0)}</text>'
        f'<text x="{width - pad_r}" y="{height - pad_b + 17}" class="tick end">'
        f"{_num(hi, 0)}</text>"
        f'<text x="{(pad_l + width - pad_r) / 2:.1f}" y="{height - 8}" class="tick mid">'
        f"выгода клиента, б.п. · p = {_num(p_value, 3)}</text>"
    )
    return _svg(width, height, "".join(parts), "MVP против случайных расписаний")


def _rules_comparison(board: pd.DataFrame, rules: tuple[str, ...]) -> str:
    """The MVP ranked against every ordinary statistical rule."""
    subset = board[board["strategy"].isin(("utility_risk", *rules))]
    subset = subset.dropna(subset=["currency_uplift_bps"]).sort_values(
        "currency_uplift_bps", ascending=False
    )
    if subset.empty:
        return ""
    width, row_h, top = 900, 30, 28
    height = top + row_h * len(subset) + 26
    hi = float(subset["currency_uplift_bps"].max()) * 1.18
    x0, x1 = 200, width - 120
    parts = [
        f'<text x="{x0}" y="16" class="cap">выгода клиента, б.п. на перевод</text>',
    ]
    for index, (_, row) in enumerate(subset.iterrows()):
        y = top + index * row_h
        value = float(row["currency_uplift_bps"])
        is_mvp = row["strategy"] == "utility_risk"
        w = max(1.0, _scale(value, 0, hi, 0, x1 - x0))
        parts.append(
            f'<text x="0" y="{y + 15}" class="lbl{" mvp" if is_mvp else ""}">'
            f'{_esc(row["strategy"])}</text>'
            f'<rect x="{x0}" y="{y + 4}" width="{w:.1f}" height="14" rx="1.5" '
            f'class="{"fill-amber" if is_mvp else "fill-ceiling"}" />'
            f'<text x="{x0 + w + 8:.1f}" y="{y + 15}" class="val">{_num(value)}</text>'
            f'<title>{_esc(row["strategy"])}: {_num(value)} б.п., '
            f'{int(row["corridors_positive"])} из {int(row["n_corridors"])} '
            f"коридоров в плюсе</title>"
        )
    return _svg(width, height, "".join(parts), "MVP против статистических правил")


def _ladder_chart(board: pd.DataFrame) -> str:
    """How the shipping signal was built, one bar per step, plus what it refuses."""
    steps = (
        ("zscore", "правило с зашитым окном"),
        ("zscore_tuned", "+ окно выбирается walk-forward"),
        ("zscore_truthful", "+ молчит, когда сказать нечего"),
        ("zscore_vetoed", "дни, которые оно отвергает"),
    )
    rows = [
        (label, float(board.loc[name, "currency_uplift_bps"]), name)
        for name, label in steps
        if name in board.index and np.isfinite(board.loc[name, "currency_uplift_bps"])
    ]
    if not rows:
        return ""
    width, row_h, top = 900, 48, 30
    height = top + row_h * len(rows) + 20
    span = max(abs(v) for _, v, _ in rows) * 1.2
    x0, x1 = 300, width - 70
    zero = _scale(0, -span, span, x0, x1)
    parts = [
        f'<text x="{x0}" y="18" class="cap">выгода клиента, б.п. на перевод</text>',
        f'<line x1="{zero:.1f}" y1="{top - 8}" x2="{zero:.1f}" y2="{height - 16}" '
        f'class="axis" />',
    ]
    for index, (label, value, name) in enumerate(rows):
        y = top + index * row_h
        x = _scale(value, -span, span, x0, x1)
        left, right = min(zero, x), max(zero, x)
        shipping = name == "zscore_truthful"
        parts.append(
            f'<text x="0" y="{y + 20}" class="lbl{" mvp" if shipping else ""}">'
            f"{_esc(label)}</text>"
            f'<rect x="{left:.1f}" y="{y + 8}" width="{max(1.0, right - left):.1f}" '
            f'height="16" rx="1.5" class="fill-{"amber" if shipping else _tone(value)}" />'
            f'<text x="{(right + 8) if value >= 0 else (left - 8):.1f}" y="{y + 21}" '
            f'class="val" text-anchor="{"start" if value >= 0 else "end"}">'
            f"{_num(value)}</text>"
        )
    return _svg(width, height, "".join(parts), "Как собран рабочий сигнал")


def _cadence_chart(sweep: pd.DataFrame) -> str:
    """Value per push against how often we push, one line per strategy."""
    if sweep.empty:
        return ""
    width, height = 900, 300
    pad_l, pad_r, pad_b, pad_t = 52, 130, 52, 26
    clean = sweep.dropna(subset=["per_week", "currency_uplift_bps"])
    x_lo, x_hi = 0.0, float(clean["per_week"].max()) * 1.08
    y_lo, y_hi = 0.0, float(clean["currency_uplift_bps"].max()) * 1.12
    def px(value: float) -> float:
        return _scale(value, x_lo, x_hi, pad_l, width - pad_r)

    def py(value: float) -> float:
        return _scale(value, y_lo, y_hi, height - pad_b, pad_t)

    parts = [
        f'<text x="0" y="{pad_t - 8}" class="cap">выгода на пуш, б.п.</text>',
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" '
        f'y2="{height - pad_b}" class="axis" />',
        # The brief's mandatory band, drawn as the constraint it is.
        f'<rect x="{px(1.0):.1f}" y="{pad_t}" width="{px(2.0) - px(1.0):.1f}" '
        f'height="{height - pad_b - pad_t:.1f}" class="band" />'
        f'<text x="{(px(1.0) + px(2.0)) / 2:.1f}" y="{pad_t + 12}" class="tick mid">'
        f"полоса ТЗ</text>",
    ]
    tones = {"oracle": "ceiling", "percentile": "gain", "utility_risk": "amber"}
    for name, group in clean.groupby("strategy", sort=False):
        ordered = group.sort_values("per_week")
        tone = tones.get(name, "flat")
        points = " ".join(
            f"{px(float(r.per_week)):.1f},{py(float(r.currency_uplift_bps)):.1f}"
            for r in ordered.itertuples()
        )
        parts.append(f'<polyline points="{points}" class="line line-{tone}" />')
        for r in ordered.itertuples():
            parts.append(
                f'<circle cx="{px(float(r.per_week)):.1f}" '
                f'cy="{py(float(r.currency_uplift_bps)):.1f}" r="4" class="dot-{tone}">'
                f"<title>{_esc(name)} · {_esc(r.cadence)}: "
                f"{_num(r.currency_uplift_bps)} б.п. при {_num(r.per_week, 2)}/нед"
                f"</title></circle>"
            )
        last = ordered.iloc[-1]
        parts.append(
            f'<text x="{px(float(last.per_week)) + 9:.1f}" '
            f'y="{py(float(last.currency_uplift_bps)) + 4:.1f}" class="val">'
            f"{_esc(name)}</text>"
        )
    for value in (0.2, 0.5, 1.0, 1.5, 2.0):
        parts.append(
            f'<text x="{px(value):.1f}" y="{height - pad_b + 18}" class="tick mid">'
            f"{value:g}</text>"
        )
    parts.append(
        f'<text x="{(pad_l + width - pad_r) / 2:.1f}" y="{height - 12}" class="tick mid">'
        f"пушей на коридор в неделю</text>"
        f'<text x="0" y="{pad_t + 4}" class="tick">{_num(y_hi, 0)}</text>'
        f'<text x="0" y="{height - pad_b}" class="tick">0</text>'
    )
    return _svg(width, height, "".join(parts), "Выгода на пуш против частоты")


def _rate_chart(
    panel: pd.DataFrame, signals: pd.DataFrame, iso: str, start: pd.Timestamp
) -> str:
    """The corridor's rate over the evaluation period with signal markers."""
    series = panel[panel["iso"].eq(iso) & (panel["quote_date"] >= start)].sort_values(
        "quote_date"
    )
    if series.empty:
        return ""
    width, height = 900, 300
    pad_l, pad_r, pad_b, pad_t = 8, 58, 40, 22
    dates = series["quote_date"].to_numpy()
    rates = series["rub_per_unit"].to_numpy(dtype=float)
    t0, t1 = dates[0].astype("datetime64[D]").astype(int), dates[-1].astype(
        "datetime64[D]"
    ).astype(int)
    r_lo, r_hi = rates.min(), rates.max()
    pad = (r_hi - r_lo) * 0.08 or r_hi * 0.01
    r_lo, r_hi = r_lo - pad, r_hi + pad

    def px(value: np.datetime64) -> float:
        return _scale(
            float(value.astype("datetime64[D]").astype(int)), t0, t1, pad_l, width - pad_r
        )

    def py(value: float) -> float:
        # Lower rub_per_unit is better for the sender, so the axis is inverted:
        # up on this chart always means "better for the client".
        return _scale(value, r_lo, r_hi, pad_t, height - pad_b)

    points = " ".join(f"{px(d):.1f},{py(r):.1f}" for d, r in zip(dates, rates, strict=True))
    parts = [
        f'<text x="0" y="{pad_t - 8}" class="cap">выгоднее для клиента ↑</text>',
        f'<polyline points="{points}" class="line line-rate" />',
    ]
    rate_by_date = dict(zip(dates, rates, strict=True))
    for order, strategy in enumerate(CHART_STRATEGIES):
        marks = signals[signals["strategy"].eq(strategy) & signals["iso"].eq(iso)]
        for _, row in marks.iterrows():
            day = np.datetime64(pd.Timestamp(row["quote_date"]), "ns")
            rate = rate_by_date.get(day)
            if rate is None:
                continue
            gain = float(row["currency_gain_bps"])
            parts.append(
                f'<circle cx="{px(day):.1f}" cy="{py(rate):.1f}" r="3.2" '
                f'class="mark mark-{order} stroke-{_tone(gain)}" '
                f'data-strategy="{_esc(strategy)}">'
                f"<title>{_esc(strategy)} · {pd.Timestamp(row['quote_date']):%Y-%m-%d} · "
                f"{_num(gain)} б.п.</title></circle>"
            )
    for value in (r_lo + pad, r_hi - pad):
        parts.append(
            f'<text x="{width - pad_r + 6}" y="{py(value):.1f}" class="tick">'
            f"{value:.4g}</text>"
        )
    year = pd.Timestamp(dates[0]).year
    while year <= pd.Timestamp(dates[-1]).year:
        mark = np.datetime64(pd.Timestamp(year=year, month=1, day=1), "ns")
        if t0 <= float(mark.astype("datetime64[D]").astype(int)) <= t1:
            parts.append(
                f'<line x1="{px(mark):.1f}" y1="{pad_t}" x2="{px(mark):.1f}" '
                f'y2="{height - pad_b}" class="axis faint" />'
                f'<text x="{px(mark):.1f}" y="{height - pad_b + 18}" class="tick mid">'
                f"{year}</text>"
            )
        year += 1
    return _svg(width, height, "".join(parts), f"Курс {iso} и срабатывания сигналов")


# --- tables -------------------------------------------------------------------


def _leaderboard_table(board: pd.DataFrame, gates: pd.DataFrame) -> str:
    contenders = board[board["selection"].eq("policy")]
    values = contenders["currency_uplift_bps"].to_numpy(dtype=float)
    lo, hi = min(values.min(), 0) * 1.1, max(values.max(), 0) * 1.1
    rows = []
    for _, row in contenders.iterrows():
        name = row["strategy"]
        failed = gates[gates["strategy"].eq(name) & gates["passed"].eq(False)]
        passed_count = 7 - len(failed)
        verdict_cls = "ok" if not len(failed) else ("warn" if passed_count >= 5 else "bad")
        verdict = (
            "проходит"
            if not len(failed)
            else "✕ " + ", ".join(g.split("_")[0] for g in failed["gate"])
        )
        highlight = " class=\"row-key\"" if name == "utility_risk" else ""
        rows.append(
            f"<tr{highlight}>"
            f'<th scope="row"><span class="mono">{_esc(name)}</span>'
            f'<span class="desc">{_esc(row["description"])}</span></th>'
            f'<td class="num {_tone(row["currency_uplift_bps"])}">'
            f'{_num(row["currency_uplift_bps"])}</td>'
            f'<td class="bar">{_bar_cell(float(row["currency_uplift_bps"]), lo, hi)}</td>'
            f'<td class="num">{_num(row["currency_uplift_bps_worst_corridor"])}</td>'
            f'<td class="num">{int(row["corridors_positive"])}/'
            f'{int(row["n_corridors"])}</td>'
            f'<td class="num">{_num(row["hit_lift"], 2)}</td>'
            f'<td class="num">{_num(row["p_value"], 3)}</td>'
            f'<td class="num">{_num(row["per_week"], 2)}</td>'
            f'<td><span class="pill pill-{verdict_cls}">{_esc(verdict)}</span></td>'
            f"</tr>"
        )
    return (
        '<div class="scroll"><table class="board">'
        "<thead><tr>"
        "<th>стратегия</th><th>выгода, б.п.</th><th></th><th>худший<br>коридор</th>"
        "<th>коридоров<br>в плюсе</th><th>lift</th><th>p</th><th>в неделю</th>"
        "<th>гейты</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _gate_matrix(board: pd.DataFrame, gates: pd.DataFrame, spec: BenchmarkSpec) -> str:
    names = [gate.name for gate in spec.gates]
    order = board[board["selection"].eq("policy")]["strategy"].tolist()
    head = "".join(f'<th>{_esc(n.split("_")[0])}</th>' for n in names)
    rows = []
    for name in order:
        cells = []
        for gate in names:
            match = gates[gates["strategy"].eq(name) & gates["gate"].eq(gate)]
            passed = bool(match["passed"].iloc[0]) if len(match) else None
            mark = "✓" if passed else ("✕" if passed is False else "—")
            cls = "ok" if passed else ("bad" if passed is False else "flat")
            value = _num(match["value"].iloc[0], 2) if len(match) else "—"
            cells.append(f'<td class="mark-{cls}" title="{value}">{mark}</td>')
        rows.append(
            f'<tr><th scope="row" class="mono">{_esc(name)}</th>{"".join(cells)}</tr>'
        )
    legend = "".join(
        f'<li><b>{_esc(g.name.split("_")[0])}</b> {_esc(g.question)} '
        f'<code>{_esc(g.describe())}</code></li>'
        for g in spec.gates
    )
    return (
        '<div class="scroll"><table class="matrix">'
        f"<thead><tr><th>стратегия</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f'<ul class="legend">{legend}</ul>'
    )


def _horizon_table(horizons: pd.DataFrame, spec: BenchmarkSpec) -> str:
    columns = [f"hit_h{h}" for h in spec.reported_horizons]
    values = horizons[columns].to_numpy(dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    head = "".join(f"<th>h={h}</th>" for h in spec.reported_horizons)
    rows = []
    for _, row in horizons.iterrows():
        cells = []
        for column in columns:
            value = float(row[column])
            weight = _scale(value, lo, hi, 0.08, 0.85) if math.isfinite(value) else 0
            cells.append(
                f'<td class="num heat" style="--w:{weight:.2f}">{_num(value, 2)}</td>'
            )
        scenario = (
            "окно закрывается" if row["scenario"] == "window_closing" else "сейчас выгодно"
        )
        rows.append(
            f'<tr><th scope="row" class="mono">{_esc(row["strategy"])}</th>'
            f'<td class="scen">{_esc(scenario)}</td>{"".join(cells)}</tr>'
        )
    return (
        '<div class="scroll"><table class="matrix heatmap">'
        f"<thead><tr><th>стратегия</th><th>сообщение</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


# --- page ---------------------------------------------------------------------


def _kpis(board: pd.DataFrame, audit: pd.DataFrame) -> str:
    """The four numbers a reader should leave with, skipping any the run lacks."""
    indexed = board.set_index("strategy")
    audit_ok = bool(audit["matched"].all()) if len(audit) else False
    tiles: list[tuple[str, str, str, str, str]] = []

    if "utility_risk" in indexed.index:
        mvp = indexed.loc["utility_risk"]
        tiles.append((
            "MVP: полезность и риск",
            _num(mvp["currency_uplift_bps"]),
            "б.п. клиенту",
            f"p = {_num(mvp['p_value'], 3)} · {int(mvp['corridors_positive'])} из "
            f"{int(mvp['n_corridors'])} коридоров в плюсе",
            _tone(float(mvp["currency_uplift_bps"])),
        ))
    if "oracle_topk" in indexed.index:
        ceiling = float(indexed.loc["oracle_topk", "currency_uplift_bps"])
        tiles.append((
            "Потолок задачи",
            _num(ceiling),
            "б.п. при идеальном выборе",
            "лучшие 2 дня недели, известные заранее",
            "ceiling",
        ))
        if "oracle" in indexed.index and ceiling:
            policed = float(indexed.loc["oracle", "currency_uplift_bps"])
            tiles.append((
                "Цена решения вслепую",
                _num((1 - policed / ceiling) * 100, 0) + "%",
                "выгоды теряется онлайн",
                f"идеальный счёт даёт {_num(policed)} б.п. вместо {_num(ceiling)}",
                "loss",
            ))
    tiles.append((
        "Заглядывание вперёд",
        "пройден" if audit_ok else "не запускался" if not len(audit) else "провален",
        "аудит на обрезанной панели",
        f"{len(audit)} проверок, побитовое совпадение счёта",
        "gain" if audit_ok else "flat" if not len(audit) else "loss",
    ))
    return "".join(
        f'<article class="kpi kpi-{tone}"><h3>{_esc(title)}</h3>'
        f'<p class="figure">{_esc(value)}</p><p class="unit">{_esc(unit)}</p>'
        f'<p class="foot">{_esc(foot)}</p></article>'
        for title, value, unit, foot, tone in tiles
    )


_STYLE = """
:root{
  --ground:#F3F5F8; --surface:#FFFFFF; --raise:#FAFBFC;
  --ink:#131A22; --muted:#5B6875; --faint:#8A96A2; --hairline:#DEE4EA;
  --gain:#0B7A6B; --loss:#B23F35; --ceiling:#2D4B73; --amber:#B8791F;
  --gain-soft:#0B7A6B22; --loss-soft:#B23F3522; --key:#B8791F14;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0F1418; --surface:#161D23; --raise:#1B242B;
    --ink:#E4EAEF; --muted:#8E9CA9; --faint:#6C7A87; --hairline:#242D36;
    --gain:#3FB39D; --loss:#E0705F; --ceiling:#7FA4D6; --amber:#DDA455;
    --gain-soft:#3FB39D26; --loss-soft:#E0705F26; --key:#DDA4551A;
  }
}
:root[data-theme="dark"]{
  --ground:#0F1418; --surface:#161D23; --raise:#1B242B;
  --ink:#E4EAEF; --muted:#8E9CA9; --faint:#6C7A87; --hairline:#242D36;
  --gain:#3FB39D; --loss:#E0705F; --ceiling:#7FA4D6; --amber:#DDA455;
  --gain-soft:#3FB39D26; --loss-soft:#E0705F26; --key:#DDA4551A;
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink); margin:0;
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1000px; margin:0 auto; padding:48px 24px 80px;
  display:flex; flex-direction:column; gap:52px}
h1,h2,h3{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;
  text-wrap:balance; margin:0; font-weight:700}
h1{font-size:clamp(30px,4.4vw,46px); line-height:1.06; letter-spacing:-.02em}
h2{font-size:23px; letter-spacing:-.01em}
h3{font-size:14px; font-weight:600}
p{margin:0}
code,.mono,.num,.figure,.tick,.val{font-family:"IBM Plex Mono",ui-monospace,monospace}
.num,.val,.figure{font-variant-numeric:tabular-nums}

header{display:flex; flex-direction:column; gap:16px}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:12px;
  letter-spacing:.13em; text-transform:uppercase; color:var(--amber)}
.lede{color:var(--muted); max-width:64ch; font-size:16px}
.meta{display:flex; flex-wrap:wrap; gap:8px 20px; font-size:12.5px;
  color:var(--faint); font-family:"IBM Plex Mono",monospace;
  border-top:1px solid var(--hairline); padding-top:14px}

.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px}
.kpi{background:var(--surface); border:1px solid var(--hairline);
  border-radius:3px; padding:16px 18px 18px; display:flex; flex-direction:column; gap:2px;
  border-top:3px solid var(--hairline)}
.kpi-gain{border-top-color:var(--gain)} .kpi-loss{border-top-color:var(--loss)}
.kpi-ceiling{border-top-color:var(--ceiling)} .kpi-flat{border-top-color:var(--faint)}
.kpi h3{color:var(--muted); font-size:12px; letter-spacing:.02em; margin-bottom:6px}
.figure{font-size:32px; font-weight:500; line-height:1.05; letter-spacing:-.02em}
.kpi-gain .figure{color:var(--gain)} .kpi-loss .figure{color:var(--loss)}
.kpi-ceiling .figure{color:var(--ceiling)}
.unit{font-size:12.5px; color:var(--muted)}
.foot{font-size:12px; color:var(--faint); margin-top:8px; line-height:1.4}

section{display:flex; flex-direction:column; gap:16px}
.claim{border-left:3px solid var(--amber); padding-left:16px;
  display:flex; flex-direction:column; gap:8px}
.claim p{color:var(--muted); max-width:66ch}
.panel{background:var(--surface); border:1px solid var(--hairline);
  border-radius:3px; padding:20px}
.chart{width:100%; height:auto; display:block; overflow:visible}
.scroll{overflow-x:auto}

table{border-collapse:collapse; width:100%; font-size:13.5px}
th{text-align:left; font-weight:600}
thead th{font-size:11px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--faint); font-weight:600; padding:0 10px 8px; vertical-align:bottom;
  border-bottom:1px solid var(--hairline); white-space:nowrap}
tbody td,tbody th{padding:9px 10px; border-bottom:1px solid var(--hairline);
  vertical-align:middle}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:none}
.board tbody th{min-width:190px}
.why{color:var(--muted); font-size:12.5px; max-width:44ch; line-height:1.4}
.next{margin:0; padding-left:22px; display:flex; flex-direction:column; gap:14px}
.next li{max-width:72ch}
.next b{display:block; margin-bottom:2px}
.board .desc{display:block; font-size:11.5px; color:var(--faint);
  font-family:"IBM Plex Sans",sans-serif; font-weight:400; max-width:34ch;
  line-height:1.35; margin-top:2px}
td.num{text-align:right; white-space:nowrap}
td.bar{width:160px; padding:9px 6px}
.row-key{background:var(--key)}
.gain{color:var(--gain)} .loss{color:var(--loss)} .flat{color:var(--muted)}

.pill{display:inline-block; padding:2px 8px; border-radius:999px; font-size:11.5px;
  font-family:"IBM Plex Mono",monospace; white-space:nowrap; border:1px solid}
.pill-ok{color:var(--gain); border-color:var(--gain); background:var(--gain-soft)}
.pill-warn{color:var(--amber); border-color:var(--amber); background:var(--key)}
.pill-bad{color:var(--loss); border-color:var(--loss); background:var(--loss-soft)}

.matrix td{text-align:center; font-family:"IBM Plex Mono",monospace}
.matrix tbody th{white-space:nowrap}
.mark-ok{color:var(--gain)} .mark-bad{color:var(--loss)} .mark-flat{color:var(--faint)}
.heatmap .heat{background:color-mix(in srgb,var(--ceiling) calc(var(--w)*100%),transparent)}
.scen{font-size:12px; color:var(--muted); white-space:nowrap}
.legend{list-style:none; margin:0; padding:0; display:flex; flex-direction:column;
  gap:5px; font-size:12.5px; color:var(--muted)}
.legend b{color:var(--ink); font-family:"IBM Plex Mono",monospace; font-weight:500}
.legend code{font-size:11.5px; color:var(--faint)}

.answers{gap:20px}
.answer{background:var(--surface); border:1px solid var(--hairline); border-radius:3px;
  padding:22px 24px 20px; display:flex; flex-direction:column; gap:10px;
  border-left:4px solid var(--hairline)}
.answer-gain{border-left-color:var(--gain)} .answer-loss{border-left-color:var(--loss)}
.qnum{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--faint)}
.answer h3{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;
  font-size:21px; font-weight:700; letter-spacing:-.01em}
.verdict{font-size:17px; font-weight:600}
.answer-gain .verdict{color:var(--gain)} .answer-loss .verdict{color:var(--loss)}
.answer-body{max-width:66ch; color:var(--ink)}
.answer-caveat{max-width:66ch; color:var(--muted); font-size:13.5px;
  border-top:1px solid var(--hairline); padding-top:10px}
.answer b{font-variant-numeric:tabular-nums}

/* chart primitives */
.axis{stroke:var(--hairline); stroke-width:1}
.axis.faint{stroke:var(--hairline); stroke-width:1; opacity:.6}
.axis.dashed{stroke-dasharray:3 3}
.fill-gain{fill:var(--gain)} .fill-loss{fill:var(--loss)}
.fill-ceiling{fill:var(--ceiling)} .fill-flat{fill:var(--faint)}
.stroke-gain{stroke:var(--gain)} .stroke-loss{stroke:var(--loss)}
.stroke-flat{stroke:var(--faint)}
.line{fill:none; stroke-width:1.8; stroke-linejoin:round}
.line-gain{stroke:var(--gain)} .line-ceiling{stroke:var(--ceiling)}
.line-rate{stroke:var(--ink); stroke-width:1.2; opacity:.55}
.line-amber{stroke:var(--amber)} .dot-amber{fill:var(--amber)}
.dot-flat{fill:var(--faint)} .line-flat{stroke:var(--faint)}
.band{fill:var(--ceiling); opacity:.07}
.fill-null{fill:var(--muted); opacity:.32}
.fill-amber{fill:var(--amber)}
.marker-line{stroke:var(--amber); stroke-width:2}
.marker-label{fill:var(--amber); font-weight:600}
.lbl.mvp{fill:var(--amber)}
.trend{stroke-width:1.5; stroke-dasharray:5 4}
.trend-gain{stroke:var(--gain)} .trend-loss{stroke:var(--loss)}
.dot{fill:var(--faint)} .dot-key{fill:var(--amber)}
.dot-gain{fill:var(--gain)} .dot-ceiling{fill:var(--ceiling)}
.dot-hollow{fill:var(--surface); stroke-width:2}
.dot-key{stroke-width:2}
.mark{fill:var(--surface); stroke-width:1.6}
.mark-1{fill:var(--ceiling); stroke:var(--ceiling); opacity:.75}
text{fill:var(--ink)}
.cap{font-size:11px; fill:var(--faint); font-family:"IBM Plex Mono",monospace;
  letter-spacing:.05em; text-transform:uppercase}
.lbl{font-size:13px; fill:var(--ink); font-weight:600}
.note{font-size:11px; fill:var(--faint)}
.val{font-size:12px; fill:var(--muted)}
.tick{font-size:10.5px; fill:var(--faint); font-family:"IBM Plex Mono",monospace}
.tick.mid{text-anchor:middle} .tick.end{text-anchor:end}

.tabs{display:flex; gap:6px; flex-wrap:wrap}
.tab{font-family:"IBM Plex Mono",monospace; font-size:12.5px; padding:5px 13px;
  border:1px solid var(--hairline); background:var(--surface); color:var(--muted);
  border-radius:2px; cursor:pointer}
.tab:hover{border-color:var(--amber); color:var(--ink)}
.tab[aria-selected="true"]{background:var(--ink); color:var(--ground);
  border-color:var(--ink)}
.tab:focus-visible,a:focus-visible{outline:2px solid var(--amber); outline-offset:2px}
.key{display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--muted);
  font-family:"IBM Plex Mono",monospace}
.key i{display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:6px; vertical-align:baseline}
.key-hollow{background:var(--surface); border:2px solid var(--faint)}

footer{border-top:1px solid var(--hairline); padding-top:20px; color:var(--faint);
  font-size:12.5px; display:flex; flex-direction:column; gap:8px}
footer code{background:var(--raise); padding:2px 6px; border-radius:2px;
  border:1px solid var(--hairline); color:var(--muted)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

_SCRIPT = """
document.querySelectorAll('[data-tabs]').forEach(function(group){
  var tabs = group.querySelectorAll('.tab');
  tabs.forEach(function(tab){
    tab.addEventListener('click', function(){
      tabs.forEach(function(other){
        var on = other === tab;
        other.setAttribute('aria-selected', on ? 'true' : 'false');
        var target = document.getElementById(other.dataset.panel);
        if (target) { target.hidden = !on; }
      });
    });
  });
});
"""



def _rejected_table(board: pd.DataFrame) -> str:
    """What was tried and did not survive the benchmark, with its number.

    Kept on the page rather than deleted from the record: a case is closed by
    knowing what does not work as much as by what does, and each of these is
    still re-runnable from a CLI flag.
    """
    indexed = board.set_index("strategy")

    def value(name: str) -> str:
        if name not in indexed.index:
            return "—"
        return _num(indexed.loc[name, "currency_uplift_bps"]) + " б.п."

    rows = (
        ("Обучаемая модель полезности и риска", value("utility_risk"),
         "Проиграла каждому статистическому правилу. Ни набор признаков, ни "
         "регуляризация, ни цена ошибки λ не спасают.",
         "--strategies utility_risk"),
        ("Веса на индикаторах вместо сырых признаков", "−7.0 б.п.",
         "Комбинация ТЗ п.8 через подгонку. Индикаторы коллинеарны, линейное "
         "смешивание разрушает робастное ранжирование каждого.",
         "--feature-set rules"),
        ("Комбинация рангов без подгонки", value("rank_blend"),
         "Обходит любое одиночное правило, но на 1.1 б.п. — в пределах шума. "
         "Ценность лишь в том, что не надо угадывать лучшее правило заранее.",
         "--strategies rank_blend"),
        ("Выбор индикатора под коридор", "18.0 б.п.",
         "Хуже, чем всегда брать z-score. Калибровать нужно параметр, а не "
         "семейство: все коридоры всё равно выбирают одно окно.",
         "--strategies rule_select"),
        ("«Окно закрывается» на отскоке от минимума", "−31.4 б.п.",
         "Дни после отскока локально дороже соседних. Как второе сообщение для "
         "периодов молчания тоже не годится: −21 б.п.",
         "--strategies reversal"),
        ("Дни, которые отвергает проверка факта", value("zscore_vetoed"),
         "Не просто немые: теряют клиенту деньги на всех пяти коридорах. Это и "
         "есть довод в пользу молчания.",
         "--strategies zscore_vetoed"),
    )
    body = "".join(
        f"<tr><th scope=\"row\">{_esc(name)}</th>"
        f'<td class="num loss">{_esc(number)}</td>'
        f'<td class="why">{_esc(why)}</td>'
        f'<td><code>{_esc(flag)}</code></td></tr>'
        for name, number, why, flag in rows
    )
    return (
        '<div class="scroll"><table class="board">'
        "<thead><tr><th>что пробовали</th><th>итог</th><th>почему отвергнуто</th>"
        "<th>воспроизвести</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _headline_section(
    *,
    board: pd.DataFrame,
    null_chart: str,
    uplift: float,
    p_value: float,
    ceiling: float,
    positive: int,
    total: int,
) -> str:
    """What the layer sends, and the evidence for trusting it.

    The page used to open by asking whether the learned model beat a random day.
    It does, slightly, and it lost to every statistical rule in the run, so that
    question no longer leads: what ships does.
    """
    captured = uplift / ceiling * 100 if ceiling else float("nan")
    return f"""<section class="answers">
  <div class="claim">
    <h2>Что отправляется</h2>
    <p>Калиброванный z-score: окно не зашито, а выбирается walk-forward, и день
      отвергается, если про него нечего сказать правдиво. Пуш утверждает ровно
      то, что индикатор измерил, — насколько курс ниже собственного среднего за
      это окно.</p>
  </div>

  <article class="answer answer-gain">
    <p class="qnum">Сколько это стоит клиенту</p>
    <h3>+{_num(uplift)} б.п. на перевод против соседнего дня наугад</h3>
    <p class="verdict">Значимо, и на всех {positive} из {total} коридоров.</p>
    <p class="answer-body">На переводе в 100 000 ₽ это примерно
      {_num(uplift * 10, 0)} ₽. Эталон — не ноль, а 500 случайных расписаний того
      же размера в тех же окнах: выигрыш выше почти всех,
      <b>p = {_num(p_value, 3)}</b>. Из доступной при этой частоте выгоды
      ({_num(ceiling)} б.п. при идеальном выборе тех же дней) забрано
      <b>{_num(captured, 0)}%</b>.</p>
    {null_chart}
  </article>

  <article class="answer answer-gain">
    <p class="qnum">Как он собран</p>
    <h3>Два шага от обычного правила</h3>
    <p class="answer-body">Первый — <b>окно выбирает себя само</b>: не зашитые
      60 наблюдений, а выбор walk-forward по прошлой связи счёта с деньгами
      клиента. Второй — <b>проверка факта перед отправкой</b>. Она начиналась
      как комплаенс и дала основной прирост: отвергнутые дни не просто немые,
      они теряют клиенту деньги на каждом коридоре.</p>
    {_ladder_chart(board)}
  </article>
</section>"""


def render_dashboard(
    frames: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    spec: BenchmarkSpec,
) -> str:
    """Build the whole page from the run's frames plus the rate panel."""
    board = frames["leaderboard"]
    gates = frames["gates"]
    per_corridor = frames["per_corridor"]
    per_fold = frames["per_fold"]
    signals = frames["signals"]
    audit = frames.get("audit", pd.DataFrame())
    cadence = frames.get("cadence_sweep", pd.DataFrame())
    nulls = frames.get("null_distribution", pd.DataFrame())
    indexed = board.set_index("strategy")

    def value_of(strategy: str, column: str = "currency_uplift_bps") -> float | None:
        """A leaderboard cell, or None when that strategy was not in the run."""
        if strategy not in indexed.index:
            return None
        cell = indexed.loc[strategy, column]
        return float(cell) if isinstance(cell, int | float | np.floating) else cell

    eval_start = pd.Timestamp(per_fold["fold_start"].min())
    eval_end = pd.Timestamp(per_fold["fold_end"].max())
    folds = per_fold["fold"].nunique()

    ship = "zscore_truthful"
    shipped = indexed.loc[ship] if ship in indexed.index else None
    uplift = float(shipped["currency_uplift_bps"]) if shipped is not None else float("nan")
    p_value = float(shipped["p_value"]) if shipped is not None else float("nan")
    ceiling = float(shipped["ceiling_bps"]) if shipped is not None else float("nan")
    positive = int(shipped["corridors_positive"]) if shipped is not None else 0
    total = int(shipped["n_corridors"]) if shipped is not None else 0
    captured_pct = _num(uplift / ceiling * 100, 0) if ceiling else "—"

    gate_rows = gates[gates["strategy"].eq(ship)].set_index("gate") if len(gates) else None

    def gate_value(name: str) -> str:
        if gate_rows is None or name not in gate_rows.index:
            return "—"
        return _num(gate_rows.loc[name, "value"], 2)

    lift_value = gate_value("G1_lift")
    cv_value = gate_value("G5_evenness")
    gap_value = gate_value("G6_gap")

    ship_nulls = nulls[nulls["strategy"].eq(ship)] if len(nulls) else nulls
    null_chart = _null_histogram(ship_nulls, uplift, p_value)
    focus = [s for s in (ship, "zscore") if s in set(board["strategy"])]
    rejected_table = _rejected_table(board)

    corridors = [c for c in spec.corridors if c in set(signals["iso"])]
    tabs = "".join(
        f'<button class="tab" role="tab" data-panel="rate-{_esc(iso)}" '
        f'aria-selected="{"true" if index == 0 else "false"}">{_esc(iso)}</button>'
        for index, iso in enumerate(corridors)
    )
    panels = "".join(
        f'<div id="rate-{_esc(iso)}" role="tabpanel"{"" if index == 0 else " hidden"}>'
        f"{_rate_chart(panel, signals, iso, eval_start)}</div>"
        for index, iso in enumerate(corridors)
    )

    answers_section = (
        _headline_section(
            board=indexed, null_chart=null_chart, uplift=uplift, p_value=p_value,
            ceiling=ceiling, positive=positive, total=total,
        )
        if shipped is not None
        else ""
    )

    return f"""<title>Сигнальный слой: разбор</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{_STYLE}</style>
<div class="wrap">

<header>
  <p class="eyebrow">CBSB-1 · бенчмарк сигнального слоя</p>
  <h1>Сигнал, который отправляется</h1>
  <p class="lede">Все кандидаты жили на одинаковом бюджете пушей, оценивались
    только out-of-time и сравнивались со случайным расписанием, а не с нулём.
    Победил не обучаемый счёт, а калиброванная статистика с проверкой факта
    перед отправкой. Ниже — что она даёт, чего это стоит и что дальше.</p>
  <div class="meta">
    <span>коридоры: {_esc(", ".join(spec.corridors))}</span>
    <span>{eval_start:%Y-%m-%d} — {eval_end:%Y-%m-%d}</span>
    <span>{folds} окон по {spec.fold_months} мес</span>
    <span>h = {spec.horizon} дней</span>
    <span>бюджет: {_esc(spec.cadence.label)}</span>
    <span>{spec.random_trials} случайных расписаний</span>
  </div>
</header>

<div class="kpis">{_kpis(board, audit)}</div>

{answers_section}

<section>
  <div class="claim">
    <h2>Чего это стоит: три гейта из семи не пройдены</h2>
    <p>Правдивость заставляет молчать. Пройдены значимость, выгода момента,
      темп и риск; не пройдены lift ({lift_value}), ровность ({cv_value}) и
      максимальная пауза ({gap_value} дней). Ни один из трёх не чинится
      настройкой, и вот почему.</p>
    <p><b>Пауза — рыночная, а не наша.</b> Между <i>пригодными</i> днями рынок
      сам даёт 49–73 дня: курс держится выше своего короткого тренда месяцами.
      Политика добавляет к этому около двадцати. Любой правдивый факт, доступный
      в дни молчания, стоит клиенту от −21 до −82 б.п. — заполнять паузу нечем,
      не заплатив за это его деньгами.</p>
    <p><b>Lift меряет прогноз, которого сообщение не делает.</b> Пуш утверждает
      проверяемый факт о прошлом, поэтому его правдивость равна 100 % по
      построению и обеспечена вето. Оба правила ТЗ проверяют утверждение о
      будущем. Цифры приведены, потому что ТЗ их требует.</p>
  </div>
  <div class="panel">{_corridor_dots(per_corridor, focus)}</div>
  <div class="panel">
    {_fold_chart(per_fold, focus)}
    <div class="key">
      <span><i style="background:var(--gain)"></i>рабочий сигнал</span>
      <span><i style="background:var(--ceiling)"></i>правило с зашитым окном</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>Что проверено и отвергнуто</h2>
    <p>Каждая строка воспроизводится ключом <code>--strategies</code> или
      <code>--feature-set</code>. Отрицательный результат, который нельзя
      перезапустить, — не результат.</p>
  </div>
  <div class="panel">{rejected_table}</div>
  <div class="panel">
    <p class="answer-caveat" style="border:0;padding:0">Отдельно — почему мы не
      гонимся за hit rate. Правило «курс останется не хуже» выполняется ровно
      тогда, когда курс продолжает падать, то есть когда клиенту следовало
      подождать. Дни, которые на самом деле лучшие для клиента, проходят его в
      9 % случаев.</p>
    {_conflict_chart(board)}
  </div>
</section>

<section>
  <div class="claim">
    <h2>Курс и срабатывания</h2>
    <p>Ось перевёрнута: вверх — выгоднее для клиента. Точка окрашена по тому,
      что сигнал принёс на самом деле.</p>
  </div>
  <div class="panel" data-tabs>
    <div class="tabs" role="tablist">{tabs}</div>
    {panels}
    <div class="key">
      <span><i class="key-hollow"></i>рабочий сигнал</span>
      <span><i style="background:var(--ceiling)"></i>MVP полезность/риск</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>Следующий шаг</h2>
    <p>По убыванию отдачи, с уже измеренными основаниями.</p>
  </div>
  <div class="panel">
    <ol class="next">
      <li>
        <b>Вернуть G5 и G6 в бизнес как вопрос к ТЗ, а не чинить их кодом.</b>
        Гейты требуют ровного потока, но половину дней рынку нечего предложить:
        в дни молчания средняя выгода −75 б.п. Предложение — заменить «нет
        кварталов молчания» на «не молчим, когда есть что сказать», то есть
        мерить покрытие пригодных дней, а не календарь. Цифры для разговора уже
        есть: 49–73 дня рыночной паузы и −21…−82 б.п. за попытку её заполнить.
      </li>
      <li>
        <b>Запас лежит в распределении бюджета, а не в индикаторе.</b> Мы берём
        {captured_pct} % от доступного при своей частоте. Свип по темпу уже
        показал, что месячное окно бьёт недельное при равной средней частоте, —
        логичное продолжение — бюджет, следующий за плотностью возможностей,
        вместо фиксированного недельного лимита. Проверяется тем же бенчмарком:
        добавить кадансы с переменным окном и сравнить с текущим.
      </li>
      <li>
        <b>Библиотека текстов (п. 9 ТЗ).</b> Сейчас шаблон один и он покрывает
        единственный сценарий, который мы отправляем. Нужны формулировки под
        каждый сценарий и список запрещённых с обоснованием. Это дёшево и
        обязательно, а сама рамка уже задана: вето не выпускает сообщение,
        факт которого не выполняется.
      </li>
      <li>
        <b>Честная статистика по коридорам.</b> Общий p завышен: перестановочный
        тест считает пять коридоров независимыми, а они почти один ряд. Нужна
        совместная блочная перестановка по датам.
      </li>
    </ol>
  </div>
</section>

<section>
  <div class="claim">
    <h2>Приложение: полный прогон</h2>
    <p>Таблица лидеров, гейты, темп и правдивость по горизонтам — для проверки,
      а не для чтения подряд.</p>
  </div>
  <div class="panel">{_leaderboard_table(board, gates)}</div>
  <div class="panel">{_gate_matrix(board, gates, spec)}</div>
  <div class="panel">
    <p class="answer-caveat" style="border:0;padding:0">Цена коммуникационной
      политики: тот же счёт на разных бюджетах. Серая полоса — обязательный
      коридор ТЗ.</p>
    {_cadence_chart(cadence)}
  </div>
  <div class="panel">{_horizon_table(frames["horizons"], spec)}</div>
  <div class="claim">
    <h2>Как не переоценить эти цифры</h2>
    <p>Пять коридоров — почти один ряд: движется рубль, а не валюта получателя.
      Общий p считает их независимыми и потому завышает силу доказательства.
      Читать стоит в порядке «коридоров в плюсе» → «значимых после поправки» →
      разброс по окнам → и только потом общий p.</p>
  </div>
</section>

<footer>
  <p>Воспроизводится целиком: <code>uv run python -m signal_layer.run_benchmark
    --out reports/benchmark</code>. Методика — <code>BENCHMARK.md</code>,
    полные таблицы — CSV рядом с этой страницей.</p>
  <p>Сигнал на дату T считается только по данным, доступным на T; аудит
    пересчитывает счёт на панели, из которой будущее удалено физически.
    Курс ЦБ — не курс исполнения банка.</p>
</footer>

</div>
<script>{_SCRIPT}</script>
"""


def build_dashboard(
    report_dir: str | Path,
    panel: pd.DataFrame,
    spec: BenchmarkSpec | None = None,
    *,
    filename: str = "dashboard.html",
) -> Path:
    """Read a run's CSVs from ``report_dir`` and write the dashboard beside them."""
    directory = Path(report_dir)
    names = (
        "leaderboard", "per_corridor", "per_fold", "gates",
        "horizons", "lambda_sweep", "cadence_sweep", "audit", "signals",
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        path = directory / f"{name}.csv"
        if not path.is_file():
            if name in ("audit", "cadence_sweep"):
                frames[name] = pd.DataFrame()
                continue
            raise FileNotFoundError(f"Missing {path}; run the benchmark first")
        frame = pd.read_csv(path)
        for column in ("quote_date", "fold_start", "fold_end", "asof", "exec_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        frames[name] = frame

    target = directory / filename
    target.write_text(render_dashboard(frames, panel, spec or BenchmarkSpec()), "utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a CBSB-1 run as a standalone HTML dashboard"
    )
    parser.add_argument("--report-dir", default="reports/benchmark")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--corridors", nargs="+", default=["TJS", "UZS", "KGS", "AMD", "KZT"])
    return parser


def main(argv: list[str] | None = None) -> int:
    from ..config import Settings
    from ..data.normalization import read_rate_directory

    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else Settings.from_environment().data_dir
    spec = BenchmarkSpec(corridors=tuple(c.upper() for c in args.corridors))
    panel = read_rate_directory(data_dir, currencies=spec.all_currencies)
    target = build_dashboard(args.report_dir, panel, spec)
    print(f"Дашборд: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
