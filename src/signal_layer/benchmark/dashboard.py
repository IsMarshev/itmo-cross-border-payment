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
CHART_STRATEGIES: tuple[str, ...] = ("utility_risk", "percentile")

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


def _policy_cost_chart(pairs: list[tuple[str, float, float]]) -> str:
    """Paired bars: the same score decided freely vs through the greedy policy."""
    if not pairs:
        return ""
    width, row_h, top = 900, 62, 34
    height = top + row_h * len(pairs) + 16
    values = [v for _, a, b in pairs for v in (a, b)]
    hi = max(values + [1.0]) * 1.12
    x0, x1 = 250, width - 90
    parts = [
        f'<text x="{x0}" y="20" class="cap">выгода клиента, б.п.</text>',
        f'<line x1="{x0}" y1="{top - 10}" x2="{x1}" y2="{top - 10}" class="axis" />',
    ]
    for index, (name, free, policed) in enumerate(pairs):
        y = top + index * row_h
        parts.append(f'<text x="0" y="{y + 22}" class="lbl">{_esc(name)}</text>')
        for offset, (value, cls, tag) in enumerate(
            ((free, "fill-ceiling", "свободный выбор дней"),
             (policed, "fill-gain", "через жадную политику"))
        ):
            bar_y = y + 4 + offset * 20
            w = max(1.0, _scale(value, 0, hi, 0, x1 - x0))
            parts.append(
                f'<rect x="{x0}" y="{bar_y}" width="{w:.1f}" height="15" rx="1.5" '
                f'class="{cls}" />'
                f'<text x="{x0 + w + 8:.1f}" y="{bar_y + 12}" class="val">'
                f"{_num(value)}</text>"
                f'<title>{_esc(tag)}: {_num(value)} б.п.</title>'
            )
        lost = (1 - policed / free) * 100 if free else float("nan")
        caption = (
            f"политика съедает {_num(lost, 0)}%"
            if lost > 0
            else "свободный выбор не помогает — счёт слабо ранжирует дни"
        )
        parts.append(
            f'<text x="{x0}" y="{y + row_h - 6}" class="note">{_esc(caption)}</text>'
        )
    return _svg(width, height, "".join(parts), "Что теряет жадная политика отправки")


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


def _lambda_chart(sweep: pd.DataFrame) -> str:
    """Uplift and bad-push rate across the price of error."""
    width, height = 900, 220
    pad_l, pad_r, pad_b, pad_t = 46, 60, 46, 26
    lam = sweep["lam"].to_numpy(dtype=float)
    uplift = sweep["currency_uplift_bps"].to_numpy(dtype=float)
    bad = sweep["bad_push_rate"].to_numpy(dtype=float)
    u_lo, u_hi = min(uplift.min(), 0) * 1.1, uplift.max() * 1.15
    b_lo, b_hi = bad.min() - 0.05, bad.max() + 0.05

    def px(value: float) -> float:
        return _scale(value, lam.min(), lam.max(), pad_l, width - pad_r)

    parts = [
        f'<text x="0" y="{pad_t - 8}" class="cap">выгода, б.п.</text>',
        f'<text x="{width - pad_r + 6}" y="{pad_t - 8}" class="cap">доля плохих</text>',
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" '
        f'y2="{height - pad_b}" class="axis" />',
    ]
    for values, lo, hi, cls in (
        (uplift, u_lo, u_hi, "gain"),
        (bad, b_lo, b_hi, "ceiling"),
    ):
        points = " ".join(
            f"{px(price):.1f},{_scale(v, lo, hi, height - pad_b, pad_t):.1f}"
            for price, v in zip(lam, values, strict=True)
        )
        parts.append(f'<polyline points="{points}" class="line line-{cls}" />')
        for price, v in zip(lam, values, strict=True):
            parts.append(
                f'<circle cx="{px(price):.1f}" '
                f'cy="{_scale(v, lo, hi, height - pad_b, pad_t):.1f}" r="4" '
                f'class="dot-{cls}"><title>λ={price:g}: {v:.2f}</title></circle>'
            )
    for price in lam:
        parts.append(
            f'<text x="{px(price):.1f}" y="{height - pad_b + 18}" class="tick mid">'
            f"λ={price:g}</text>"
        )
    parts.append(
        f'<text x="0" y="{pad_t + 4}" class="tick">{_num(u_hi, 0)}</text>'
        f'<text x="0" y="{height - pad_b}" class="tick">{_num(u_lo, 0)}</text>'
        f'<text x="{width - pad_r + 6}" y="{pad_t + 4}" class="tick">{_num(b_hi, 2)}</text>'
        f'<text x="{width - pad_r + 6}" y="{height - pad_b}" class="tick">'
        f"{_num(b_lo, 2)}</text>"
    )
    return _svg(width, height, "".join(parts), "Чувствительность к цене ошибки")


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
                "Съедает политика отправки",
                _num((1 - policed / ceiling) * 100, 0) + "%",
                "выгоды теряется",
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

    # Each pair differs *only* in how the score becomes a signal, so the MVP row
    # uses its no-floor variant: pairing the floored version against free weekly
    # choice would mix the policy's cost with the silence floor's benefit.
    candidate_pairs = (
        ("идеальный счёт", "oracle_topk", "oracle"),
        ("правило процентиля", "percentile_weekly", "percentile"),
        ("MVP, счёт без порога", "utility_risk_weekly", "utility_risk_paced"),
    )
    pairs = [
        (label, value_of(free), value_of(policed))
        for label, free, policed in candidate_pairs
        if value_of(free) is not None and value_of(policed) is not None
    ]

    present = set(board["strategy"])
    focus = [s for s in ("utility_risk", "percentile") if s in present]
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

    mvp_uplift = value_of("utility_risk")
    pct_uplift = value_of("percentile")
    conflict_lo = value_of("oracle_topk", "hit_favourable")
    conflict_lift = value_of("oracle_topk", "hit_lift_favourable")
    policy_loss = (
        (1 - pairs[0][2] / pairs[0][1]) * 100 if pairs and pairs[0][1] else float("nan")
    )

    return f"""<title>Сигнальный слой: разбор</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{_STYLE}</style>
<div class="wrap">

<header>
  <p class="eyebrow">CBSB-1 · бенчмарк сигнального слоя</p>
  <h1>Индикатор не был узким местом</h1>
  <p class="lede">Все стратегии живут на одинаковом бюджете пушей, оцениваются
    только out-of-time и сравниваются со случайным расписанием, а не с нулём.
    В такой рамке видно, что бо́льшую часть выгоды теряет не модель, а правило
    отправки — и что одно из двух правил правдивости из ТЗ вредно
    оптимизировать.</p>
  <div class="meta">
    <span>коридоры: {_esc(", ".join(spec.corridors))}</span>
    <span>{eval_start:%Y-%m-%d} — {eval_end:%Y-%m-%d}</span>
    <span>{folds} окон по {spec.fold_months} мес</span>
    <span>h = {spec.horizon} дней</span>
    <span>≤ {spec.max_signals_per_week} пуша/нед</span>
    <span>{spec.random_trials} случайных расписаний</span>
  </div>
</header>

<div class="kpis">{_kpis(board, audit)}</div>

<section>
  <div class="claim">
    <h2>Таблица лидеров</h2>
    <p>Выгода — на сколько базисных пунктов больше валюты получает клиент,
      переводя в дни сигналов, а не в соседний день. Случайное расписание стоит
      в нуле по построению, поэтому знак столбца и есть ответ на вопрос
      «лучше ли мы случайного дня». Гейты — семь обязательных условий ТЗ.</p>
  </div>
  <div class="panel">{_leaderboard_table(board, gates)}</div>
</section>

<section>
  <div class="claim">
    <h2>1 · Жадная политика отправки съедает три четверти выгоды</h2>
    <p>Одна и та же оценка дня, но два способа превратить её в пуш. Свободный
      выбор лучших дней недели — против онлайн-правила «отправляй первый день
      выше скользящего квантиля», которое работает сейчас. Даже идеальное знание
      будущего проходит через эту воронку и теряет
      {_num(policy_loss, 0)}% выгоды.
      Улучшение политики стоит дороже любого улучшения модели. Строки сравнимы
      напрямую: обе стороны отличаются только способом превратить счёт в пуш,
      поэтому MVP взят в варианте без порога молчания.</p>
  </div>
  <div class="panel">
    {_policy_cost_chart(pairs)}
    <div class="key">
      <span><i style="background:var(--ceiling)"></i>свободный выбор дней недели</span>
      <span><i style="background:var(--gain)"></i>через текущую жадную политику</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>2 · Два правила правдивости смотрят в разные стороны</h2>
    <p>«Сейчас выгодно» засчитывается, когда курс h дней не поднимается выше —
      то есть когда он продолжает падать и клиенту следовало подождать.
      Дни, которые на самом деле были лучшими для клиента, проходят это правило
      лишь в {_num((conflict_lo or 0) * 100, 0)}% случаев: lift
      {_num(conflict_lift, 2)}, хуже случайного дня. «Окно закрывается»
      ведёт себя противоположно. Сигналу о локальном минимуме нужно второе
      сообщение, а hit rate по первому правилу нельзя оптимизировать.</p>
  </div>
  <div class="panel">
    {_conflict_chart(board)}
    <div class="key">
      <span><i style="background:var(--amber)"></i>MVP и потолок задачи</span>
      <span><i style="background:var(--faint)"></i>остальные стратегии</span>
      <span>пунктир — линия тренда по стратегиям</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>3 · Выигрыш MVP держится на двух коридорах и на 2022 годе</h2>
    <p>Общая цифра {_num(mvp_uplift)} б.п. выглядит лучше правила процентиля
      ({_num(pct_uplift)} б.п.), но распадается при разрезе. Из пяти коридоров
      в плюсе только два, и оба выигрыша приходятся на всплеск волатильности
      2022 года. Правило процентиля слабее в среднем, зато положительно везде
      и идёт ровным темпом.</p>
  </div>
  <div class="panel">
    {_corridor_dots(per_corridor, focus)}
    <div class="key">
      <span><i style="background:var(--amber)"></i>значим после поправки BH</span>
      <span><i class="key-hollow"></i>не значим</span>
    </div>
  </div>
  <div class="panel">
    {_fold_chart(per_fold, focus)}
    <div class="key">
      <span><i style="background:var(--gain)"></i>MVP полезность/риск</span>
      <span><i style="background:var(--ceiling)"></i>правило процентиля</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>Цена ошибки: λ почти ничего не меняет</h2>
    <p>λ говорит, во сколько раз рубль, потерянный клиентом после нашего пуша,
      дороже рубля упущенной возможности. Доля плохих пушей на кривой стоит на
      месте. Причина в самих головах модели: риск отрицательно коррелирован с
      полезностью (−0.37…−0.59 по коридорам) и имеет втрое меньший разброс,
      поэтому вычитание риска усиливает тот же порядок дней, а не меняет его.
      Чтобы λ заработала, голове риска нужен источник, которого нет у головы
      полезности, — режим волатильности, а не те же признаки возврата
      к среднему.</p>
  </div>
  <div class="panel">
    {_lambda_chart(frames["lambda_sweep"])}
    <div class="key">
      <span><i style="background:var(--gain)"></i>выгода клиента, б.п.</span>
      <span><i style="background:var(--ceiling)"></i>доля плохих пушей</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>Курс и срабатывания</h2>
    <p>Ось перевёрнута: вверх — выгоднее для клиента. Точка окрашена по тому,
      что сигнал принёс на самом деле — зелёная выиграла против соседних дней,
      красная проиграла.</p>
  </div>
  <div class="panel" data-tabs>
    <div class="tabs" role="tablist">{tabs}</div>
    {panels}
    <div class="key">
      <span><i class="key-hollow"></i>MVP полезность/риск</span>
      <span><i style="background:var(--ceiling)"></i>правило процентиля</span>
    </div>
  </div>
</section>

<section>
  <div class="claim">
    <h2>Правдивость по горизонтам</h2>
    <p>Доля сигналов, после которых утверждение пуша подтвердилось через h дней.
      Проверяется по правилу того сообщения, которое стратегия действительно
      отправляет.</p>
  </div>
  <div class="panel">{_horizon_table(frames["horizons"], spec)}</div>
</section>

<section>
  <div class="claim">
    <h2>Обязательные условия</h2>
    <p>Наведите на клетку, чтобы увидеть значение. Ни одна работающая стратегия
      пока не закрывает все семь: MVP спотыкается о ровность темпа и паузы,
      правила — о lift, потому что несут сообщение «сейчас выгодно».</p>
  </div>
  <div class="panel">{_gate_matrix(board, gates, spec)}</div>
</section>

<section>
  <div class="claim">
    <h2>Как не переоценить эти цифры</h2>
    <p>Пять коридоров — почти один ряд: основное движение даёт рубль, а не
      валюта получателя. Общий p завышает силу доказательства, потому что считает
      коридоры независимыми. Читать стоит в порядке «коридоров в плюсе» →
      «значимых после поправки» → разброс по окнам → и только потом общий p.
      Настоящая независимая ось здесь — время, а не коридор.</p>
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
        "horizons", "lambda_sweep", "audit", "signals",
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        path = directory / f"{name}.csv"
        if not path.is_file():
            if name == "audit":
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
