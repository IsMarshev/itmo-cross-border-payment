# ruff: noqa: E501
"""Self-contained HTML dashboard generated from a Stage-4 backtest report."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SUMMARY_FILE = "summary.json"
DECISION_LOG_FILE = "decision_log.jsonl"
DEFAULT_DASHBOARD_FILE = "dashboard.html"


def generate_dashboard(
    report_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Build an interactive, standalone dashboard from one backtest run.

    The dashboard deliberately does not load ``random_baseline.jsonl``: it can
    be very large, while the matched-baseline aggregates needed for comparison
    are already persisted in ``summary.json``.
    """
    source_dir = Path(report_dir)
    summary_path = source_dir / SUMMARY_FILE
    decisions_path = source_dir / DECISION_LOG_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f"Backtest summary does not exist: {summary_path}")
    if not decisions_path.is_file():
        raise FileNotFoundError(f"Decision log does not exist: {decisions_path}")

    summary = pd.read_json(summary_path)
    decisions = pd.read_json(decisions_path, lines=True)
    output = Path(output_path) if output_path is not None else source_dir / DEFAULT_DASHBOARD_FILE
    output.parent.mkdir(parents=True, exist_ok=True)

    dashboard_html = _render_document(summary, decisions)
    output.write_text(dashboard_html, encoding="utf-8")
    return output


def _render_document(summary: pd.DataFrame, decisions: pd.DataFrame) -> str:
    corridor_summary = summary.loc[summary["iso"].ne("ALL")].copy()
    aggregate = summary.loc[summary["iso"].eq("ALL")]
    aggregate_row = aggregate.iloc[0] if not aggregate.empty else pd.Series(dtype=object)
    selected = decisions.loc[
        decisions["decision"].astype(bool) & decisions["outcome_complete"].astype(bool)
    ].copy()
    selected["decision_date"] = pd.to_datetime(selected["decision_date"])

    figures = [
        _advantage_figure(corridor_summary),
        _risk_figure(corridor_summary),
        _volume_figure(selected),
        _distribution_figure(selected),
    ]
    fragments = [
        figure.to_html(
            full_html=False,
            include_plotlyjs=True if index == 0 else False,
            config={"responsive": True, "displaylogo": False},
        )
        for index, figure in enumerate(figures)
    ]
    table = _summary_table(corridor_summary)
    cards = _kpi_cards(aggregate_row)
    selected_table = _recent_signals_table(selected)

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Сигнальный слой — отчёт бэктеста</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; color: #172033; background: #f6f8fc; }}
    body {{ margin: 0; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 36px 24px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 34px 0 14px; font-size: 20px; }}
    .subtitle {{ margin: 0; color: #586174; max-width: 900px; line-height: 1.5; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 24px; }}
    .card, .panel {{ background: white; border: 1px solid #e2e7f0; border-radius: 14px; box-shadow: 0 2px 8px #1720330a; }}
    .card {{ padding: 18px; }}
    .card-label {{ color: #6c7585; font-size: 13px; }}
    .card-value {{ font-size: 25px; font-weight: 700; margin-top: 7px; }}
    .card-note {{ color: #6c7585; font-size: 12px; margin-top: 7px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 18px; }}
    .panel {{ padding: 6px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; }}
    th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #edf0f5; white-space: nowrap; }}
    th {{ color: #5f6877; background: #f9fafc; font-weight: 600; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #6c7585; font-size: 13px; line-height: 1.5; margin-top: 14px; }}
    .plotly-graph-div {{ min-height: 380px; }}
  </style>
</head>
<body>
  <main>
    <h1>Сигнальный слой: бэктест</h1>
    <p class="subtitle">Историческая оценка политики под коммуникационным бюджетом. Положительный advantage означает, что курс в день сигнала был выгоднее медианного курса следующих H наблюдений. Это оценка качества тайминга, а не доказательство влияния пуша на конверсию.</p>
    <section class="cards">{cards}</section>
    <h2>Польза и matched random baseline</h2>
    <section class="grid"><div class="panel">{fragments[0]}</div><div class="panel">{fragments[1]}</div></section>
    <h2>Динамика и распределение исходов</h2>
    <section class="grid"><div class="panel">{fragments[2]}</div><div class="panel">{fragments[3]}</div></section>
    <h2>Сводка по коридорам</h2>
    <section class="panel">{table}</section>
    <h2>Последние отобранные сигналы</h2>
    <section class="panel">{selected_table}</section>
    <p class="note">CI — 95% moving-block bootstrap. Random baseline выбирает то же число доступных дат в каждом коридоре и коммуникационном окне. Дашборд не загружает массивный random-log: использует его агрегаты из summary.json.</p>
  </main>
</body>
</html>"""


def _advantage_figure(summary: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if summary.empty:
        return _empty_figure("Сравнение advantage", "Нет данных для отображения")
    figure.add_bar(
        name="Политика",
        x=summary["iso"],
        y=summary["mean_advantage_bps"],
        error_y={
            "type": "data",
            "symmetric": False,
            "array": (summary["advantage_ci_high"] - summary["mean_advantage_bps"]),
            "arrayminus": (summary["mean_advantage_bps"] - summary["advantage_ci_low"]),
        },
        marker_color="#2563eb",
    )
    figure.add_bar(
        name="Matched random",
        x=summary["iso"],
        y=summary["random_mean_advantage_bps"],
        marker_color="#94a3b8",
    )
    figure.update_layout(
        title="Средняя выгода на сигнал, б.п.",
        barmode="group",
        yaxis_title="б.п.",
        legend_title_text="",
        margin={"l": 55, "r": 25, "t": 55, "b": 40},
    )
    return figure


def _risk_figure(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return _empty_figure("Риск ранней отправки", "Нет данных для отображения")
    figure = px.scatter(
        summary,
        x="early_send_rate",
        y="mean_advantage_bps",
        size="n_signals",
        color="iso",
        text="iso",
        hover_data=["p90_regret_bps", "hit_rate", "advantage_delta_bps"],
        labels={
            "early_send_rate": "Риск ранней отправки",
            "mean_advantage_bps": "Средняя выгода, б.п.",
            "n_signals": "Сигналов",
        },
        title="Польза против риска",
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#64748b")
    figure.update_traces(textposition="top center")
    figure.update_layout(margin={"l": 55, "r": 25, "t": 55, "b": 40}, showlegend=False)
    return figure


def _volume_figure(selected: pd.DataFrame) -> go.Figure:
    if selected.empty:
        return _empty_figure("Частота сигналов", "Нет отобранных сигналов")
    monthly = (
        selected.assign(month=selected["decision_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "iso"], as_index=False)
        .size()
        .rename(columns={"size": "signals"})
    )
    figure = px.line(
        monthly,
        x="month",
        y="signals",
        color="iso",
        labels={"month": "Месяц", "signals": "Сигналов"},
        title="Частота отобранных сигналов по месяцам",
    )
    figure.update_layout(margin={"l": 55, "r": 25, "t": 55, "b": 40})
    return figure


def _distribution_figure(selected: pd.DataFrame) -> go.Figure:
    if selected.empty:
        return _empty_figure("Распределение advantage", "Нет отобранных сигналов")
    figure = px.box(
        selected,
        x="iso",
        y="advantage_bps",
        color="iso",
        points="outliers",
        labels={"iso": "Коридор", "advantage_bps": "Выгода, б.п."},
        title="Распределение исхода на сигнал",
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#64748b")
    figure.update_layout(margin={"l": 55, "r": 25, "t": 55, "b": 40}, showlegend=False)
    return figure


def _empty_figure(title: str, annotation: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=annotation, showarrow=False, font={"size": 16})
    figure.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def _kpi_cards(row: pd.Series) -> str:
    cards = [
        ("Отобрано сигналов", _format_number(row.get("n_signals"), digits=0), "полный период"),
        (
            "Средняя выгода",
            _format_bps(row.get("mean_advantage_bps")),
            "на один сигнал",
        ),
        (
            "95% CI выгоды",
            f"{_format_bps(row.get('advantage_ci_low'))} … {_format_bps(row.get('advantage_ci_high'))}",
            "moving-block bootstrap",
        ),
        ("Риск ранней отправки", _format_percent(row.get("early_send_rate")), "ниже ε в горизонте H"),
        ("Hit-rate lift", _format_number(row.get("hit_rate_lift"), digits=2), "против matched random"),
    ]
    return "".join(
        "<article class=\"card\">"
        f"<div class=\"card-label\">{html.escape(label)}</div>"
        f"<div class=\"card-value\">{html.escape(value)}</div>"
        f"<div class=\"card-note\">{html.escape(note)}</div>"
        "</article>"
        for label, value, note in cards
    )


def _summary_table(summary: pd.DataFrame) -> str:
    columns = [
        ("iso", "Коридор", lambda value: str(value)),
        ("n_signals", "Сигналов", lambda value: _format_number(value, digits=0)),
        ("mean_advantage_bps", "Advantage", _format_bps),
        ("advantage_delta_bps", "Δ к random", _format_bps),
        ("hit_rate_lift", "Hit lift", lambda value: _format_number(value, digits=2)),
        ("early_send_rate", "Ранняя отправка", _format_percent),
        ("p90_regret_bps", "p90 regret", _format_bps),
        ("per_week", "В неделю", lambda value: _format_number(value, digits=2)),
    ]
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    rows = []
    for _, row in summary.iterrows():
        values = "".join(
            f"<td>{html.escape(formatter(row.get(column)))}</td>"
            for column, _, formatter in columns
        )
        rows.append(f"<tr>{values}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _recent_signals_table(selected: pd.DataFrame) -> str:
    if selected.empty:
        return "<p class=\"note\">Нет отобранных сигналов.</p>"
    columns = [
        ("decision_date", "Дата", lambda value: pd.Timestamp(value).date().isoformat()),
        ("iso", "Коридор", str),
        ("score", "Score", lambda value: _format_number(value, digits=3)),
        ("threshold", "Порог", lambda value: _format_number(value, digits=3)),
        ("advantage_bps", "Advantage", _format_bps),
        ("regret_bps", "Regret", _format_bps),
    ]
    recent = selected.sort_values("decision_date", ascending=False).head(20)
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    rows = []
    for _, row in recent.iterrows():
        values = "".join(
            f"<td>{html.escape(formatter(row[column]))}</td>"
            for column, _, formatter in columns
        )
        rows.append(f"<tr>{values}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _format_number(value: object, *, digits: int) -> str:
    numeric = _to_finite_float(value)
    if numeric is None:
        return "—"
    return f"{numeric:.{digits}f}"


def _format_bps(value: object) -> str:
    numeric = _to_finite_float(value)
    if numeric is None:
        return "—"
    return f"{numeric:+.1f} б.п."


def _format_percent(value: object) -> str:
    numeric = _to_finite_float(value)
    if numeric is None:
        return "—"
    return f"{numeric * 100:.1f}%"


def _to_finite_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a static signal-layer dashboard")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/backtest"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output = generate_dashboard(args.report_dir, output_path=args.out)
    print(f"Dashboard written to {output}")


if __name__ == "__main__":
    main()
