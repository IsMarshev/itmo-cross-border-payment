"""Offline HTML dashboard (Plotly): visualise signals and replay strategies.

Generates a single self-contained ``.html`` file (Plotly embedded, no network)
that for each corridor shows, over the trailing year:

* the rate history with a zoomable/scalable plot;
* buy markers for three strategies on the same budget — model (green), DCA
  (yellow, fixed cadence), random (red) — so the timing of each is visible;
* how much currency each strategy bought for the same monthly budget, with
  uplift vs DCA and vs random;
* hit-rate of model signals at h=5 and h=15 days (did the rate rise by then).

Business scenario: a client in RF transfers money home to CIS 1–3 times a
month on a fixed monthly budget. The signal layer's job is to pick the best
days within a 5–15 day horizon. All strategies spend the same total roubles;
only the timing differs.

Usage::

    uv run python -m signal_layer.dashboard \
        --corridors USD TJS UZS KGS AMD KZT \
        --monthly-budget 50000 --cadence-days 5 \
        --out reports/dashboard.html
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import models, simulation
from .run_m0 import DEFAULT_CORRIDORS, _load_panel, _signals_from_predictions

# USD is included by default so the user can sanity-check the chart against a
# corridor they know intuitively.
DASHBOARD_CORRIDORS = ("USD", *DEFAULT_CORRIDORS)


def _model_hit_rate_at_h(rates: np.ndarray, sig_pos: np.ndarray, h: int) -> float:
    """Share of model signals after which the median future rate rose by h days."""
    if len(sig_pos) == 0:
        return float("nan")
    hits = 0
    for p in sig_pos:
        future = rates[p + 1 : p + 1 + h]
        if len(future) == 0:
            continue
        if np.median(future) > rates[p]:
            hits += 1
    return hits / len(sig_pos)


def _build_corridor_data(
    panel: pd.DataFrame,
    iso: str,
    signals: pd.DataFrame,
    res: dict[str, simulation.StrategyResult],
    start: pd.Timestamp,
) -> dict:
    """Assemble the data bundle one corridor needs for the Plotly figure.

    The chart line and markers are filtered to ``[start, last]`` (the trailing
    year) so the Y-axis auto-fits that window instead of being flattened by the
    full 2000-onward history. Hit rates use the *full* series because they need
    future observations past the window's end.
    """
    full = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    full_dates = pd.to_datetime(full["quote_date"])
    full_rates = full["rub_per_unit"].to_numpy(dtype=float)

    # Hit rate over the full series (needs future data).
    sig_set = set(pd.Timestamp(d) for d in signals["signal_date"]) if len(signals) else set()
    full_sig_pos = np.where(full_dates.isin(sig_set).to_numpy())[0]

    # Chart data: trailing year only.
    win = full[full["quote_date"] >= start].reset_index(drop=True)
    dates = pd.to_datetime(win["quote_date"])
    rates = win["rub_per_unit"].to_numpy(dtype=float)

    def _date_idx(arr: np.ndarray) -> np.ndarray:
        out = []
        for d in arr:
            i = np.searchsorted(dates.to_numpy(), np.datetime64(pd.Timestamp(d)))
            if i < len(dates):
                out.append(i)
        return np.array(out, dtype=int)

    m_pos = _date_idx(res["model"].buy_dates) if len(res["model"].buy_dates) else np.array([], dtype=int)
    d_pos = _date_idx(res["dca"].buy_dates) if len(res["dca"].buy_dates) else np.array([], dtype=int)
    r_pos = _date_idx(res["random"].buy_dates) if len(res["random"].buy_dates) else np.array([], dtype=int)

    return {
        "iso": iso,
        "dates": dates.dt.strftime("%Y-%m-%d").to_list(),
        "rates": rates.tolist(),
        "model_pos": m_pos.tolist(),
        "dca_pos": d_pos.tolist(),
        "random_pos": r_pos.tolist(),
        "strategies": {
            k: {
                "name": v.name,
                "n_buys": v.n_buys,
                "currency": v.total_currency,
                "avg_rate": v.avg_rate if v.avg_rate == v.avg_rate else 0.0,
                "total_rub": v.total_rub,
            }
            for k, v in res.items()
        },
        "hit_rate_h5": _model_hit_rate_at_h(full_rates, full_sig_pos, 5),
        "hit_rate_h15": _model_hit_rate_at_h(full_rates, full_sig_pos, 15),
        "n_signals": int(len(full_sig_pos)),
    }


def build_dashboard(
    panel: pd.DataFrame,
    corridors: list[str],
    *,
    monthly_budget: float = 50_000.0,
    cadence_days: int = 5,
    slots_per_week: float = 1.5,
    h: int = 20,
    out_path: str = "reports/dashboard.html",
) -> None:
    """Generate the Plotly HTML dashboard for a set of corridors (trailing year)."""
    last = panel["quote_date"].max()
    start = last - pd.DateOffset(years=1)

    corridor_data = []
    for iso in corridors:
        print(f"-> {iso}: walk-forward + signals...", end=" ", flush=True)
        pred = models.walk_forward_predict(panel, iso, h=h, alpha=1.0, min_train=500)
        signals = _signals_from_predictions(pred, slots_per_week=slots_per_week)
        if len(signals):
            rmap = panel[panel["iso"] == iso].set_index("quote_date")["rub_per_unit"]
            signals["rub_per_unit"] = signals["signal_date"].map(rmap).astype(float)
        res = simulation.simulate_strategies(
            panel, iso, signals, monthly_budget=monthly_budget, cadence_days=cadence_days,
            start=start, end=last,
        )
        corridor_data.append(_build_corridor_data(panel, iso, signals, res, start))
        print(f"{len(signals)} signals")

    html = _render_html(corridor_data, monthly_budget=monthly_budget)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nDashboard -> {out_path}")


def _render_html(corridor_data: list[dict], *, monthly_budget: float) -> str:
    """Render the self-contained HTML with Plotly figures per corridor."""
    from plotly.io import to_html

    # Embed plotly.js inline once (first figure), reuse it for the rest so the
    # file is fully offline but not bloated by N copies of the library.
    figs_html = []
    for i, cd in enumerate(corridor_data):
        fig = _make_figure(cd)
        include = "inline" if i == 0 else False
        figs_html.append(
            to_html(
                fig, include_plotlyjs=include, full_html=False, div_id=f"plot_{cd['iso']}",
            )
        )
    body = "\n".join(figs_html)
    summary_html = _summary_table(corridor_data, monthly_budget)

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-border signal dashboard</title>
<style>
  :root {{ --bg:#0f1115; --panel:#171a21; --ink:#e6e8eb; --dim:#8b929d;
          --accent:#5b9dff; --green:#3fb950; --red:#f85149; --amber:#e3b341;
          --grid:#262b33; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:18px 22px; border-bottom:1px solid var(--grid); }}
  header h1 {{ margin:0 0 4px; font-size:18px; font-weight:600; }}
  header p {{ margin:0; color:var(--dim); font-size:13px; }}
  main {{ padding:18px 22px; max-width:1200px; margin:0 auto; }}
  .corridor {{ margin-bottom:30px; }}
  .corr-title {{ font-size:16px; font-weight:600; margin-bottom:8px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
           gap:10px; margin:12px 0; }}
  .card {{ background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:12px; }}
  .card h3 {{ margin:0 0 4px; font-size:12px; color:var(--dim); font-weight:500; }}
  .card .big {{ font-size:20px; font-weight:600; }}
  .card .sub {{ font-size:11px; color:var(--dim); }}
  .pos {{ color:var(--green); }} .neg {{ color:var(--red); }} .neu {{ color:var(--amber); }}
  table {{ width:100%; border-collapse:collapse; margin:10px 0; font-size:13px; }}
  th, td {{ padding:6px 9px; text-align:right; border-bottom:1px solid var(--grid); }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--dim); font-weight:500; }}
  .summary {{ background:var(--panel); border:1px solid var(--grid); border-radius:10px;
              padding:16px; margin-bottom:24px; }}
  .summary h2 {{ margin:0 0 10px; font-size:15px; }}
  .hint {{ color:var(--dim); font-size:12px; margin-top:8px; }}
</style></head><body>
<header>
  <h1>Сигнальный слой трансграничных переводов — Ridge m0</h1>
  <p>Курс валюты получателя в рублях (меньше = выгоднее). Зум — колесо мыши, а
     перетаскивание рамки zoom по обеим осям (двойной клик — сброс). Точки — дни покупок:
     <span style="color:var(--green)">●</span> модель,
     <span style="color:var(--amber)">●</span> DCA (раз в неделю),
     <span style="color:var(--red)">●</span> случайные дни. Окно — последний год.</p>
</header>
<main>
{summary_html}
{body}
</main>
</body></html>
"""


def _summary_table(corridor_data: list[dict], monthly_budget: float) -> str:
    """Cross-corridor summary table: currency bought + uplift + hit rates."""
    rows = []
    for cd in corridor_data:
        s = cd["strategies"]
        m, d, r = s["model"], s["dca"], s["random"]
        m_dca = (m["currency"] - d["currency"]) / d["currency"] * 100 if d["currency"] else float("nan")
        m_rand = (m["currency"] - r["currency"]) / r["currency"] * 100 if r["currency"] else float("nan")
        rows.append(
            f"""<tr><td>{cd['iso']}</td>
            <td>{m['n_buys']}</td><td>{d['n_buys']}</td>
            <td>{m['currency']:,.0f}</td>
            <td class="{'pos' if m_dca>0 else 'neg'}">{m_dca:+.2f}%</td>
            <td class="{'pos' if m_rand>0 else 'neg'}">{m_rand:+.2f}%</td>
            <td>{cd['hit_rate_h5']*100:.0f}%</td>
            <td>{cd['hit_rate_h15']*100:.0f}%</td></tr>"""
        )
    return f"""<div class="summary">
  <h2>Сводка за последний год — бюджет {monthly_budget:,.0f} ₽/мес на коридор</h2>
  <table>
    <tr><th>Коридор</th><th>Покупок (модель)</th><th>Покупок (DCA)</th>
    <th>Валюты куплено (модель)</th><th>Модель vs DCA</th><th>Модель vs random</th>
    <th>Попаданий h=5</th><th>Попаданий h=15</th></tr>
    {''.join(rows)}
  </table>
  <div class="hint">Попадания: доля сигналов модели, после которых медиана курса за h дней
    оказалась выше сигнального дня (дно «закрылось»). Месячный бюджет делится поровну между
    покупками каждой стратегии — всего потрачено одинаково, отличается только тайминг.</div>
</div>"""


def _make_figure(cd: dict) -> go.Figure:
    """One Plotly figure: rate line + strategy markers + stats annotation."""
    dates = cd["dates"]
    rates = cd["rates"]
    fig = go.Figure()

    # Rate line.
    fig.add_trace(go.Scatter(
        x=dates, y=rates, mode="lines", name="Курс",
        line=dict(color="#5b9dff", width=1.2), hovertemplate="%{x}<br>%{y:.4f} ₽<extra></extra>",
    ))

    def _marker_trace(pos_list, name, color):
        if not pos_list:
            return None
        xs = [dates[i] for i in pos_list]
        ys = [rates[i] for i in pos_list]
        return go.Scatter(
            x=xs, y=ys, mode="markers", name=name,
            marker=dict(color=color, size=5, opacity=0.75, line=dict(width=0.5, color="#0f1115")),
            hovertemplate="%{x}<br>%{y:.4f} ₽<extra>" + name + "</extra>",
        )

    for tr, nm, col in [
        (_marker_trace(cd["model_pos"], "Модель", "#3fb950"), "Модель", "#3fb950"),
        (_marker_trace(cd["dca_pos"], "DCA", "#e3b341"), "DCA", "#e3b341"),
        (_marker_trace(cd["random_pos"], "Random", "#f85149"), "Random", "#f85149"),
    ]:
        if tr is not None:
            fig.add_trace(tr)

    s = cd["strategies"]
    m, d = s["model"], s["dca"]
    m_dca = (m["currency"] - d["currency"]) / d["currency"] * 100 if d["currency"] else float("nan")

    fig.update_layout(
        title=dict(text=f"{cd['iso']} — последний год", font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="#0f1115", plot_bgcolor="#171a21",
        font=dict(color="#e6e8eb", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        height=380,
        dragmode="zoom",  # drag a rectangle to zoom both X and Y; double-click resets
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1м", step="month", stepmode="backward"),
                    dict(count=3, label="3м", step="month", stepmode="backward"),
                    dict(count=6, label="6м", step="month", stepmode="backward"),
                    dict(step="all", label="Всё"),
                ]
            ),
            gridcolor="#262b33",
        ),
        yaxis=dict(
            title="₽ за единицу валюты (меньше = выгоднее)",
            autorange=True,  # fits the trailing year, not the full history
            gridcolor="#262b33",
            fixedrange=False,  # allow vertical box-zoom
        ),
        annotations=[dict(
            x=0.99, y=0.98, xref="paper", yref="paper", showarrow=False, align="right",
            text=(
                f"<b>Модель:</b> {m['currency']:,.0f} {cd['iso']} "
                f"({m['n_buys']} пок.)<br>"
                f"<b>DCA:</b> {d['currency']:,.0f} ({d['n_buys']} пок.)<br>"
                f"<b>vs DCA:</b> <b>{'+' if m_dca>=0 else ''}{m_dca:.2f}%</b><br>"
                f"<b>попадания:</b> h=5 {cd['hit_rate_h5']*100:.0f}%, "
                f"h=15 {cd['hit_rate_h15']*100:.0f}%"
            ),
            bgcolor="rgba(23,26,33,0.85)", bordercolor="#262b33",
            font=dict(size=11, color="#e6e8eb"),
        )],
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline Plotly signal dashboard")
    p.add_argument("--corridors", nargs="+", default=list(DASHBOARD_CORRIDORS))
    p.add_argument("--data-dir", default="currency_data")
    p.add_argument(
        "--monthly-budget", type=float, default=50_000.0,
        help="RUB per calendar month per corridor (total spent = this x months)",
    )
    p.add_argument("--cadence-days", type=int, default=5, help="DCA buy every N trading days (5≈weekly)")
    p.add_argument("--slots-per-week", type=float, default=1.5)
    p.add_argument("--h", type=int, default=20)
    p.add_argument("--out", default="reports/dashboard.html")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    panel = _load_panel(args.corridors, args.data_dir)
    build_dashboard(
        panel, args.corridors, monthly_budget=args.monthly_budget,
        cadence_days=args.cadence_days, slots_per_week=args.slots_per_week,
        h=args.h, out_path=args.out,
    )


if __name__ == "__main__":
    main()
