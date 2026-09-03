"""Offline HTML dashboard: visualise signals and replay transfer strategies.

Generates a single self-contained ``.html`` file (no network needed) that:

* plots the full rate history per corridor with model signal markers, so it is
  visible how often and how well the model times the lows;
* replays three spending strategies — model signals, DCA (fixed cadence), and
  random days — on the same budget, showing total currency bought and uplift;
* walks through a single push scenario: the rate on the signal day, the rate
  the client would see ``h`` days later, and what 50 000 RUB bought in each.

Usage::

    uv run python -m signal_layer.dashboard \
        --corridors TJS UZS KGS AMD KZT \
        --budget 600000 --cadence-days 5 \
        --out reports/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from . import models, simulation
from .run_m0 import DEFAULT_CORRIDORS, _load_panel, _signals_from_predictions

HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-border signal dashboard</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --ink:#e6e8eb; --dim:#8b929d;
          --accent:#5b9dff; --green:#3fb950; --red:#f85149; --amber:#e3b341;
          --grid:#262b33; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--grid); }
  header h1 { margin:0 0 4px; font-size:18px; font-weight:600; }
  header p { margin:0; color:var(--dim); font-size:13px; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center;
              padding:12px 22px; border-bottom:1px solid var(--grid); }
  .controls label { color:var(--dim); font-size:13px; }
  select, input[type=range], button { background:var(--panel); color:var(--ink);
         border:1px solid var(--grid); border-radius:6px; padding:5px 9px; }
  main { padding:18px 22px; }
  .chart-wrap { background:var(--panel); border:1px solid var(--grid);
                border-radius:10px; padding:14px; margin-bottom:18px; }
  .chart-title { display:flex; justify-content:space-between; align-items:baseline;
                 margin-bottom:8px; }
  .chart-title h2 { margin:0; font-size:15px; font-weight:600; }
  .chart-title .legend { font-size:12px; color:var(--dim); }
  canvas { width:100%; height:300px; display:block; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
           gap:12px; margin-bottom:18px; }
  .card { background:var(--panel); border:1px solid var(--grid); border-radius:10px;
          padding:14px; }
  .card h3 { margin:0 0 8px; font-size:13px; color:var(--dim); font-weight:500; }
  .card .big { font-size:24px; font-weight:600; }
  .card .sub { font-size:12px; color:var(--dim); margin-top:4px; }
  .pos { color:var(--green); } .neg { color:var(--red); } .neu { color:var(--amber); }
  .strat-table { width:100%; border-collapse:collapse; }
  .strat-table th, .strat-table td { padding:7px 10px; text-align:right;
           border-bottom:1px solid var(--grid); }
  .strat-table th:first-child, .strat-table td:first-child { text-align:left; }
  .strat-table th { color:var(--dim); font-weight:500; font-size:12px; }
  .barrow { display:flex; align-items:center; gap:10px; margin:6px 0; }
  .barrow .name { width:64px; font-size:12px; color:var(--dim); }
  .barrow .bar { height:18px; border-radius:4px; }
  .scenario { background:var(--panel); border:1px solid var(--grid);
              border-radius:10px; padding:16px; }
  .scenario .row { display:flex; gap:16px; flex-wrap:wrap; }
  .scenario .kv { flex:1; min-width:140px; }
  .scenario .kv .k { font-size:12px; color:var(--dim); }
  .scenario .kv .v { font-size:17px; font-weight:600; }
  .hint { color:var(--dim); font-size:12px; margin-top:10px; }
</style></head><body>
<header>
  <h1>Сигнальный слой трансграничных переводов — Ridge m0</h1>
  <p>Курс валюты получателя в рублях (меньше = выгоднее). Точки — сигналы модели.
     Одинаковыми цветами — как «сработало» ли сообщение на горизонте h.</p>
</header>
<div class="controls">
  <label>Коридор <select id="iso"></select></label>
  <label>Окно симуляции
    <select id="window"></select>
  </label>
  <label>Бюджет, ₽ <input type="number" id="budget" value="__BUDGET__" step="10000"></label>
  <label>Горизонт h <input type="range" id="h" min="1" max="20" value="5">
    <span id="hval">5</span></label>
  <button id="rerun">Пересчитать</button>
</div>
<main>
  <div class="chart-wrap">
    <div class="chart-title">
      <h2 id="chart-h">Курс</h2>
      <div class="legend"><span style="color:var(--green)">●</span> сигнал сработал (курс вырос к h)
        &nbsp; <span style="color:var(--red)">●</span> не сработал
        &nbsp; <span style="color:var(--accent)">—</span> курс</div>
    </div>
    <canvas id="chart"></canvas>
  </div>
  <div class="stats" id="stats"></div>
  <div class="chart-wrap">
    <div class="chart-title"><h2>Стратегии: валюты куплено на одинаковый бюджет</h2>
      <div class="legend">model = по сигналам · dca = раз в неделю · random = случайные дни</div></div>
    <div id="strat"></div>
  </div>
  <div class="scenario" id="scenario"></div>
</main>
<script>
const DATA = __DATA__;
const BUDGET = __BUDGET__;
const CADENCE = __CADENCE__;
let curIso = Object.keys(DATA)[0], curWin = 0;

// populate selectors
const isoSel = document.getElementById('iso');
for (const k of Object.keys(DATA)) { const o=document.createElement('option'); o.value=k; o.textContent=k; isoSel.appendChild(o); }
const winSel = document.getElementById('window');
function fillWindows(){ winSel.innerHTML='';
  const wins = DATA[curIso].windows;
  wins.forEach((w,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=w.label; winSel.appendChild(o); });
  winSel.value = curWin < wins.length ? curWin : 0;
}

const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
function resize(){ const r=canvas.getBoundingClientRect(); canvas.width=r.width*devicePixelRatio; canvas.height=300*devicePixelRatio; }
window.addEventListener('resize', ()=>{ resize(); draw(); });

function draw(){
  const d = DATA[curIso]; const W=canvas.width, H=canvas.height, dpr=devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pts = d.series; // [{t, r, sig, hit}]
  if(!pts.length) return;
  const rates = pts.map(p=>p.r);
  let rmin=Math.min(...rates), rmax=Math.max(...rates);
  const pad=(rmax-rmin)*0.06 || 1; rmin-=pad; rmax+=pad;
  const n=pts.length;
  const x = i => 40*dpr + (W-50*dpr) * i/(n-1);
  const y = r => (H-30*dpr) - (H-50*dpr) * (r-rmin)/(rmax-rmin);
  // grid + axis labels
  ctx.strokeStyle='#262b33'; ctx.fillStyle='#8b929d'; ctx.font=10*dpr+'px sans-serif';
  for(let g=0; g<=4; g++){ const yy=y(rmin+(rmax-rmin)*g/4);
    ctx.beginPath(); ctx.moveTo(40*dpr,yy); ctx.lineTo(W-10*dpr,yy); ctx.stroke();
    ctx.fillText((rmax-(rmax-rmin)*g/4).toFixed(3), 2*dpr, yy-2*dpr); }
  // date labels (first, mid, last)
  const dts=[0,Math.floor(n/2),n-1];
  dts.forEach(i=>{ ctx.fillText(pts[i].t, x(i)-20*dpr, H-12*dpr); });
  // rate line
  ctx.strokeStyle='#5b9dff'; ctx.lineWidth=1*dpr; ctx.beginPath();
  pts.forEach((p,i)=>{ i? ctx.lineTo(x(i),y(p.r)) : ctx.moveTo(x(i),y(p.r)); }); ctx.stroke();
  // signals
  const h = +document.getElementById('h').value;
  pts.forEach((p,i)=>{ if(!p.sig) return;
    const hit = p.hitH && h<=p.hitH.length ? p.hitH[h-1] : null;
    ctx.fillStyle = hit===null ? '#e3b341' : (hit?'#3fb950':'#f85149');
    ctx.beginPath(); ctx.arc(x(i), y(p.r), 3*dpr, 0, 2*Math.PI); ctx.fill();
  });
}

function fmt(n,d=0){ if(!isFinite(n)) return '—'; return n.toLocaleString('ru-RU',{maximumFractionDigits:d}); }

function render(){
  const d = DATA[curIso]; const w = d.windows[curWin];
  document.getElementById('chart-h').textContent = `Курс ${curIso} — ${w.label}`;
  // strategies
  const s = w.strategies; const maxv = Math.max(s.model.currency, s.dca.currency, s.random.currency);
  const colors = {model:'#3fb950', dca:'#5b9dff', random:'#8b929d'};
  let html = '<div class="strat-table"><table><tr><th>Стратегия</th><th>Покупок</th><th>Валюты куплено</th><th>Средний курс</th><th>vs random</th><th>vs DCA</th></tr>';
  for(const k of ['model','dca','random']){ const r=s[k];
    const vr = s.random.currency? (r.currency-s.random.currency)/s.random.currency*100 : NaN;
    const vd = s.dca.currency? (r.currency-s.dca.currency)/s.dca.currency*100 : NaN;
    html += `<tr><td>${k}</td><td>${r.n_buys}</td><td>${fmt(r.currency,0)}</td><td>${r.avg_rate.toFixed(4)}</td>
      <td class="${vr>0?'pos':vr<0?'neg':'neu'}">${isFinite(vr)?vr.toFixed(2)+'%':'—'}</td>
      <td class="${vd>0?'pos':vd<0?'neg':'neu'}">${isFinite(vd)?vd.toFixed(2)+'%':'—'}</td></tr>`;
  }
  html += '</table>';
  for(const k of ['model','dca','random']){ const r=s[k];
    html += `<div class="barrow"><div class="name">${k}</div>
      <div class="bar" style="width:${r.currency/maxv*70}%;background:${colors[k]}"></div>
      <div style="font-size:12px">${fmt(r.currency,0)} ${curIso}</div></div>`; }
  document.getElementById('strat').innerHTML = html;
  // push scenario: first model signal in window
  const sigs = d.series.filter(p=>p.sig);
  const winStart = w.start, winEnd = w.end;
  const inWin = sigs.filter(p=>p.t>=winStart && p.t<=winEnd);
  let scen = '<div style="margin-bottom:10px;font-weight:600">Сценарий пуша → покупка</div>';
  if(!inWin.length){ scen += '<div class="hint">Нет сигналов в этом окне.</div>'; }
  else {
    const p = inWin[Math.floor(inWin.length/2)]; // a representative middle signal
    const h = +document.getElementById('h').value;
    const futureRate = p.futureH && h<=p.futureH.length ? p.futureH[h-1] : null;
    const buy50 = 50000/p.r;
    const buy50future = futureRate ? 50000/futureRate : null;
    const diff = futureRate ? (futureRate-p.r)/p.r*100 : null;
    scen += `<div class="row">
      <div class="kv"><div class="k">Дата пуша</div><div class="v">${p.t}</div></div>
      <div class="kv"><div class="k">Курс в день пуша</div><div class="v">${p.r.toFixed(4)}</div></div>
      <div class="kv"><div class="k">Курс через ${h} дн.</div><div class="v">${futureRate?futureRate.toFixed(4):'—'}</div></div>
      <div class="kv"><div class="k">Изменение курса</div><div class="v ${diff>0?'pos':diff<0?'neg':'neu'}">${diff!==null?diff.toFixed(2)+'%':'—'}</div></div>
      <div class="kv"><div class="k">Куплено на 50к ₽ в пуш</div><div class="v">${fmt(buy50,0)}</div></div>
      <div class="kv"><div class="k">Купил бы через ${h} дн.</div><div class="v">${buy50future?fmt(buy50future,0):'—'}</div></div>
    </div>
    <div class="hint">${diff>0?'Курс вырос — пуш сработал, клиент купил выгоднее, чем если бы подождал.':
       diff<0?'Курс упал — пуш не сработал: клиент купил дороже, чем если бы подождал.':'Курс не изменился.'}</div>`;
  }
  document.getElementById('scenario').innerHTML = scen;
  draw();
}

isoSel.onchange = ()=>{ curIso=isoSel.value; fillWindows(); curWin=0; render(); };
winSel.onchange = ()=>{ curWin=+winSel.value; render(); };
document.getElementById('h').oninput = e=>{ document.getElementById('hval').textContent=e.target.value; render(); };
document.getElementById('rerun').onclick = render;
fillWindows(); resize(); render();
</script></body></html>
"""


def _series_for_corridor(
    panel: pd.DataFrame, pred: pd.DataFrame, signals: pd.DataFrame, iso: str, h_max: int = 20
) -> list[dict]:
    """Build the per-date chart series for one corridor."""
    s = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    rates = s["rub_per_unit"].to_numpy(dtype=float)
    dates = s["quote_date"].to_numpy()
    sig_dates = set(pd.Timestamp(d) for d in signals["signal_date"]) if len(signals) else set()
    series = []
    for i, d in enumerate(dates):
        dts = pd.Timestamp(d)
        is_sig = dts in sig_dates
        hit_h = None
        future_h = None
        if is_sig:
            future = rates[i + 1 : i + 1 + h_max]
            hit_h = [bool(fv > rates[i]) for fv in future]
            future_h = future.tolist()
        series.append(
            {
                "t": dts.strftime("%Y-%m-%d"),
                "r": round(float(rates[i]), 6),
                "sig": is_sig,
                "hitH": hit_h,
                "futureH": future_h,
            }
        )
    return series


def _windows_for_corridor(
    panel: pd.DataFrame, signals: pd.DataFrame, iso: str, budget: float, cadence_days: int
) -> list[dict]:
    """Build strategy comparison results for several evaluation windows."""
    s = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    dates = s["quote_date"]
    if dates.empty:
        return []
    last = pd.Timestamp(dates.iloc[-1])
    # Three windows: trailing 1y, trailing 2y, full history.
    windows = [
        ("Последний год", last - pd.DateOffset(years=1), last),
        ("Последние 2 года", last - pd.DateOffset(years=2), last),
        ("Вся история", pd.Timestamp(dates.iloc[0]), last),
    ]
    out = []
    for label, start, end in windows:
        start = max(start, pd.Timestamp(dates.iloc[0]))
        try:
            res = simulation.simulate_strategies(
                panel, iso, signals, budget=budget, cadence_days=cadence_days,
                start=start, end=end,
            )
        except ValueError:
            continue
        out.append(
            {
                "label": f"{label} ({start.strftime('%Y-%m')}–{end.strftime('%Y-%m')})",
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "strategies": {
                    k: {
                        "name": v.name,
                        "n_buys": v.n_buys,
                        "currency": v.total_currency,
                        "avg_rate": v.avg_rate if v.avg_rate == v.avg_rate else 0.0,
                    }
                    for k, v in res.items()
                },
            }
        )
    return out


def build_dashboard(
    panel: pd.DataFrame,
    corridors: list[str],
    *,
    budget: float = 600_000.0,
    cadence_days: int = 5,
    slots_per_week: float = 1.5,
    h: int = 20,
    out_path: str = "reports/dashboard.html",
) -> None:
    """Generate the full HTML dashboard for a set of corridors."""
    data: dict[str, dict] = {}
    for iso in corridors:
        print(f"-> {iso}: walk-forward + signals...", end=" ", flush=True)
        pred = models.walk_forward_predict(panel, iso, h=h, alpha=1.0, min_train=500)
        signals = _signals_from_predictions(pred, slots_per_week=slots_per_week)
        if len(signals):
            rmap = panel[panel["iso"] == iso].set_index("quote_date")["rub_per_unit"]
            signals["rub_per_unit"] = signals["signal_date"].map(rmap).astype(float)
        series = _series_for_corridor(panel, pred, signals, iso, h_max=20)
        windows = _windows_for_corridor(panel, signals, iso, budget, cadence_days)
        data[iso] = {"series": series, "windows": windows}
        print(f"{len(signals)} signals, {len(series)} days")

    payload = json.dumps(data, ensure_ascii=False)
    html = (
        HTML_TEMPLATE
        .replace("__DATA__", payload)
        .replace("__BUDGET__", str(int(budget)))
        .replace("__CADENCE__", str(cadence_days))
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nDashboard -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline HTML signal dashboard")
    p.add_argument("--corridors", nargs="+", default=list(DEFAULT_CORRIDORS))
    p.add_argument("--data-dir", default="currency_data")
    p.add_argument("--budget", type=float, default=600_000.0, help="RUB spent per strategy per window")
    p.add_argument("--cadence-days", type=int, default=5, help="DCA buy every N trading days (5≈weekly)")
    p.add_argument("--slots-per-week", type=float, default=1.5)
    p.add_argument("--h", type=int, default=20)
    p.add_argument("--out", default="reports/dashboard.html")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    panel = _load_panel(args.corridors, args.data_dir)
    build_dashboard(
        panel, args.corridors, budget=args.budget, cadence_days=args.cadence_days,
        slots_per_week=args.slots_per_week, h=args.h, out_path=args.out,
    )


if __name__ == "__main__":
    main()
