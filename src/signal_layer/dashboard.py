"""An offline, self-contained HTML report; no CDN, server or training required."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


def _read(directory, name):
    try:
        return pd.read_csv(directory / name)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=5))


def generate_dashboard(run, out=None):
    directory = Path(run)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Dashboard requires a completed backtest")
    payload = {"manifest": manifest}
    for key, filename in {
        "summary": "summary.csv",
        "folds": "fold_metrics.csv",
        "diagnostics": "diagnostics.csv",
        "calibration": "calibration.csv",
        "waiting": "waiting_episodes.csv",
        "random": "random_policy_draws.csv",
        "random_days": "random_day_draws.csv",
    }.items():
        payload[key] = _records(_read(directory, filename))
    decisions = _read(directory, "decisions.csv.gz")
    targets = _read(directory, "outcomes.csv.gz")
    decisions = decisions.merge(
        targets[["date", "iso", "y_regret_bps", "y_stale_bps"]], on=["date", "iso"], how="left"
    )
    columns = [
        "date",
        "iso",
        "method",
        "decision",
        "reason",
        "scenario",
        "rub_per_unit",
        "utility_bps",
        "pred_local_min",
        "pred_no_regret",
        "pred_hold",
        "pred_close",
        "upper_regret_bps",
        "upper_stale_bps",
        "y_regret_bps",
        "y_stale_bps",
        "episode_id",
        "push_text",
        "phase",
    ]
    for col in columns:
        if col not in decisions:
            decisions[col] = None
    # Columnar JSON keeps long multi-method reports reasonably small.
    payload["decision_columns"] = columns
    payload["decision_rows"] = json.loads(
        decisions[columns].to_json(orient="values", double_precision=5)
    )
    rates = (
        decisions[["date", "iso", "rub_per_unit"]]
        .drop_duplicates(["date", "iso"])
        .sort_values(["date", "iso"])
    )
    payload["rates"] = _records(rates)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = TEMPLATE.replace("__PLOTLY__", get_plotlyjs()).replace("__PAYLOAD__", data)
    output = Path(out) if out else directory / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output.resolve()


TEMPLATE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FX Signals · Backtest</title>
<style>
:root{--bg:#f4f6fa;--ink:#17233d;--muted:#65738b;--line:#dde4ee;--blue:#245ceb;--green:#117e65;--red:#b94444}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,-apple-system,sans-serif}header{background:#14223e;color:white;padding:32px max(24px,calc((100vw - 1440px)/2))}header small{letter-spacing:2px;color:#adbedc}h1{font-size:32px;font-weight:650;margin:10px 0}header p{color:#c3cfe0;margin:6px 0;line-height:1.6}.badge{display:inline-block;border:1px solid #627398;border-radius:5px;padding:4px 8px;font-size:12px;margin-top:10px}main{max-width:1488px;margin:auto;padding:24px}.controls{display:flex;gap:16px;flex-wrap:wrap;background:white;border:1px solid var(--line);padding:16px 20px;border-radius:12px;align-items:end}.control label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}select,button{font:inherit;padding:9px 12px;border:1px solid #cbd5e5;border-radius:6px;background:white;color:var(--ink)}button{cursor:pointer}button:hover{border-color:var(--blue)}.tabs{display:flex;gap:8px;margin:24px 0 16px;flex-wrap:wrap}.tabs button{background:transparent;border-color:transparent}.tabs .active{background:var(--blue);color:white}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.card,.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:18px}.card .label{font-size:12px;color:var(--muted)}.card .value{font-size:26px;font-weight:650;margin:8px 0}.card .note{font-size:11px;color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.full{grid-column:1/-1}.panel h2{font-size:16px;margin:0 0 8px}.panel p,.note{color:var(--muted);font-size:12px;line-height:1.6;margin:4px 0 12px}.chart{height:360px}.tall{height:480px}.hidden{display:none}.tablewrap{max-height:540px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 8px;vertical-align:top}th{position:sticky;top:0;background:#edf2f9;white-space:nowrap}td.message{min-width:250px;max-width:400px}td.num{font-variant-numeric:tabular-nums}.status{font-size:11px;padding:3px 6px;border-radius:4px;background:#edf2f9}.send{background:#e5f4ee;color:var(--green)}.wait{background:#fff3d8;color:#866015}.alert{margin:14px 0;padding:12px 16px;border-left:3px solid #dba446;background:#fff8e9;line-height:1.6;font-size:13px}.foot{color:var(--muted);font-size:12px;line-height:1.7;margin:24px 0}details{background:white;border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}summary{cursor:pointer;font-weight:600}.empty{padding:24px;color:var(--muted)}.toolbar{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}@media(max-width:950px){.cards{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}h1{font-size:26px}}@media(max-width:550px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr)}.card .value{font-size:22px}.control{width:100%}select{width:100%}}
</style><script>__PLOTLY__</script></head><body>
<header><small>RESEARCH / CROSS-BORDER PAYMENTS</small><h1>Валютные сигналы: проверка на истории</h1><p id="period"></p><p>Качество момента, цена ошибки и ограниченный бюджет уведомлений.</p><span class="badge" id="run-badge"></span></header>
<main><div class="controls">
<div class="control"><label for="iso">Коридор</label><select id="iso"></select></div>
<div class="control"><label for="method">Метод для детального просмотра</label><select id="method"></select></div>
<div class="control"><label for="horizon">Горизонт, календарные дни</label><select id="horizon"></select></div>
<div class="control"><label for="scenario">Тип сигнала</label><select id="scenario"><option value="all">Все сценарии</option><option value="favourable_now">Выгодный уровень</option><option value="window_closing">Наблюдаемый отскок</option></select></div>
</div>
<div class="tabs"><button data-tab="overview" class="active">Обзор и сравнение</button><button data-tab="risk">Риски и калибровка</button><button data-tab="waiting">Цена ожидания</button><button data-tab="journal">Журнал решений</button></div>
<div id="overview" class="tab">
<div class="cards" id="cards"></div><div class="alert" id="evidence"></div>
<div class="grid">
<section class="panel full"><h2>Курс и отправленные сигналы</h2><p id="rate-note"></p><div id="rates" class="chart"></div></section>
<section class="panel"><h2>Lift относительно случайного дня</h2><p>Та же валюта и календарный месяц, тот же сценарий. Полосы — 95% блочные интервалы; отсутствие полос означает недостаток данных.</p><div id="lift" class="chart tall"></div></section>
<section class="panel"><h2>Выгода момента, б.п.</h2><p>Относительное снижение цены против среднего в окне ±h дней. Показатель рассчитан по ЦБ, не является доходностью клиента.</p><div id="gain" class="chart tall"></div></section>
<section class="panel"><h2>Устойчивость по будущим периодам</h2><p>Каждый столбец — отдельный walk-forward период. Отсутствие сигналов отображается пропуском.</p><div id="folds" class="chart tall"></div></section>
<section class="panel"><h2>Частота уведомлений</h2><p>Недельное число сигналов. Для ALL — среднее по коридорам. Целевая полоса: 1–2.</p><div id="frequency" class="chart tall"></div></section>
<section class="panel full"><h2>Сводная таблица сравнения</h2><p>«Не доказано» включает недостаток наблюдений, lift ниже цели или неподходящую частоту. Нулевой поток не считается успешным.</p><div id="summary-table" class="tablewrap"></div></section>
</div></div>
<div id="risk" class="tab hidden"><div class="grid">
<section class="panel"><h2>Калибровка вероятностей</h2><div class="control"><label for="head">Событие</label><select id="head"><option value="no_regret">Нет существенного дальнейшего удешевления</option><option value="local_min">Близость к локальному минимуму</option><option value="hold">Сохранение условий до открытия</option><option value="close">Рост после отскока</option></select></div><p>Диагональ — соответствие предсказанной вероятности наблюдаемой частоте. Выбранные пуши проверяются отдельно.</p><div id="calibration-chart" class="chart"></div></section>
<section class="panel"><h2>Риск преждевременного перевода</h2><p>Наблюдаемая упущенная возможность за основной горизонт и оценённая верхняя граница. Не финансовый убыток по счёту.</p><div id="risk-chart" class="chart"></div></section>
<section class="panel full"><h2>Проверка вероятностей и границ риска</h2><p>Brier/ECE: меньше лучше. Coverage: доля исходов под односторонней границей. Целевое покрытие не является гарантией точности уведомлений.</p><div id="diagnostics-table" class="tablewrap"></div></section></div></div>
<div id="waiting" class="tab hidden"><div class="alert" id="waiting-note"></div><div class="grid">
<section class="panel"><h2>Цена ожидания подтверждения</h2><p>Положительное значение: получатель получил бы меньше за ту же сумму рублей. Здесь показаны только эпизоды с подтверждением.</p><div id="waiting-chart" class="chart"></div></section>
<section class="panel"><h2>Все эпизоды, включая отсутствие подтверждения</h2><p>Для ожидания без подтверждения действие — пропустить; его полезность равна нулю. Сравнение учитывает также пропущенную возможность.</p><div id="waiting-all" class="chart"></div></section>
<section class="panel full"><h2>Разбор эпизодов</h2><div id="waiting-table" class="tablewrap"></div></section></div></div>
<div id="journal" class="tab hidden"><section class="panel"><div class="toolbar"><h2>Почему система отправила, подождала или пропустила</h2><button id="download">Скачать выбранные решения CSV</button></div><p>Последние 200 записей выбранного метода. Выгрузка содержит все отфильтрованные записи.</p><div class="control"><label for="action">Решение</label><select id="action"><option value="all">Все решения</option><option value="send">Отправить</option><option value="wait">Подождать</option><option value="abstain">Пропустить</option></select></div><div id="journal-table" class="tablewrap"></div></section></div>
<details><summary>Методика и ограничения</summary><div class="foot">Все признаки используют только прошлые данные. Цели размечаются на будущих окнах и допускаются в обучение после label_known_on. Обучение, калибровка вероятностей и выбор порогов разделены во времени. Коммуникационный бюджет и эпизоды сохраняются при переходе между фолдами. Дневной ЦБ не позволяет оценить внутридневное устаревание курса приложения. Параметры holdout заранее фиксируются, но плановые переобучения на уже созревших данных продолжаются. Блочные интервалы учитывают временную зависимость; при малом числе блоков интервалы не выводятся. Случайный день сопоставлен по валюте и месяцу; отдельная случайная политика соблюдает лимиты отправки. Правила служат прозрачными контрольными методами и не проходят ML-фильтр риска. Исследовательские абляции показывают вклад ожидания, оценки неопределённости и явных признаков режима.</div><pre id="provenance" style="white-space:pre-wrap;font-size:11px"></pre></details>
<p class="foot">Официальный курс ЦБ — аналитический ориентир. Продуктовый эффект и условия исполнения проверяются отдельно в пилоте. HTML работает автономно; данные и Plotly включены в файл.</p>
</main><script id="data" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), M=D.manifest;
const rows=D.decision_rows.map(a=>Object.fromEntries(D.decision_columns.map((c,i)=>[c,a[i]])));
const $=id=>document.getElementById(id), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={catboost:'CatBoost',linear:'Линейная модель',random_walk:'Random walk',random_walk_drift:'Random walk + drift',ar1:'AR(1)',ets:'ETS: log returns',rule_value:'Правило: уровень',rule_momentum:'Правило: моментум',rule_reversal:'Правило: отскок',rule_seasonal:'Правило: сезонность',random_policy:'Случайная политика',catboost_no_wait:'CatBoost без ожидания',catboost_no_uncertainty:'CatBoost без фильтра границ',catboost_no_regime:'CatBoost без признаков режима'};
const f=(v,n=2)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString('ru-RU',{maximumFractionDigits:n,minimumFractionDigits:n});
const pct=v=>v===null||v===undefined?'—':f(v*100,1)+'%';
const avg=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const methods=[...new Set(D.summary.map(r=>r.method))].sort((a,b)=>(a==='catboost'?-1:b==='catboost'?1:a.localeCompare(b)));
const isos=M.configuration.data.corridors;
function options(id,values,display=v=>v){$(id).innerHTML=values.map(v=>`<option value="${esc(v)}">${esc(display(v))}</option>`).join('')}
options('iso',[...isos,'ALL']);options('method',methods,v=>labels[v]||v);options('horizon',M.configuration.targets.horizons);
$('horizon').value=M.configuration.targets.primary_horizon;
$('period').textContent=M.eval_start+' — '+M.eval_end+' · '+isos.map(x=>'RUB→'+x).join(' / ');
$('run-badge').textContent=M.configuration.model.iterations<=20?'ТЕХНИЧЕСКИЙ ПРОГОН · короткое обучение':'WALK-FORWARD · сохранённые модели каждого периода';
$('provenance').textContent=JSON.stringify({version:M.environment.version,code_sha256:M.environment.code_sha256,seed:M.configuration.seed,holdout_start:M.holdout_start,data:M.environment.data},null,2);
function state(){return {iso:$('iso').value,method:$('method').value,h:Number($('horizon').value),scenario:$('scenario').value}}
function filter(r,s,method=true){return (s.iso==='ALL'||r.iso===s.iso)&&(!method||r.method===s.method)&&(s.scenario==='all'||r.scenario===s.scenario||r.scenario===undefined)}
function plot(id,data,layout={}){Plotly.react(id,data,{paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#65738b',size:11},margin:{l:60,r:24,t:15,b:50},xaxis:{gridcolor:'#edf1f7',zeroline:false},yaxis:{gridcolor:'#edf1f7',zeroline:false},legend:{orientation:'h',y:-.22},...layout},{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']})}
function table(id,headers,data){$(id).innerHTML=data.length?'<table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+data.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table>':'<div class="empty">Нет наблюдений для выбранного фильтра.</div>'}
function selectedRows(){const s=state();return rows.filter(r=>filter(r,s)&&($('action').value==='all'||r.decision===$('action').value))}
function render(){const s=state(), summaries=D.summary.filter(r=>r.iso===s.iso&&r.horizon===s.h&&r.scenario===s.scenario), item=summaries.find(r=>r.method===s.method)||{};
const cards=[['Сигналов',f(item.n_signals,0),'С полным исходом на h'],['Hit rate',pct(item.hit_rate),'Правило зависит от сценария'],['Lift',f(item.lift),'Случайный день = 1'],['Выгода, б.п.',f(item.gain_bps),'Против среднего в ±h'],['В неделю',f(item.frequency_per_week),'В среднем на коридор'],['Ошибка, p95',f(item.regret_p95_bps),'Дальнейшее удешевление, б.п.']];
$('cards').innerHTML=cards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value">${c[1]}</div><div class="note">${c[2]}</div></div>`).join('');
$('evidence').textContent=item.evidence==='promising'?'В этом срезе выполнены критерии lift, выгоды и частоты. Проверьте устойчивость по периодам и другим коридорам.':item.n_signals===0?'Метод не отправил сигналов. Это отказ от коммуникации, а не подтверждение высокой точности. Подробности — в журнале и policy_trials.csv.':'Преимущество по всем критериям ещё не доказано. Оценивайте lift вместе с интервалами, выгодой, частотой и числом наблюдений.';
const active=rows.filter(r=>filter(r,s)), sent=active.filter(r=>r.decision==='send'), rates=D.rates.filter(r=>s.iso==='ALL'||r.iso===s.iso), trace=[];
for(const iso of (s.iso==='ALL'?isos:[s.iso])){const rr=rates.filter(r=>r.iso===iso), base=rr[0]?.rub_per_unit||1, norm=v=>s.iso==='ALL'?100*v/base:v;trace.push({x:rr.map(r=>r.date),y:rr.map(r=>norm(r.rub_per_unit)),type:'scatter',mode:'lines',name:iso,line:{width:2}});const ss=sent.filter(r=>r.iso===iso);trace.push({x:ss.map(r=>r.date),y:ss.map(r=>norm(r.rub_per_unit)),type:'scatter',mode:'markers',name:iso+' · пуш',marker:{size:9,symbol:'diamond',color:'#117e65'},text:ss.map(r=>r.scenario),hovertemplate:'%{x}<br>%{text}<extra></extra>'})}
$('rate-note').textContent=s.iso==='ALL'?'Индекс курса: первое наблюдение = 100. Меньше — выгоднее для отправителя.':'Рубли за единицу валюты. Меньше — выгоднее для отправителя.';plot('rates',trace,{yaxis:{title:{text:s.iso==='ALL'?'Индекс, 100 = начало':'RUB / единица'}}});
const ss=summaries.slice().sort((a,b)=>(b.lift??-1)-(a.lift??-1));
for(const [id,metric,lo,hi] of [['lift','lift','lift_lo','lift_hi'],['gain','gain_bps','gain_lo','gain_hi']]){plot(id,[{type:'bar',orientation:'h',y:ss.map(r=>labels[r.method]||r.method),x:ss.map(r=>r[metric]),marker:{color:ss.map(r=>r.method===s.method?'#245ceb':'#aebfdc')},error_x:{type:'data',array:ss.map(r=>r[hi]===null?0:Math.max(0,r[hi]-r[metric])),arrayminus:ss.map(r=>r[lo]===null?0:Math.max(0,r[metric]-r[lo])),color:'#526789',thickness:1}}],{margin:{l:215,r:25,t:10,b:45},yaxis:{autorange:'reversed'},shapes:[{type:'line',xref:'x',yref:'paper',x0:id==='lift'?1:0,x1:id==='lift'?1:0,y0:0,y1:1,line:{dash:'dot',color:'#c77a40'}}]})}
const fs=D.folds.filter(r=>r.iso===s.iso&&r.scenario===s.scenario&&r.horizon===s.h), foldIds=[...new Set(fs.map(r=>r.fold))];
plot('folds',[{type:'heatmap',x:foldIds.map(i=>'Период '+(i+1)),y:methods.map(m=>labels[m]||m),z:methods.map(m=>foldIds.map(i=>fs.find(r=>r.method===m&&r.fold===i)?.lift??null)),colorscale:[[0,'#f5d4cf'],[.5,'#f6f7fb'],[1,'#8dc8b7']],zmid:1,hoverongaps:false,colorbar:{title:{text:'Lift'}}}],{margin:{l:215,r:30,t:10,b:60},yaxis:{autorange:'reversed'}});
const weeks={};for(const r of sent){const dt=new Date(r.date+'T12:00:00Z');dt.setUTCDate(dt.getUTCDate()-((dt.getUTCDay()+6)%7));const key=dt.toISOString().slice(0,10);weeks[key]=(weeks[key]||0)+1}
let start=new Date(M.eval_start+'T12:00:00Z');start.setUTCDate(start.getUTCDate()-((start.getUTCDay()+6)%7));const wx=[];for(let dt=new Date(start);dt<=new Date(M.eval_end+'T23:59:59Z');dt.setUTCDate(dt.getUTCDate()+7))wx.push(dt.toISOString().slice(0,10));plot('frequency',[{x:wx,y:wx.map(w=>(weeks[w]||0)/(s.iso==='ALL'?isos.length:1)),type:'bar',marker:{color:'#245ceb'}}],{shapes:[{type:'rect',xref:'paper',yref:'y',x0:0,x1:1,y0:1,y1:2,fillcolor:'#117e6515',line:{width:0},layer:'below'}],yaxis:{title:{text:'Сигналов / неделю'},rangemode:'tozero'}});
table('summary-table',['Метод','N','Hit','Random hit','Lift [95%]','Выгода, б.п.','В неделю','Пустые недели','Ошибка p95','Вывод'],ss.map(r=>[esc(labels[r.method]||r.method),f(r.n_signals,0),pct(r.hit_rate),pct(r.random_day_hit),f(r.lift)+' ['+f(r.lift_lo)+'; '+f(r.lift_hi)+']',f(r.gain_bps),f(r.frequency_per_week),pct(r.empty_week_share),f(r.regret_p95_bps),r.evidence==='promising'?'Критерии выполнены':r.evidence==='no_signals'?'Нет сигналов':'Не доказано']));
const head=$('head').value, cal=D.calibration.filter(r=>r.method===s.method&&r.head===head&&(s.iso==='ALL'||r.iso===s.iso)), ct=[{x:[0,1],y:[0,1],mode:'lines',name:'Идеальная калибровка',line:{color:'#adb9cc',dash:'dot'}}];
for(const scope of ['all_predictions','sent_only']){const a=[];for(let b=0;b<10;b++){const g=cal.filter(r=>r.scope===scope&&r.bin===b), n=g.reduce((z,r)=>z+r.n,0);if(n)a.push({p:g.reduce((z,r)=>z+r.predicted*r.n,0)/n,y:g.reduce((z,r)=>z+r.observed*r.n,0)/n,n})}ct.push({x:a.map(r=>r.p),y:a.map(r=>r.y),text:a.map(r=>'N='+r.n),mode:'lines+markers',name:scope==='all_predictions'?'Все прогнозы':'Отправленные пуши'})}plot('calibration-chart',ct,{xaxis:{title:{text:'Прогноз'},range:[0,1]},yaxis:{title:{text:'Наблюдаемая частота'},range:[0,1]}});
plot('risk-chart',[{x:active.map(r=>r.date),y:active.map(r=>r.upper_regret_bps),type:'scatter',mode:'lines',name:'Верхняя оценка',line:{color:'#245ceb',width:1}},{x:sent.map(r=>r.date),y:sent.map(r=>r.y_regret_bps),type:'scatter',mode:'markers',name:'Факт после пуша',marker:{color:'#b94444',size:7}}],{yaxis:{title:{text:'Базисные пункты'}}});
table('diagnostics-table',['Коридор','Выборка','Метрика','Значение','N'],D.diagnostics.filter(r=>r.method===s.method&&(s.iso==='ALL'||r.iso===s.iso)).map(r=>[esc(r.iso),r.scope==='sent_only'?'Отправленные':'Все прогнозы',esc(r.metric),f(r.value,4),f(r.n,0)]));
const ww=D.waiting.filter(r=>r.method===s.method&&(s.iso==='ALL'||r.iso===s.iso)), yes=ww.filter(r=>r.confirmed===true), no=ww.filter(r=>r.confirmed===false);
$('waiting-note').textContent=`Эпизодов: ${ww.length}. С подтверждением: ${yes.length}; без подтверждения: ${no.length}. Средняя цена ожидания среди подтвердившихся: ${f(avg(yes.map(r=>r.waiting_cost_bps).filter(x=>x!==null)))} б.п. Это диагностическое сравнение быстрых и медленных правил, а не гарантированная ценность ожидания.`;
plot('waiting-chart',[{x:yes.map(r=>r.days_waited),y:yes.map(r=>r.waiting_cost_bps),mode:'markers',type:'scatter',text:yes.map(r=>r.iso+' '+r.fast_date),marker:{color:'#245ceb',size:8},hovertemplate:'%{text}<br>%{x} дней · %{y:.1f} б.п.<extra></extra>'}],{xaxis:{title:{text:'Дней до подтверждения'}},yaxis:{title:{text:'Цена ожидания, б.п.'}}});
plot('waiting-all',[{x:ww.map(r=>r.confirmed?'Подтверждение':'Без подтверждения'),y:ww.map(r=>r.wait_minus_fast_bps),type:'box',boxpoints:'all',jitter:.3,marker:{color:'#245ceb',size:4},name:'Ожидание − сразу'}],{yaxis:{title:{text:'Разница полезности, б.п.'}}});
table('waiting-table',['Коридор','Быстрый сигнал','Подтверждение','Первое решение','Цена ожидания','Ожидание − сразу'],ww.slice(-200).reverse().map(r=>[esc(r.iso),esc(r.fast_date),esc(r.confirmation_date||'Не было'),esc(r.selected_action),f(r.waiting_cost_bps),f(r.wait_minus_fast_bps)]));
renderJournal();}
function renderJournal(){table('journal-table',['Дата','Коридор','Решение','Причина','Сценарий','Курс','Оценка, б.п.','Текст'],selectedRows().slice(-200).reverse().map(r=>[esc(r.date),esc(r.iso),`<span class="status ${esc(r.decision)}">${esc(r.decision)}</span>`,esc(r.reason),esc(r.scenario),f(r.rub_per_unit,5),f(r.utility_bps),esc(r.push_text)]))}
for(const id of ['iso','method','horizon','scenario','head'])$(id).addEventListener('change',render);
$('action').addEventListener('change',renderJournal);
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('hidden',x.id!==b.dataset.tab));document.querySelectorAll('#'+b.dataset.tab+' .chart').forEach(x=>Plotly.Plots.resize(x))}));
$('download').addEventListener('click',()=>{const cols=D.decision_columns, quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"', csv='\ufeff'+[cols.map(quote).join(','),...selectedRows().map(r=>cols.map(c=>quote(r[c])).join(','))].join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'})), a=document.createElement('a');a.href=url;a.download='fx-decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
render();window.__FX_REPORT_READY__=true;
</script></body></html>"""
