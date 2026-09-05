/* Симуляция сигнального слоя.
 *
 * Фронт не знает ни одного правила: он двигает дату, спрашивает backend
 * «сегодня есть повод написать?» и «то, что мы написали, ещё верно?», и рисует
 * ответ. Все пороги, отбор дня и уровень раскрытия живут в
 * signal_layer.services.simulation.
 */
(function () {
  "use strict";

  var MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
  var DAYS = ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"];
  var RECIPIENTS = {
    TJS: ["Фарход А.", "ФА"], UZS: ["Азиз Р.", "АР"], KGS: ["Нурбек Т.", "НТ"],
    AMD: ["Ануш М.", "АМ"], KZT: ["Айгуль С.", "АС"]
  };
  var AMOUNT = 50000;
  var SEND_TIME = "09:30";
  var TICK_MS = 650;
  var VISIBLE = 120;

  /* ── Состояние ───────────────────────────────────────── */
  var corridors = [];
  var meta = null;
  var iso = null;
  var series = [];
  var byDate = {};
  var day = null;            // текущий день симуляции, "YYYY-MM-DD"
  var playing = false;
  var timer = null;
  var stepped = 0;
  var fired = [];            // сигналы, до которых симуляция уже дошла
  var pushes = [];           // что лежит на экране блокировки, свежие сверху
  var openPush = null;       // открытый пуш
  var fresh = null;          // ответ backend о свежести открытого пуша
  var view = "lock";
  var sheetOpen = false;
  var lastSignalDate = null;
  var atEnd = false;

  /* ── Формат ──────────────────────────────────────────── */
  function fmt(x, d) {
    return x.toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function ru(s) { var p = s.split("-"); return p[2] + "." + p[1] + "." + p[0]; }
  function ruLong(s) { var p = s.split("-"); return Number(p[2]) + " " + MONTHS[Number(p[1]) - 1] + " " + p[0]; }
  function weekday(s) { var p = s.split("-"); return DAYS[new Date(Date.UTC(+p[0], +p[1] - 1, +p[2])).getUTCDay()]; }
  function pct(bps) { return fmt(Math.abs(bps) / 100, 2) + " %"; }
  function digitsFor(code) { return code === "UZS" ? 6 : 4; }
  function addDays(s, n) {
    var p = s.split("-");
    var t = new Date(Date.UTC(+p[0], +p[1] - 1, +p[2] + n));
    return t.toISOString().slice(0, 10);
  }
  function daysBetween(a, b) {
    var pa = a.split("-"), pb = b.split("-");
    var ta = Date.UTC(+pa[0], +pa[1] - 1, +pa[2]), tb = Date.UTC(+pb[0], +pb[1] - 1, +pb[2]);
    return Math.round((tb - ta) / 86400000);
  }
  function ago(n) {
    if (n <= 0) return "сейчас";
    if (n === 1) return "вчера";
    var last = n % 10, tens = n % 100;
    var word = (last === 1 && tens !== 11) ? "день" : (last >= 2 && last <= 4 && (tens < 12 || tens > 14)) ? "дня" : "дней";
    return n + " " + word + " назад";
  }

  /* ── Backend ─────────────────────────────────────────── */
  function api(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error(path + " → " + r.status);
      return r.json();
    });
  }

  /* ── График ──────────────────────────────────────────── */
  function drawChart() {
    var svg = document.getElementById("chart");
    var seen = series.filter(function (p) { return p.d <= day; });
    if (!seen.length) { svg.innerHTML = ""; return; }
    var pts = seen.slice(Math.max(0, seen.length - VISIBLE));

    var W = 900, H = 380, L = 64, R = 18, T = 18, B = 30;
    var lo = Infinity, hi = -Infinity;
    pts.forEach(function (p) { lo = Math.min(lo, p.r, p.e); hi = Math.max(hi, p.r, p.e); });
    var pad = (hi - lo) * 0.12 || Math.max(hi * 0.001, 1e-6);
    lo -= pad; hi += pad;
    var span = Math.max(pts.length - 1, 1);
    var x = function (i) { return L + i * (W - L - R) / span; };
    var y = function (v) { return T + (hi - v) * (H - T - B) / (hi - lo); };
    var index = {};
    pts.forEach(function (p, i) { index[p.d] = i; });

    var out = [];
    for (var t = 0; t <= 4; t++) {
      var v = lo + (hi - lo) * t / 4, yy = y(v);
      out.push('<line x1="' + L + '" y1="' + yy.toFixed(1) + '" x2="' + (W - R) + '" y2="' + yy.toFixed(1) + '" style="stroke:var(--line)" stroke-width="1"/>');
      out.push('<text x="' + (L - 10) + '" y="' + (yy + 4).toFixed(1) + '" text-anchor="end" style="font-family:var(--mono);font-size:11px;fill:var(--muted)">' + fmt(v, digitsFor(iso)) + "</text>");
    }
    var month = null;
    pts.forEach(function (p, i) {
      var m = p.d.slice(0, 7);
      if (m !== month) {
        if (i > 2 && i < pts.length - 2) {
          out.push('<text x="' + x(i).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle" style="font-family:var(--mono);font-size:11px;fill:var(--muted)">' + MONTHS[Number(p.d.slice(5, 7)) - 1] + "</text>");
        }
        month = m;
      }
    });

    var line = pts.map(function (p, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.r).toFixed(1); }).join(" ");
    var trend = pts.map(function (p, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.e).toFixed(1); }).join(" ");
    out.push('<defs><linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" style="stop-color:var(--ink-2);stop-opacity:.14"/>' +
      '<stop offset="100%" style="stop-color:var(--ink-2);stop-opacity:0"/></linearGradient></defs>');
    out.push('<path d="' + line + " L" + x(pts.length - 1).toFixed(1) + " " + (H - B) + " L" + L + " " + (H - B) + ' Z" style="fill:url(#fillGrad)"/>');
    out.push('<path d="' + trend + '" style="fill:none;stroke:var(--muted);stroke-width:1.4;stroke-dasharray:4 4;opacity:.9"/>');
    out.push('<path d="' + line + '" style="fill:none;stroke:var(--ink-2);stroke-width:1.8;stroke-linejoin:round"/>');

    fired.forEach(function (s) {
      var i = index[s.date];
      if (i === undefined) return;
      var last = s.date === lastSignalDate;
      if (last) {
        out.push('<line x1="' + x(i).toFixed(1) + '" y1="' + T + '" x2="' + x(i).toFixed(1) + '" y2="' + (H - B) + '" style="stroke:var(--red);stroke-width:1;opacity:.35"/>');
      }
      out.push('<circle cx="' + x(i).toFixed(1) + '" cy="' + y(s.rate).toFixed(1) + '" r="' + (last ? 6.5 : 3.8) + '" style="fill:var(--red);stroke:var(--paper);stroke-width:' + (last ? 2.5 : 1.4) + '"><title>' + ru(s.date) + "</title></circle>");
    });

    // Голова симуляции: последняя котировка, до которой дошёл календарь.
    var head = pts[pts.length - 1];
    out.push('<circle cx="' + x(pts.length - 1).toFixed(1) + '" cy="' + y(head.r).toFixed(1) + '" r="4" style="fill:var(--paper);stroke:var(--ink-2);stroke-width:2"/>');

    svg.innerHTML = out.join("");
  }

  /* ── Телефон ─────────────────────────────────────────── */
  function lockScreen() {
    var list = pushes.map(function (n, i) {
      var age = daysBetween(n.date, day);
      return '<button class="push' + (age > 0 ? " aged" : "") + '" type="button" data-push="' + i + '">' +
        '<span class="push-head"><span class="logo" aria-hidden="true">А</span>Альфа-Банк<span class="when">' + ago(age) + "</span></span>" +
        "<b>Переводы за рубеж</b><p>" + n.signal.client_message + "</p></button>";
    }).join("");

    return '<div class="statusbar"><span>' + SEND_TIME + '</span><span class="sig">●●●● Alfa 5G</span></div>' +
      '<div class="lock">' +
      '<div class="lock-time">' + SEND_TIME + "</div>" +
      '<div class="lock-date">' + ruLong(day) + ", " + weekday(day) + "</div>" +
      '<div class="pushes">' + list + "</div>" +
      '<div class="lock-hint">' + (pushes.length ? "Нажмите на уведомление" : "Уведомлений нет") + "</div></div>";
  }

  function freshnessBlock() {
    var stamp = ru(fresh.signal_date) + ", " + SEND_TIME;
    var gotThen = AMOUNT / fresh.signal_rate, gotNow = AMOUNT / fresh.current_rate;
    if (fresh.level === "same") {
      return '<div class="freshness ok"><span class="dot"></span><span><b>Данные актуальны на ' + stamp +
        "</b><br>Курс с момента уведомления не менялся.</span></div>";
    }
    if (fresh.level === "better") {
      return '<div class="freshness ok"><span class="dot"></span><span><b>Данные были актуальны на ' + stamp +
        "</b><br>С тех пор курс изменился в вашу пользу на " + pct(fresh.delta_bps) + ": получатель получит на " +
        fmt(gotNow - gotThen, 2) + " " + iso + " больше.</span></div>";
    }
    if (fresh.level === "mild") {
      return '<div class="freshness note"><span class="dot"></span><span><b>Данные были актуальны на ' + stamp +
        "</b><br>Актуальные данные могли измениться: с тех пор курс изменился на " + pct(fresh.delta_bps) +
        ", форма пересчитана по курсу на " + ru(fresh.current_date) + ".</span></div>";
    }
    return '<div class="freshness warn"><span class="dot"></span><span><b>Данные были актуальны на ' + stamp +
      "</b><br>Актуальные данные могли измениться — курс изменился на " + pct(fresh.delta_bps) + " не в вашу пользу.</span></div>";
  }

  function transferScreen() {
    var d = digitsFor(iso);
    var who = RECIPIENTS[iso] || ["Получатель", "П"];
    var s = openPush.signal;
    var gotNow = AMOUNT / fresh.current_rate;
    return '<div class="statusbar"><span>' + SEND_TIME + '</span><span class="sig">●●●● Alfa 5G</span></div>' +
      '<div class="app">' +
      '<div class="appbar"><button class="back" id="backBtn" type="button" aria-label="Назад">‹</button><h4>Перевод за рубеж</h4></div>' +
      '<div class="appbody">' +
      freshnessBlock() +
      '<div class="tile"><div class="tile-row"><span class="avatar">' + who[1] + "</span>" +
      '<span class="grow"><span class="tile-label">Получателю в ' + meta.country + '</span><br><span class="tile-value">' + who[0] + " · карта •• 4417</span></span></div></div>" +
      '<div class="tile"><div class="tile-label">Сумма перевода</div>' +
      '<div class="amount">' + fmt(AMOUNT, 0) + " ₽</div>" +
      '<div class="gets" style="margin-top:6px">Получатель получит ≈ <b>' + fmt(gotNow, 2) + " " + iso + "</b></div>" +
      '<div class="small" style="margin-top:8px">Курс ЦБ на ' + ru(fresh.current_date) + ": 1 " + iso + " = " + fmt(fresh.current_rate, d) + " ₽</div></div>" +
      '<div class="tile"><div class="tile-label" style="margin-bottom:6px">Почему мы написали</div>' +
      '<div class="fact">' + s.client_message + " Данные ЦБ на " + ru(s.date) + ".</div></div>" +
      '<p class="small">Сумма и получатель подставлены из ваших прошлых переводов — их можно изменить.</p>' +
      "</div>" +
      '<div class="appfoot"><button class="btn" id="sendBtn" type="button">Перевести ' + fmt(AMOUNT, 0) + " ₽</button></div>" +
      "</div>" + (sheetOpen ? sheet() : "");
  }

  function sheet() {
    var d = digitsFor(iso);
    var gotThen = AMOUNT / fresh.signal_rate, gotNow = AMOUNT / fresh.current_rate;
    return '<div class="sheet-scrim" id="scrim"><div class="sheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle">' +
      '<div class="grabber"></div>' +
      '<h3 id="sheetTitle">Курс изменился с момента уведомления</h3>' +
      '<p class="stale-line">Данные были актуальны на <b>' + ru(fresh.signal_date) + ", " + SEND_TIME +
      "</b>. Актуальные данные могли измениться — ниже курс на " + ru(fresh.current_date) + ".</p>" +
      '<div class="compare">' +
      '<div class="side"><div class="cap">в уведомлении</div><div class="val">' + fmt(fresh.signal_rate, d) +
      ' ₽</div><div class="sub">≈ ' + fmt(gotThen, 2) + " " + iso + "</div></div>" +
      '<div class="arrow" aria-hidden="true">→</div>' +
      '<div class="side now"><div class="cap">сейчас</div><div class="val">' + fmt(fresh.current_rate, d) +
      ' ₽</div><div class="sub">≈ ' + fmt(gotNow, 2) + " " + iso + "</div></div></div>" +
      '<p class="delta-line">Курс изменился на <b>' + pct(fresh.delta_bps) + "</b> не в вашу пользу: при переводе " +
      fmt(AMOUNT, 0) + " ₽ получатель получит на <b>" + fmt(gotThen - gotNow, 2) + " " + iso +
      "</b> меньше, чем было в уведомлении.</p>" +
      '<div class="sheet-actions">' +
      '<button class="btn" id="acceptBtn" type="button">Перевести по текущему курсу</button>' +
      '<button class="btn grey" id="notifyBtn" type="button">Напомнить при следующем сигнале</button>' +
      '<button class="btn ghost" id="closeSheet" type="button">Не сейчас</button></div>' +
      '<p class="why">Курс ЦБ РФ, без спреда и комиссии перевода</p>' +
      "</div></div>";
  }

  function renderPhone() {
    var screen = document.getElementById("screen");
    screen.innerHTML = view === "lock" || !fresh ? lockScreen() : transferScreen();

    Array.prototype.forEach.call(screen.querySelectorAll("[data-push]"), function (el) {
      el.addEventListener("click", function () { open(pushes[+el.getAttribute("data-push")]); });
    });
    var back = screen.querySelector("#backBtn");
    if (back) back.addEventListener("click", function () { view = "lock"; sheetOpen = false; renderPhone(); });
    var close = screen.querySelector("#closeSheet");
    if (close) close.addEventListener("click", function () { sheetOpen = false; renderPhone(); });
    var accept = screen.querySelector("#acceptBtn");
    if (accept) accept.addEventListener("click", function () { sheetOpen = false; renderPhone(); });
    var send = screen.querySelector("#sendBtn");
    if (send) send.addEventListener("click", function () { dismiss(openPush); });
    var notify = screen.querySelector("#notifyBtn");
    if (notify) {
      notify.addEventListener("click", function () {
        notify.textContent = "Напомним при следующем сигнале";
        notify.disabled = true;
        notify.style.opacity = ".7";
      });
    }
    var scrim = screen.querySelector("#scrim");
    if (scrim) {
      scrim.addEventListener("click", function (e) {
        if (e.target === scrim) { sheetOpen = false; renderPhone(); }
      });
    }
  }

  function open(n) {
    if (!n) return;
    pause();
    openPush = n;
    api("/api/freshness/" + iso + "?signal_date=" + n.date + "&as_of=" + day).then(function (f) {
      fresh = f;
      view = "app";
      sheetOpen = f.level === "stale";
      renderPhone();
      renderStatus();
    });
  }

  function dismiss(n) {
    pushes = pushes.filter(function (p) { return p !== n; });
    openPush = null;
    fresh = null;
    view = "lock";
    sheetOpen = false;
    renderPhone();
    renderStatus();
  }

  /* ── Индикаторы ──────────────────────────────────────── */
  function renderClock() {
    document.getElementById("clockDate").textContent = ru(day);
    var quote = byDate[day];
    var last = null;
    for (var i = series.length - 1; i >= 0; i--) {
      if (series[i].d <= day) { last = series[i]; break; }
    }
    document.getElementById("clockRate").textContent = last
      ? (quote ? "1 " + iso + " = " + fmt(last.r, digitsFor(iso)) + " ₽"
               : "выходной · курс от " + ru(last.d))
      : "нет котировок";
  }

  function renderStatus() {
    var box = document.getElementById("status");
    var text = document.getElementById("statusText");
    box.className = "status" + (playing ? " run" : lastSignalDate && !playing && pushes.length ? " hit" : "");
    var unread = pushes.length;
    var parts = [];
    if (atEnd) parts.push("данные закончились — «Симуляция» начнёт заново");
    else if (playing) parts.push("идёт симуляция");
    else if (lastSignalDate) parts.push("сигнал " + ru(lastSignalDate) + " — пауза");
    else parts.push("пауза");
    if (unread) {
      var age = daysBetween(pushes[0].date, day);
      parts.push(unread === 1 ? "1 пуш на экране, " + ago(age) : unread + " пуша на экране");
    }
    parts.push("шагов: " + stepped);
    text.textContent = parts.join(" · ");
    document.getElementById("play").disabled = playing;
    document.getElementById("pause").disabled = !playing;
  }

  function render() {
    renderClock();
    drawChart();
    renderPhone();
    renderStatus();
  }

  /* ── Ход времени ─────────────────────────────────────── */
  function step() {
    if (day >= meta.last_date) { atEnd = true; pause(); renderStatus(); return Promise.resolve(); }
    day = addDays(day, 1);
    stepped += 1;
    return api("/api/day/" + iso + "?on=" + day).then(function (res) {
      if (res.decision === "candidate" && res.signal) {
        fired.push({ date: res.signal.date, rate: res.signal.rate });
        pushes.unshift({ date: res.signal.date, signal: res.signal });
        pushes = pushes.slice(0, 3);   // экран блокировки не резиновый
        lastSignalDate = res.signal.date;
        // Сигнал останавливает симуляцию: дальше решает человек.
        pause();
        view = "lock";
        sheetOpen = false;
      }
      render();
    }).catch(function () { pause(); });
  }

  function play() {
    if (playing) return;
    if (atEnd) {           // повторный запуск с начала: других кнопок нет
      atEnd = false;
      resetRun(meta.sim_start);
      render();
    }
    playing = true;
    renderStatus();
    tick();
  }

  function tick() {
    if (!playing) return;
    step().then(function () {
      if (playing) timer = setTimeout(tick, TICK_MS);
    });
  }

  function pause() {
    playing = false;
    if (timer) { clearTimeout(timer); timer = null; }
    renderStatus();
  }

  function resetRun(from) {
    day = from;
    stepped = 0;
    fired = [];
    pushes = [];
    openPush = null;
    fresh = null;
    lastSignalDate = null;
    view = "lock";
    sheetOpen = false;
  }

  /* ── Коридоры ────────────────────────────────────────── */
  function selectCorridor(code) {
    pause();
    iso = code;
    meta = corridors.filter(function (c) { return c.iso === code; })[0];
    atEnd = false;
    return api("/api/series/" + code).then(function (res) {
      series = res.series;
      byDate = {};
      series.forEach(function (p) { byDate[p.d] = p; });
      resetRun(meta.sim_start);
      Array.prototype.forEach.call(document.getElementById("tabs").children, function (el) {
        el.setAttribute("aria-selected", String(el.getAttribute("data-iso") === code));
      });
      render();
    });
  }

  function boot() {
    api("/api/corridors").then(function (list) {
      corridors = list;
      var tabs = document.getElementById("tabs");
      list.forEach(function (c) {
        var b = document.createElement("button");
        b.className = "tab";
        b.type = "button";
        b.setAttribute("role", "tab");
        b.setAttribute("data-iso", c.iso);
        b.setAttribute("aria-selected", "false");
        b.textContent = "RUB → " + c.iso;
        b.title = c.country;
        b.addEventListener("click", function () { selectCorridor(c.iso); });
        tabs.appendChild(b);
      });
      return selectCorridor(list[0].iso);
    }).catch(function (error) {
      document.getElementById("statusText").textContent = "backend недоступен: " + error.message;
    });
  }

  document.getElementById("play").addEventListener("click", play);
  document.getElementById("pause").addEventListener("click", pause);
  boot();
})();
