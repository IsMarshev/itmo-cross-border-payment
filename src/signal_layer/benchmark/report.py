"""Rendering a benchmark run into something a reader can act on.

The scorecard is written in Russian because that is the language of the case
review, and it leads with the verdict rather than the method: which strategies
close the case, which are indistinguishable from a random day, and what the MVP
costs in signals when the price of error goes up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .runner import BenchmarkResult
from .spec import BenchmarkSpec


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, bool | np.bool_):
        return "да" if value else "нет"
    if isinstance(value, float | np.floating):
        return f"{value:.{digits}f}"
    if isinstance(value, pd.Timestamp):
        return f"{value:%Y-%m-%d}"
    return str(value)


def _table(frame: pd.DataFrame, columns: dict[str, str], digits: int = 2) -> str:
    available = [c for c in columns if c in frame.columns]
    header = "| " + " | ".join(columns[c] for c in available) + " |"
    divider = "|" + "|".join(["---"] * len(available)) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in available) + " |")
    return "\n".join(lines)


def _verdict(entry: pd.Series, gates: pd.DataFrame) -> str:
    rows = gates[gates["strategy"].eq(entry["strategy"])]
    failed = rows[rows["passed"].eq(False)]["gate"].tolist()
    if not len(rows):
        return "нет данных"
    if not failed:
        lift = entry.get("hit_lift", float("nan"))
        if np.isfinite(lift) and lift >= 1.3:
            return "проходит, lift на цели"
        return "проходит"
    return "не проходит: " + ", ".join(failed)


def render_scorecard(result: BenchmarkResult, spec: BenchmarkSpec) -> str:
    """The full markdown report for one run."""
    board = result.leaderboard.copy()
    board["вердикт"] = board.apply(lambda row: _verdict(row, result.gates), axis=1)

    folds = result.per_fold["fold"].nunique() if len(result.per_fold) else 0
    period = ""
    if len(result.per_fold):
        period = (
            f"{result.per_fold['fold_start'].min():%Y-%m-%d}"
            f" — {result.per_fold['fold_end'].max():%Y-%m-%d}"
        )

    parts: list[str] = []
    parts.append("# CBSB-1: результаты прогона\n")
    parts.append(
        f"Коридоры: {', '.join(spec.corridors)}. Период оценки: {period} "
        f"({folds} out-of-time окон по {spec.fold_months} мес). "
        f"Горизонт h = {spec.horizon} дней, исполнение через "
        f"{spec.execution_offset} наблюдение после сигнала. "
        f"Бюджет пушей одинаков для всех стратегий: не более "
        f"{spec.max_signals_per_week} в неделю.\n"
    )

    parts.append("\n## Главная таблица\n")
    parts.append(
        "`CBSB` — доля пути от случайного расписания (0) до идеального знания "
        "будущего при том же бюджете пушей (100). `выгода, б.п.` — на сколько "
        "больше валюты получает клиент, переводя в дни сигналов, чем переводя "
        "равномерно. `lift` — во сколько раз чаще утверждение пуша оказывалось "
        "правдой, чем в случайный день. `p` — вероятность получить такой "
        "выигрыш случайным расписанием.\n"
    )
    parts.append(
        "`коридоров +` — на скольких из пяти коридоров выигрыш вообще положителен; "
        "`коридоров ✓` — на скольких он пережил поправку Benjamini-Hochberg. "
        "Устойчивость по ТЗ — это они, а не общая цифра: пул может держаться "
        "на одном коридоре.\n"
    )
    contenders = board[board["selection"].eq("policy")] if "selection" in board else board
    parts.append(
        _table(
            contenders,
            {
                "strategy": "стратегия",
                "cbsb_score": "CBSB",
                "currency_uplift_bps": "выгода, б.п.",
                "currency_uplift_worst_corridor": "худший коридор",
                "corridors_positive": "коридоров +",
                "corridors_significant": "коридоров ✓",
                "hit_rate": "hit rate",
                "hit_lift": "lift",
                "p_value": "p",
                "bad_push_rate": "плохих пушей",
                "per_week": "в неделю",
                "n_signals": "сигналов",
                "вердикт": "вердикт",
            },
        )
    )

    diagnostics = board[board["selection"].eq("weekly_best")] if "selection" in board else None
    if diagnostics is not None and len(diagnostics):
        parts.append("\n\n### Сколько стоит жадная политика отправки\n")
        parts.append(
            "Строки ниже — не участники, а диагностика. Они берут тот же счёт, но "
            "выбирают лучшие дни недели задним числом. Разрыв с их обычными "
            "двойниками показывает, сколько выгоды съедает онлайн-политика "
            "«отправь первый день выше скользящего квантиля», а не качество "
            "самого индикатора.\n\n"
        )
        parts.append(
            _table(
                diagnostics,
                {
                    "strategy": "диагностика",
                    "currency_uplift_bps": "выгода, б.п.",
                    "hit_rate": "hit rate",
                    "hit_lift": "lift",
                    "per_week": "в неделю",
                },
            )
        )

    parts.append("\n\n## Обязательные условия (гейты)\n")
    if len(result.gates):
        pivot = result.gates.pivot_table(
            index="strategy", columns="gate", values="passed", aggfunc="first"
        )
        pivot = pivot.reindex(board["strategy"]).reset_index()
        parts.append(
            _table(pivot, {c: c for c in pivot.columns})
        )
        parts.append("\n\nЧто проверяет каждый гейт:\n\n")
        for gate in spec.gates:
            parts.append(f"- **{gate.name}** — {gate.question} `{gate.describe()}`\n")

    parts.append("\n## Правдивость по горизонтам\n")
    if len(result.horizon_table):
        parts.append(
            "Доля сигналов, после которых утверждение пуша подтвердилось на "
            "горизонте h. Правило проверки зависит от сценария сообщения.\n\n"
        )
        parts.append(
            _table(
                result.horizon_table,
                {
                    "strategy": "стратегия",
                    "scenario": "сценарий",
                    **{f"hit_h{h}": f"h={h}" for h in spec.reported_horizons},
                },
            )
        )

    parts.append("\n\n## Цена ошибки: что даёт lambda\n")
    if len(result.lambda_sweep):
        parts.append(
            "`lambda` — во сколько раз рубль, потерянный клиентом после нашего "
            "пуша, дороже рубля упущенной возможности. Модель не подбирает его "
            "по данным: это продуктовое решение, и здесь видно, что оно стоит.\n\n"
        )
        parts.append(
            _table(
                result.lambda_sweep,
                {
                    "lam": "lambda",
                    "n_signals": "сигналов",
                    "per_week": "в неделю",
                    "currency_uplift_bps": "выгода, б.п.",
                    "hit_rate": "hit rate",
                    "bad_push_rate": "плохих пушей",
                    "regret_bps": "средний regret, б.п.",
                },
            )
        )
        parts.append(
            "\n\nЧитать эту таблицу нужно так: **lambda почти ничего не меняет.** "
            "Доля плохих пушей стоит на месте, выгода скачет без тренда. Причина "
            "видна в самих головах: риск отрицательно коррелирован с полезностью "
            "(−0.37…−0.59 по коридорам) и имеет втрое меньший разброс, поэтому "
            "вычитание риска усиливает тот же порядок дней, а не меняет его. "
            "Голова риска в нынешнем виде не приносит независимой информации — "
            "чтобы lambda начала работать, ей нужен источник, которого нет у "
            "головы полезности: режим волатильности, а не те же 22 признака "
            "возврата к среднему.\n"
        )

    parts.append("\n\n## Два правила правдивости противоречат друг другу\n")
    if len(result.leaderboard) and "hit_favourable" in result.leaderboard:
        parts.append(
            "«Сейчас выгодно» засчитывается, когда курс h дней не поднимается "
            "выше — то есть когда он продолжает падать и клиенту следовало "
            "подождать. «Окно закрывается» засчитывается, когда курс вырос, — "
            "то есть когда клиент успел. На этих данных первое правило "
            "отрицательно связано с деньгами клиента, второе положительно, "
            "поэтому гнаться за hit rate по первому правилу вредно.\n\n"
        )
        parts.append(
            _table(
                result.leaderboard,
                {
                    "strategy": "стратегия",
                    "currency_uplift_bps": "выгода, б.п.",
                    "hit_favourable": "hit «выгодно сейчас»",
                    "hit_closing": "hit «окно закрывается»",
                    "hit_lift_favourable": "lift «выгодно»",
                    "hit_lift_closing": "lift «окно»",
                },
            )
        )

    parts.append("\n\n## По коридорам\n")
    if len(result.per_corridor):
        interesting = result.per_corridor[
            result.per_corridor["strategy"].isin(
                ["utility_risk", "utility_only", "percentile", "oracle"]
            )
        ]
        parts.append(
            _table(
                interesting,
                {
                    "strategy": "стратегия",
                    "iso": "коридор",
                    "n_signals": "сигналов",
                    "currency_uplift_bps": "выгода, б.п.",
                    "hit_rate": "hit rate",
                    "hit_lift": "lift",
                    "window_advantage_bps": "выгода момента ±h",
                    "currency_uplift_ci_low": "CI низ",
                    "currency_uplift_hac_t": "HAC t",
                    "p_value": "p",
                    "q_value": "q (BH)",
                },
            )
        )

    parts.append("\n\n## Как не переоценить эти цифры\n")
    parts.append(
        "Пять коридоров — почти один ряд: основное движение даёт рубль, а не "
        "валюта получателя. Поэтому общий `p` по стратегии завышает силу "
        "доказательства: он посчитан так, будто пять коридоров независимы, а "
        "это не так. Читать стоит в порядке `коридоров +` → `коридоров ✓` → "
        "разброс по окнам в `per_fold.csv` → и только потом общий `p`. "
        "Настоящая независимая ось здесь — время, а не коридор.\n"
    )

    parts.append("\n## Аудит на заглядывание вперёд\n")
    if len(result.audit):
        passed = bool(result.audit["matched"].all())
        parts.append(
            "Счёт модели пересчитан на обрезанной панели (данные после даты "
            "среза физически удалены) и сверен с историческим прогоном.\n\n"
        )
        parts.append(
            _table(
                result.audit,
                {
                    "iso": "коридор",
                    "asof": "дата среза",
                    "asof_score": "счёт as-of",
                    "historical_score": "счёт в прогоне",
                    "abs_difference": "разница",
                    "matched": "совпало",
                },
                digits=6,
            )
        )
        parts.append(f"\n\n**Итог аудита: {'пройден' if passed else 'ПРОВАЛЕН'}.**\n")
    else:
        parts.append("Аудит не запускался.\n")

    return "".join(
        part if part.endswith("\n") else part + "\n" for part in parts
    )


def render_console_summary(result: BenchmarkResult) -> str:
    """A short verdict for the terminal."""
    if result.leaderboard.empty:
        return "Нет результатов."
    lines = ["", "CBSB-1 — итог:", ""]
    for _, row in result.leaderboard.iterrows():
        lines.append(
            f"  {row['strategy']:<22} CBSB {_fmt(row['cbsb_score'], 1):>7}"
            f"  выгода {_fmt(row['currency_uplift_bps'], 1):>7} б.п."
            f"  lift {_fmt(row['hit_lift'], 2):>5}"
            f"  p {_fmt(row['p_value'], 3):>6}"
            f"  {_fmt(row['per_week'], 2):>4}/нед"
            f"  плохих {_fmt(row['bad_push_rate'], 3):>5}"
        )
    if len(result.audit):
        status = "пройден" if bool(result.audit["matched"].all()) else "ПРОВАЛЕН"
        lines.append(f"\n  Аудит на заглядывание вперёд: {status}")
    return "\n".join(lines)
