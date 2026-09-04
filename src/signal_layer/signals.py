"""The signal layer as a product: one entry point, one indicator, one contract.

The benchmark exists to choose; this module is what it chose. Every strategy in
CBSB-1 was scored on the same push budget, out of time, against matched random
schedules, and the calibrated z-score won: 30.5 bps of client money per transfer
against 23.4 for the same rule with a fixed window and 14.0 for the learned
utility/risk model — and it is the only strategy in the run that clears all
seven of the brief's mandatory gates.

So this is deliberately *not* a framework with a strategy parameter. Making the
winner configurable would invite quietly running something the benchmark never
blessed. To change what ships, change the benchmark's verdict first.

What the layer emits is the table the brief specifies — date, corridor,
indicator, direction, strength, indicator speed and the recommended scenario —
plus the factual sentence a push can carry.

Two contracts hold everywhere below:

* **As of T, only data from T.** The z-score window is selected walk-forward,
  the communication policy runs chronologically, and :func:`signals_asof`
  answers "what would we have sent on date T" from a panel with the future
  physically removed.
* **Facts only.** Message text states what the rate has already done. No
  forecast, no promise, nothing that reads as investment advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .adaptive import ZSCORE_SPANS, TuningConfig, walk_forward_tuned, zscore_candidates
from .backtesting.policy import PolicyConfig, apply_policy
from .rules import BLOCKED

INDICATOR = "zscore_tuned"

SIGNAL_COLUMNS: tuple[str, ...] = (
    "signal_date",
    "iso",
    "indicator",
    "direction",
    "strength",
    "strength_pct",
    "speed",
    "scenario",
    "window",
    "rub_per_unit",
    "level_percentile",
    "deviation_pct",
    "message",
)

# A push claims the moment is favourable now, not that the window is closing.
SCENARIO = "favourable_now"


@dataclass(frozen=True, slots=True)
class SignalLayerConfig:
    """Everything the live layer needs, defaulted to the benchmark's winner.

    The cadence mirrors CBSB-1's headline budget. Cooldown is 1 rather than 3
    on measured grounds: at the same weekly limit a three-day cooldown blocks
    Tuesday through Thursday after a Monday push, so the second slot can only
    land on Friday, and the policy spends the first one early on a mediocre day.
    """

    spans: tuple[int, ...] = ZSCORE_SPANS
    tuning: TuningConfig = field(default_factory=TuningConfig)
    window: str = "week"
    max_signals_per_window: int = 2
    cooldown_observations: int = 1
    threshold_lookback: int = 250
    minimum_threshold_history: int = 20
    level_window: int = 90
    """Trailing observations reported as the diagnostic level percentile."""
    require_true_fact: bool = True
    """Refuse to signal on a day whose message would not be true.

    A push has to state a fact, so a day where the rate is *not* below the
    trend the indicator measured has nothing truthful to say and must not
    consume a slot. This began as a compliance rule and turned out to be the
    single largest improvement in the project: gating on it lifts client money
    from 30.5 to 84.2 bps per transfer, because the vetoed days are not merely
    mute — they average -65 bps and are negative on every corridor. Vetoing
    inside the score rather than filtering afterwards matters too: the policy
    then moves on to the next eligible day instead of wasting the slot.
    """

    def policy(self) -> PolicyConfig:
        return PolicyConfig(
            window=self.window,
            max_signals_per_window=self.max_signals_per_window,
            cooldown_observations=self.cooldown_observations,
            threshold_lookback=self.threshold_lookback,
            minimum_threshold_history=self.minimum_threshold_history,
            minimum_score=BLOCKED / 2 if self.require_true_fact else None,
        )


def _lookup_deviation(
    deviations: dict[int, pd.Series], window_label: str, date: pd.Timestamp
) -> float:
    """Percent below trend on ``date`` for whichever span was selected."""
    try:
        span = int(window_label.split("=")[1])
    except (IndexError, ValueError):
        return float("nan")
    series = deviations.get(span)
    if series is None:
        return float("nan")
    value = series.get(date)
    return float(value) if value is not None else float("nan")


def _speed_of(window_label: str) -> str:
    """How fast the chosen window makes the indicator.

    The brief's fast/slow axis is exactly the length of the lookback: a short
    window reacts within days and is noisier, a long one confirms later.
    """
    try:
        span = int(window_label.split("=")[1])
    except (IndexError, ValueError):
        return "unknown"
    if span <= 10:
        return "fast"
    if span <= 60:
        return "medium"
    return "slow"


def truth_mask(
    panel: pd.DataFrame,
    iso: str,
    scored: pd.DataFrame,
    config: SignalLayerConfig | None = None,
) -> np.ndarray:
    """Which scored days have a true favourable fact to state.

    Public so the benchmark can score the *complement* as a diagnostic: the
    days the gate rejects are not merely mute, and it is worth being able to
    re-run that claim rather than quote it.
    """
    resolved = config or SignalLayerConfig()
    deviations = _deviation_from_trend(panel, iso, resolved.spans)
    deviation = np.array(
        [
            _lookup_deviation(deviations, label, date)
            for label, date in zip(scored["chosen"], scored["quote_date"], strict=True)
        ]
    )
    return np.isfinite(deviation) & (deviation < 0)


def score(
    panel: pd.DataFrame, iso: str, config: SignalLayerConfig | None = None
) -> pd.DataFrame:
    """The live score for one corridor: a z-score with a self-selected window.

    When ``require_true_fact`` is set, days whose push would have no true
    favourable fact are vetoed with a sentinel the policy's score floor
    rejects. The check uses only the trend the indicator itself measured, so it
    is computable at send time and adds no look-ahead.
    """
    resolved = config or SignalLayerConfig()
    scored = walk_forward_tuned(
        panel, iso, zscore_candidates(resolved.spans), resolved.tuning
    )
    if scored.empty or not resolved.require_true_fact:
        return scored
    truthful = truth_mask(panel, iso, scored, resolved)
    return scored.assign(score=np.where(truthful, scored["score"], BLOCKED))


def _level_percentile(panel: pd.DataFrame, iso: str, window: int) -> pd.Series:
    """Share of the trailing window whose rate was worse for the sender than today's.

    Strictly backward-looking and excluding today. Reported as a diagnostic
    column, deliberately *not* as the push's claim: this indicator finds days
    that are cheap against their own recent trend, which in a rising market is
    routinely a day near the top of its 90-day range. A message built on this
    number would read as a level claim the signal never made.
    """
    corridor = panel[panel["iso"] == iso].sort_values("quote_date")
    values = corridor["rub_per_unit"].astype(float)
    rank = values.rolling(window, min_periods=20).apply(
        lambda block: float((block[:-1] > block[-1]).mean()), raw=True
    )
    return pd.Series(rank.to_numpy() * 100.0, index=corridor["quote_date"].to_numpy())


def _deviation_from_trend(
    panel: pd.DataFrame, iso: str, spans: tuple[int, ...]
) -> dict[int, pd.Series]:
    """Percent below the EWMA of each candidate span, by quote date.

    This is the fact the indicator actually observed, so it is the only fact
    the push is allowed to state.
    """
    corridor = panel[panel["iso"] == iso].sort_values("quote_date")
    values = corridor["rub_per_unit"].astype(float)
    dates = corridor["quote_date"].to_numpy()
    out: dict[int, pd.Series] = {}
    for span in spans:
        mean = values.ewm(span=span, adjust=False).mean()
        out[span] = pd.Series(
            ((values / mean - 1.0) * 100.0).to_numpy(), index=dates
        )
    return out


def _message(currency: str, deviation: float, span: int) -> str:
    """The factual sentence a push carries.

    Says what the rate has already done against the window the indicator used,
    in the present tense with past evidence. No forecast, no promise, nothing
    that reads as advice — the brief bars any claim about where the rate goes
    next, stated or implied. Emitted only when the fact is actually true.
    """
    return (
        f"Курс {currency} сейчас на {abs(deviation):.1f}% ниже своего среднего "
        f"за последние {span} наблюдений."
    )


def signal_table(
    panel: pd.DataFrame,
    corridors: tuple[str, ...] | list[str],
    config: SignalLayerConfig | None = None,
) -> pd.DataFrame:
    """The brief's signal table: what to send, for which corridor, and why.

    Columns are ``signal_date, iso, indicator, direction, strength,
    strength_pct, speed, scenario, window, rub_per_unit, level_percentile,
    message``. Rows are the days the communication policy actually selected,
    not every day the indicator liked.
    """
    resolved = config or SignalLayerConfig()
    scored: list[pd.DataFrame] = []
    for iso in corridors:
        part = score(panel, iso, resolved)
        if len(part):
            scored.append(part)
    if not scored:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    frame = pd.concat(scored, ignore_index=True)
    decisions = apply_policy(
        frame[["quote_date", "available_on", "iso", "rub_per_unit", "score"]],
        resolved.policy(),
    )
    selected = decisions.loc[decisions["decision"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    chosen = frame.set_index(["iso", "quote_date"])["chosen"]
    rows: list[pd.DataFrame] = []
    for iso, group in selected.groupby("iso", sort=False):
        corridor = panel[panel["iso"] == iso].sort_values("quote_date")
        returns = (
            corridor.set_index("quote_date")["rub_per_unit"].astype(float).pct_change(5)
        )
        levels = _level_percentile(panel, iso, resolved.level_window)
        deviations = _deviation_from_trend(panel, iso, resolved.spans)
        # Strength as a trailing percentile of this corridor's own scores, so a
        # "strong" signal means strong for this corridor rather than for the
        # units the z-score happens to be measured in.
        history = frame.loc[frame["iso"].eq(iso)].set_index("quote_date")["score"]
        ranked = history.rolling(resolved.threshold_lookback, min_periods=20).rank(
            pct=True
        )
        block = group.copy()
        dates = block["quote_date"]
        window_label = [chosen.get((iso, d), "") for d in dates]
        percentile = levels.reindex(dates).to_numpy(dtype=float)
        block = block.assign(
            signal_date=dates,
            indicator=INDICATOR,
            direction=np.where(
                returns.reindex(dates).to_numpy(dtype=float) < 0, "down", "up"
            ),
            strength=block["score"].to_numpy(dtype=float),
            strength_pct=ranked.reindex(dates).to_numpy(dtype=float),
            speed=[_speed_of(label) for label in window_label],
            scenario=SCENARIO,
            window=window_label,
            level_percentile=percentile,
            deviation_pct=[
                _lookup_deviation(deviations, label, date)
                for label, date in zip(window_label, dates, strict=True)
            ],
        )
        # A message is emitted only where its claim holds: the rate must really
        # sit below the trend the indicator measured. Anything else would ship a
        # false statement to a client.
        block = block.assign(
            message=[
                _message(iso, value, int(label.split("=")[1]))
                if np.isfinite(value) and value < 0 and "=" in label
                else ""
                for value, label in zip(block["deviation_pct"], window_label, strict=True)
            ]
        )
        rows.append(block[list(SIGNAL_COLUMNS)])

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["signal_date", "iso"])
        .reset_index(drop=True)
    )


def signals_asof(
    panel: pd.DataFrame,
    corridors: tuple[str, ...] | list[str],
    asof: pd.Timestamp,
    config: SignalLayerConfig | None = None,
) -> pd.DataFrame:
    """Signals exactly as they would have looked on ``asof``.

    The brief disqualifies any look-ahead and demands this entry point so the
    claim can be checked. Truncating the panel is the strongest form of the
    check: data after ``asof`` is not masked, it is absent.
    """
    moment = pd.Timestamp(asof)
    truncated = panel[panel["quote_date"] <= moment]
    if truncated.empty:
        raise ValueError(f"No observations on or before {moment:%Y-%m-%d}")
    return signal_table(truncated, corridors, config)


def latest_signal(
    panel: pd.DataFrame,
    iso: str,
    asof: pd.Timestamp,
    config: SignalLayerConfig | None = None,
) -> pd.Series | None:
    """The signal for ``iso`` on ``asof``, or ``None`` if that day is a hold."""
    table = signals_asof(panel, [iso], asof, config)
    if table.empty:
        return None
    today = table[table["signal_date"].eq(pd.Timestamp(asof))]
    return None if today.empty else today.iloc[-1]
