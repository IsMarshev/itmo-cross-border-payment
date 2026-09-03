"""Sequential communication policy with an auditable decision log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Window = Literal["week", "month"]


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Parameters known before a backtest starts."""

    window: Window = "week"
    max_signals_per_window: int = 2
    expected_observations_per_window: int | None = None
    cooldown_observations: int = 3
    threshold_lookback: int = 250
    minimum_threshold_history: int = 20
    minimum_score: float | None = None

    def __post_init__(self) -> None:
        if self.window not in {"week", "month"}:
            raise ValueError("window must be 'week' or 'month'")
        if self.max_signals_per_window <= 0:
            raise ValueError("max_signals_per_window must be positive")
        if self.cooldown_observations < 0:
            raise ValueError("cooldown_observations must be non-negative")
        if (
            self.expected_observations_per_window is not None
            and self.expected_observations_per_window <= 0
        ):
            raise ValueError("expected_observations_per_window must be positive")
        if self.threshold_lookback <= 0 or self.minimum_threshold_history <= 0:
            raise ValueError("threshold history lengths must be positive")

    @property
    def expected_observations(self) -> int:
        if self.expected_observations_per_window is not None:
            return self.expected_observations_per_window
        return 5 if self.window == "week" else 21


def _window_id(value: pd.Timestamp, window: Window) -> str:
    frequency = "W-SUN" if window == "week" else "M"
    return str(value.to_period(frequency))


def apply_policy(scores: pd.DataFrame, config: PolicyConfig) -> pd.DataFrame:
    """Evaluate scores strictly in chronological order.

    The ``1 - slots_remaining / expected_days_remaining`` quantile is a pacing
    heuristic, not an optimal-stopping theorem. A score floor prevents unused
    slots from forcing a weak signal near the end of a window.
    """
    required = {"quote_date", "iso", "rub_per_unit", "score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {', '.join(sorted(missing))}")
    if scores.empty:
        raise ValueError("No score observations are available for the backtest")

    decisions: list[dict[str, object]] = []
    for _, corridor_scores in scores.sort_values(["iso", "quote_date"]).groupby(
        "iso", sort=False
    ):
        history: list[float] = []
        used_by_window: dict[str, int] = {}
        seen_by_window: dict[str, int] = {}
        last_signal_position: int | None = None

        for position, (_, row) in enumerate(corridor_scores.reset_index(drop=True).iterrows()):
            decision_date = pd.Timestamp(row.get("available_on", row["quote_date"]))
            window_id = _window_id(decision_date, config.window)
            seen_by_window[window_id] = seen_by_window.get(window_id, 0) + 1
            slots_before = config.max_signals_per_window - used_by_window.get(window_id, 0)
            expected_remaining = max(
                1, config.expected_observations - seen_by_window[window_id] + 1
            )
            quantile = float(np.clip(1.0 - slots_before / expected_remaining, 0.0, 1.0))
            trailing_history = history[-config.threshold_lookback :]
            threshold = float("nan")
            if len(trailing_history) >= config.minimum_threshold_history:
                threshold = float(np.quantile(trailing_history, quantile))
                if config.minimum_score is not None:
                    threshold = max(threshold, config.minimum_score)

            since_last = (
                position - last_signal_position if last_signal_position is not None else None
            )
            cooldown_remaining = (
                max(0, config.cooldown_observations - since_last + 1)
                if since_last is not None
                else 0
            )
            decision = False
            if len(trailing_history) < config.minimum_threshold_history:
                reason = "threshold_warmup"
            elif slots_before <= 0:
                reason = "window_budget_exhausted"
            elif cooldown_remaining > 0:
                reason = "cooldown"
            elif float(row["score"]) < threshold:
                reason = "below_threshold"
            else:
                decision = True
                reason = "selected"
                used_by_window[window_id] = used_by_window.get(window_id, 0) + 1
                last_signal_position = position

            slots_after = config.max_signals_per_window - used_by_window.get(window_id, 0)
            record = row.to_dict()
            record.update(
                {
                    "decision_date": decision_date,
                    "window_id": window_id,
                    "observation_position": position,
                    "score_history_count": len(trailing_history),
                    "pacing_quantile": quantile,
                    "threshold": threshold,
                    "slots_before": slots_before,
                    "slots_after": slots_after,
                    "cooldown_remaining": cooldown_remaining,
                    "decision": decision,
                    "decision_reason": reason,
                }
            )
            decisions.append(record)
            history.append(float(row["score"]))

    return pd.DataFrame(decisions).sort_values(["iso", "quote_date"]).reset_index(drop=True)
