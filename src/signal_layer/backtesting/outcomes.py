"""Future outcomes used only after a historical decision has been recorded."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_outcomes(
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
    epsilon_bps: float = 30.0,
) -> pd.DataFrame:
    """Build the Stage-4 advantage and early-send outcomes.

    The last ``horizon`` rows of every corridor remain incomplete. They are kept
    in the output so the decision log remains exhaustive, but they must never be
    included in the final metric denominator.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if epsilon_bps < 0:
        raise ValueError("epsilon_bps must be non-negative")

    parts: list[pd.DataFrame] = []
    for _, group in panel.sort_values(["iso", "quote_date"]).groupby("iso", sort=False):
        current = group.reset_index(drop=True).copy()
        if "available_on" not in current.columns:
            current["available_on"] = current["quote_date"]
        rates = current["rub_per_unit"].to_numpy(dtype=float)
        advantage = np.full(len(current), np.nan)
        future_median = np.full(len(current), np.nan)
        future_minimum = np.full(len(current), np.nan)
        regret_bps = np.full(len(current), np.nan)
        early_send: list[object] = [pd.NA] * len(current)
        outcome_available_on = pd.Series(pd.NaT, index=current.index, dtype="datetime64[ns]")

        for position in range(len(current) - horizon):
            rate = rates[position]
            future = rates[position + 1 : position + 1 + horizon]
            median = float(np.median(future))
            minimum = float(np.min(future))
            advantage[position] = (median - rate) / rate * 10_000.0
            future_median[position] = median
            future_minimum[position] = minimum
            regret_bps[position] = max(0.0, (rate - minimum) / rate * 10_000.0)
            early_send[position] = minimum < rate * (1.0 - epsilon_bps / 10_000.0)
            outcome_available_on.iloc[position] = current.loc[
                position + horizon, "available_on"
            ]

        current["future_median"] = future_median
        current["future_minimum"] = future_minimum
        current["advantage_bps"] = advantage
        current["early_send"] = pd.array(early_send, dtype="boolean")
        current["regret_bps"] = regret_bps
        current["outcome_available_on"] = outcome_available_on
        current["outcome_complete"] = current["advantage_bps"].notna()
        parts.append(
            current[
                [
                    "quote_date",
                    "iso",
                    "future_median",
                    "future_minimum",
                    "advantage_bps",
                    "early_send",
                    "regret_bps",
                    "outcome_available_on",
                    "outcome_complete",
                ]
            ]
        )

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["iso", "quote_date"])
