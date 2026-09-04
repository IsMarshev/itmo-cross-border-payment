"""The rule indicators, in one place so both the benchmark and the model use them.

Each rule turns the feature frame into a single "how favourable is today" score
for one corridor. They are the brief's own indicator library — level, momentum,
seasonality, reversal — plus textbook mean reversion.

Two consumers share these definitions, and it matters that they share them
rather than each keeping a copy:

* ``benchmark.strategies`` scores each rule on its own, as a contender.
* ``utility_risk`` can use the rule scores as its *feature set*. With a signal
  of roughly 15 bps against a 300 bps standard deviation, fitting 22 correlated
  raw features mostly fits noise; six individually-informative indicators is a
  far better ratio, and it is what the brief actually asks for in its
  combination step — weight the indicators, don't re-derive them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Score for a day a rule refuses outright (e.g. the rate did not turn up).
# Finite, so quantile arithmetic downstream stays well defined; paired with a
# `minimum_score` floor so such a day can never be selected.
BLOCKED = -1e6

RULE_NAMES: tuple[str, ...] = (
    "percentile",
    "zscore",
    "momentum",
    "drawdown",
    "seasonal",
    "reversal",
)


def rule_score(name: str, features: pd.DataFrame) -> pd.Series:
    """One rule's score over the feature frame. Higher means more favourable."""
    if name == "percentile":
        return 1.0 - features["pct_rank_90"]
    if name == "zscore":
        return -features["ewma_zscore"]
    if name == "momentum":
        # Integer streaks tie constantly; the 5-day return breaks ties inside a
        # streak level without ever reordering across levels.
        return features["down_streak"] - features["ret_5"]
    if name == "drawdown":
        return features["drawdown_60"]
    if name == "seasonal":
        return -features["seasonal_zscore"]
    if name == "reversal":
        turned_up = features["ret_1"] > 0
        return pd.Series(
            np.where(turned_up, -features["dist_to_min_90"], BLOCKED),
            index=features.index,
        )
    raise KeyError(f"No rule expression for {name!r}")


def rule_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Every rule's score as columns, for use as a model feature set.

    ``reversal`` is split rather than passed through. Its contender score uses a
    large negative sentinel to veto days the rate did not turn up, which is right
    for a score but wrong for a feature: the sentinel would swamp every scale,
    and collapsing it onto the series minimum would read the whole series —
    including its future — to pick that floor. The two pieces of information it
    encodes are therefore carried as two leak-free columns instead: how close the
    rate sits to its 90-day minimum, and whether it turned up today.
    """
    columns: dict[str, pd.Series] = {}
    for name in RULE_NAMES:
        if name == "reversal":
            columns["rule_near_min"] = -features["dist_to_min_90"].astype(float)
            columns["rule_turned_up"] = (features["ret_1"] > 0).astype(float)
            continue
        columns[f"rule_{name}"] = rule_score(name, features).astype(float)
    return pd.DataFrame(columns, index=features.index)


RULE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rule_percentile",
    "rule_zscore",
    "rule_momentum",
    "rule_drawdown",
    "rule_seasonal",
    "rule_near_min",
    "rule_turned_up",
)
