"""Matched random baseline, block bootstrap and report construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def matched_random_schedules(
    decision_log: pd.DataFrame,
    *,
    trials: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample the same number of dates in every corridor/communication window."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    complete = decision_log.loc[decision_log["outcome_complete"]].copy()
    selected_counts = (
        complete.loc[complete["decision"]]
        .groupby(["iso", "window_id"], sort=False)
        .size()
        .rename("count")
    )
    if selected_counts.empty:
        return pd.DataFrame(columns=[*decision_log.columns, "trial"])

    rng = np.random.default_rng(seed)
    candidates_by_window = {
        key: group.index.to_numpy()
        for key, group in complete.groupby(["iso", "window_id"], sort=False)
    }
    samples: list[pd.DataFrame] = []
    for trial in range(trials):
        trial_indices: list[int] = []
        for (iso, window_id), count in selected_counts.items():
            candidate_indices = candidates_by_window[(iso, window_id)]
            chosen_indices = rng.choice(candidate_indices, size=int(count), replace=False)
            trial_indices.extend(chosen_indices.tolist())
        chosen = complete.loc[trial_indices].copy()
        chosen["trial"] = trial
        samples.append(chosen)
    return pd.concat(samples, ignore_index=True)


def moving_block_mean_ci(
    values: np.ndarray,
    *,
    block_length: int,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI for a mean using contiguous moving blocks."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    length = min(max(1, block_length), len(clean))
    possible_starts = np.arange(len(clean) - length + 1)
    blocks_needed = int(np.ceil(len(clean) / length))
    rng = np.random.default_rng(seed)
    means = np.empty(trials)
    for trial in range(trials):
        starts = rng.choice(possible_starts, size=blocks_needed, replace=True)
        sample = np.concatenate([clean[start : start + length] for start in starts])[: len(clean)]
        means[trial] = sample.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _statistics(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {
            "mean_advantage_bps": float("nan"),
            "median_advantage_bps": float("nan"),
            "p10_advantage_bps": float("nan"),
            "hit_rate": float("nan"),
            "negative_share": float("nan"),
            "early_send_rate": float("nan"),
            "p90_regret_bps": float("nan"),
        }
    advantage = rows["advantage_bps"].to_numpy(dtype=float)
    return {
        "mean_advantage_bps": float(np.mean(advantage)),
        "median_advantage_bps": float(np.median(advantage)),
        "p10_advantage_bps": float(np.quantile(advantage, 0.1)),
        "hit_rate": float(np.mean(advantage > 0)),
        "negative_share": float(np.mean(advantage < 0)),
        "early_send_rate": float(rows["early_send"].astype(bool).mean()),
        "p90_regret_bps": float(np.quantile(rows["regret_bps"], 0.9)),
    }


def _random_statistics(random_log: pd.DataFrame, iso: str) -> dict[str, float]:
    sample = random_log if iso == "ALL" else random_log.loc[random_log["iso"].eq(iso)]
    if sample.empty:
        return {key: float("nan") for key in _statistics(sample)}
    by_trial = []
    for _, trial_rows in sample.groupby("trial"):
        by_trial.append(_statistics(trial_rows))
    return {
        key: float(np.mean([trial[key] for trial in by_trial]))
        for key in by_trial[0]
    }


def build_report(
    decision_log: pd.DataFrame,
    random_log: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_trials: int,
    seed: int,
) -> pd.DataFrame:
    """Summarise value, downside and communication behaviour per corridor."""
    complete = decision_log.loc[decision_log["outcome_complete"]]
    selected = complete.loc[complete["decision"]]
    currencies = [*sorted(decision_log["iso"].unique()), "ALL"]
    report_rows: list[dict[str, object]] = []
    for iso in currencies:
        evaluated = complete if iso == "ALL" else complete.loc[complete["iso"].eq(iso)]
        model_rows = selected if iso == "ALL" else selected.loc[selected["iso"].eq(iso)]
        model = _statistics(model_rows)
        random = _random_statistics(random_log, iso)
        ci_low, ci_high = moving_block_mean_ci(
            model_rows["advantage_bps"].to_numpy(dtype=float),
            block_length=block_length,
            trials=bootstrap_trials,
            seed=seed,
        )
        baseline_advantage = random["mean_advantage_bps"]
        baseline_hit = random["hit_rate"]
        advantage_lift = (
            model["mean_advantage_bps"] / baseline_advantage
            if np.isfinite(baseline_advantage) and baseline_advantage > 0
            else float("nan")
        )
        hit_rate_lift = (
            model["hit_rate"] / baseline_hit
            if np.isfinite(baseline_hit) and baseline_hit > 0
            else float("nan")
        )

        intervals = model_rows.sort_values(["iso", "observation_position"]).groupby("iso")[
            "observation_position"
        ].diff()
        series_share = (
            float((intervals <= 5).sum() / len(model_rows))
            if len(model_rows)
            else float("nan")
        )
        evaluation_weeks = len(evaluated) / 5.0
        report_rows.append(
            {
                "iso": iso,
                "n_signals": len(model_rows),
                "per_week": (
                    len(model_rows) / evaluation_weeks if evaluation_weeks else float("nan")
                ),
                "series_share": series_share,
                **model,
                "advantage_ci_low": ci_low,
                "advantage_ci_high": ci_high,
                "random_mean_advantage_bps": baseline_advantage,
                "random_hit_rate": baseline_hit,
                "advantage_delta_bps": model["mean_advantage_bps"] - baseline_advantage,
                "advantage_lift": advantage_lift,
                "hit_rate_lift": hit_rate_lift,
            }
        )
    return pd.DataFrame(report_rows)
