"""Execution of CBSB-1.

The shape of a run, in order:

1. Load the panel, compute features once, compute outcome labels once.
2. Every strategy scores every trading day of every corridor.
3. Every strategy's scores go through the **same** communication policy, run
   chronologically over the whole history exactly as it would run live.
4. The resulting signals are sliced into out-of-time folds and joined to labels.
5. For every (strategy, corridor) a matched-random null is sampled: the same
   number of days, in the same folds, drawn uniformly without replacement. That
   null is the benchmark's reference, and it supplies the p-value.
6. Gates and the composite score turn the table into a verdict.

The rule that keeps this honest: steps 2 and 3 never see a label, and step 5
compares against the *same* push budget rather than against zero.

Aggregation convention: a fold contributes to a corridor in proportion to the
number of signals the strategy actually sent in it, and a corridor contributes
to a strategy the same way. Folds a strategy sat out therefore neither help nor
hurt it, and the per-corridor spread is reported next to the pooled figure so a
single lucky corridor is visible rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtesting.policy import PolicyConfig, apply_policy
from ..features import compute_features
from ..labels import build_labels, hit_column
from ..utility_risk import UtilityRiskConfig, scores_asof, walk_forward_scores
from . import stats
from .spec import BenchmarkSpec, Cadence
from .strategies import (
    DEFAULT_STRATEGY_NAMES,
    ModelScoreCache,
    Strategy,
    build_scores,
    get_strategy,
)

_NULL_METRICS = (
    "hit_rate",
    "hit_favourable",
    "hit_closing",
    "currency_uplift_bps",
    "window_advantage_bps",
    "bad_push_rate",
)
_WEIGHTED_METRICS = (
    "ceiling_bps",
    "hit_rate",
    "hit_favourable",
    "hit_closing",
    "currency_uplift_bps",
    "window_advantage_bps",
    "bad_push_rate",
    "regret_bps",
)


@dataclass(slots=True)
class BenchmarkResult:
    """Everything one run produces."""

    signals: pd.DataFrame
    per_fold: pd.DataFrame
    per_corridor: pd.DataFrame
    leaderboard: pd.DataFrame
    gates: pd.DataFrame
    horizon_table: pd.DataFrame
    lambda_sweep: pd.DataFrame
    cadence_sweep: pd.DataFrame
    null_distribution: pd.DataFrame
    audit: pd.DataFrame
    coefficients: pd.DataFrame


# --- helpers -----------------------------------------------------------------


def _policy_for(
    strategy: Strategy, spec: BenchmarkSpec, cadence: Cadence | None = None
) -> PolicyConfig:
    point = cadence or spec.cadence
    return PolicyConfig(
        window=point.window,
        max_signals_per_window=point.max_per_window,
        cooldown_observations=point.cooldown,
        threshold_lookback=spec.threshold_lookback,
        minimum_threshold_history=spec.minimum_threshold_history,
        minimum_score=strategy.minimum_score,
    )


def _emit_signals(
    strategy: Strategy,
    spec: BenchmarkSpec,
    scores: pd.DataFrame,
    cadence: Cadence | None = None,
) -> pd.DataFrame:
    """Run the shared communication policy and keep the days it selected.

    A ``weekly_best`` strategy bypasses the policy and takes the highest scoring
    days of each week with hindsight over the week. Those rows are diagnostics,
    not contenders: comparing one against its ``policy`` twin separates "the
    score is bad" from "the policy throws the score away".
    """
    if scores.empty:
        return pd.DataFrame(columns=["iso", "quote_date", "score", "threshold"])
    if strategy.selection == "weekly_best":
        ranked = scores.copy()
        point = cadence or spec.cadence
        period = "W-SUN" if point.window == "week" else "M"
        ranked["window_id"] = ranked["quote_date"].dt.to_period(period)
        best = (
            ranked.sort_values("score", ascending=False)
            .groupby(["iso", "window_id"], sort=False)
            .head(point.max_per_window)
        )
        best = best.assign(threshold=float("nan"))
        return best[["iso", "quote_date", "score", "threshold"]].reset_index(drop=True)
    decisions = apply_policy(scores, _policy_for(strategy, spec, cadence))
    selected = decisions.loc[decisions["decision"], ["iso", "quote_date", "score", "threshold"]]
    return selected.reset_index(drop=True)


def _cadence(signals: pd.DataFrame) -> dict[str, float]:
    """How evenly a corridor's pushes are spread over the evaluation period.

    ``interval_cv`` is the coefficient of variation of the gaps between pushes:
    a memoryless (Poisson) stream sits near 1.0, a metronome at 0.0, and a
    strategy that empties its budget into two weeks and then goes quiet climbs
    well above 1.0. ``max_gap_days`` catches the quarter of silence directly.
    """
    if len(signals) < 2:
        return {"interval_cv": float("nan"), "max_gap_days": float("nan"),
                "series_share": float("nan")}
    gaps = pd.to_datetime(signals["quote_date"]).sort_values().diff().dt.days.dropna()
    values = gaps.to_numpy(dtype=float)
    mean = float(values.mean())
    return {
        "interval_cv": float(values.std(ddof=1) / mean) if mean > 0 else float("nan"),
        "max_gap_days": float(values.max()),
        "series_share": float((values <= 5).mean()),
    }


def _cadence_by_corridor(signals: pd.DataFrame) -> dict[str, float]:
    """Cadence metrics computed per corridor, then weighted by signal count.

    Gaps are only meaningful inside one corridor: a client on the TJS corridor
    never sees the KGS pushes, so pooling all five before differencing would
    interleave unrelated streams and report a burstiness nobody experiences.
    """
    per_iso, weights = [], []
    for _, group in signals.groupby("iso", sort=False):
        stats_for_iso = _cadence(group)
        if np.isfinite(stats_for_iso["interval_cv"]):
            per_iso.append(stats_for_iso)
            weights.append(float(len(group)))
    if not per_iso:
        return {"interval_cv": float("nan"), "max_gap_days": float("nan"),
                "series_share": float("nan")}
    weight_array = np.asarray(weights)
    out = {
        key: _weighted(np.asarray([entry[key] for entry in per_iso]), weight_array)
        for key in ("interval_cv", "series_share")
    }
    out["max_gap_days"] = float(max(entry["max_gap_days"] for entry in per_iso))
    return out


def _fold_statistics(
    signal_labels: pd.DataFrame, universe: pd.DataFrame, hit_col: str
) -> dict[str, float]:
    """Metrics for one (strategy, corridor, fold) cell."""
    if signal_labels.empty or universe.empty:
        return {"n_signals": 0, **{key: float("nan") for key in _WEIGHTED_METRICS}}
    # The ceiling is matched to this cell's own budget, the same way the random
    # null is. Value per push rises steeply as pushes get rarer, so a ceiling
    # fixed at some other frequency would flatter a selective strategy and
    # penalise a talkative one for a difference that is purely cadence.
    best = np.sort(universe["currency_gain_bps"].to_numpy(dtype=float))[::-1]
    best = best[np.isfinite(best)][: len(signal_labels)]
    return {
        "n_signals": len(signal_labels),
        "ceiling_bps": float(best.mean()) if len(best) else float("nan"),
        "hit_rate": float(signal_labels[hit_col].mean()),
        "hit_favourable": float(signal_labels["held_favourable"].mean()),
        "hit_closing": float(signal_labels["held_window_closing"].mean()),
        "currency_uplift_bps": float(np.nanmean(signal_labels["currency_gain_bps"])),
        "window_advantage_bps": float(np.nanmean(signal_labels["window_advantage_bps"])),
        "regret_bps": float(np.nanmean(signal_labels["regret_bps"])),
        "bad_push_rate": float(signal_labels["bad_push"].mean()),
    }


def _random_fold_trials(
    universe: pd.DataFrame,
    count: int,
    hit_col: str,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Matched-random null for one fold: ``trials`` schedules of ``count`` days."""
    m = len(universe)
    if count <= 0 or m == 0:
        return {key: np.full(trials, np.nan) for key in _NULL_METRICS}
    take = min(count, m)
    # Sampling without replacement inside a trial makes the null a random
    # *schedule* spending the same budget, not a draw with repeats.
    picks = rng.random((trials, m)).argsort(axis=1)[:, :take]

    hit = universe[hit_col].to_numpy(dtype=float)
    favourable = universe["held_favourable"].to_numpy(dtype=float)
    closing = universe["held_window_closing"].to_numpy(dtype=float)
    gain = universe["currency_gain_bps"].to_numpy(dtype=float)
    advantage = universe["window_advantage_bps"].to_numpy(dtype=float)
    bad = universe["bad_push"].to_numpy(dtype=float)
    return {
        "hit_rate": hit[picks].mean(axis=1),
        "hit_favourable": favourable[picks].mean(axis=1),
        "hit_closing": closing[picks].mean(axis=1),
        "currency_uplift_bps": np.nanmean(gain[picks], axis=1),
        "window_advantage_bps": np.nanmean(advantage[picks], axis=1),
        "bad_push_rate": bad[picks].mean(axis=1),
    }


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    """Signal-count weighted mean, ignoring cells the strategy sat out."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not usable.any():
        return float("nan")
    return float(np.sum(values[usable] * weights[usable]) / np.sum(weights[usable]))


def _null_comparison(observed: dict[str, object], null: dict[str, np.ndarray]) -> dict[str, float]:
    """Lift, delta and permutation p-value of a cell against its matched null."""
    def mean_of(key: str) -> float:
        values = null.get(key)
        if values is None or not np.isfinite(values).any():
            return float("nan")
        return float(np.nanmean(values))

    random_hit = mean_of("hit_rate")
    random_uplift = mean_of("currency_uplift_bps")
    out: dict[str, float] = {
        "random_hit_rate": random_hit,
        "random_currency_uplift_bps": random_uplift,
        "random_window_advantage_bps": mean_of("window_advantage_bps"),
        "random_bad_push_rate": mean_of("bad_push_rate"),
    }

    hit = float(observed.get("hit_rate", float("nan")))
    uplift = float(observed.get("currency_uplift_bps", float("nan")))
    bad = float(observed.get("bad_push_rate", float("nan")))
    out["hit_lift"] = (
        hit / random_hit if np.isfinite(random_hit) and random_hit > 0 else float("nan")
    )
    for scenario in ("favourable", "closing"):
        reference = mean_of(f"hit_{scenario}")
        observed_hit = float(observed.get(f"hit_{scenario}", float("nan")))
        out[f"random_hit_{scenario}"] = reference
        out[f"hit_lift_{scenario}"] = (
            observed_hit / reference
            if np.isfinite(reference) and reference > 0
            else float("nan")
        )
    out["currency_uplift_delta_bps"] = uplift - random_uplift
    out["bad_push_delta"] = bad - out["random_bad_push_rate"]
    out["p_value"] = stats.permutation_pvalue(
        uplift, null.get("currency_uplift_bps", np.array([])), alternative="greater"
    )
    out["p_value_hit"] = stats.permutation_pvalue(
        hit, null.get("hit_rate", np.array([])), alternative="greater"
    )

    uplift_null = null.get("currency_uplift_bps")
    if uplift_null is not None and np.isfinite(uplift_null).any():
        low, high = np.nanquantile(uplift_null, [0.025, 0.975])
        out["random_uplift_q025"] = float(low)
        out["random_uplift_q975"] = float(high)
        # How far the strategy clears the luckiest 2.5% of random schedules.
        out["uplift_over_null_q975_bps"] = uplift - float(high)
    else:
        out["random_uplift_q025"] = float("nan")
        out["random_uplift_q975"] = float("nan")
        out["uplift_over_null_q975_bps"] = float("nan")
    return out


def _pool_nulls(
    name: str,
    isos: list[str],
    weights: np.ndarray,
    null_by_cell: dict[tuple[str, str], dict[str, np.ndarray]],
    trials: int,
) -> dict[str, np.ndarray]:
    """Combine per-corridor nulls with the same weights the observed metrics use."""
    pooled: dict[str, np.ndarray] = {}
    for key in _NULL_METRICS:
        stack: list[np.ndarray] = []
        used: list[float] = []
        for iso, weight in zip(isos, weights, strict=True):
            null = null_by_cell.get((name, iso))
            if null is None or weight <= 0 or not np.isfinite(null[key]).any():
                continue
            stack.append(np.nan_to_num(null[key]) * weight)
            used.append(float(weight))
        pooled[key] = (
            np.sum(stack, axis=0) / np.sum(used) if stack else np.full(trials, np.nan)
        )
    return pooled


# --- the run -----------------------------------------------------------------


def run_benchmark(
    panel: pd.DataFrame,
    spec: BenchmarkSpec | None = None,
    strategy_names: tuple[str, ...] = DEFAULT_STRATEGY_NAMES,
    *,
    model_config: UtilityRiskConfig | None = None,
    lambda_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
    run_audit: bool = True,
) -> BenchmarkResult:
    """Score every strategy against CBSB-1 and return the full evidence set."""
    resolved = spec or BenchmarkSpec()
    config = model_config or UtilityRiskConfig(
        horizon=resolved.horizon,
        execution_offset=resolved.execution_offset,
        bad_push_bps=resolved.bad_push_bps,
        local_min_tolerance_bps=resolved.local_min_tolerance_bps,
    )

    features = compute_features(panel)
    labels = build_labels(
        panel,
        horizon=resolved.horizon,
        execution_offset=resolved.execution_offset,
        epsilon_bps=resolved.epsilon_bps,
        bad_push_bps=resolved.bad_push_bps,
        local_min_tolerance_bps=resolved.local_min_tolerance_bps,
    )
    complete = labels[labels["outcome_complete"] & labels["iso"].isin(resolved.corridors)]
    folds = resolved.folds(pd.Timestamp(complete["quote_date"].max()))
    eval_start, eval_end = folds[0][1], folds[-1][2]
    cache = ModelScoreCache(config)

    signal_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    corridor_rows: list[dict[str, object]] = []
    null_by_cell: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for name in strategy_names:
        strategy = get_strategy(name)
        scores = build_scores(strategy, resolved, panel, features, labels, cache)
        emitted = _emit_signals(strategy, resolved, scores)
        joined = emitted.merge(complete, on=["iso", "quote_date"], how="inner")
        joined = joined[joined["quote_date"].between(eval_start, eval_end)].reset_index(drop=True)
        joined["strategy"] = name
        signal_rows.append(joined)

        hit_col = hit_column(strategy.scenario)
        rng = np.random.default_rng(resolved.seed)

        for iso in resolved.corridors:
            corridor_signals = joined[joined["iso"].eq(iso)]
            fold_values: dict[str, list[float]] = {key: [] for key in _WEIGHTED_METRICS}
            fold_weights: list[float] = []
            accumulator = {key: np.zeros(resolved.random_trials) for key in _NULL_METRICS}
            null_weight = 0.0

            for fold_id, start, end in folds:
                universe = complete[
                    complete["iso"].eq(iso) & complete["quote_date"].between(start, end)
                ]
                in_fold = corridor_signals[
                    corridor_signals["quote_date"].between(start, end)
                ]
                cell = _fold_statistics(in_fold, universe, hit_col)
                trading_days = len(universe)
                cell.update(
                    {
                        "strategy": name,
                        "iso": iso,
                        "fold": fold_id,
                        "fold_start": start,
                        "fold_end": end,
                        "trading_days": trading_days,
                        "per_week": (
                            cell["n_signals"] / (trading_days / 5.0)
                            if trading_days
                            else float("nan")
                        ),
                    }
                )
                fold_rows.append(cell)

                weight = float(cell["n_signals"])
                fold_weights.append(weight)
                for key in _WEIGHTED_METRICS:
                    fold_values[key].append(float(cell.get(key, float("nan"))))
                if weight > 0:
                    trials = _random_fold_trials(
                        universe, int(weight), hit_col, resolved.random_trials, rng
                    )
                    for key, values in trials.items():
                        accumulator[key] += np.nan_to_num(values) * weight
                    null_weight += weight

            weights = np.asarray(fold_weights, dtype=float)
            summary: dict[str, object] = {"strategy": name, "iso": iso}
            for key in _WEIGHTED_METRICS:
                summary[key] = _weighted(np.asarray(fold_values[key]), weights)
            summary["n_signals"] = int(weights.sum())

            evaluated_days = len(
                complete[
                    complete["iso"].eq(iso)
                    & complete["quote_date"].between(eval_start, eval_end)
                ]
            )
            summary["per_week"] = (
                summary["n_signals"] / (evaluated_days / 5.0) if evaluated_days else float("nan")
            )
            summary.update(_cadence(corridor_signals))

            null = (
                {key: values / null_weight for key, values in accumulator.items()}
                if null_weight > 0
                else {key: np.full(resolved.random_trials, np.nan) for key in _NULL_METRICS}
            )
            null_by_cell[(name, iso)] = null
            summary.update(_null_comparison(summary, null))
            summary.update(_advantage_significance(corridor_signals, resolved))
            corridor_rows.append(summary)

    signals = pd.concat(signal_rows, ignore_index=True) if signal_rows else pd.DataFrame()
    per_fold = pd.DataFrame(fold_rows)
    per_corridor = _apply_fdr(pd.DataFrame(corridor_rows), resolved.fdr_alpha)
    leaderboard, null_distribution = _build_leaderboard(
        per_corridor, signals, null_by_cell, resolved
    )
    gates = _evaluate_gates(leaderboard, resolved)
    horizon_table = _horizon_table(panel, signals, resolved)
    lambda_sweep = _lambda_sweep(
        panel, features, labels, complete, (eval_start, eval_end), cache, resolved, lambda_grid
    )
    sweep_strategies = tuple(
        name for name in ("oracle", "percentile", "utility_risk") if name in strategy_names
    )
    cadence_sweep = (
        _cadence_sweep(
            panel, features, labels, complete, (eval_start, eval_end),
            cache, resolved, sweep_strategies,
        )
        if sweep_strategies
        else pd.DataFrame()
    )
    audit = _audit_no_lookahead(panel, resolved, config) if run_audit else pd.DataFrame()
    return BenchmarkResult(
        signals=signals,
        per_fold=per_fold,
        per_corridor=per_corridor,
        leaderboard=leaderboard,
        gates=gates,
        horizon_table=horizon_table,
        lambda_sweep=lambda_sweep,
        cadence_sweep=cadence_sweep,
        null_distribution=null_distribution,
        audit=audit,
        coefficients=cache.coefficients,
    )


def _advantage_significance(signals: pd.DataFrame, spec: BenchmarkSpec) -> dict[str, float]:
    """Block-bootstrap CI and HAC t-statistic for the per-signal outcomes.

    Applied to both the brief's "выгода момента" (rates) and the client-money
    version (currency per rouble). Blocks are ``horizon`` long so overlapping
    outcome windows cannot masquerade as independent evidence.
    """
    out: dict[str, float] = {}
    columns = {
        "window_advantage": "window_advantage_bps",
        "currency_uplift": "currency_gain_bps",
    }
    for prefix, column in columns.items():
        values = (
            signals[column].to_numpy(dtype=float)
            if len(signals) and column in signals
            else np.array([])
        )
        low, high = stats.moving_block_bootstrap_ci(
            values,
            block_length=spec.horizon,
            trials=spec.bootstrap_trials,
            seed=spec.seed,
        )
        _, stderr, tstat = stats.newey_west_tstat(values)
        out[f"{prefix}_ci_low"] = low
        out[f"{prefix}_ci_high"] = high
        out[f"{prefix}_hac_se"] = stderr
        out[f"{prefix}_hac_t"] = tstat
    return out


def _apply_fdr(per_corridor: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Benjamini-Hochberg across the corridors of each strategy.

    The family is one strategy's corridors, not every cell in the run: whether
    this strategy holds up on TJS should not depend on how many other
    strategies happened to be scored alongside it. Note the correction is
    conservative here for a second reason — these corridors are close to the
    same series, since most of the move is the rouble rather than the
    recipient currency, so the five tests are far from independent.
    """
    if per_corridor.empty or "p_value" not in per_corridor:
        return per_corridor
    out = per_corridor.copy()
    out["q_value"] = float("nan")
    out["significant_fdr"] = False
    for name, group in out.groupby("strategy", sort=False):
        rejected, qvalues = stats.benjamini_hochberg(
            group["p_value"].to_numpy(dtype=float), alpha=alpha
        )
        out.loc[group.index, "q_value"] = qvalues
        out.loc[group.index, "significant_fdr"] = rejected
    return out


def _build_leaderboard(
    per_corridor: pd.DataFrame,
    signals: pd.DataFrame,
    null_by_cell: dict[tuple[str, str], dict[str, np.ndarray]],
    spec: BenchmarkSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per strategy: pooled metrics, the null test, and the CBSB score.

    Also returns the pooled null itself, one row per random schedule. Reporting
    a p-value without the distribution it came from asks the reader to take the
    test on faith; with the draws in hand they can see how far outside the cloud
    of random schedules a strategy actually lands.
    """
    if per_corridor.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    null_rows: list[pd.DataFrame] = []
    for name, group in per_corridor.groupby("strategy", sort=False):
        weights = group["n_signals"].to_numpy(dtype=float)
        row: dict[str, object] = {
            "strategy": name,
            "description": get_strategy(name).description,
            "scenario": get_strategy(name).scenario,
            "n_signals": int(weights.sum()),
            "n_corridors": int(len(group)),
        }
        for key in _WEIGHTED_METRICS:
            values = group[key].to_numpy(dtype=float)
            row[key] = _weighted(values, weights)
            row[f"{key}_worst_corridor"] = float(np.nanmin(values)) if len(values) else float("nan")
        row["per_week"] = float(np.nanmean(group["per_week"].to_numpy(dtype=float)))
        row["interval_cv"] = _weighted(group["interval_cv"].to_numpy(dtype=float), weights)
        row["series_share"] = _weighted(group["series_share"].to_numpy(dtype=float), weights)
        gaps = group["max_gap_days"].to_numpy(dtype=float)
        row["max_gap_days"] = float(np.nanmax(gaps)) if np.isfinite(gaps).any() else float("nan")
        row["corridors_significant"] = int(
            group["significant_fdr"].sum() if "significant_fdr" in group else 0
        )
        row["corridors_positive"] = int(
            (group["currency_uplift_bps"].to_numpy(dtype=float) > 0).sum()
        )

        pooled = _pool_nulls(
            name, list(group["iso"]), weights, null_by_cell, spec.random_trials
        )
        row.update(_null_comparison(row, pooled))
        null_rows.append(
            pd.DataFrame(
                {
                    "strategy": name,
                    "trial": np.arange(len(pooled["currency_uplift_bps"])),
                    "currency_uplift_bps": pooled["currency_uplift_bps"],
                    "hit_rate": pooled["hit_rate"],
                }
            )
        )
        strategy_signals = (
            signals[signals["strategy"].eq(name)] if len(signals) else pd.DataFrame()
        )
        row.update(_advantage_significance(strategy_signals, spec))
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    leaderboard["selection"] = leaderboard["strategy"].map(
        lambda name: get_strategy(name).selection
    )
    # 0 = a matched random schedule (zero expected uplift by construction),
    # 100 = perfect foresight spending *this strategy's own* budget. Both ends
    # of the scale are therefore frequency-matched, so a strategy that stays
    # quiet is not credited for the cadence effect alone.
    ceiling = leaderboard["ceiling_bps"].to_numpy(dtype=float)
    value = leaderboard["currency_uplift_bps"].to_numpy(dtype=float)
    leaderboard["cbsb_score"] = np.where(
        np.isfinite(ceiling) & (ceiling > 0), value / ceiling * 100.0, np.nan
    )
    nulls = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    return (
        leaderboard.sort_values("cbsb_score", ascending=False).reset_index(drop=True),
        nulls,
    )


def _evaluate_gates(leaderboard: pd.DataFrame, spec: BenchmarkSpec) -> pd.DataFrame:
    """Pass/fail the brief's mandatory conditions, one row per strategy per gate."""
    rows: list[dict[str, object]] = []
    for _, entry in leaderboard.iterrows():
        for gate in spec.gates:
            value = entry.get(gate.metric, float("nan"))
            rows.append(
                {
                    "strategy": entry["strategy"],
                    "gate": gate.name,
                    "question": gate.question,
                    "rule": gate.describe(),
                    "value": value,
                    "passed": gate.check(value),
                    "target": gate.target,
                }
            )
    return pd.DataFrame(rows)


def _horizon_table(
    panel: pd.DataFrame, signals: pd.DataFrame, spec: BenchmarkSpec
) -> pd.DataFrame:
    """Hit rate of each strategy's signal set at h in {1,3,5,10,20}."""
    if signals.empty:
        return pd.DataFrame()
    by_horizon = {
        h: build_labels(
            panel,
            horizon=h,
            execution_offset=spec.execution_offset,
            epsilon_bps=spec.epsilon_bps,
            bad_push_bps=spec.bad_push_bps,
        )
        for h in spec.reported_horizons
    }
    rows: list[dict[str, object]] = []
    for name, group in signals.groupby("strategy", sort=False):
        column = hit_column(get_strategy(name).scenario)
        keys = group[["iso", "quote_date"]].drop_duplicates()
        entry: dict[str, object] = {"strategy": name, "scenario": get_strategy(name).scenario}
        for h, table in by_horizon.items():
            joined = keys.merge(table, on=["iso", "quote_date"], how="inner")
            joined = joined[joined["outcome_complete"]]
            entry[f"hit_h{h}"] = float(joined[column].mean()) if len(joined) else float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


def _cadence_sweep(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    complete: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
    cache: ModelScoreCache,
    spec: BenchmarkSpec,
    strategy_names: tuple[str, ...],
) -> pd.DataFrame:
    """What the communication policy costs, priced in client money.

    The same scores are pushed through a range of budgets. Two effects separate
    here and they are usually confused with each other: the *cooldown* silently
    shrinks a weekly budget the policy still paces against, and the *frequency
    itself* determines how far up the value distribution a push has to reach.
    Neither is a defect of any indicator, which is why the sweep lives beside
    the leaderboard rather than inside it.
    """
    eval_start, eval_end = window
    universe = complete[complete["quote_date"].between(eval_start, eval_end)]
    n_corridors = max(1, len(spec.corridors))
    days_per_corridor = len(universe) / n_corridors
    rows: list[dict[str, object]] = []
    for cadence in spec.cadence_grid:
        for name in strategy_names:
            strategy = get_strategy(name)
            scores = build_scores(strategy, spec, panel, features, labels, cache)
            emitted = _emit_signals(strategy, spec, scores, cadence)
            sent = emitted.merge(complete, on=["iso", "quote_date"], how="inner")
            sent = sent[sent["quote_date"].between(eval_start, eval_end)]
            row: dict[str, object] = {
                "cadence": cadence.label,
                "window": cadence.window,
                "max_per_window": cadence.max_per_window,
                "cooldown": cadence.cooldown,
                "strategy": name,
                "n_signals": len(sent),
            }
            if sent.empty:
                row["per_week"] = 0.0
                rows.append(row)
                continue
            row.update(
                {
                    "per_week": (len(sent) / n_corridors) / (days_per_corridor / 5.0),
                    "currency_uplift_bps": float(np.nanmean(sent["currency_gain_bps"])),
                    "hit_closing": float(sent["held_window_closing"].mean()),
                    "bad_push_rate": float(sent["bad_push"].mean()),
                    **_cadence_by_corridor(sent),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _lambda_sweep(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    complete: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
    cache: ModelScoreCache,
    spec: BenchmarkSpec,
    lambda_grid: tuple[float, ...],
) -> pd.DataFrame:
    """What the price of error buys: the MVP's metrics across lambda.

    The heads never see lambda, so the whole curve reuses one walk-forward fit.
    This is the numeric answer to "how much should a bad push cost us?" — the
    business picks a point on the curve rather than accepting a fitted constant.
    """
    eval_start, eval_end = window
    universe = complete[complete["quote_date"].between(eval_start, eval_end)]
    n_corridors = max(1, len(spec.corridors))
    days_per_corridor = len(universe) / n_corridors
    rows: list[dict[str, object]] = []
    for lam in lambda_grid:
        variant = Strategy(
            name=f"utility_risk_lam{lam:g}",
            kind="model",
            scenario="favourable_now",
            description=f"MVP, lambda = {lam:g}",
            minimum_score=0.0,
            lam=lam,
        )
        scores = build_scores(variant, spec, panel, features, labels, cache)
        emitted = _emit_signals(variant, spec, scores)
        sent = emitted.merge(complete, on=["iso", "quote_date"], how="inner")
        sent = sent[sent["quote_date"].between(eval_start, eval_end)]
        if sent.empty:
            rows.append({"lam": lam, "n_signals": 0, "per_week": 0.0})
            continue
        rows.append(
            {
                "lam": lam,
                "n_signals": len(sent),
                "per_week": (len(sent) / n_corridors) / (days_per_corridor / 5.0),
                "hit_rate": float(sent["held_favourable"].mean()),
                "currency_uplift_bps": float(np.nanmean(sent["currency_gain_bps"])),
                "window_advantage_bps": float(np.nanmean(sent["window_advantage_bps"])),
                "bad_push_rate": float(sent["bad_push"].mean()),
                "regret_bps": float(np.nanmean(sent["regret_bps"])),
            }
        )
    return pd.DataFrame(rows)


def _audit_no_lookahead(
    panel: pd.DataFrame,
    spec: BenchmarkSpec,
    config: UtilityRiskConfig,
    n_dates: int = 2,
    n_corridors: int = 2,
) -> pd.DataFrame:
    """Recompute model scores from a truncated panel and require an exact match.

    The brief disqualifies any look-ahead, so masking future rows is not enough:
    the audit deletes them. If the score for date ``T`` computed on a panel that
    *ends* at ``T`` differs from the same date's score in the full historical
    run, the model is reading the future.
    """
    rows: list[dict[str, object]] = []
    for iso in spec.corridors[:n_corridors]:
        full, _ = walk_forward_scores(panel, iso, config)
        evaluated = full[full["quote_date"] >= spec.eval_start] if len(full) else full
        if evaluated.empty:
            continue
        picks = evaluated["quote_date"].iloc[
            np.linspace(0, len(evaluated) - 1, n_dates, dtype=int)
        ]
        for asof in picks:
            live = scores_asof(panel, iso, asof, config)
            live = live[live["quote_date"].eq(asof)]
            historical = full[full["quote_date"].eq(asof)]
            if live.empty or historical.empty:
                rows.append(
                    {"iso": iso, "asof": asof, "matched": False,
                     "reason": "score missing in one of the runs"}
                )
                continue
            difference = abs(float(live["score"].iloc[0]) - float(historical["score"].iloc[0]))
            rows.append(
                {
                    "iso": iso,
                    "asof": asof,
                    "asof_score": float(live["score"].iloc[0]),
                    "historical_score": float(historical["score"].iloc[0]),
                    "abs_difference": difference,
                    "matched": bool(difference < 1e-9),
                    "reason": "",
                }
            )
    return pd.DataFrame(rows)
