"""CBSB-1: the statistics, the gate logic and one end-to-end run."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from signal_layer.benchmark import BenchmarkSpec, run_benchmark
from signal_layer.benchmark.spec import Gate
from signal_layer.benchmark.stats import (
    benjamini_hochberg,
    moving_block_bootstrap_ci,
    newey_west_tstat,
    permutation_pvalue,
)

# --- statistics --------------------------------------------------------------


def test_newey_west_widens_the_error_on_autocorrelated_data():
    rng = np.random.default_rng(0)
    independent = rng.normal(1.0, 1.0, 500)
    correlated = pd.Series(rng.normal(1.0, 1.0, 500)).rolling(20).mean().dropna().to_numpy()

    _, plain_se, _ = newey_west_tstat(independent, lags=0)
    _, hac_se, _ = newey_west_tstat(correlated)
    _, naive_se, _ = newey_west_tstat(correlated, lags=0)

    assert hac_se > naive_se  # the correction is doing something
    assert plain_se == pytest.approx(1.0 / np.sqrt(500), rel=0.3)


def test_newey_west_handles_degenerate_input():
    assert all(np.isnan(v) for v in newey_west_tstat(np.array([])))
    mean, se, t = newey_west_tstat(np.array([2.0]))
    assert mean == 2.0 and np.isnan(se) and np.isnan(t)


def test_block_bootstrap_brackets_the_sample_mean():
    values = np.random.default_rng(1).normal(5.0, 2.0, 400)
    low, high = moving_block_bootstrap_ci(values, block_length=10, trials=500, seed=0)
    assert low < values.mean() < high
    assert high - low < 2.0


def test_permutation_pvalue_is_never_zero_and_ranks_correctly():
    null = np.random.default_rng(2).normal(0.0, 1.0, 999)
    assert permutation_pvalue(100.0, null) == pytest.approx(1 / 1000)
    assert permutation_pvalue(-100.0, null) == pytest.approx(1.0)
    assert 0.4 < permutation_pvalue(0.0, null) < 0.6


def test_benjamini_hochberg_is_stricter_than_raw_pvalues():
    pvalues = np.array([0.001, 0.02, 0.03, 0.5, np.nan])
    rejected, qvalues = benjamini_hochberg(pvalues, alpha=0.05)

    assert rejected.tolist() == [True, True, True, False, False]
    assert np.isnan(qvalues[4])
    # q-values are monotone in p and never below the raw value.
    finite = np.isfinite(qvalues)
    assert np.all(qvalues[finite] >= pvalues[finite] - 1e-12)
    assert np.all(np.diff(qvalues[:4]) >= -1e-12)


def test_benjamini_hochberg_rejects_nothing_when_all_pvalues_are_large():
    rejected, _ = benjamini_hochberg(np.array([0.2, 0.4, 0.9]), alpha=0.05)
    assert not rejected.any()


# --- gates -------------------------------------------------------------------


def test_gate_operators():
    assert Gate("g", "?", "m", ">=", 1.0).check(1.0) is True
    assert Gate("g", "?", "m", ">", 1.0).check(1.0) is False
    assert Gate("g", "?", "m", "<=", 1.0).check(0.5) is True
    assert Gate("g", "?", "m", "between", (0.5, 2.0)).check(2.5) is False
    assert Gate("g", "?", "m", ">=", 1.0).check(float("nan")) is None


def test_spec_folds_tile_the_evaluation_period_without_gaps():
    spec = BenchmarkSpec(eval_start=pd.Timestamp("2021-01-01"), fold_months=6)
    folds = spec.folds(pd.Timestamp("2022-12-31"))
    assert len(folds) == 4
    for (_, _, end), (_, start, _) in zip(folds, folds[1:], strict=False):
        assert start == end + pd.Timedelta(days=1)
    assert folds[0][1] == pd.Timestamp("2021-01-01")
    assert folds[-1][2] == pd.Timestamp("2022-12-31")


# --- end to end --------------------------------------------------------------


@pytest.fixture(scope="module")
def small_run():
    rng = np.random.default_rng(11)
    n = 1200
    level = np.zeros(n)
    for i in range(1, n):
        level[i] = 0.97 * level[i - 1] + rng.normal(0, 0.012)
    dates = pd.bdate_range("2016-01-01", periods=n)
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "quote_date": dates,
                    "available_on": dates + pd.Timedelta(days=1),
                    "iso": iso,
                    "rub_per_unit": 10.0 * scale * np.exp(level),
                }
            )
            for iso, scale in (("TJS", 1.0), ("USD", 8.0))
        ],
        ignore_index=True,
    )
    spec = BenchmarkSpec(
        corridors=("TJS",),
        horizon=5,
        eval_start=dates[900],
        fold_months=6,
        random_trials=50,
        bootstrap_trials=100,
    )
    return run_benchmark(
        panel,
        spec,
        ("percentile", "utility_risk", "oracle", "oracle_topk"),
        model_config=None,
        lambda_grid=(0.0, 2.0),
    )


def test_run_produces_every_artefact(small_run):
    for frame in (
        small_run.leaderboard,
        small_run.per_corridor,
        small_run.per_fold,
        small_run.gates,
        small_run.horizon_table,
        small_run.lambda_sweep,
        small_run.signals,
    ):
        assert len(frame) > 0


def test_no_lookahead_audit_passes(small_run):
    assert len(small_run.audit)
    assert small_run.audit["matched"].all()


def test_the_oracle_beats_every_real_strategy(small_run):
    board = small_run.leaderboard.set_index("strategy")["currency_uplift_bps"]
    assert board["oracle_topk"] > board["percentile"]
    assert board["oracle_topk"] > board["utility_risk"]
    # The ceiling defines the scale, so it scores exactly 100.
    assert small_run.leaderboard.set_index("strategy").loc["oracle_topk", "cbsb_score"] == (
        pytest.approx(100.0)
    )


def test_every_contender_respects_the_shared_push_budget(small_run):
    contenders = small_run.leaderboard[small_run.leaderboard["selection"].eq("policy")]
    assert (contenders["per_week"] <= 2.0 + 1e-9).all()


def test_random_baseline_has_no_expected_edge(small_run):
    """The matched-random null must sit at zero, or the metric is mis-centred."""
    random_uplift = small_run.per_corridor["random_currency_uplift_bps"]
    assert random_uplift.abs().max() < 5.0


def test_signals_carry_realised_outcomes_only_for_complete_horizons(small_run):
    assert small_run.signals["outcome_complete"].all()
    assert small_run.signals["exec_rate"].notna().all()


# --- dashboard ---------------------------------------------------------------


def _panel_for(signals: pd.DataFrame) -> pd.DataFrame:
    """A rate panel covering exactly the dates a run's signals reference."""
    dates = pd.bdate_range("2016-01-01", periods=1200)
    level = np.cumsum(np.random.default_rng(11).normal(0, 0.012, 1200))
    return pd.DataFrame(
        {
            "quote_date": dates,
            "available_on": dates + pd.Timedelta(days=1),
            "iso": "TJS",
            "rub_per_unit": 10.0 * np.exp(level),
        }
    )


def test_dashboard_renders_from_a_partial_strategy_set(small_run, tmp_path):
    """A run with `--strategies` must still produce a page, not a KeyError."""
    from signal_layer.benchmark.dashboard import render_dashboard

    frames = {
        "leaderboard": small_run.leaderboard,
        "per_corridor": small_run.per_corridor,
        "per_fold": small_run.per_fold,
        "gates": small_run.gates,
        "horizons": small_run.horizon_table,
        "lambda_sweep": small_run.lambda_sweep,
        "audit": small_run.audit,
        "signals": small_run.signals,
    }
    spec = BenchmarkSpec(corridors=("TJS",), horizon=5)
    html = render_dashboard(frames, _panel_for(small_run.signals), spec)

    assert html.lstrip().startswith("<title>")
    # The run had no percentile_weekly / utility_risk_paced, so those rows are
    # dropped rather than crashing the page.
    assert "utility_risk" in html
    for tag in ("svg", "table", "section"):
        assert html.count(f"</{tag}>") == len(re.findall(rf"<{tag}[ >]", html))


def test_dashboard_defines_every_colour_token_in_both_themes(small_run, tmp_path):
    """The classic unreadable-artifact bug: a colour that exists in one theme only."""
    from signal_layer.benchmark.dashboard import render_dashboard

    frames = {
        "leaderboard": small_run.leaderboard,
        "per_corridor": small_run.per_corridor,
        "per_fold": small_run.per_fold,
        "gates": small_run.gates,
        "horizons": small_run.horizon_table,
        "lambda_sweep": small_run.lambda_sweep,
        "audit": small_run.audit,
        "signals": small_run.signals,
    }
    html = render_dashboard(frames, _panel_for(small_run.signals), BenchmarkSpec())
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    light = set(re.findall(r"--([\w-]+):", re.search(r":root\{(.*?)\}", style, re.S).group(1)))
    for guard in (r':root:not\(\[data-theme="light"\]\)\{(.*?)\}',
                  r':root\[data-theme="dark"\]\{(.*?)\}'):
        dark = set(re.findall(r"--([\w-]+):", re.search(guard, style, re.S).group(1)))
        assert dark == light, f"theme token mismatch: {dark ^ light}"
    assert "background:var(--ground)" in style.replace(" ", "")


def test_build_dashboard_writes_a_file(small_run, tmp_path):
    from signal_layer.benchmark.dashboard import build_dashboard

    for name, frame in (
        ("leaderboard", small_run.leaderboard),
        ("per_corridor", small_run.per_corridor),
        ("per_fold", small_run.per_fold),
        ("gates", small_run.gates),
        ("horizons", small_run.horizon_table),
        ("lambda_sweep", small_run.lambda_sweep),
        ("audit", small_run.audit),
        ("signals", small_run.signals),
    ):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)

    target = build_dashboard(
        tmp_path, _panel_for(small_run.signals), BenchmarkSpec(corridors=("TJS",))
    )
    assert target.is_file()
    assert target.stat().st_size > 5_000


# --- cadence sweep -----------------------------------------------------------


def test_cadence_sweep_prices_every_budget_in_the_grid(small_run):
    sweep = small_run.cadence_sweep
    assert len(sweep)
    assert set(sweep["cadence"]) == {c.label for c in BenchmarkSpec().cadence_grid}
    # Rarer pushes must clear a higher bar, so value per push rises as the rate
    # falls. Checked on the oracle, where the ordering is not obscured by noise.
    oracle = sweep[sweep["strategy"].eq("oracle")].dropna(subset=["currency_uplift_bps"])
    if len(oracle) > 2:
        ordered = oracle.sort_values("per_week")
        assert ordered["currency_uplift_bps"].iloc[0] > ordered["currency_uplift_bps"].iloc[-1]


def test_cadence_metrics_are_computed_per_corridor_not_pooled():
    """Gaps between signals of different corridors are not gaps a client sees."""
    from signal_layer.benchmark.runner import _cadence, _cadence_by_corridor

    dates = pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24"])
    # Two corridors, each perfectly regular, but interleaved one day apart.
    signals = pd.DataFrame(
        {
            "iso": ["TJS"] * 4 + ["KGS"] * 4,
            "quote_date": list(dates) + list(dates + pd.Timedelta(days=1)),
        }
    )
    per_corridor = _cadence_by_corridor(signals)
    pooled = _cadence(signals)

    # Each corridor is a metronome: zero variation in its own gaps.
    assert per_corridor["interval_cv"] == pytest.approx(0.0, abs=1e-9)
    # Pooling interleaves them into an alternating 1/6-day pattern and reports
    # burstiness no client experiences.
    assert pooled["interval_cv"] > 0.5
    assert pooled["interval_cv"] > per_corridor["interval_cv"]


def test_headline_cadence_stays_inside_the_briefs_band():
    """The default budget must satisfy the gate the brief mandates."""
    spec = BenchmarkSpec()
    gate = next(g for g in spec.gates if g.name == "G4_frequency")
    assert gate.bound == (1.0, 2.0), "the brief mandates 1-2 signals per week"
    assert spec.cadence.per_week <= 2.0


def test_null_distribution_is_persisted_and_centred(small_run):
    """The p-value's evidence must ship with the p-value."""
    nulls = small_run.null_distribution
    assert len(nulls)
    assert set(nulls["strategy"]) <= set(small_run.leaderboard["strategy"])
    for _, group in nulls.groupby("strategy"):
        # A matched random schedule has no expected edge by construction.
        assert abs(float(group["currency_uplift_bps"].mean())) < 10.0
        assert group["trial"].is_unique
