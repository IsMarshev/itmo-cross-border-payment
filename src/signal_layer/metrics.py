"""Message-specific hits, matched random days, block uncertainty and waiting episodes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policy import PolicyParameters, PolicyReplay, SignalPolicy
from .targets import CLASS_HEADS


def _bootstrap(frame, hit_column, config, start, end):
    length = config.backtest.bootstrap_block_days
    nblocks = (end - start).days // length + 1
    result = {f"{m}_{side}": np.nan for m in ("hit", "lift", "gain") for side in ("lo", "hi")}
    if frame.empty or nblocks < 3 or config.backtest.bootstrap_samples < 20:
        return result
    f = frame.copy()
    f["block"] = (f.date - start).dt.days // length
    a = f.groupby("block").agg(
        n=(hit_column, "count"),
        hit=(hit_column, "sum"),
        random=("random_hit", "sum"),
        gain=("gain", "sum"),
    )
    a = a.reindex(range(nblocks), fill_value=0).to_numpy(float)
    # Same date-block draws for all currencies preserve dependence in ALL aggregates.
    rng = np.random.default_rng(config.seed)
    draws = rng.integers(0, nblocks, size=(config.backtest.bootstrap_samples, nblocks))
    sums = a[draws].sum(axis=1)
    good = (sums[:, 0] > 0) & (sums[:, 2] > 0)
    if good.sum() < 20:
        return result
    values = {
        "hit": sums[good, 1] / sums[good, 0],
        "lift": sums[good, 1] / sums[good, 2],
        "gain": sums[good, 3] / sums[good, 0],
    }
    for name, v in values.items():
        result[f"{name}_lo"], result[f"{name}_hi"] = np.quantile(v, [0.025, 0.975])
    return result


def _frequency(sent, iso_count, start, end):
    days = max(1, (end - start).days + 1)
    weeks = pd.date_range(start, end, freq="D").to_period("W")
    counts = sent.groupby(["iso", sent.date.dt.to_period("W")]).size().to_numpy(float)
    total_cells = len(weeks.unique()) * iso_count
    weekly = pd.Series(np.concatenate([counts, np.zeros(max(0, total_cells - len(counts)))]))
    gaps = []
    for _, g in sent.groupby("iso"):
        gaps.extend(g.date.sort_values().diff().dt.days.dropna().tolist())
    gaps = np.asarray(gaps, float)
    return {
        "frequency_per_week": len(sent) / (days / 7) / iso_count,
        "frequency_per_month": len(sent) / (days / 30.4375) / iso_count,
        "empty_week_share": float((weekly == 0).mean()),
        "weekly_count_cv": float(weekly.std(ddof=0) / weekly.mean()) if weekly.mean() else np.nan,
        "series_share": float(np.mean(gaps <= 3)) if len(gaps) else np.nan,
        "median_gap_days": float(np.median(gaps)) if len(gaps) else np.nan,
        "gap_cv": float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() else np.nan,
    }


def summarize(decisions, targets, config, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    universe = decisions[["date", "iso"]].drop_duplicates().merge(targets, on=["date", "iso"])
    universe["month"] = universe.date.dt.to_period("M").astype(str)
    joined = decisions.merge(targets, on=["date", "iso"], validate="many_to_one")
    joined["month"] = joined.date.dt.to_period("M").astype(str)
    rows = []
    for method in sorted(decisions.method.unique()):
        method_rows = joined.loc[joined.method == method]
        for iso in config.data.corridors + ["ALL"]:
            subset = method_rows if iso == "ALL" else method_rows.loc[method_rows.iso == iso]
            sent_all = subset.loc[subset.decision == "send"]
            for scenario in ("all", "favourable_now", "window_closing"):
                sent = (
                    sent_all if scenario == "all" else sent_all.loc[sent_all.scenario == scenario]
                )
                for h in config.targets.horizons:
                    f = sent.copy()
                    f["hit"] = np.where(
                        f.scenario == "window_closing", f[f"close_h{h}"], f[f"hold_h{h}"]
                    )
                    f["gain"] = f[f"gain_bps_h{h}"]
                    f = f.dropna(subset=["hit", "gain"])
                    pool = universe.groupby(["iso", "month"])[
                        [f"hold_h{h}", f"close_h{h}", f"gain_bps_h{h}"]
                    ].mean()
                    f["random_hit"] = [
                        pool.loc[
                            (r.iso, r.month),
                            f"{'close' if r.scenario == 'window_closing' else 'hold'}_h{h}",
                        ]
                        for r in f.itertuples()
                    ]
                    f["random_gain"] = [
                        pool.loc[(r.iso, r.month), f"gain_bps_h{h}"] for r in f.itertuples()
                    ]
                    n = len(f)
                    hit, random = (f.hit.mean(), f.random_hit.mean()) if n else (np.nan, np.nan)
                    row = dict(
                        method=method,
                        iso=iso,
                        scenario=scenario,
                        horizon=h,
                        n_signals=n,
                        hit_rate=hit,
                        random_day_hit=random,
                        lift=hit / random if random > 0 else np.nan,
                        gain_bps=f.gain.mean(),
                        random_day_gain_bps=f.random_gain.mean(),
                        excess_gain_bps=(f.gain - f.random_gain).mean(),
                        local_min_hit=f[f"local_min_h{h}"].mean(),
                        no_regret_hit=f[f"no_regret_h{h}"].mean(),
                        regret_mean_bps=f[f"regret_bps_h{h}"].mean(),
                        regret_p95_bps=f[f"regret_bps_h{h}"].quantile(0.95),
                        stale_mean_bps=f[f"stale_bps_h{h}"].mean(),
                        eval_start=str(start.date()),
                        eval_end=str(end.date()),
                    )
                    row.update(
                        _frequency(
                            sent, len(config.data.corridors) if iso == "ALL" else 1, start, end
                        )
                    )
                    row.update(_bootstrap(f, "hit", config, start, end))
                    row["max_possible_lift"] = 1 / random if random > 0 else np.nan
                    row["lift_target_feasible"] = bool(random > 0 and random <= 1 / 1.3)
                    row["frequency_in_target"] = (
                        config.policy.frequency_target_min
                        <= row["frequency_per_week"]
                        <= config.policy.max_per_7_days
                    )
                    row["evidence"] = (
                        "no_signals"
                        if n == 0
                        else (
                            "promising"
                            if row["lift"] >= 1.3
                            and row["lift_lo"] > 1
                            and row["gain_lo"] > 0
                            and row["frequency_in_target"]
                            else "not_demonstrated"
                        )
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def prediction_diagnostics(decisions, targets):
    f = decisions.merge(targets, on=["date", "iso"], validate="many_to_one")
    rows, bins = [], []
    for (method, iso), g in f.groupby(["method", "iso"]):
        for scope in ("all_predictions", "sent_only"):
            s = g if scope == "all_predictions" else g.loc[g.decision == "send"]
            for head in CLASS_HEADS:
                valid = s.dropna(subset=[f"y_{head}", f"pred_{head}"]).copy()
                if valid.empty:
                    continue
                y, p = valid[f"y_{head}"], valid[f"pred_{head}"]
                valid["bin"] = np.minimum((p * 10).astype(int), 9)
                ece = 0
                for b, block in valid.groupby("bin"):
                    predicted, actual = block[f"pred_{head}"].mean(), block[f"y_{head}"].mean()
                    ece += len(block) / len(valid) * abs(predicted - actual)
                    bins.append(
                        dict(
                            method=method,
                            iso=iso,
                            scope=scope,
                            head=head,
                            bin=b,
                            n=len(block),
                            predicted=predicted,
                            observed=actual,
                        )
                    )
                rows.append(
                    dict(
                        method=method,
                        iso=iso,
                        scope=scope,
                        metric=f"brier_{head}",
                        value=np.mean((p - y) ** 2),
                        n=len(valid),
                    )
                )
                rows.append(
                    dict(
                        method=method,
                        iso=iso,
                        scope=scope,
                        metric=f"ece_{head}",
                        value=ece,
                        n=len(valid),
                    )
                )
            for head in ("regret_bps", "stale_bps"):
                v = s.dropna(subset=[f"y_{head}"])
                if v.empty:
                    continue
                rows.append(
                    dict(
                        method=method,
                        iso=iso,
                        scope=scope,
                        metric=f"coverage_{head}",
                        value=np.mean(v[f"y_{head}"] <= v[f"upper_{head}"]),
                        n=len(v),
                    )
                )
                rows.append(
                    dict(
                        method=method,
                        iso=iso,
                        scope=scope,
                        metric=f"mae_{head}",
                        value=np.mean(abs(v[f"y_{head}"] - v[f"pred_{head}"])),
                        n=len(v),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(bins)


def waiting_episodes(decisions, targets, config):
    truth = targets.set_index(["date", "iso"])
    rows = []
    for (method, iso, episode), g in decisions.dropna(subset=["episode_id"]).groupby(
        ["method", "iso", "episode_id"]
    ):
        g = g.sort_values("date")
        candidates = g.loc[g.is_candidate]
        if candidates.empty:
            continue
        first = candidates.iloc[0]
        if first.scenario == "window_closing":
            continue
        later = g.loc[g.date > first.date].head(config.policy.max_wait_updates)
        later = later.loc[(later.date - first.date).dt.days <= config.policy.max_wait_days]
        confirmations = later.loc[later.scenario == "window_closing"]
        slow = None if confirmations.empty else confirmations.iloc[0]
        sent = g.loc[g.decision == "send"]
        row = dict(
            method=method,
            iso=iso,
            episode_id=episode,
            fast_date=first.date,
            confirmation_date=None if slow is None else slow.date,
            confirmed=slow is not None,
            selected_action=first.decision,
            actual_send_date=None if sent.empty else sent.iloc[0].date,
            waiting_cost_bps=np.nan
            if slow is None
            else (1 - first.rub_per_unit / slow.rub_per_unit) * 10000,
            days_waited=np.nan if slow is None else (slow.date - first.date).days,
        )
        start_truth = truth.loc[(first.date, iso)]
        utility_fast = (
            start_truth.y_gain_bps
            - config.risk.regret_penalty * start_truth.y_regret_bps
            - config.risk.stale_penalty * start_truth.y_stale_bps
        )
        utility_wait = 0.0
        if slow is not None:
            slow_truth = truth.loc[(slow.date, iso)]
            # Same initial benefit reference, adjusted by the observed cost of waiting.
            utility_wait = (
                start_truth.y_gain_bps
                - row["waiting_cost_bps"]
                - config.risk.regret_penalty * slow_truth.y_regret_bps
                - config.risk.stale_penalty * slow_truth.y_stale_bps
            )
        row["fast_utility_bps"], row["wait_utility_bps"] = utility_fast, utility_wait
        row["wait_minus_fast_bps"] = utility_wait - utility_fast
        rows.append(row)
    columns = [
        "method",
        "iso",
        "episode_id",
        "fast_date",
        "confirmation_date",
        "confirmed",
        "selected_action",
        "actual_send_date",
        "waiting_cost_bps",
        "days_waited",
        "fast_utility_bps",
        "wait_utility_bps",
        "wait_minus_fast_bps",
    ]
    return pd.DataFrame(rows, columns=columns)


def random_day_draws(decisions, targets, config):
    """Uniform same-corridor/month days; this control deliberately has no cooldown."""
    h = config.targets.primary_horizon
    universe = decisions[["date", "iso"]].drop_duplicates().merge(targets, on=["date", "iso"])
    universe["month"] = universe.date.dt.to_period("M").astype(str)
    pools = {
        (iso, month): g.dropna(subset=[f"hold_h{h}", f"close_h{h}", f"gain_bps_h{h}"])
        for (iso, month), g in universe.groupby(["iso", "month"])
    }
    sent = decisions.loc[decisions.decision == "send"].copy()
    sent["month"] = sent.date.dt.to_period("M").astype(str)
    rows = []
    for (method, iso), group in sent.groupby(["method", "iso"]):
        for repeat in range(config.backtest.random_repeats):
            rng = np.random.default_rng(config.seed + repeat)
            hits, gains = [], []
            for (month, scenario), part in group.groupby(["month", "scenario"]):
                pool = pools[(iso, month)]
                if pool.empty:
                    continue
                sample = pool.iloc[
                    rng.choice(len(pool), size=len(part), replace=len(part) > len(pool))
                ]
                hits.extend(
                    sample[f"{'close' if scenario == 'window_closing' else 'hold'}_h{h}"].tolist()
                )
                gains.extend(sample[f"gain_bps_h{h}"].tolist())
            if hits:
                rows.append(
                    dict(
                        method=method,
                        iso=iso,
                        repeat=repeat,
                        n=len(hits),
                        random_hit=np.mean(hits),
                        random_gain_bps=np.mean(gains),
                    )
                )
    return pd.DataFrame(
        rows, columns=["method", "iso", "repeat", "n", "random_hit", "random_gain_bps"]
    )


def random_policy_draws(reference_predictions, targets, config, start, end):
    """Sequential Bernoulli control, same hard weekly/cooldown limits, many seeds."""
    rows = []
    params = {
        iso: PolicyParameters(probability=0, status="random_control")
        for iso in config.data.corridors
    }
    h = config.targets.primary_horizon
    for repeat in range(config.backtest.random_repeats):
        policy = SignalPolicy(config, "random_policy", seed=config.seed + repeat)
        # Random decisions do not use estimates or uncertainty.
        replay = PolicyReplay(config, "random_policy", [], policy=policy)
        d = replay.run(reference_predictions, params)
        s = d.loc[d.decision == "send"].merge(targets, on=["date", "iso"])
        for iso in config.data.corridors:
            g = s.loc[s.iso == iso].dropna(subset=[f"hold_h{h}"])
            rows.append(
                dict(
                    iso=iso,
                    repeat=repeat,
                    n=len(g),
                    hit_rate=g[f"hold_h{h}"].mean(),
                    gain_bps=g[f"gain_bps_h{h}"].mean(),
                    frequency_per_week=len(g) / max(1, (end - start).days + 1) * 7,
                )
            )
    return pd.DataFrame(rows)
