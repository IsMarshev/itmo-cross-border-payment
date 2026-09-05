"""Stateful, sequential send/wait/abstain policy shared by serving and backtests."""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import RiskTracker


@dataclass
class PolicyParameters:
    level: float = 0.25
    probability: float = 0.70
    contact_cost_bps: float = 0.0
    enabled: bool = True
    status: str = "configured"


@dataclass
class CorridorState:
    sent: deque = field(default_factory=deque)
    episode: str | None = None
    phase: str = "idle"
    start: pd.Timestamp | None = None
    anchor_price: float = float("inf")
    wait_updates: int = 0
    quiet_updates: int = 0


def push_text(row, scenario, method):
    if method == "random_policy":
        return "", "analytical_baseline"
    currency = {"TJS": "сомони", "UZS": "сума", "KGS": "сома", "AMD": "драма", "KZT": "тенге"}.get(
        row["iso"], row["iso"]
    )
    if scenario == "window_closing":
        change = (np.exp(row["ret_1"]) - 1) * 100
        return (
            f"Курс {currency} по данным ЦБ вырос на {change:.2f}% при последнем обновлении.",
            "observed_rebound",
        )
    if np.isfinite(row.get("text_better_pct", np.nan)) and row["text_better_pct"] >= 60:
        pct = int(np.floor(row["text_better_pct"]))
        return (
            f"По данным ЦБ, курс {currency} ниже, чем в {pct}% дней за последние {row.get('text_window_days', 90)} дней.",
            "historical_percentile",
        )
    if row["down_streak"] >= 2:
        return (
            f"Курс {currency} снижался при последних {int(row['down_streak'])} обновлениях ЦБ.",
            "down_streak",
        )
    change = row.get("week_change_bps", np.nan) / 100
    if np.isfinite(change):
        verb = "снизился" if change < 0 else "вырос"
        return (
            f"Курс {currency} по данным ЦБ {verb} на {abs(change):.2f}% за неделю.",
            "weekly_change",
        )
    return (
        f"Обновлён официальный курс {currency}. Условия перевода доступны в приложении.",
        "rate_update",
    )


class SignalPolicy:
    def __init__(self, config, method, seed=None):
        self.config, self.method = config, method
        self.seed = config.seed if seed is None else seed
        self.states = defaultdict(CorridorState)

    def decide(self, row, params):
        pc, rc = self.config.policy, self.config.risk
        iso, dt, price = row["iso"], pd.Timestamp(row["date"]), row["rub_per_unit"]
        state = self.states[iso]
        while state.sent and (dt - state.sent[0]).days >= 7:
            state.sent.popleft()
        rule = self.method.startswith("rule_")
        random = self.method == "random_policy"
        low, momentum = row["level_rank"] <= params.level, bool(row["fast_momentum"])
        slow = low and bool(row["slow_confirmed"])
        candidate = low or momentum
        if self.method == "rule_value":
            candidate = low
        elif self.method == "rule_momentum":
            candidate = momentum
        elif self.method == "rule_reversal":
            candidate = slow
        elif self.method == "rule_seasonal":
            candidate = row.get("seasonal_score", 0) > pc.min_expected_gain_bps
        elif random:
            key = f"{self.seed}|{iso}|{dt.date()}".encode()
            u = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") / 2**64
            candidate = u < pc.random_send_probability
        scenario = "window_closing" if slow and not random else "favourable_now"
        utility = (
            row["pred_gain_bps"]
            - rc.regret_penalty * row["pred_regret_bps"]
            - rc.stale_penalty * row["pred_stale_bps"]
            - params.contact_cost_bps
        )
        if len(state.sent) == pc.max_per_7_days - 1 and pc.max_per_7_days > 1:
            utility -= pc.last_slot_extra_bps
        result = dict(
            row,
            method=self.method,
            scenario=scenario,
            decision="abstain",
            reason="no_candidate",
            utility_bps=float(utility),
            episode_id=state.episode,
            threshold_probability=params.probability,
            threshold_level=params.level,
            threshold_contact_bps=params.contact_cost_bps,
            available_slots=pc.max_per_7_days - len(state.sent),
            indicator="reversal" if slow else ("level" if low else "momentum"),
            speed="slow" if slow else "fast",
            direction="down" if row["ret_1"] < 0 else "up",
            strength=float(1 - row["level_rank"]),
            policy_status=params.status,
        )
        result["is_candidate"] = bool(candidate)
        if not params.enabled:
            result["reason"] = "insufficient_calibration_evidence"
            return result
        if not candidate:
            state.quiet_updates += 1
            if state.quiet_updates >= pc.reset_updates and state.phase != "pending":
                state.episode, state.phase = None, "idle"
            if state.phase == "pending":
                state.wait_updates += 1
                if (
                    state.wait_updates >= pc.max_wait_updates
                    or (dt - state.start).days >= pc.max_wait_days
                ):
                    state.phase = "expired"
                    result["reason"] = "wait_expired"
                else:
                    result["decision"], result["reason"] = "wait", "awaiting_confirmation"
            return result
        state.quiet_updates = 0
        improvement = (state.anchor_price / price - 1) * 10000
        if (
            random
            or state.episode is None
            or (state.phase in ("sent", "expired") and improvement >= pc.rearm_improvement_bps)
        ):
            state.episode = f"{iso}:{dt.date()}"
            state.phase, state.start, state.anchor_price, state.wait_updates = "open", dt, price, 0
        result["episode_id"] = state.episode
        if state.phase in ("sent", "expired"):
            result["reason"] = (
                "episode_already_used" if state.phase == "sent" else "episode_expired"
            )
            return result
        if state.phase == "pending":
            state.wait_updates += 1
            if (dt - state.start).days > pc.max_wait_days:
                state.phase, result["reason"] = "expired", "wait_expired"
                return result
        if len(state.sent) >= pc.max_per_7_days:
            result["reason"] = "weekly_budget"
            return result
        if state.sent and (dt - state.sent[-1]).days < pc.cooldown_days:
            result["reason"] = "cooldown"
            return result
        if not rule and not random:
            probability = (
                row["pred_close"]
                if slow
                else min(row["pred_local_min"], row["pred_no_regret"], row["pred_hold"])
            )
            if row["regime"] == "shock" and not self.method.endswith("no_regime"):
                required = min(0.99, params.probability + 0.10)
            else:
                required = params.probability
            risk_ok = self.method.endswith("no_uncertainty") or (
                row["upper_regret_bps"] <= pc.regret_cap_bps
                and row["upper_stale_bps"] <= pc.stale_cap_bps
            )
            qualifies = (
                probability >= required
                and risk_ok
                and utility > 0
                and row["pred_gain_bps"] >= pc.min_expected_gain_bps
            )
            can_wait = (
                not self.method.endswith("no_wait") and state.wait_updates < pc.max_wait_updates
            )
            prefer_wait = row["pred_wait_delta_bps"] > pc.wait_margin_bps and not slow
            if can_wait and (prefer_wait or (not qualifies and low and not slow)):
                state.phase = "pending"
                result["decision"], result["reason"] = (
                    "wait",
                    "waiting_has_value" if prefer_wait else "awaiting_confirmation",
                )
                return result
            if not qualifies:
                result["reason"] = (
                    "risk_bound"
                    if not risk_ok
                    else ("low_probability" if probability < required else "nonpositive_value")
                )
                if state.phase == "pending" and state.wait_updates >= pc.max_wait_updates:
                    state.phase, result["reason"] = "expired", "wait_expired"
                return result
        state.sent.append(dt)
        state.phase = "sent"
        result["decision"], result["reason"] = (
            "send",
            "slow_confirmation" if slow else "qualified_fast",
        )
        result["push_text"], result["fact_type"] = push_text(row, scenario, self.method)
        result["reference_source"] = "CBR official reference; not an executable bank quote"
        result["valid_until"] = str(
            (dt + pd.Timedelta(days=self.config.targets.opening_horizon)).date()
        )
        return result


class PolicyReplay:
    """Only matured residuals are revealed before each decision, including in replay."""

    def __init__(self, config, method, seeds, policy=None):
        self.config, self.method = config, method
        self.policy = policy or SignalPolicy(config, method)
        self.risk = RiskTracker(seeds, config)
        self.pending, self.sequence = [], 0

    def run(self, predictions, parameters, targets=None):
        truth_map = {} if targets is None else targets.set_index(["date", "iso"]).to_dict("index")
        results = []
        for row in predictions.sort_values(["date", "iso"]).to_dict("records"):
            dt = pd.Timestamp(row["date"])
            if self.method.endswith("no_regime"):
                row["regime"] = "range"
            while self.pending and self.pending[0][0] <= dt:
                _, _, previous, truth = heapq.heappop(self.pending)
                self.risk.observe(previous, truth)
            for head in ("regret_bps", "stale_bps"):
                row[f"upper_{head}"] = self.risk.upper(
                    row, head, not self.method.endswith("no_uncertainty")
                )
            row["text_window_days"] = self.config.features.text_window_days
            decision = self.policy.decide(row, parameters[row["iso"]])
            results.append(decision)
            truth = truth_map.get((dt, row["iso"]))
            if (
                truth
                and pd.notna(truth["label_known_on"])
                and np.isfinite(truth.get("y_regret_bps", np.nan))
            ):
                self.sequence += 1
                heapq.heappush(
                    self.pending, (pd.Timestamp(truth["label_known_on"]), self.sequence, row, truth)
                )
        return pd.DataFrame(results)
