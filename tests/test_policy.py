from __future__ import annotations

import pandas as pd

from signal_layer.config import Config
from signal_layer.policy import PolicyParameters, PolicyReplay, SignalPolicy


def row(day, **changes):
    r = dict(
        date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
        iso="TJS",
        rub_per_unit=10,
        level_rank=0.1,
        fast_momentum=True,
        slow_confirmed=False,
        regime="range",
        ret_1=-0.01,
        down_streak=3,
        pred_gain_bps=200,
        pred_regret_bps=10,
        pred_stale_bps=10,
        pred_wait_delta_bps=0,
        pred_local_min=0.95,
        pred_no_regret=0.95,
        pred_hold=0.95,
        pred_close=0.95,
        q_regret_bps=10,
        q_stale_bps=10,
        upper_regret_bps=20,
        upper_stale_bps=20,
        text_better_pct=85,
        week_change_bps=-100,
    )
    return dict(r, **changes)


def test_send_wait_expire_and_confirmation():
    c = Config()
    p, par = SignalPolicy(c, "catboost"), PolicyParameters()
    assert p.decide(row(0, pred_wait_delta_bps=100), par)["decision"] == "wait"
    signal = p.decide(row(1, slow_confirmed=True, ret_1=0.01), par)
    assert signal["decision"] == "send" and signal["scenario"] == "window_closing"
    assert "вырос" in signal["push_text"] and "снижал" not in signal["push_text"]
    p = SignalPolicy(c, "catboost")
    assert p.decide(row(0, pred_wait_delta_bps=100), par)["decision"] == "wait"
    assert p.decide(row(1, level_rank=0.9, fast_momentum=False), par)["decision"] == "wait"
    expired = p.decide(row(2, level_rank=0.9, fast_momentum=False), par)
    assert expired["reason"] == "wait_expired"


def test_budget_survives_fold_boundary_and_random_has_same_limits():
    c = Config()
    c.policy.random_send_probability = 1
    policy = SignalPolicy(c, "random_policy")
    first = PolicyReplay(c, "random_policy", [], policy).run(
        pd.DataFrame([row(i) for i in range(8)]), {"TJS": PolicyParameters()}
    )
    second = PolicyReplay(c, "random_policy", [], policy).run(
        pd.DataFrame([row(i) for i in range(8, 30)]), {"TJS": PolicyParameters()}
    )
    sent = pd.concat([first, second]).query("decision == 'send'").date
    assert (sent.diff().dropna().dt.days >= 3).all()
    for dt in pd.date_range("2024-01-01", periods=30):
        assert sent.between(dt - pd.Timedelta(days=6), dt).sum() <= 2


def test_same_episode_not_repeated_and_new_improvement_rearms():
    p, par = SignalPolicy(Config(), "catboost"), PolicyParameters()
    assert p.decide(row(0), par)["decision"] == "send"
    assert p.decide(row(4), par)["reason"] == "episode_already_used"
    assert p.decide(row(7, rub_per_unit=9.8), par)["decision"] == "send"


def test_delayed_risk_feedback_is_not_revealed_early():
    c = Config()
    seeds = [{"iso": "TJS", "regime": "range", "regret_bps": 0, "stale_bps": 0}] * 40
    replay = PolicyReplay(c, "random_policy", seeds)
    truth = pd.DataFrame(
        [
            dict(
                date=row(0)["date"],
                iso="TJS",
                label_known_on=row(5)["date"],
                y_regret_bps=1000,
                y_stale_bps=1000,
            )
        ]
    )
    replay.run(pd.DataFrame([row(0), row(3)]), {"TJS": PolicyParameters()}, truth)
    assert len(replay.risk.buffers[("TJS", "all", "regret_bps")]) == 40
    assert replay.risk.alpha[("TJS", "all", "regret_bps")] == c.risk.alpha
    replay.run(pd.DataFrame([row(6)]), {"TJS": PolicyParameters()}, truth)
    assert len(replay.risk.buffers[("TJS", "all", "regret_bps")]) == 41
    assert replay.risk.alpha[("TJS", "all", "regret_bps")] < c.risk.alpha
