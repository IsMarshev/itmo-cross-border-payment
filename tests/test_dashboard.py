from __future__ import annotations

import json
from pathlib import Path

from signal_layer.dashboard import generate_dashboard


def test_generate_dashboard_from_backtest_artifacts(tmp_path: Path) -> None:
    summary = [
        {
            "iso": "AMD",
            "n_signals": 2,
            "per_week": 0.5,
            "series_share": 0.0,
            "mean_advantage_bps": 10.0,
            "median_advantage_bps": 8.0,
            "p10_advantage_bps": -5.0,
            "hit_rate": 0.6,
            "negative_share": 0.4,
            "early_send_rate": 0.2,
            "p90_regret_bps": 20.0,
            "advantage_ci_low": 1.0,
            "advantage_ci_high": 20.0,
            "random_mean_advantage_bps": 2.0,
            "random_hit_rate": 0.5,
            "advantage_delta_bps": 8.0,
            "advantage_lift": 5.0,
            "hit_rate_lift": 1.2,
        },
        {
            "iso": "ALL",
            "n_signals": 2,
            "per_week": 0.5,
            "series_share": 0.0,
            "mean_advantage_bps": 10.0,
            "median_advantage_bps": 8.0,
            "p10_advantage_bps": -5.0,
            "hit_rate": 0.6,
            "negative_share": 0.4,
            "early_send_rate": 0.2,
            "p90_regret_bps": 20.0,
            "advantage_ci_low": 1.0,
            "advantage_ci_high": 20.0,
            "random_mean_advantage_bps": 2.0,
            "random_hit_rate": 0.5,
            "advantage_delta_bps": 8.0,
            "advantage_lift": 5.0,
            "hit_rate_lift": 1.2,
        },
    ]
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    decision = {
        "decision_date": "2026-01-10T00:00:00.000",
        "iso": "AMD",
        "decision": True,
        "outcome_complete": True,
        "score": 0.9,
        "threshold": 0.8,
        "advantage_bps": 12.0,
        "regret_bps": 0.0,
    }
    (tmp_path / "decision_log.jsonl").write_text(
        json.dumps(decision) + "\n", encoding="utf-8"
    )

    output = generate_dashboard(tmp_path)

    assert output == tmp_path / "dashboard.html"
    page = output.read_text(encoding="utf-8")
    assert "Сигнальный слой: бэктест" in page
    assert "AMD" in page
    assert "Последние отобранные сигналы" in page
