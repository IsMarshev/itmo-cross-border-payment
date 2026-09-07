from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from signal_layer.data import (
    CANONICAL_COLUMNS,
    RateDataError,
    normalize_rate_frame,
    read_rate_directory,
)


def test_normalize_rate_frame_converts_nominal_to_one_currency_unit() -> None:
    source = pd.DataFrame(
        {
            "date": ["2026-09-02"],
            "iso": [" amd "],
            "nominal": [100],
            "rate": [23.8162],
            "rate_per_unit": [0.238162],
        }
    )

    actual = normalize_rate_frame(source)

    assert actual.columns.tolist() == list(CANONICAL_COLUMNS)
    assert actual.loc[0, "iso"] == "AMD"
    assert actual.loc[0, "rub_per_unit"] == pytest.approx(0.238162)
    assert actual.loc[0, "available_on"] == pd.Timestamp("2026-09-03")


def test_normalize_rate_frame_rejects_duplicate_currency_date() -> None:
    source = pd.DataFrame(
        {
            "date": ["2026-09-02", "2026-09-02"],
            "iso": ["USD", "USD"],
            "nominal": [1, 1],
            "rate": [86.75, 86.80],
        }
    )

    with pytest.raises(RateDataError, match="Duplicate"):
        normalize_rate_frame(source)


def test_normalize_rate_frame_rejects_inconsistent_derived_rate() -> None:
    source = pd.DataFrame(
        {
            "date": ["2026-09-02"],
            "iso": ["KZT"],
            "nominal": [100],
            "rate": [18.7651],
            "rate_per_unit": [1.0],
        }
    )

    with pytest.raises(RateDataError, match="does not match"):
        normalize_rate_frame(source)


def test_read_rate_directory_normalizes_repository_sources() -> None:
    project_root = Path(__file__).resolve().parents[1]

    panel = read_rate_directory(project_root / "currency_data", currencies=["AMD", "USD"])

    assert set(panel["iso"]) == {"AMD", "USD"}
    assert panel.duplicated(["quote_date", "iso"]).sum() == 0
    assert (panel["rub_per_unit"] > 0).all()
    amd_last = panel.loc[panel["iso"].eq("AMD")].iloc[-1]
    assert amd_last["rub_per_unit"] == pytest.approx(0.238162)
