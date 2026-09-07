"""Runtime configuration for the signal-layer service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Service settings resolved once during application startup."""

    data_dir: Path

    @classmethod
    def from_environment(cls) -> Settings:
        """Load settings without requiring a local dotenv file."""
        default_data_dir = Path(__file__).resolve().parents[2] / "currency_data"
        configured_data_dir = os.getenv("SIGNAL_LAYER_DATA_DIR")
        data_dir = Path(configured_data_dir) if configured_data_dir else default_data_dir
        return cls(data_dir=data_dir.expanduser().resolve())
