"""Strict, serializable experiment configuration; all horizons are calendar days."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    directory: str = "currency_data"
    start: str = "2011-01-01"
    end: str | None = None
    corridors: list[str] = field(default_factory=lambda: ["TJS", "UZS", "KGS", "AMD", "KZT"])
    context: list[str] = field(default_factory=lambda: ["USD", "EUR", "CNY"])
    availability_lag_days: int = 0
    max_stale_days: int = 21
    holidays_file: str | None = None


@dataclass
class FeatureConfig:
    windows: list[int] = field(default_factory=lambda: [5, 20, 60, 120])
    level_window: int = 60
    text_window_days: int = 90
    momentum_streak: int = 3
    reversal_bps: float = 20.0
    shock_z: float = 4.0


@dataclass
class TargetConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    primary_horizon: int = 5
    opening_horizon: int = 1
    near_min_bps: float = 50.0
    regret_tolerance_bps: float = 75.0
    hold_tolerance_bps: float = 75.0
    closing_bps: float = 25.0


@dataclass
class ModelConfig:
    methods: list[str] = field(
        default_factory=lambda: [
            "catboost",
            "linear",
            "random_walk",
            "random_walk_drift",
            "ar1",
            "ets",
            "rule_value",
            "rule_momentum",
            "rule_reversal",
            "rule_seasonal",
            "random_policy",
        ]
    )
    iterations: int = 300
    depth: int = 5
    learning_rate: float = 0.04
    l2_leaf_reg: float = 10.0
    threads: int = 4
    ridge_alpha: float = 20.0
    statistical_window: int = 504
    simulation_paths: int = 256
    ets_alpha: float = 0.2
    min_train_rows: int = 500
    min_calibration_rows: int = 80
    train_window_days: int = 1825
    calibration_days: int = 180
    tuning_days: int = 180


@dataclass
class RiskConfig:
    alpha: float = 0.10
    adaptive: bool = True
    adaptation_rate: float = 0.005
    residual_window: int = 250
    regime_min_samples: int = 40
    regret_penalty: float = 3.0
    stale_penalty: float = 1.0


@dataclass
class PolicyConfig:
    max_per_7_days: int = 2
    cooldown_days: int = 3
    max_wait_updates: int = 2
    max_wait_days: int = 7
    reset_updates: int = 2
    rearm_improvement_bps: float = 100.0
    level_thresholds: list[float] = field(default_factory=lambda: [0.15, 0.25, 0.35])
    probability_thresholds: list[float] = field(default_factory=lambda: [0.55, 0.70, 0.85])
    contact_costs_bps: list[float] = field(default_factory=lambda: [0.0, 20.0])
    regret_cap_bps: float = 200.0
    stale_cap_bps: float = 200.0
    min_expected_gain_bps: float = 10.0
    wait_margin_bps: float = 15.0
    last_slot_extra_bps: float = 10.0
    min_tuning_signals: int = 10
    frequency_target_min: float = 1.0
    random_send_probability: float = 0.30


@dataclass
class BacktestConfig:
    start: str = "2023-01-01"
    end: str | None = None
    fold_days: int = 90
    holdout_days: int = 180
    bootstrap_samples: int = 300
    bootstrap_block_days: int = 28
    random_repeats: int = 100
    ablations: bool = True


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def validate(self):
        if not self.data.corridors or len(set(self.data.corridors)) != len(self.data.corridors):
            raise ValueError("data.corridors must be a nonempty unique list")
        if self.data.availability_lag_days < 0:
            raise ValueError("Negative availability lags would introduce look-ahead")
        if any(h < 1 for h in self.targets.horizons):
            raise ValueError("Horizons must be positive calendar days")
        if self.targets.primary_horizon not in self.targets.horizons:
            raise ValueError("primary_horizon must belong to horizons")
        if self.targets.opening_horizon not in self.targets.horizons:
            raise ValueError("opening_horizon must belong to horizons")
        if self.policy.cooldown_days < 1 or self.policy.max_per_7_days not in (1, 2):
            raise ValueError("Use cooldown >= 1 and a weekly limit of 1 or 2")
        if not 0 < self.risk.alpha < 0.5:
            raise ValueError("risk.alpha must lie in (0, .5)")
        if min(self.features.windows) < 2 or self.features.level_window < 2:
            raise ValueError("Feature windows must be >= 2")
        if self.backtest.fold_days < 1 or self.model.min_train_rows < 10:
            raise ValueError("Invalid fold length or minimum training size")
        if self.model.calibration_days <= max(self.targets.horizons):
            raise ValueError("Calibration period must exceed the label horizon")
        if self.model.tuning_days <= max(self.targets.horizons):
            raise ValueError("Tuning period must exceed the label horizon")
        if self.model.simulation_paths < 20 or self.model.iterations < 1:
            raise ValueError("Use >=20 simulation paths and >=1 boosting iterations")
        if not self.policy.level_thresholds or not self.policy.probability_thresholds:
            raise ValueError("Policy grids cannot be empty")
        if any(not 0 < p <= 1 for p in self.policy.level_thresholds):
            raise ValueError("Level thresholds must lie in (0,1]")
        if not 0 <= self.policy.random_send_probability <= 1:
            raise ValueError("Random send probability must lie in [0,1]")
        if self.backtest.random_repeats < 1 or self.backtest.bootstrap_samples < 0:
            raise ValueError("Use >=1 random repetition and >=0 bootstrap samples")
        if self.backtest.holdout_days < 1:
            raise ValueError("holdout_days must be positive")
        if (
            min(self.policy.max_wait_updates, self.policy.max_wait_days, self.policy.reset_updates)
            < 1
        ):
            raise ValueError("Waiting and reset windows must be positive")
        if any(not 0 < p < 1 for p in self.policy.probability_thresholds):
            raise ValueError("Probability thresholds must lie in (0,1)")
        if self.backtest.bootstrap_block_days < max(self.targets.horizons):
            raise ValueError("Bootstrap block must be at least the largest horizon")
        allowed = {
            "catboost",
            "linear",
            "random_walk",
            "random_walk_drift",
            "ar1",
            "ets",
            "rule_value",
            "rule_momentum",
            "rule_reversal",
            "rule_seasonal",
            "random_policy",
        }
        if not self.model.methods or set(self.model.methods) - allowed:
            raise ValueError(f"Unknown/empty methods; choose from {sorted(allowed)}")
        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def config_from_dict(raw: dict) -> Config:
    sections = {
        "data": DataConfig,
        "features": FeatureConfig,
        "targets": TargetConfig,
        "model": ModelConfig,
        "risk": RiskConfig,
        "policy": PolicyConfig,
        "backtest": BacktestConfig,
    }
    unknown = set(raw) - {"seed", *sections}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    kwargs = {"seed": raw.get("seed", 42)}
    for name, cls in sections.items():
        values = raw.get(name, {})
        extra = set(values) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {name} configuration keys: {sorted(extra)}")
        kwargs[name] = cls(**values)
    return Config(**kwargs).validate()


def load_config(path: str | Path | None = None) -> Config:
    return (
        config_from_dict(yaml.safe_load(Path(path).read_text()) or {})
        if path
        else Config().validate()
    )
