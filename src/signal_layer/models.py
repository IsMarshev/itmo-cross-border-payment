"""Small pooled supervised heads and classical stochastic forecasting baselines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .targets import CLASS_HEADS, VALUE_HEADS


class Encoder:
    def __init__(self, columns, config):
        self.columns = list(columns)
        self.categories = {
            "iso": list(config.data.corridors),
            "regime": ["range", "trend", "volatile", "shock"],
        }

    def transform(self, frame):
        arrays = []
        for column in self.columns:
            if column in self.categories:
                arrays.extend(
                    (frame[column] == value).to_numpy(float) for value in self.categories[column]
                )
            else:
                arrays.append(frame[column].to_numpy(float))
        return np.column_stack(arrays)

    def names(self):
        return [
            name
            for col in self.columns
            for name in (
                [f"{col}={v}" for v in self.categories[col]] if col in self.categories else [col]
            )
        ]


class SupervisedPredictor:
    def __init__(self, method, columns, config):
        self.method, self.config = method, config
        self.encoder = Encoder(columns, config)
        self.models, self.quantile_models = {}, {}

    def fit(self, frame):
        x = self.encoder.transform(frame)
        cfg = self.config.model
        weights = 1 / frame.groupby("date").date.transform("size").to_numpy()
        weights /= weights.mean()
        if self.method == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor

            params = dict(
                iterations=cfg.iterations,
                depth=cfg.depth,
                learning_rate=cfg.learning_rate,
                l2_leaf_reg=cfg.l2_leaf_reg,
                random_seed=self.config.seed,
                thread_count=cfg.threads,
                verbose=False,
                allow_writing_files=False,
            )
        for head in CLASS_HEADS + VALUE_HEADS:
            y = frame[f"y_{head}"].to_numpy(float)
            if np.unique(y).size == 1:
                self.models[head] = float(y[0])
                continue
            if self.method == "catboost":
                cls = CatBoostClassifier if head in CLASS_HEADS else CatBoostRegressor
                model = cls(**params, loss_function="Logloss" if head in CLASS_HEADS else "RMSE")
                model.fit(x, y, sample_weight=weights)
            else:
                estimator = (
                    LogisticRegression(C=0.2, max_iter=1500, random_state=self.config.seed)
                    if head in CLASS_HEADS
                    else Ridge(alpha=cfg.ridge_alpha)
                )
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    estimator,
                )
                fit_kw = {f"{model.steps[-1][0]}__sample_weight": weights}
                model.fit(x, y, **fit_kw)
            self.models[head] = model
        for head in ("regret_bps", "stale_bps"):
            y = frame[f"y_{head}"].to_numpy(float)
            if self.method == "catboost" and np.unique(y).size > 1:
                model = CatBoostRegressor(
                    **params, loss_function=f"Quantile:alpha={1 - self.config.risk.alpha}"
                )
                model.fit(x, y, sample_weight=weights)
                self.quantile_models[head] = model
        return self

    def predict(self, frame):
        x = self.encoder.transform(frame)
        out = frame.copy()
        for head, model in self.models.items():
            if isinstance(model, float):
                values = np.full(len(frame), model)
            else:
                values = model.predict_proba(x)[:, 1] if head in CLASS_HEADS else model.predict(x)
            if head in ("regret_bps", "stale_bps"):
                values = np.maximum(0, values)
            out[f"pred_{head}"] = values
        for head in ("regret_bps", "stale_bps"):
            model = self.quantile_models.get(head)
            out[f"q_{head}"] = (
                np.maximum(0, model.predict(x)) if model is not None else out[f"pred_{head}"]
            )
        return out

    def importance(self):
        rows = []
        for head, model in self.models.items():
            if isinstance(model, float):
                continue
            weights = (
                model.feature_importances_
                if self.method == "catboost"
                else model.steps[-1][1].coef_.ravel()
            )
            rows.extend(
                {"head": head, "feature": feature, "importance": float(w)}
                for feature, w in zip(self.encoder.names(), weights)
            )
        return rows


class StatisticalPredictor:
    """Gaussian random walk / drift / AR(1) / ETS(A,N,N) on update log returns.

    Paths share a deterministic random stream between methods and dates. Future
    update weekdays are estimated from training only; unscheduled holidays are
    not claimed to be known. No future realized update dates enter the forecast.
    """

    def __init__(self, method, config):
        self.method, self.config, self.parameters = method, config, {}

    def fit(self, frame):
        cfg = self.config.model
        for iso, group in frame.groupby("iso"):
            g = group.sort_values("date").tail(cfg.statistical_window)
            r = g.ret_1.to_numpy(float)
            intercept, phi = float(np.mean(r)), 0.0
            if len(r) > 10:
                phi = float(
                    np.clip(
                        np.cov(r[:-1], r[1:], ddof=0)[0, 1] / max(np.var(r[:-1]), 1e-12),
                        -0.95,
                        0.95,
                    )
                )
                intercept = float(np.mean(r[1:]) - phi * np.mean(r[:-1]))
            residual = r[1:] - intercept - phi * r[:-1]
            frequencies = g.date.dt.dayofweek.value_counts(normalize=True)
            self.parameters[iso] = dict(
                mean=float(r.mean()),
                sigma=max(float(r.std()), 1e-6),
                ar_sigma=max(float(residual.std()), 1e-6),
                phi=phi,
                intercept=intercept,
                weekdays=set(frequencies.index[frequencies > 0.05]),
            )
        return self

    def predict(self, frame):
        rows = []
        tc, mc = self.config.targets, self.config.model
        h, opening = tc.primary_horizon, tc.opening_horizon
        length = max(tc.horizons) + 8
        for row in frame.itertuples(index=False):
            par = self.parameters[row.iso]
            seed = int.from_bytes(
                hashlib.blake2b(
                    f"{self.config.seed}|{row.iso}|{row.date}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            rng = np.random.default_rng(seed)
            noise = rng.normal(size=(mc.simulation_paths, length))
            price = float(row.rub_per_unit)
            paths = np.full((mc.simulation_paths, length + 1), price)
            previous_return = np.full(mc.simulation_paths, row.ret_1)
            first_update = None
            for day in range(1, length + 1):
                if (row.date + pd.Timedelta(days=day)).dayofweek not in par["weekdays"]:
                    paths[:, day] = paths[:, day - 1]
                    continue
                first_update = day if first_update is None else first_update
                if self.method == "ar1":
                    mean, sigma = (
                        row.stat_ar_intercept + row.stat_ar_phi * previous_return,
                        row.stat_ar_sigma,
                    )
                elif self.method == "random_walk_drift":
                    mean, sigma = row.stat_mean, row.stat_sigma
                elif self.method == "ets":
                    mean, sigma = row.ewma_return, row.stat_sigma
                else:
                    mean, sigma = 0.0, row.stat_sigma
                innovation = mean + sigma * noise[:, day - 1]
                paths[:, day] = paths[:, day - 1] * np.exp(np.clip(innovation, -0.5, 0.5))
                previous_return = innovation
            future = paths[:, 1 : h + 1]
            regret = np.maximum(0, price / np.minimum(price, future.min(axis=1)) - 1) * 10000
            stale = (
                np.maximum(0, 1 - price / np.maximum(price, paths[:, 1 : opening + 1].max(axis=1)))
                * 10000
            )
            local = (price / np.minimum(row.past_min_primary, future.min(axis=1)) - 1) * 10000
            gain = (1 - price / ((row.past_sum_primary + future.sum(axis=1)) / (2 * h + 1))) * 10000
            k = first_update or 1
            nxt = paths[:, k]
            regret_next = (
                np.maximum(0, nxt / np.minimum(nxt, paths[:, k + 1 : k + h + 1].min(axis=1)) - 1)
                * 10000
            )
            stale_next = (
                np.maximum(
                    0, 1 - nxt / np.maximum(nxt, paths[:, k + 1 : k + opening + 1].max(axis=1))
                )
                * 10000
            )
            wait = (
                -(1 - price / nxt) * 10000
                + self.config.risk.regret_penalty * (regret - regret_next)
                + self.config.risk.stale_penalty * (stale - stale_next)
            )
            rows.append(
                {
                    "pred_local_min": np.mean(local <= tc.near_min_bps),
                    "pred_no_regret": np.mean(regret <= tc.regret_tolerance_bps),
                    "pred_hold": np.mean(stale <= tc.hold_tolerance_bps),
                    "pred_close": np.mean((future[:, -1] / price - 1) * 10000 >= tc.closing_bps),
                    "pred_gain_bps": gain.mean(),
                    "pred_regret_bps": regret.mean(),
                    "pred_stale_bps": stale.mean(),
                    "pred_wait_delta_bps": wait.mean(),
                    "q_regret_bps": np.quantile(regret, 1 - self.config.risk.alpha),
                    "q_stale_bps": np.quantile(stale, 1 - self.config.risk.alpha),
                }
            )
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    def importance(self):
        return []


class RulePredictor:
    """Past-only empirical rates for diagnostics; rules do not masquerade as ML."""

    def __init__(self, method, config):
        self.method, self.config = method, config

    def fit(self, frame):
        self.global_means = {h: float(frame[f"y_{h}"].mean()) for h in CLASS_HEADS + VALUE_HEADS}
        self.season = {}
        for (iso, month), g in frame.groupby(["iso", frame.date.dt.month]):
            weight = len(g) / (len(g) + 60)
            self.season[(iso, month)] = (
                weight * g.y_gain_bps.mean() + (1 - weight) * self.global_means["gain_bps"]
            )
        return self

    def predict(self, frame):
        out = frame.copy()
        for head, mean in self.global_means.items():
            out[f"pred_{head}"] = mean
        out["seasonal_score"] = [
            self.season.get((r.iso, r.date.month), 0.0) for r in out.itertuples()
        ]
        out["q_regret_bps"] = out.pred_regret_bps
        out["q_stale_bps"] = out.pred_stale_bps
        return out

    def importance(self):
        return []


@dataclass
class ProbabilityMap:
    model: object = None
    constant: float | None = None

    @classmethod
    def fit(cls, probabilities, labels):
        y = np.asarray(labels, float)
        if np.unique(y).size < 2 or np.ptp(probabilities) < 1e-10:
            return cls(constant=float((y.sum() + 1) / (len(y) + 2)))
        x = logit(np.clip(probabilities, 1e-5, 1 - 1e-5)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, max_iter=500).fit(x, y)
        return cls(model=model)

    def predict(self, probabilities):
        if self.constant is not None:
            return np.full(len(probabilities), self.constant)
        x = logit(np.clip(probabilities, 1e-5, 1 - 1e-5)).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


@dataclass
class FittedModel:
    method: str
    predictor: object
    probability_maps: dict = field(default_factory=dict)
    risk_seed: list[dict] = field(default_factory=list)

    def predict(self, frame, calibrated=True):
        out = self.predictor.predict(frame)
        for head in CLASS_HEADS:
            col = f"pred_{head}"
            out[f"raw_{head}"] = out[col]
            if calibrated and head in self.probability_maps:
                for iso, ids in out.groupby("iso").groups.items():
                    mapping = self.probability_maps[head].get(
                        iso, self.probability_maps[head]["pooled"]
                    )
                    out.loc[ids, col] = mapping.predict(out.loc[ids, col].to_numpy())
        return out


def fit_model(method, train, calibration, columns, config):
    if method in ("catboost", "linear"):
        predictor = SupervisedPredictor(method, columns, config)
    elif method.startswith("rule_") or method == "random_policy":
        predictor = RulePredictor(method, config)
    else:
        predictor = StatisticalPredictor(method, config)
    predictor.fit(train)
    fitted = FittedModel(method, predictor)
    raw = predictor.predict(calibration.reset_index(drop=True))
    if not method.startswith("rule_") and method != "random_policy":
        for head in CLASS_HEADS:
            maps = {"pooled": ProbabilityMap.fit(raw[f"pred_{head}"].to_numpy(), raw[f"y_{head}"])}
            for iso, group in raw.groupby("iso"):
                if len(group) >= config.model.min_calibration_rows:
                    maps[iso] = ProbabilityMap.fit(
                        group[f"pred_{head}"].to_numpy(), group[f"y_{head}"]
                    )
            fitted.probability_maps[head] = maps
    for r in raw.itertuples():
        fitted.risk_seed.append(
            {
                "date": r.date,
                "iso": r.iso,
                "regime": r.regime,
                "regret_bps": r.y_regret_bps - r.q_regret_bps,
                "stale_bps": r.y_stale_bps - r.q_stale_bps,
            }
        )
    return fitted
