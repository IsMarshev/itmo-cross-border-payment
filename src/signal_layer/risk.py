"""Empirical one-sided conformal residual correction with delayed feedback.

Coverage is measured, not promised under arbitrary dependence or selection.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class RiskTracker:
    def __init__(self, seeds, config):
        self.config = config
        self.buffers = defaultdict(lambda: deque(maxlen=config.risk.residual_window))
        self.alpha = defaultdict(lambda: config.risk.alpha)
        for row in seeds:
            self._append(row)

    def _append(self, row):
        for head in ("regret_bps", "stale_bps"):
            for key in [(row["iso"], "all", head), (row["iso"], row["regime"], head)]:
                self.buffers[key].append(float(row[head]))

    def upper(self, row, head, enabled=True):
        base = float(row[f"q_{head}"])
        if not enabled:
            return max(0, base)
        key = (row["iso"], row["regime"], head)
        if len(self.buffers[key]) < self.config.risk.regime_min_samples:
            key = (row["iso"], "all", head)
        values = np.asarray(self.buffers[key])
        if len(values) < 10:
            return float("inf")
        # Finite-sample correction; higher order statistic, one-sided.
        level = min(1.0, np.ceil((len(values) + 1) * (1 - self.alpha[key])) / len(values))
        correction = float(np.quantile(values, level, method="higher"))
        return max(0, base + correction, float(row[f"pred_{head}"]))

    def observe(self, prediction, truth):
        residual = {"iso": prediction["iso"], "regime": prediction["regime"]}
        for head in ("regret_bps", "stale_bps"):
            value = float(truth[f"y_{head}"])
            if not np.isfinite(value):
                return
            residual[head] = value - prediction[f"q_{head}"]
            if self.config.risk.adaptive:
                missed = value > prediction[f"upper_{head}"]
                for key in [
                    (prediction["iso"], "all", head),
                    (prediction["iso"], prediction["regime"], head),
                ]:
                    self.alpha[key] = float(
                        np.clip(
                            self.alpha[key]
                            + self.config.risk.adaptation_rate * (self.config.risk.alpha - missed),
                            0.01,
                            0.40,
                        )
                    )
        self._append(residual)
