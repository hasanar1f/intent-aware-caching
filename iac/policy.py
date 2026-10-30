from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iac.stats import OnlineReuseStats

DEFAULT_HORIZON_S = 40.0
DEFAULT_SHRINKAGE = 20.0
DEFAULT_WINDOW = 50_000
DEFAULT_UPDATE_INTERVAL_S = 60.0
_SQRT2 = math.sqrt(2.0)


def lognormal_sf(x: float, mu: float, sigma: float) -> float:
    if x <= 0.0:
        return 1.0
    if sigma <= 0.0:
        return 0.0 if x >= math.exp(mu) else 1.0
    return 0.5 * math.erfc((math.log(x) - mu) / (sigma * _SQRT2))


def reuse_within(p_next: float, mu: float, sigma: float, age: float, horizon: float) -> float:
    age = max(age, 0.0)
    s_now = lognormal_sf(age, mu, sigma)
    s_later = lognormal_sf(age + horizon, mu, sigma)
    den = (1.0 - p_next) + p_next * s_now
    return p_next * (s_now - s_later) / den if den > 0.0 else 0.0


class IACPolicy:
    def __init__(self, stats: OnlineReuseStats, horizon: float = DEFAULT_HORIZON_S):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.stats = stats
        self.horizon = horizon

    def lookup(self, intent: str, turn: int) -> tuple[float, float, float]:
        return self.stats.lookup_dist(intent, turn)

    def priority(self, intent: str, turn: int, age: float, size: int) -> tuple[float, int]:
        p_next, mu, sigma = self.lookup(intent, turn)
        return reuse_within(p_next, mu, sigma, age, self.horizon), size
