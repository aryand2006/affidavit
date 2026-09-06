"""Sharpe utilities used by the gate.

Kept small and explicit. Every formula here is something a reader should be
able to recompute from the report without opening another library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Moments:
    n: int
    mean: float
    std: float
    skew: float
    kurtosis: float  # excess kurtosis

    @property
    def sharpe(self) -> float:
        if self.std <= 0 or self.n < 2:
            return float("nan")
        return self.mean / self.std

    def annualized_sharpe(self, periods_per_year: int = TRADING_DAYS) -> float:
        s = self.sharpe
        if math.isnan(s):
            return float("nan")
        return s * math.sqrt(periods_per_year)


def moments(returns: np.ndarray) -> Moments:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n == 0:
        return Moments(0, float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    if n < 3 or std == 0:
        return Moments(n, mean, std, float("nan"), float("nan"))
    centered = r - mean
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = m3 / (std**3)
    kurtosis = m4 / (std**4) - 3.0
    return Moments(n, mean, std, skew, kurtosis)


def expected_max_sharpe(n_trials: int, years: float) -> float:
    """Noise-floor Sharpe from searching N candidates over T years.

    E[max SR] ≈ sqrt(2 · ln(N) / T) under the usual IID null. Logarithmic in N,
    brutal in short samples.
    """
    if n_trials < 1 or years <= 0:
        return float("nan")
    return math.sqrt(2.0 * math.log(n_trials) / years)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 0.0,
) -> float:
    """Bailey & López de Prado Probabilistic Sharpe Ratio."""
    if n < 2 or not math.isfinite(observed_sr):
        return float("nan")
    skew = 0.0 if not math.isfinite(skew) else skew
    kurtosis = 0.0 if not math.isfinite(kurtosis) else kurtosis
    denom = 1.0 - skew * observed_sr + ((kurtosis + 3.0) / 4.0) * observed_sr**2
    if denom <= 0:
        return float("nan")
    se = math.sqrt(denom / (n - 1))
    if se == 0:
        return float("nan")
    z = (observed_sr - benchmark_sr) / se
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 0.0,
    years: float | None = None,
) -> float:
    """PSR against the expected maximum Sharpe under N trials."""
    if years is None:
        years = n / TRADING_DAYS if n else float("nan")
    bench = expected_max_sharpe(n_trials, years)
    return probabilistic_sharpe_ratio(observed_sr, bench, n, skew, kurtosis)


def min_track_record_length(
    observed_sr: float,
    skew: float = 0.0,
    kurtosis: float = 0.0,
    confidence: float = CONFIDENCE,
) -> float:
    """Minimum years to claim observed_sr > 0 at the given confidence."""
    if not math.isfinite(observed_sr) or observed_sr <= 0:
        return float("inf")
    skew = 0.0 if not math.isfinite(skew) else skew
    kurtosis = 0.0 if not math.isfinite(kurtosis) else kurtosis
    # Inverse erf for one-sided normal quantile.
    z = math.sqrt(2.0) * _erfinv(2.0 * confidence - 1.0)
    denom = observed_sr**2
    if denom == 0:
        return float("inf")
    numer = (1.0 - skew * observed_sr + ((kurtosis + 3.0) / 4.0) * observed_sr**2) * (z**2)
    return numer / denom


def _erfinv(x: float) -> float:
    # Abramowitz & Stegun approximation, good enough for gate thresholds.
    a = 0.147
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return float(math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), x))
