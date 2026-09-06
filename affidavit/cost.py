"""Cost tilt and placebo batteries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sharpe import TRADING_DAYS, moments


@dataclass(frozen=True)
class CostTilt:
    """How fast the edge dies as costs rise.

    `gross_returns` are per-period strategy returns before costs.
    `turnover` is the fraction of NAV traded per period (0.02 = 2%/period).
    Costs are applied as `turnover * bps / 1e4` subtracted from each period.
    """

    gross_returns: np.ndarray
    turnover: float
    bps_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
    periods_per_year: int = TRADING_DAYS

    def net_returns(self, bps: float) -> np.ndarray:
        r = np.asarray(self.gross_returns, dtype=float).ravel()
        drag = self.turnover * (bps / 1e4)
        return r - drag

    def sharpe_curve(self) -> dict[float, float]:
        out: dict[float, float] = {}
        for bps in self.bps_grid:
            m = moments(self.net_returns(bps))
            out[float(bps)] = m.annualized_sharpe(self.periods_per_year)
        return out

    def mean_curve(self) -> dict[float, float]:
        out: dict[float, float] = {}
        for bps in self.bps_grid:
            r = self.net_returns(bps)
            out[float(bps)] = float(np.mean(r)) if r.size else float("nan")
        return out

    def break_even_bps(self) -> float | None:
        """Cost where edge crosses zero.

        Prefer Sharpe-cross when volatility is present. Fall back to mean
        return when the series is constant (Sharpe undefined).
        """
        curve = self.sharpe_curve()
        items = [(b, s) for b, s in sorted(curve.items()) if np.isfinite(s)]
        if len(items) >= 2:
            crossed = self._cross_zero(items)
            if crossed is not None:
                return crossed
        means = [(b, m) for b, m in sorted(self.mean_curve().items()) if np.isfinite(m)]
        return self._cross_zero(means)

    @staticmethod
    def _cross_zero(items: list[tuple[float, float]]) -> float | None:
        if not items:
            return None
        if items[0][1] <= 0:
            return float(items[0][0])
        for (b0, s0), (b1, s1) in zip(items, items[1:]):
            if s0 > 0 >= s1:
                if s0 == s1:
                    return float(b1)
                return float(b0 + (b1 - b0) * (s0 / (s0 - s1)))
        return None

    def sensitivity_ratio(self) -> float | None:
        """|ΔSharpe| between 0bp and 10bp, relative to |Sharpe@0|.

        Large values mean the result is an artifact of the cost assumption.
        """
        curve = self.sharpe_curve()
        s0 = curve.get(0.0)
        s10 = curve.get(10.0)
        if s0 is None or s10 is None:
            return None
        if not np.isfinite(s0) or abs(s0) < 1e-12:
            return None
        return abs(s10 - s0) / abs(s0)

    def to_dict(self) -> dict:
        return {
            "turnover": self.turnover,
            "sharpe_curve": self.sharpe_curve(),
            "break_even_bps": self.break_even_bps(),
            "sensitivity_ratio": self.sensitivity_ratio(),
        }


@dataclass(frozen=True)
class PlaceboResult:
    observed_sharpe: float
    null_mean: float
    null_p95: float
    percentile: float
    n_placebos: int
    kind: str

    @property
    def beats_null(self) -> bool:
        return self.percentile >= 0.95

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "observed_sharpe": self.observed_sharpe,
            "null_mean": self.null_mean,
            "null_p95": self.null_p95,
            "percentile": self.percentile,
            "n_placebos": self.n_placebos,
            "beats_null": self.beats_null,
        }


def _ann_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    return moments(returns).annualized_sharpe(periods_per_year)


def shuffle_placebo(
    returns: np.ndarray,
    n_placebos: int = 500,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS,
) -> PlaceboResult:
    """IID permutation null: destroy temporal structure, keep the marginal."""
    r = np.asarray(returns, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    observed = _ann_sharpe(r, periods_per_year)
    null = np.empty(n_placebos, dtype=float)
    for i in range(n_placebos):
        null[i] = _ann_sharpe(rng.permutation(r), periods_per_year)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return PlaceboResult(observed, float("nan"), float("nan"), float("nan"), 0, "shuffle")
    percentile = float(np.mean(null <= observed))
    return PlaceboResult(
        observed_sharpe=float(observed),
        null_mean=float(np.mean(null)),
        null_p95=float(np.quantile(null, 0.95)),
        percentile=percentile,
        n_placebos=int(null.size),
        kind="shuffle",
    )


def ar1_placebo(
    returns: np.ndarray,
    n_placebos: int = 500,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS,
) -> PlaceboResult:
    """AR(1)-matched null: preserve lag-1 autocorrelation, kill higher structure."""
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n < 3:
        return PlaceboResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, "ar1")

    mean = float(np.mean(r))
    centered = r - mean
    denom = float(np.dot(centered[:-1], centered[:-1]))
    phi = float(np.dot(centered[:-1], centered[1:]) / denom) if denom > 0 else 0.0
    phi = float(np.clip(phi, -0.99, 0.99))
    resid = centered[1:] - phi * centered[:-1]
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else float(np.std(centered, ddof=1))

    rng = np.random.default_rng(seed)
    observed = _ann_sharpe(r, periods_per_year)
    null = np.empty(n_placebos, dtype=float)
    for i in range(n_placebos):
        eps = rng.normal(0.0, sigma, size=n)
        sim = np.empty(n, dtype=float)
        sim[0] = eps[0]
        for t in range(1, n):
            sim[t] = phi * sim[t - 1] + eps[t]
        null[i] = _ann_sharpe(sim + mean, periods_per_year)

    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return PlaceboResult(observed, float("nan"), float("nan"), float("nan"), 0, "ar1")
    percentile = float(np.mean(null <= observed))
    return PlaceboResult(
        observed_sharpe=float(observed),
        null_mean=float(np.mean(null)),
        null_p95=float(np.quantile(null, 0.95)),
        percentile=percentile,
        n_placebos=int(null.size),
        kind="ar1",
    )
