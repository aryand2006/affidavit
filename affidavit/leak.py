"""Lookahead probe via prefix invariance.

Run a decision function twice at the same date: once with the full series,
once with everything after that date hidden. Code that only uses information
available at the time cannot tell the difference.

This is black-box. It needs no instrumentation of the strategy beyond being
callable as `decision(data, asof_index) -> array-like`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

DecisionFn = Callable[[np.ndarray, int], Any]


def outputs_match(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(outputs_match(a[k], b[k], tolerance) for k in a)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(outputs_match(x, y, tolerance) for x, y in zip(a, b))

    try:
        left = np.asarray(a, dtype=float)
        right = np.asarray(b, dtype=float)
        if left.shape == right.shape:
            return bool(np.allclose(left, right, atol=tolerance, rtol=0, equal_nan=True))
    except (TypeError, ValueError):
        pass

    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(
        b, (int, float, np.floating, np.integer)
    ):
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(float(a) - float(b)) <= tolerance

    return bool(a == b)


def truncate(data: np.ndarray, asof: int) -> np.ndarray:
    """Hide everything strictly after `asof`."""
    if asof < 0 or asof >= len(data):
        raise IndexError(f"asof={asof} out of range for length {len(data)}")
    return np.asarray(data[: asof + 1]).copy()


@dataclass
class Leak:
    asof: int
    with_future: Any
    without_future: Any
    horizon: int | None = None

    def describe(self) -> str:
        if self.horizon is None:
            return f"asof={self.asof}"
        if self.horizon < 0:
            return f"asof={self.asof}, needs the rest of the sample"
        return f"asof={self.asof}, needs {self.horizon} future row(s)"


@dataclass
class LeakReport:
    tested: int = 0
    leaks: list[Leak] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.leaks and not self.errors

    def to_dict(self) -> dict:
        return {
            "tested": self.tested,
            "n_leaks": len(self.leaks),
            "leaks": [leak.describe() for leak in self.leaks],
            "errors": list(self.errors),
            "clean": self.clean,
        }


def _horizon_search(
    decision: DecisionFn,
    data: np.ndarray,
    asof: int,
    with_future: Any,
    tolerance: float,
) -> int:
    """Smallest future window that reproduces the uncensored answer.

    Returns -1 if even the full remaining sample is required in a way that
    binary search cannot isolate to a finite prefix (effectively unbounded).
    """
    remaining = len(data) - asof - 1
    if remaining <= 0:
        return 0

    lo, hi = 1, remaining
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        window = np.asarray(data[: asof + 1 + mid]).copy()
        try:
            ans = decision(window, asof)
        except Exception:
            lo = mid + 1
            continue
        if outputs_match(ans, with_future, tolerance):
            found = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return found if found is not None else -1


def probe_leak(
    decision: DecisionFn,
    data: np.ndarray,
    asof_indices: list[int] | None = None,
    tolerance: float = 1e-9,
    measure_horizon: bool = True,
) -> LeakReport:
    """Differential lookahead test across decision dates."""
    data = np.asarray(data)
    if data.ndim != 1:
        # Strategies may take 2d panels; we only require row-major time on axis 0.
        if data.ndim < 1:
            raise ValueError("data must be array-like with a time axis")
    n = len(data)
    if asof_indices is None:
        if n < 5:
            asof_indices = list(range(max(n - 1, 0)))
        else:
            # Spread probes across the sample; dense enough to catch systemic leaks.
            step = max(n // 8, 1)
            asof_indices = list(range(step, n - 1, step))
            if (n - 2) not in asof_indices:
                asof_indices.append(n - 2)

    report = LeakReport()
    for asof in asof_indices:
        if asof < 0 or asof >= n - 1:
            continue
        report.tested += 1
        try:
            with_future = decision(data, asof)
            censored = truncate(data if data.ndim == 1 else data, asof)
            # For multi-dim, truncate along axis 0.
            if data.ndim > 1:
                censored = np.asarray(data[: asof + 1]).copy()
            without_future = decision(censored, asof)
        except Exception as exc:  # noqa: BLE001 — black-box probe must survive bad strategies
            report.errors.append(f"asof={asof}: {exc}")
            continue

        if outputs_match(with_future, without_future, tolerance):
            continue

        horizon = None
        if measure_horizon:
            horizon = _horizon_search(decision, data, asof, with_future, tolerance)
        report.leaks.append(
            Leak(
                asof=asof,
                with_future=with_future,
                without_future=without_future,
                horizon=horizon,
            )
        )
    return report
