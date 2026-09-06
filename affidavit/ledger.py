"""Degrees-of-freedom ledger.

Selection bias is not an abstract N. It is every free choice that entered the
search: parameters, universes, lookbacks, features, cost assumptions, the day
you decided to stop looking. An affidavit that does not count those choices is
swearing to a fiction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FreeChoice:
    name: str
    levels: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.levels < 1:
            raise ValueError(f"{self.name}: levels must be >= 1")


@dataclass
class DegreesOfFreedom:
    """Explicit product of free choices that defined the search."""

    choices: list[FreeChoice] = field(default_factory=list)

    def add(self, name: str, levels: int, note: str = "") -> DegreesOfFreedom:
        self.choices.append(FreeChoice(name, levels, note))
        return self

    @property
    def n_trials(self) -> int:
        if not self.choices:
            return 1
        n = 1
        for c in self.choices:
            n *= c.levels
        return n

    def to_dict(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "choices": [
                {"name": c.name, "levels": c.levels, "note": c.note} for c in self.choices
            ],
        }


@dataclass
class TrialLedger:
    """Named candidates that were tested, and which ones were kept.

    The blunt check: you tested N things and kept K. What would running all N
    have returned? That number is denominated in money, not in probability, and
    it usually lands harder than the theory does.
    """

    name: str
    candidates: dict[str, float]
    selected: list[str]
    capital: float = 1.0
    years: float | None = None
    risk_free_rate: float = 0.0
    benchmark_annual_return: float | None = None

    def __post_init__(self) -> None:
        missing = [s for s in self.selected if s not in self.candidates]
        if missing:
            raise ValueError(f"selected candidates not in ledger: {missing}")

    @property
    def n_tested(self) -> int:
        return len(self.candidates)

    @property
    def n_kept(self) -> int:
        return len(self.selected)

    @property
    def reported_pnl(self) -> float:
        return float(sum(self.candidates[s] for s in self.selected))

    @property
    def ex_ante_pnl(self) -> float:
        return float(sum(self.candidates.values()))

    def reported_annual_return(self) -> float | None:
        if self.years is None or self.years <= 0 or self.capital <= 0:
            return None
        return self.reported_pnl / (self.capital * self.years)

    def ex_ante_annual_return(self) -> float | None:
        if self.years is None or self.years <= 0 or self.capital <= 0:
            return None
        return self.ex_ante_pnl / (self.capital * self.years)

    def concentration(self) -> float | None:
        if not self.selected:
            return None
        profits = [max(self.candidates[s], 0.0) for s in self.selected]
        total = sum(profits)
        if total <= 0:
            return None
        return max(profits) / total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_tested": self.n_tested,
            "n_kept": self.n_kept,
            "reported_pnl": self.reported_pnl,
            "ex_ante_pnl": self.ex_ante_pnl,
            "reported_annual_return": self.reported_annual_return(),
            "ex_ante_annual_return": self.ex_ante_annual_return(),
            "concentration": self.concentration(),
            "capital": self.capital,
            "years": self.years,
            "risk_free_rate": self.risk_free_rate,
            "benchmark_annual_return": self.benchmark_annual_return,
            "candidates": dict(self.candidates),
            "selected": list(self.selected),
        }


def combinatorial_trials(*level_counts: int) -> int:
    """Product of level counts; empty product is 1."""
    n = 1
    for k in level_counts:
        if k < 1:
            raise ValueError("each factor must be >= 1")
        n *= k
    return n


def shannon_bits(n_trials: int) -> float:
    """log2 of the search size — how many bits of freedom the search spent."""
    if n_trials < 1:
        return float("nan")
    return math.log2(n_trials)
