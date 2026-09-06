"""affidavit — swear whether a strategy result survives falsification.

An affidavit is a sworn statement of fact. Before capital moves, a backtest
result needs one: not a vibe, not a screenshot of an equity curve, but a
record of every check that was run, every free choice that entered the search,
and a single PASS / WARN / FAIL that shows its working.

This package is the gate. It does not invent strategies. It attacks them.
"""

from __future__ import annotations

from .gate import Finding, Verdict, evaluate
from .ledger import DegreesOfFreedom, TrialLedger

__all__ = [
    "DegreesOfFreedom",
    "Finding",
    "TrialLedger",
    "Verdict",
    "evaluate",
]

__version__ = "0.1.0"
