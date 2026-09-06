"""The deployment gate: every check, one verdict, one sworn artifact.

Each check answers a question that has to be answered before capital moves, and
each one states the number that decided it. A gate that says "fail" without
showing its working gets overridden the first time it is inconvenient.

Severity is deliberately coarse. `fail` means the result is not distinguishable
from the search, the leak, or the null that produced it. `warn` means it
survives but rests on something fragile. `pass` means the check found nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from .cost import CostTilt, ar1_placebo, shuffle_placebo
from .leak import DecisionFn, probe_leak
from .ledger import DegreesOfFreedom, TrialLedger
from .sharpe import (
    TRADING_DAYS,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
)

CONCENTRATION_LIMIT = 0.6
COST_SENSITIVITY_WARN = 0.5
COST_SENSITIVITY_FAIL = 1.0


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str  # pass | warn | fail | skip
    headline: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "headline": self.headline,
            "detail": self.detail,
        }


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def severity(self) -> str:
        ranks = {"fail": 3, "warn": 2, "pass": 1, "skip": 0}
        if not self.findings:
            return "skip"
        return max(self.findings, key=lambda f: ranks.get(f.severity, 0)).severity

    @property
    def sworn(self) -> bool:
        """True only when every applicable check passed (no fail, no warn)."""
        applicable = [f for f in self.findings if f.severity != "skip"]
        return bool(applicable) and all(f.severity == "pass" for f in applicable)

    def to_dict(self) -> dict:
        return {
            "verdict": self.severity,
            "sworn": self.sworn,
            "findings": [f.to_dict() for f in self.findings],
            "stats": self.stats,
        }


def _fmt_pct(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def evaluate(
    *,
    ledger: TrialLedger | None = None,
    dof: DegreesOfFreedom | None = None,
    returns: np.ndarray | None = None,
    turnover: float | None = None,
    decision: DecisionFn | None = None,
    decision_data: np.ndarray | None = None,
    periods_per_year: int = TRADING_DAYS,
    placebo_n: int = 400,
    placebo_seed: int = 0,
    subject: str = "strategy",
) -> Verdict:
    """Run every applicable check. Missing inputs are skipped, never guessed."""
    verdict = Verdict()
    stats: dict[str, Any] = {"subject": subject}

    # --- selection / ex-ante reconstitution ---
    if ledger is None:
        verdict.findings.append(
            Finding("selection", "skip", "no trial ledger provided", "skipped")
        )
    else:
        stats["ledger"] = ledger.to_dict()
        reported = ledger.reported_annual_return()
        ex_ante = ledger.ex_ante_annual_return()
        if reported is not None and ex_ante is not None:
            drop = reported - ex_ante
            if ex_ante <= ledger.risk_free_rate:
                sev = "fail"
                headline = (
                    f"ex-ante { _fmt_pct(ex_ante) }/yr does not clear "
                    f"risk-free { _fmt_pct(ledger.risk_free_rate) }"
                )
            elif drop > abs(reported) * 0.5 and ledger.n_kept < ledger.n_tested:
                sev = "fail"
                headline = (
                    f"selection inflated return from {_fmt_pct(ex_ante)}/yr "
                    f"to {_fmt_pct(reported)}/yr"
                )
            elif ledger.n_kept < ledger.n_tested:
                sev = "warn"
                headline = (
                    f"kept {ledger.n_kept}/{ledger.n_tested}; "
                    f"reported {_fmt_pct(reported)}/yr vs ex-ante {_fmt_pct(ex_ante)}/yr"
                )
            else:
                sev = "pass"
                headline = f"ex-ante matches reported at {_fmt_pct(reported)}/yr"
            verdict.findings.append(
                Finding(
                    "selection",
                    sev,
                    headline,
                    detail=(
                        f"tested={ledger.n_tested} kept={ledger.n_kept} "
                        f"reported_pnl={ledger.reported_pnl:.0f} "
                        f"ex_ante_pnl={ledger.ex_ante_pnl:.0f}"
                    ),
                )
            )
        else:
            if ledger.n_kept < ledger.n_tested:
                verdict.findings.append(
                    Finding(
                        "selection",
                        "warn",
                        f"kept {ledger.n_kept}/{ledger.n_tested} without years/capital to annualize",
                        "provide years and capital for the money-denominated check",
                    )
                )
            else:
                verdict.findings.append(
                    Finding("selection", "pass", "no selection among candidates", "")
                )

        conc = ledger.concentration()
        if conc is None:
            verdict.findings.append(
                Finding("concentration", "skip", "no positive contributor profits", "")
            )
        elif conc >= CONCENTRATION_LIMIT:
            verdict.findings.append(
                Finding(
                    "concentration",
                    "fail" if conc >= 0.75 else "warn",
                    f"{_fmt_pct(conc)} of profit from a single candidate",
                    f"limit={CONCENTRATION_LIMIT:.0%}",
                )
            )
        else:
            verdict.findings.append(
                Finding(
                    "concentration",
                    "pass",
                    f"max contributor share {_fmt_pct(conc)}",
                    "",
                )
            )

        if ledger.benchmark_annual_return is not None and ex_ante is not None:
            if ex_ante < ledger.benchmark_annual_return:
                verdict.findings.append(
                    Finding(
                        "benchmark",
                        "fail",
                        f"ex-ante {_fmt_pct(ex_ante)}/yr below benchmark "
                        f"{_fmt_pct(ledger.benchmark_annual_return)}",
                        "",
                    )
                )
            else:
                verdict.findings.append(
                    Finding(
                        "benchmark",
                        "pass",
                        f"ex-ante clears benchmark {_fmt_pct(ledger.benchmark_annual_return)}",
                        "",
                    )
                )

    # --- degrees of freedom / noise floor ---
    n_trials = None
    years = None
    if dof is not None:
        n_trials = dof.n_trials
        stats["degrees_of_freedom"] = dof.to_dict()
    elif ledger is not None:
        n_trials = ledger.n_tested
    if ledger is not None:
        years = ledger.years

    if n_trials is None or years is None or years <= 0:
        verdict.findings.append(
            Finding(
                "degrees",
                "skip",
                "need trial count and years for the noise floor",
                "provide DegreesOfFreedom or a ledger with years",
            )
        )
    else:
        floor = expected_max_sharpe(n_trials, years)
        stats["noise_floor_sharpe"] = floor
        headline = (
            f"search of {n_trials} trials over {years:g}y can manufacture "
            f"Sharpe {_fmt_num(floor)} from noise"
        )
        # Without observed SR we only report the floor as context (warn if brutal).
        sev = "warn" if floor >= 1.0 else "pass"
        if returns is not None:
            m = moments(np.asarray(returns, dtype=float))
            ann = m.annualized_sharpe(periods_per_year)
            stats["observed_sharpe"] = ann
            dsr = deflated_sharpe_ratio(
                m.sharpe, n_trials, m.n, m.skew, m.kurtosis, years=years
            )
            stats["deflated_sharpe_psr"] = dsr
            if not math.isfinite(ann) or ann <= floor:
                sev = "fail"
                headline = (
                    f"observed Sharpe {_fmt_num(ann)} does not clear "
                    f"noise floor {_fmt_num(floor)} (N={n_trials}, T={years:g}y)"
                )
            elif dsr < 0.95:
                sev = "warn"
                headline = (
                    f"observed Sharpe {_fmt_num(ann)} clears floor {_fmt_num(floor)} "
                    f"but deflated PSR={_fmt_num(dsr)}"
                )
            else:
                sev = "pass"
                headline = (
                    f"observed Sharpe {_fmt_num(ann)} clears floor {_fmt_num(floor)} "
                    f"(deflated PSR={_fmt_num(dsr)})"
                )
        verdict.findings.append(Finding("degrees", sev, headline, ""))

    # --- track record ---
    if returns is None:
        verdict.findings.append(
            Finding("track_record", "skip", "no return series provided", "")
        )
    else:
        r = np.asarray(returns, dtype=float).ravel()
        m = moments(r)
        ann = m.annualized_sharpe(periods_per_year)
        psr = probabilistic_sharpe_ratio(m.sharpe, 0.0, m.n, m.skew, m.kurtosis)
        min_trl = min_track_record_length(m.sharpe, m.skew, m.kurtosis)
        stats["track_record"] = {
            "n": m.n,
            "annualized_sharpe": ann,
            "psr_vs_zero": psr,
            "min_track_record_years": min_trl,
        }
        years_obs = m.n / periods_per_year if periods_per_year else float("nan")
        if not math.isfinite(psr) or psr < 0.95:
            sev = "fail"
            headline = f"Sharpe {_fmt_num(ann)} not distinguishable from zero (PSR={_fmt_num(psr)})"
        elif math.isfinite(min_trl) and years_obs < min_trl:
            sev = "warn"
            headline = (
                f"need ~{_fmt_num(min_trl, 1)}y to trust Sharpe {_fmt_num(ann)}; "
                f"have {_fmt_num(years_obs, 1)}y"
            )
        else:
            sev = "pass"
            headline = f"Sharpe {_fmt_num(ann)} clears zero (PSR={_fmt_num(psr)})"
        verdict.findings.append(Finding("track_record", sev, headline, ""))

    # --- cost tilt ---
    if returns is None or turnover is None:
        verdict.findings.append(
            Finding(
                "cost_tilt",
                "skip",
                "need returns and turnover for cost tilt",
                "",
            )
        )
    else:
        tilt = CostTilt(returns, turnover=float(turnover), periods_per_year=periods_per_year)
        stats["cost_tilt"] = tilt.to_dict()
        be = tilt.break_even_bps()
        sens = tilt.sensitivity_ratio()
        if be is not None and be < 2.0:
            sev = "fail"
            headline = f"edge dies by {_fmt_num(be, 1)}bp round-trip"
        elif sens is not None and sens >= COST_SENSITIVITY_FAIL:
            sev = "fail"
            headline = f"cost sensitivity {_fmt_num(sens)} — result is a cost-model artifact"
        elif sens is not None and sens >= COST_SENSITIVITY_WARN:
            sev = "warn"
            headline = (
                f"cost sensitivity {_fmt_num(sens)}; break-even {_fmt_num(be, 1)}bp"
            )
        else:
            sev = "pass"
            headline = (
                f"break-even {_fmt_num(be, 1)}bp; sensitivity {_fmt_num(sens)}"
            )
        verdict.findings.append(Finding("cost_tilt", sev, headline, f"turnover={turnover}"))

    # --- placebos ---
    if returns is None:
        verdict.findings.append(
            Finding("placebo", "skip", "no return series for placebo battery", "")
        )
    else:
        shuf = shuffle_placebo(
            returns, n_placebos=placebo_n, seed=placebo_seed, periods_per_year=periods_per_year
        )
        ar1 = ar1_placebo(
            returns, n_placebos=placebo_n, seed=placebo_seed + 1, periods_per_year=periods_per_year
        )
        stats["placebo"] = {"shuffle": shuf.to_dict(), "ar1": ar1.to_dict()}
        worst_pct = min(
            x for x in (shuf.percentile, ar1.percentile) if math.isfinite(x)
        ) if any(math.isfinite(x) for x in (shuf.percentile, ar1.percentile)) else float("nan")
        if not math.isfinite(worst_pct) or worst_pct < 0.5:
            sev = "fail"
            headline = (
                f"fails placebo battery (shuffle pctl={_fmt_num(shuf.percentile)}, "
                f"AR1 pctl={_fmt_num(ar1.percentile)})"
            )
        elif worst_pct < 0.95:
            sev = "warn"
            headline = (
                f"beats half the nulls but not 95% "
                f"(shuffle={_fmt_num(shuf.percentile)}, AR1={_fmt_num(ar1.percentile)})"
            )
        else:
            sev = "pass"
            headline = (
                f"clears 95% placebo threshold "
                f"(shuffle={_fmt_num(shuf.percentile)}, AR1={_fmt_num(ar1.percentile)})"
            )
        verdict.findings.append(Finding("placebo", sev, headline, ""))

    # --- leak probe ---
    if decision is None or decision_data is None:
        verdict.findings.append(
            Finding("leak", "skip", "no decision function / data for leak probe", "")
        )
    else:
        leak = probe_leak(decision, np.asarray(decision_data))
        stats["leak"] = leak.to_dict()
        if leak.errors and not leak.leaks:
            verdict.findings.append(
                Finding(
                    "leak",
                    "warn",
                    f"probe errors on {len(leak.errors)} date(s), no confirmed leak",
                    "; ".join(leak.errors[:3]),
                )
            )
        elif leak.leaks:
            horizons = [lk.horizon for lk in leak.leaks if lk.horizon is not None]
            detail = ", ".join(lk.describe() for lk in leak.leaks[:5])
            sev = "fail"
            headline = f"{len(leak.leaks)} lookahead leak(s) across {leak.tested} probes"
            if horizons:
                finite = [h for h in horizons if h >= 0]
                if finite:
                    headline += f"; median horizon {int(np.median(finite))} row(s)"
            verdict.findings.append(Finding("leak", sev, headline, detail))
        else:
            verdict.findings.append(
                Finding(
                    "leak",
                    "pass",
                    f"prefix-invariant across {leak.tested} probes",
                    "",
                )
            )

    verdict.stats = stats
    return verdict


def swear(verdict: Verdict, *, issuer: str = "affidavit", subject: str | None = None) -> dict:
    """Build the sworn artifact: verdict + content hash.

    The hash covers the canonical JSON of findings+stats so a later reader can
    detect tampering or silent re-runs with different inputs.
    """
    body = verdict.to_dict()
    if subject:
        body["stats"] = {**body.get("stats", {}), "subject": subject}
    canonical = json.dumps(body, sort_keys=True, default=_json_default)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "affidavit_version": "0.1.0",
        "issuer": issuer,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": digest,
        **body,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
