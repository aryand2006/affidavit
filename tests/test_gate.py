from __future__ import annotations

import math

import numpy as np
import pytest

from affidavit.cost import CostTilt, ar1_placebo, shuffle_placebo
from affidavit.gate import evaluate, swear
from affidavit.leak import probe_leak
from affidavit.ledger import DegreesOfFreedom, TrialLedger
from affidavit.sharpe import expected_max_sharpe, moments


def test_expected_max_sharpe_six_over_two_years():
    # Matches the assay README worked example: ~1.34
    assert expected_max_sharpe(6, 2) == pytest.approx(math.sqrt(2 * math.log(6) / 2), rel=1e-9)


def test_degrees_of_freedom_product():
    dof = DegreesOfFreedom().add("pairs", 6).add("entry", 4).add("exit", 3)
    assert dof.n_trials == 72


def test_selection_ex_ante_fails_risk_free():
    ledger = TrialLedger(
        name="demo",
        candidates={
            "a": 163000,
            "b": 89000,
            "c": -38000,
            "d": -72000,
            "e": -54000,
            "f": -54000,
        },
        selected=["a", "b", "c"],
        capital=1_000_000,
        years=2,
        risk_free_rate=0.065,
        benchmark_annual_return=0.12,
    )
    # reported ~10.7%/yr, ex-ante ~1.7%/yr
    assert ledger.reported_annual_return() == pytest.approx(0.107, abs=1e-3)
    assert ledger.ex_ante_annual_return() == pytest.approx(0.017, abs=1e-3)

    verdict = evaluate(ledger=ledger)
    by_check = {f.check: f for f in verdict.findings}
    assert by_check["selection"].severity == "fail"
    assert verdict.severity == "fail"
    assert verdict.sworn is False


def test_honest_ledger_passes_selection():
    ledger = TrialLedger(
        name="honest",
        candidates={"only": 200000},
        selected=["only"],
        capital=1_000_000,
        years=2,
        risk_free_rate=0.05,
    )
    verdict = evaluate(ledger=ledger)
    by_check = {f.check: f for f in verdict.findings}
    assert by_check["selection"].severity == "pass"


def test_leak_detects_full_sample_zscore():
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.normal(0, 1, size=200)) + 100

    def leaky(data, asof):
        # Uses the full-sample mean — classic lookahead.
        mu = float(np.mean(data))
        return float(data[asof] - mu)

    def clean(data, asof):
        hist = data[: asof + 1]
        mu = float(np.mean(hist))
        return float(data[asof] - mu)

    dirty = probe_leak(leaky, prices)
    honest = probe_leak(clean, prices)
    assert dirty.leaks, "full-sample mean must fail prefix invariance"
    assert honest.clean


def test_cost_tilt_break_even():
    # Constant 10bp/period gross edge, 100% turnover → dies near 10bp cost.
    r = np.full(252, 0.001)
    tilt = CostTilt(r, turnover=1.0, bps_grid=(0, 5, 10, 15))
    be = tilt.break_even_bps()
    assert be is not None
    assert 8 <= be <= 12


def test_placebo_edge_beats_shuffle():
    rng = np.random.default_rng(1)
    # Strong drift — should beat shuffle null.
    r = rng.normal(0.002, 0.01, size=500)
    result = shuffle_placebo(r, n_placebos=200, seed=2)
    assert result.percentile >= 0.95


def test_placebo_noise_fails():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=500)
    result = ar1_placebo(r, n_placebos=200, seed=4)
    assert result.percentile < 0.95


def test_swear_is_stable_hash():
    ledger = TrialLedger(
        name="hash",
        candidates={"a": 1.0, "b": -1.0},
        selected=["a"],
        capital=100.0,
        years=1.0,
        risk_free_rate=0.0,
    )
    v1 = evaluate(ledger=ledger)
    v2 = evaluate(ledger=ledger)
    a1 = swear(v1, subject="hash")
    a2 = swear(v2, subject="hash")
    assert a1["content_sha256"] == a2["content_sha256"]
    assert a1["verdict"] == "fail" or a1["verdict"] in {"fail", "warn", "pass"}


def test_moments_sharpe():
    r = np.array([0.01, -0.005, 0.002, 0.003, -0.001])
    m = moments(r)
    assert m.n == 5
    assert m.std > 0
    assert math.isfinite(m.sharpe)
