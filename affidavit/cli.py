"""CLI for affidavit.

    affidavit check --ledger examples/cointegration-pairs.json
    affidavit check --ledger run.json --returns returns.csv --turnover 0.05
    affidavit explain
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .gate import evaluate, swear
from .ledger import DegreesOfFreedom, TrialLedger
from .report import render


EXPLAIN = """
affidavit asks six questions before a result is allowed to call itself deployable.

  selection      You tested N and kept K. What would running all N have returned?
  concentration  Is this a portfolio, or one bet with decorations?
  degrees        What Sharpe can this search manufacture from pure noise?
  track_record   Is the Sharpe distinguishable from zero on this sample?
  cost_tilt      How fast does the edge die as costs rise?
  placebo        Does the result beat skill-free shuffle and AR(1) nulls?
  leak           Does hiding the future change the decision function's answer?
  benchmark      Does the ex-ante result clear the index (when provided)?

Checks whose inputs are absent are skipped and reported as skipped, never
quietly assumed. A gate that guesses is a gate that gets overridden.

A result is sworn only when every applicable check passes.
""".strip()


def _load_ledger(path: Path) -> tuple[TrialLedger, DegreesOfFreedom | None]:
    raw = json.loads(path.read_text())
    ledger = TrialLedger(
        name=raw.get("name", path.stem),
        candidates={str(k): float(v) for k, v in raw["candidates"].items()},
        selected=[str(s) for s in raw["selected"]],
        capital=float(raw.get("capital", 1.0)),
        years=float(raw["years"]) if raw.get("years") is not None else None,
        risk_free_rate=float(raw.get("risk_free_rate", 0.0)),
        benchmark_annual_return=(
            float(raw["benchmark_annual_return"])
            if raw.get("benchmark_annual_return") is not None
            else None
        ),
    )
    dof = None
    if "degrees_of_freedom" in raw:
        dof = DegreesOfFreedom()
        for item in raw["degrees_of_freedom"]:
            dof.add(item["name"], int(item["levels"]), item.get("note", ""))
    elif "n_trials" in raw:
        dof = DegreesOfFreedom().add("declared_trials", int(raw["n_trials"]))
    return ledger, dof


def _load_returns(path: Path) -> np.ndarray:
    text = path.read_text().strip()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return np.asarray(data, dtype=float).ravel()
    # CSV: single column, or a column named return/returns/pnl
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return np.asarray([], dtype=float)
    if rows[0] and not _is_float(rows[0][0]):
        header = [h.strip().lower() for h in rows[0]]
        body = rows[1:]
        for key in ("return", "returns", "ret", "pnl", "r"):
            if key in header:
                idx = header.index(key)
                return np.asarray([float(r[idx]) for r in body if r], dtype=float)
        idx = 0
        return np.asarray([float(r[idx]) for r in body if r], dtype=float)
    vals = []
    for row in rows:
        if not row:
            continue
        vals.append(float(row[0]))
    return np.asarray(vals, dtype=float)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def cmd_check(args: argparse.Namespace) -> int:
    ledger = None
    dof = None
    if args.ledger:
        ledger, dof = _load_ledger(Path(args.ledger))
    returns = _load_returns(Path(args.returns)) if args.returns else None
    if args.n_trials and dof is None:
        dof = DegreesOfFreedom().add("declared_trials", int(args.n_trials))

    verdict = evaluate(
        ledger=ledger,
        dof=dof,
        returns=returns,
        turnover=args.turnover,
        periods_per_year=args.periods_per_year,
        placebo_n=args.placebo_n,
        placebo_seed=args.seed,
        subject=args.subject,
    )
    artifact = swear(verdict, subject=args.subject)
    print(render(verdict, title=args.subject or "affidavit"), end="")

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}  sha256={artifact['content_sha256'][:12]}…")

    if verdict.severity == "fail":
        return 2
    if verdict.severity == "warn":
        return 1
    return 0


def cmd_explain(_: argparse.Namespace) -> int:
    print(EXPLAIN)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="affidavit",
        description="Swear whether a strategy result survives falsification.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="run the gate and optionally write an affidavit")
    check.add_argument("--ledger", help="JSON trial ledger (candidates + selected)")
    check.add_argument("--returns", help="CSV/JSON per-period returns of the selected book")
    check.add_argument("--turnover", type=float, help="fraction of NAV traded per period")
    check.add_argument("--n-trials", type=int, dest="n_trials", help="override search size")
    check.add_argument("--periods-per-year", type=int, default=252, dest="periods_per_year")
    check.add_argument("--placebo-n", type=int, default=400, dest="placebo_n")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--subject", default="strategy", help="name stamped on the affidavit")
    check.add_argument("--json-out", dest="json_out", help="write sworn JSON artifact here")
    check.set_defaults(func=cmd_check)

    explain = sub.add_parser("explain", help="what each check asks and why")
    explain.set_defaults(func=cmd_explain)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
