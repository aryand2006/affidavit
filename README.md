# affidavit

**Swear whether a strategy result survives falsification.**

An affidavit is a sworn statement of fact. A backtest that kept the best of
many candidates, assumed a cost model, and never checked whether the decision
function can see tomorrow needs the same treatment: the reported number
includes every degree of freedom that entered the search, and those have to be
subtracted before anything is deployable.

```
affidavit check --ledger examples/cointegration-pairs.json

3-pair cointegration

  FAIL

  sworn: no  — at least one check failed or warned

  FAIL  selection      ex-ante 1.7%/yr does not clear risk-free 6.5%
  WARN  concentration  64.7% of profit from a single candidate
  FAIL  benchmark      ex-ante 1.7%/yr below benchmark 12.0%
  WARN  degrees        search of 72 trials over 2y can manufacture Sharpe 2.07 from noise
  skip  track_record   no return series provided
  skip  cost_tilt      need returns and turnover for cost tilt
  skip  placebo        no return series for placebo battery
  skip  leak           no decision function / data for leak probe
```

That is a real shape of failure. Six cointegration pairs were tested, three
kept, and the keep/drop decision justified afterwards on grounds that sound
prior and always fit. Running all six — the portfolio actually available before
the results were known — returns 1.7%/yr against a 6.5% risk-free rate. Counting
the entry/exit grid lifts the search to 72 trials, which can manufacture a
Sharpe of ~2.1 from noise alone on a two-year sample.

---

## What it checks

| Check | Question |
|---|---|
| `selection` | You tested N and kept K. What would running all N have returned? |
| `concentration` | Is this a portfolio, or one bet with decorations? |
| `degrees` | What Sharpe can this search manufacture from pure noise? |
| `track_record` | Is the Sharpe distinguishable from zero on this sample? |
| `cost_tilt` | How fast does the edge die as costs rise? |
| `placebo` | Does it beat skill-free shuffle and AR(1) nulls? |
| `leak` | Does hiding the future change the decision function's answer? |
| `benchmark` | Does the ex-ante result clear the index? |

Checks whose inputs are absent are **skipped and reported as skipped**, never
quietly assumed. A gate that guesses is a gate that gets overridden.

A result is **sworn** only when every applicable check passes. The CLI can write
a JSON artifact with a `content_sha256` over the canonical findings — so a later
reader can tell whether the affidavit was silently re-run with different inputs.

### The numbers behind two of them

**Noise floor.** Searching N candidates over T years, noise alone is expected to
produce an annualized Sharpe of about `sqrt(2·ln(N)/T)`. Six candidates over two
years gives **1.34**. Count the real free choices — entry grid, exit grid,
universe filter — and N is usually much larger than the number of lines in the
final write-up.

**Ex-ante reconstitution.** The statistics of selection bias are well
established. The check nobody runs is the blunt one: put the losers back. It is
denominated in money rather than in probability, and it usually lands harder.

---

## Use

```bash
git clone https://github.com/aryand2006/affidavit && cd affidavit
pip install -e .        # numpy is the only dependency

# `sworn` is an alias for the same CLI
affidavit explain
sworn demo
affidavit check --ledger examples/cointegration-pairs.json --json-out affidavit.json
```

Ledger schema (only `candidates` and `selected` are required):

```json
{
  "name": "my strategy search",
  "capital": 1000000,
  "years": 2,
  "risk_free_rate": 0.065,
  "benchmark_annual_return": 0.12,
  "candidates": {"variant_a": 163000, "variant_b": -124000},
  "selected": ["variant_a"],
  "degrees_of_freedom": [
    {"name": "lookback", "levels": 5},
    {"name": "threshold", "levels": 4}
  ]
}
```

With a return series and turnover, the cost / placebo / track-record checks turn on:

```bash
affidavit check \
  --ledger run.json \
  --returns returns.csv \
  --turnover 0.05 \
  --json-out affidavit.json
```

### Library

```python
from affidavit import TrialLedger, DegreesOfFreedom, evaluate
from affidavit.gate import swear

ledger = TrialLedger(
    name="demo",
    candidates={"a": 100.0, "b": -40.0},
    selected=["a"],
    capital=1000.0,
    years=2.0,
    risk_free_rate=0.05,
)
dof = DegreesOfFreedom().add("grid", 20)
verdict = evaluate(ledger=ledger, dof=dof)
artifact = swear(verdict, subject="demo")
print(artifact["verdict"], artifact["content_sha256"][:12])
```

Prefix-invariance leak probe against a black-box decision function:

```python
import numpy as np
from affidavit.leak import probe_leak

prices = np.cumsum(np.random.default_rng(0).normal(size=300)) + 100

def decision(data, asof):
    # clean: mean only uses history through asof
    return float(data[asof] - data[: asof + 1].mean())

print(probe_leak(decision, prices).clean)
```

---

## Relationship to the rest of the stack

`affidavit` is the research-governance umbrella. The sharper single-purpose
instruments still exist on their own:

- [`assay`](https://github.com/aryand2006/assay) — selection bias and deflated Sharpe in depth
- [`clairvoyant`](https://github.com/aryand2006/clairvoyant) — differential lookahead with horizon search
- [`parallax`](https://github.com/aryand2006/parallax) — implementation-decision attribution across fill/cost models
- [`sediment`](https://github.com/aryand2006/sediment) — structural-erosion gate for machine-authored code

One thread: **don't trust a result you can't verify.**

---

## Develop

```bash
pip install -e ".[dev]"
pytest -q
```

MIT license. Issues and counterexamples welcome — especially affidavits that
passed here and failed live.
