"""Human-readable report for an affidavit verdict."""

from __future__ import annotations

from .gate import Verdict


_SEV = {
    "fail": "FAIL",
    "warn": "WARN",
    "pass": "PASS",
    "skip": "skip",
}


def render(verdict: Verdict, title: str = "affidavit") -> str:
    lines = [
        f"{title}",
        "",
        f"  {_SEV.get(verdict.severity, verdict.severity).upper()}",
        "",
    ]
    if verdict.sworn:
        lines.append("  sworn: yes — every applicable check passed")
    else:
        lines.append("  sworn: no  — at least one check failed or warned")
    lines.append("")

    width = max((len(f.check) for f in verdict.findings), default=8)
    for f in verdict.findings:
        tag = _SEV.get(f.severity, f.severity)
        lines.append(f"  {tag:<4}  {f.check:<{width}}  {f.headline}")
        if f.detail:
            lines.append(f"        {'':<{width}}  {f.detail}")

    subject = verdict.stats.get("subject")
    if subject:
        lines.extend(["", f"  subject: {subject}"])

    floor = verdict.stats.get("noise_floor_sharpe")
    if floor is not None:
        lines.append(f"  noise floor Sharpe: {floor:.3f}")

    return "\n".join(lines) + "\n"
