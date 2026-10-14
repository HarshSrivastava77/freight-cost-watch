"""Plain-English `reason` text.

reason = <verdict prefix> <explanation body> [<checked-note suffix>]

* The prefix (verdict + note id + window) and suffix are deterministic -- the LLM
  can never change the verdict or the cited note.
* The body comes from the LLM for flagged rows (temperature 0), and is accepted only
  if validate() passes; otherwise a template body is used. Non-flagged rows always
  use templates (no LLM call).
"""
from __future__ import annotations

import math
import re

from .justify import JUSTIFIED, NOT_FLAGGED, UNEXPLAINED, Verdict
from .llm import LLMError, TokenLedger
from .output import comma_free, fmt_cost, fmt_pct, is_missing

SYSTEM_PROMPT = (
    "You explain freight cost alerts in plain English for a logistics manager. "
    "Write one or two short sentences, no lists, no markdown. "
    "Use ONLY the numbers and the note text given to you. Do not mention any other "
    "note, event, cause or number. The verdict is already decided; do not change it, "
    "and if no note is given, do not suggest a cause."
)

NOTE_ID_RE = re.compile(r"\bN\d{3}\b")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CAUSAL_RE = re.compile(r"\b(due to|because|caused by|driven by|as a result of|explained by|justified by)\b", re.I)
MIN_LEN, MAX_LEN = 20, 350


# ---------------------------------------------------------------- prompt ----
def build_prompt(row: dict, verdict: Verdict) -> str:
    lines = [
        f"Route: {row['route']}",
        f"Week of: {row['week_of']}",
        f"Cost per tonne-km: {fmt_cost(row['cost_per_tonne_km'])} INR",
        f"Change vs this route's average over the previous {row['own_weeks_used']} weeks: {fmt_pct(row['vs_own_pct'])}",
        f"Change vs similar-length routes this week: {fmt_pct(row['vs_peer_pct'])}",
    ]
    if verdict.flagged == JUSTIFIED:
        n = verdict.note
        lines += [
            f"Verdict (already decided): justified by note {n.note_id}.",
            f"Note {n.note_id} (applies to {n.applies_to}; {n.window_str}): {n.text}",
            "Explain how this note accounts for the rise.",
        ]
    else:
        lines += [
            "Verdict (already decided): unexplained -- no note supports this rise.",
            "Describe the rise using the numbers only. Do not suggest any cause.",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------- validator ----
def _allowed_numbers(row: dict, verdict: Verdict) -> set[float]:
    vals = [row["cost_per_tonne_km"], abs(row["vs_own_pct"]), abs(row["vs_peer_pct"]), row["own_weeks_used"]]
    allowed: set[float] = set()
    for v in vals:
        if is_missing(v):
            continue
        for r in (round(v, 2), round(v, 1), math.floor(v), math.ceil(v)):
            allowed.add(float(r))
    texts = [row["week_of"]]
    if verdict.note is not None:
        n = verdict.note
        texts += [n.text, n.date, n.window_start.isoformat(), n.window_end.isoformat() if n.window_end else ""]
    for t in texts:
        allowed.update(float(x) for x in NUMBER_RE.findall(t))
    return allowed


def validate(body: str, row: dict, verdict: Verdict) -> tuple[bool, str]:
    """Return (ok, problem). Rejects anything the deterministic pipeline did not supply."""
    if not (MIN_LEN <= len(body) <= MAX_LEN):
        return False, f"length {len(body)} outside {MIN_LEN}-{MAX_LEN}"
    if "\n" in body or body.lstrip().startswith(("-", "*", "#")):
        return False, "not a plain sentence"
    cited = set(NOTE_ID_RE.findall(body))
    allowed_ids = {verdict.matched_note_id} if verdict.flagged == JUSTIFIED else set()
    if cited - allowed_ids:
        return False, f"mentions note ids not allowed: {sorted(cited - allowed_ids)}"
    if verdict.flagged == UNEXPLAINED and CAUSAL_RE.search(body):
        return False, "suggests a cause for an unexplained row"
    allowed_nums = _allowed_numbers(row, verdict)
    for tok in NUMBER_RE.findall(NOTE_ID_RE.sub("", body)):
        if not any(abs(float(tok) - a) < 0.051 for a in allowed_nums):
            return False, f"number {tok} not in the row or note"
    return True, ""


# -------------------------------------------------------------- templates ---
def prefix(verdict: Verdict) -> str:
    if verdict.flagged == JUSTIFIED:
        n = verdict.note
        return f"Justified by note {n.note_id} ({n.applies_to}; {n.window_str})."
    if verdict.flagged == UNEXPLAINED:
        return "Unexplained: no note supports a cost rise on this route in this week; flagged for human review."
    return "Not flagged."


def _numbers_clause(row: dict) -> str:
    return (
        f"cost was {fmt_cost(row['cost_per_tonne_km'])} INR per tonne-km "
        f"({fmt_pct(row['vs_own_pct'])} vs its own past average and "
        f"{fmt_pct(row['vs_peer_pct'])} vs similar-length routes)"
    )


def template_body(row: dict, verdict: Verdict) -> str:
    if verdict.flagged == JUSTIFIED:
        s = verdict.note.summary
        return f"The note reports that {s} and this week falls inside its window; {_numbers_clause(row)}."
    if verdict.flagged == UNEXPLAINED:
        return f"This week {_numbers_clause(row)}."

    # not flagged
    if is_missing(row["own_baseline"]):
        return (f"First week of data for this route so there is no own-history baseline; "
                f"{fmt_pct(row['vs_peer_pct'])} vs similar-length routes.")
    parts = [f"{fmt_pct(row['vs_own_pct'])} vs own history"]
    if row["own_weeks_used"] < 8:
        parts[0] += f" (baseline uses only the {row['own_weeks_used']} prior week(s) available)"
    parts.append(f"{fmt_pct(row['vs_peer_pct'])} vs similar-length routes")
    if row["vs_own_pct"] <= 0 and row["peer_trigger"]:
        tail = "above peers but not rising against its own history"
    else:
        tail = "within thresholds"
    return f"{' and '.join(parts)}; {tail}."


# Rejections of "increase" notes are the informative near-misses (right kind of note,
# wrong week / wrong scope / too small), so they are preferred over the top-ranked hit.
_NEAR_MISS_CODES = ("outside_window", "applies_to_peers", "implausible_size")


def suffix(verdict: Verdict) -> str:
    """For unexplained rows: name the most relevant retrieved note and why it was rejected."""
    if verdict.flagged != UNEXPLAINED or not verdict.checks:
        return ""
    near = [c for c in verdict.checks if c.code in _NEAR_MISS_CODES]
    c = near[0] if near else verdict.checks[0]
    return f" Closest note checked ({c.note_id}) does not count: {c.detail}."


# ---------------------------------------------------------------- driver ----
class Explainer:
    def __init__(self, client=None, ledger: TokenLedger | None = None):
        self.client = client
        self.ledger = ledger

    def reason(self, row: dict, verdict: Verdict) -> str:
        body = template_body(row, verdict)
        if verdict.flagged != NOT_FLAGGED:
            label = f"{row['route']} {row['week_of']}"
            prompt = build_prompt(row, verdict)
            if self.client is None:
                if self.ledger is not None:
                    self.ledger.projected_prompts.append((SYSTEM_PROMPT, prompt))
            else:
                try:
                    result = self.client.complete(SYSTEM_PROMPT, prompt)
                    ok, problem = validate(result.text, row, verdict)
                    self.ledger.record(label, result, ok, problem=problem)
                    if ok:
                        body = result.text
                except LLMError as exc:
                    self.ledger.record(label, None, False, error=str(exc))
        return comma_free(f"{prefix(verdict)} {body}{suffix(verdict)}")
