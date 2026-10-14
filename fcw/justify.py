"""Decide whether a flagged route-week is justified by a note.

A note justifies a flag only if ALL of these hold:
  1. effect == "increase"            (no_effect / decrease notes never justify)
  2. it applies to this route        (or to ALL routes)
  3. its window overlaps this week   (Monday-Sunday)
  4. it explains every trigger that fired:
       - own-history trigger: the rise vs own history is within the note's max_effect_pct
       - peer trigger: the note must NOT also apply to the route's peers -- a shock that
         hits every similar route cannot explain one route costing more than the others
     (notes without a figure get the UNBOUNDED_NOTE_CAP_PCT sanity cap)
Anything else -> unexplained, matched_note_id blank.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .notes import Note
from .retrieval import Hit

JUSTIFIED = "No (justified)"
UNEXPLAINED = "Yes"
NOT_FLAGGED = "No"


@dataclass(frozen=True)
class NoteCheck:
    note_id: str
    score: float
    passed: bool
    code: str     # machine-readable rejection code, "ok" if passed
    detail: str   # plain-English reason


@dataclass
class Verdict:
    flagged: str
    note: Note | None = None
    checks: list[NoteCheck] = field(default_factory=list)

    @property
    def matched_note_id(self) -> str:
        return self.note.note_id if self.note else ""


def _cap(note: Note) -> float:
    base = note.max_effect_pct if note.max_effect_pct is not None else config.UNBOUNDED_NOTE_CAP_PCT
    return base + config.PLAUSIBILITY_TOLERANCE_PCT


def check_note(note: Note, row: dict, peers: list[str], score: float = 0.0) -> NoteCheck:
    def reject(code: str, detail: str) -> NoteCheck:
        return NoteCheck(note.note_id, score, False, code, detail)

    if note.effect == "no_effect":
        return reject("no_effect", "it says costs were not affected")
    if note.effect == "decrease":
        return reject("decrease", "it describes normal or improving conditions and not a cost rise")
    if not note.covers_route(row["route"]):
        return reject("wrong_route", f"it applies to {note.applies_to}, not this route")
    if not note.overlaps_week(row["week_of"]):
        return reject("outside_window", f"its window ({note.window_str}) does not overlap this week")

    cap = _cap(note)
    if row["own_trigger"] and row["vs_own_pct"] > cap:
        return reject(
            "implausible_size",
            f"it supports a rise of about {cap - config.PLAUSIBILITY_TOLERANCE_PCT:.0f}% at most "
            f"but this route is up {row['vs_own_pct']:.1f}% on its own history",
        )
    if row["peer_trigger"]:
        if any(note.covers_route(p) for p in peers):
            return reject(
                "applies_to_peers",
                "it applies to similar-length routes too so it cannot explain this route "
                "costing more than its peers",
            )
        if row["vs_peer_pct"] > cap:
            return reject(
                "implausible_size",
                f"it supports a gap of about {cap - config.PLAUSIBILITY_TOLERANCE_PCT:.0f}% at most "
                f"but this route is {row['vs_peer_pct']:.1f}% above its peers",
            )
    return NoteCheck(note.note_id, score, True, "ok", "route and dates and effect all match")


def decide(row: dict, hits: list[Hit], peers: list[str]) -> Verdict:
    """Verdict for one route-week given its retrieved candidate notes."""
    if not row["rule_flagged"]:
        return Verdict(NOT_FLAGGED)
    checks = [check_note(h.note, row, peers, h.score) for h in hits]
    passing = [(h, c) for h, c in zip(hits, checks) if c.passed]
    if not passing:
        return Verdict(UNEXPLAINED, None, checks)
    best_hit, _ = passing[0]  # hits are already ranked by score, ties broken by note_id
    return Verdict(JUSTIFIED, best_hit.note, checks)


def valid_notes_full_scan(row: dict, notes: list[Note], peers: list[str]) -> list[str]:
    """Every note that would pass the gate, ignoring retrieval. Used to audit recall."""
    return [n.note_id for n in notes if check_note(n, row, peers).passed]
