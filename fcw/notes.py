"""Context notes: raw text from context_notes.csv joined with hand-verified structure."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import config

ALL_ROUTES = "ALL"
VALID_EFFECTS = {"increase", "no_effect", "decrease"}


@dataclass(frozen=True)
class Note:
    note_id: str
    date: str               # date column from context_notes.csv
    applies_to: str         # applies_to column from context_notes.csv
    text: str               # the note itself, verbatim
    routes: tuple[str, ...]
    window_start: date
    window_end: date | None  # None = open-ended
    effect: str
    max_effect_pct: float | None
    summary: str

    @property
    def is_all_routes(self) -> bool:
        return ALL_ROUTES in self.routes

    def covers_route(self, route: str) -> bool:
        return self.is_all_routes or route in self.routes

    def overlaps_week(self, week_of: str) -> bool:
        """True if the Monday-Sunday week starting `week_of` overlaps the note window."""
        start = date.fromisoformat(week_of)
        end = start + timedelta(days=6)
        if end < self.window_start:
            return False
        return self.window_end is None or start <= self.window_end

    @property
    def window_str(self) -> str:
        if self.window_end is None:
            return f"from {self.window_start.isoformat()}"
        return f"{self.window_start.isoformat()} to {self.window_end.isoformat()}"

    def index_text(self) -> str:
        """Text that gets embedded for retrieval."""
        return f"{self.applies_to} ({self.date}): {self.text}"


def load_notes(
    csv_path: Path = config.NOTES_CSV,
    structured_path: Path = config.NOTES_STRUCTURED_JSON,
) -> list[Note]:
    raw = pd.read_csv(csv_path, dtype=str).apply(lambda c: c.str.strip())
    structured = json.loads(Path(structured_path).read_text())["notes"]
    by_id = {n["note_id"]: n for n in structured}

    if set(by_id) != set(raw["note_id"]):
        raise ValueError(
            f"notes_structured.json ids {sorted(by_id)} != context_notes.csv ids {sorted(raw['note_id'])}"
        )

    notes = []
    for row in raw.itertuples(index=False):
        s = by_id[row.note_id]
        expected_routes = [ALL_ROUTES] if row.applies_to.lower() == "all routes" else [row.applies_to]
        if s["routes"] != expected_routes:
            raise ValueError(f"{row.note_id}: routes {s['routes']} disagree with applies_to '{row.applies_to}'")
        if s["effect"] not in VALID_EFFECTS:
            raise ValueError(f"{row.note_id}: bad effect {s['effect']!r}")
        start = date.fromisoformat(s["window_start"])
        end = date.fromisoformat(s["window_end"]) if s["window_end"] else None
        if end is not None and end < start:
            raise ValueError(f"{row.note_id}: window ends before it starts")
        notes.append(Note(
            note_id=row.note_id, date=row.date, applies_to=row.applies_to, text=row.note,
            routes=tuple(s["routes"]), window_start=start, window_end=end,
            effect=s["effect"], max_effect_pct=s["max_effect_pct"], summary=s["summary"],
        ))
    return sorted(notes, key=lambda n: n.note_id)
