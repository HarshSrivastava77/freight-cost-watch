"""Output contract: exact columns, order and string formats of sample_output_format_v2.csv."""
from __future__ import annotations

import csv
import math
import unicodedata
import re
from pathlib import Path

from . import config

COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason",
]
OWN_SUFFIX = " vs this route's past average"
PEER_SUFFIX = " vs similar-length routes this week"
NA = "N/A"

# Regexes used by tests and the eval to check every output row's format.
FORMAT_PATTERNS = {
    "week_of": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "cost_per_tonne_km": re.compile(r"^\d+\.\d{2}$"),
    "vs_own_history": re.compile(r"^(N/A|[+-]\d+\.\d%" + re.escape(OWN_SUFFIX) + r")$"),
    "vs_similar_routes": re.compile(r"^(N/A|[+-]\d+\.\d%" + re.escape(PEER_SUFFIX) + r")$"),
    "flagged": re.compile(r"^(Yes|No|No \(justified\))$"),
    "matched_note_id": re.compile(r"^(N\d{3})?$"),
}


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_cost(x: float) -> str:
    return f"{x:.2f}"


def fmt_pct(x: float) -> str:
    """'+35.5%' style; avoids '-0.0%'."""
    if is_missing(x):
        return NA
    r = round(x, 1)
    if r == 0:
        r = 0.0
    return f"{r:+.1f}%"


def fmt_vs_own(x: float) -> str:
    return NA if is_missing(x) else fmt_pct(x) + OWN_SUFFIX


def fmt_vs_peer(x: float) -> str:
    return NA if is_missing(x) else fmt_pct(x) + PEER_SUFFIX


def read_contract_rows(path: Path = config.SAMPLE_OUTPUT_CSV) -> list[dict]:
    """Read a file in the output format, tolerating unquoted commas in `reason`.

    The provided sample's third row has an unquoted comma inside `reason`, which
    strict CSV parsers reject. `reason` is the last column, so any extra fields
    are joined back into it.
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n = len(header)
        rows = []
        for fields in reader:
            if len(fields) > n:
                fields = fields[: n - 1] + [",".join(fields[n - 1:])]
            rows.append(dict(zip(header, fields)))
    return rows


def sample_header(path: Path = config.SAMPLE_OUTPUT_CSV) -> list[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


_ASCII_MAP = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": "'", "\u201d": "'",
    "\u2026": "...", "\u2192": "->", "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}


def to_ascii(text: str) -> str:
    """LLMs often emit typographic characters (non-breaking hyphens, curly quotes).
    Map them to plain ASCII so output.csv is identical on every OS and encoding."""
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def comma_free(text: str) -> str:
    """Our reasons never contain commas or quotes, so every line has exactly 8 fields
    even under a naive split(','), and no CSV quoting is ever needed."""
    return " ".join(to_ascii(text).replace(",", ";").replace('"', "'").split())


def write_output(rows: list[dict], path: Path) -> None:
    header = sample_header()
    if header != COLUMNS:
        raise ValueError(f"sample header changed: {header}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            if any("," in str(v) for v in row.values()):
                raise ValueError(f"comma in output field: {row}")
            writer.writerow(row)

