"""Score output.csv against hand-labelled gold rows and re-audit every guardrail.

    python eval/score.py [path/to/output.csv]

The guardrail audit re-checks justified rows directly against the note text's
structured fields with its own small implementation (it does not call
fcw.justify), so a bug in the pipeline's gate would show up here.
Exit code is non-zero if anything fails.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fcw.output import COLUMNS, FORMAT_PATTERNS, read_contract_rows  # noqa: E402

FORBIDDEN_ALWAYS = {"N006", "N007", "N009", "N010"}  # named in the brief's trap list


def read_str_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def score_gold(out: pd.DataFrame, gold: pd.DataFrame) -> tuple[dict, list[str]]:
    merged = gold.merge(out, on=["route", "week_of"], how="left", indicator="in_output")
    failures = []
    for r in merged.itertuples(index=False):
        if r.in_output != "both":
            failures.append(f"{r.route} {r.week_of}: missing from output")
            continue
        if r.flagged != r.expected_flagged or r.matched_note_id != r.expected_matched_note_id:
            failures.append(
                f"[{r.category}] {r.route} {r.week_of}: got ({r.flagged!r}, {r.matched_note_id!r}) "
                f"expected ({r.expected_flagged!r}, {r.expected_matched_note_id!r})"
            )
    present = merged[merged["in_output"] == "both"]
    flag_acc = (present["flagged"] == present["expected_flagged"]).mean()
    note_acc = (present["matched_note_id"] == present["expected_matched_note_id"]).mean()
    by_cat = (
        present.assign(ok=(present["flagged"] == present["expected_flagged"])
                       & (present["matched_note_id"] == present["expected_matched_note_id"]))
        .groupby("category")["ok"].agg(["sum", "count"])
    )
    stats = {
        "gold_rows": int(len(gold)),
        "flagged_accuracy": round(float(flag_acc), 4),
        "matched_note_accuracy": round(float(note_acc), 4),
        "exact_row_accuracy": round(float(by_cat["sum"].sum() / by_cat["count"].sum()), 4),
        "by_category": {k: f"{int(v['sum'])}/{int(v['count'])}" for k, v in by_cat.iterrows()},
    }
    return stats, failures


def check_sample_rows(out: pd.DataFrame, sample: pd.DataFrame) -> tuple[int, list[str]]:
    """Every column except `reason` must match the sample file character for character."""
    failures, n_ok = [], 0
    cols = [c for c in COLUMNS if c != "reason"]
    for s in sample.itertuples(index=False):
        row = out[(out["route"] == s.route) & (out["week_of"] == s.week_of)]
        if row.empty:
            failures.append(f"sample row {s.route} {s.week_of} missing")
            continue
        diffs = [f"sample {s.route} {s.week_of} {c}: {row.iloc[0][c]!r} != {getattr(s, c)!r}"
                 for c in cols if row.iloc[0][c] != getattr(s, c)]
        failures += diffs
        n_ok += not diffs
    return n_ok, failures


def audit_guardrails(out: pd.DataFrame, notes_json: dict, csv_notes: pd.DataFrame) -> list[str]:
    failures = []
    if list(out.columns) != COLUMNS:
        failures.append(f"columns {list(out.columns)} != {COLUMNS}")
    for col, pat in FORMAT_PATTERNS.items():
        bad = out[~out[col].str.match(pat)]
        if len(bad):
            failures.append(f"{len(bad)} rows with bad {col} format, e.g. {bad.iloc[0][col]!r}")

    notes = {n["note_id"]: n for n in notes_json["notes"]}
    known_ids = set(csv_notes["note_id"])
    for r in out.itertuples(index=False):
        nid = r.matched_note_id
        if r.flagged == "No (justified)":
            if nid not in known_ids:
                failures.append(f"{r.route} {r.week_of}: cites unknown note {nid!r}")
                continue
            n = notes[nid]
            wk_start = date.fromisoformat(r.week_of)
            wk_end = wk_start + timedelta(days=6)
            win_start = date.fromisoformat(n["window_start"])
            win_end = date.fromisoformat(n["window_end"]) if n["window_end"] else date.max
            if n["effect"] != "increase":
                failures.append(f"{r.route} {r.week_of}: cites {nid} whose effect is {n['effect']}")
            if "ALL" not in n["routes"] and r.route not in n["routes"]:
                failures.append(f"{r.route} {r.week_of}: cites {nid} for wrong route {n['routes']}")
            if wk_end < win_start or wk_start > win_end:
                failures.append(f"{r.route} {r.week_of}: cites {nid} outside its window")
            if nid not in r.reason:
                failures.append(f"{r.route} {r.week_of}: reason does not name {nid}")
            if nid in FORBIDDEN_ALWAYS:
                failures.append(f"{r.route} {r.week_of}: cites forbidden trap note {nid}")
        elif nid:
            failures.append(f"{r.route} {r.week_of}: flagged={r.flagged!r} but matched_note_id={nid!r}")
    return failures


def main(out_path: Path) -> int:
    out = read_str_csv(out_path)
    gold = read_str_csv(ROOT / "eval" / "gold_labels.csv")
    sample = pd.DataFrame(read_contract_rows(ROOT / "data" / "sample_output_format_v2.csv"))
    notes_json = json.loads((ROOT / "data" / "notes_structured.json").read_text())
    csv_notes = read_str_csv(ROOT / "data" / "context_notes.csv")

    stats, gold_fail = score_gold(out, gold)
    sample_ok, sample_fail = check_sample_rows(out, sample)
    guard_fail = audit_guardrails(out, notes_json, csv_notes)

    stats.update({
        "sample_rows_exact_match": f"{sample_ok}/{len(sample)}",
        "output_rows": int(len(out)),
        "flag_counts": out["flagged"].value_counts().to_dict(),
        "cited_notes": sorted(set(out["matched_note_id"]) - {""}),
        "guardrail_violations": len(guard_fail),
    })
    print(json.dumps(stats, indent=2))
    for f in gold_fail + sample_fail + guard_fail:
        print("FAIL:", f)
    ok = not (gold_fail or sample_fail or guard_fail)
    print("RESULT:", "PASS" if ok else "FAIL")
    (ROOT / "eval" / "results.json").write_text(json.dumps({**stats, "failures": gold_fail + sample_fail + guard_fail,
                                                            "result": "PASS" if ok else "FAIL"}, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output.csv"))
