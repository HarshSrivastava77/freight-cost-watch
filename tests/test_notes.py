import pandas as pd
import pytest

from fcw import config
from fcw.costs import build_weekly_table, peer_routes
from fcw.justify import JUSTIFIED, NOT_FLAGGED, UNEXPLAINED, check_note, decide
from fcw.notes import load_notes
from fcw.retrieval import NoteRetriever, build_query

TRAP_NOTES = {"N006", "N007", "N009", "N010"}


@pytest.fixture(scope="module")
def notes():
    return {n.note_id: n for n in load_notes()}


@pytest.fixture(scope="module")
def weekly():
    return build_weekly_table()


@pytest.fixture(scope="module")
def peers(weekly):
    return peer_routes(weekly)


@pytest.fixture(scope="module")
def retriever(notes):
    return NoteRetriever(list(notes.values()))


def _rec(weekly, route, week):
    return weekly[(weekly.route == route) & (weekly.week_of == week)].iloc[0].to_dict()


def test_structured_notes_match_csv(notes):
    raw = pd.read_csv(config.NOTES_CSV, dtype=str)
    assert set(raw.note_id) == set(notes)
    for r in raw.itertuples():
        n = notes[r.note_id]
        assert n.text == r.note.strip()
        # the note's own date always lies inside its window
        assert n.window_start.isoformat() <= r.date
        assert n.window_end is None or r.date <= n.window_end.isoformat()


def test_only_increase_notes_are_n001_to_n003(notes):
    assert {k for k, n in notes.items() if n.effect == "increase"} == {"N001", "N002", "N003"}
    assert notes["N003"].max_effect_pct == 7.0


def test_week_overlap_boundaries(notes):
    n1 = notes["N001"]  # 2025-02-24 .. 2025-03-08
    assert n1.overlaps_week("2025-02-24")
    assert n1.overlaps_week("2025-03-03")      # 03-03..03-09 touches 03-08
    assert not n1.overlaps_week("2025-03-10")  # roads passable by Mar 9
    assert not n1.overlaps_week("2025-02-17")
    assert notes["N003"].overlaps_week("2025-12-22")  # open-ended


@pytest.mark.parametrize("route,week,expected,note", [
    ("Ahmedabad-Mumbai", "2025-01-20", JUSTIFIED, "N002"),
    ("Chennai-Bangalore", "2025-02-24", JUSTIFIED, "N001"),
    ("Chennai-Bangalore", "2025-03-03", JUSTIFIED, "N001"),
    ("Chennai-Bangalore", "2025-03-10", UNEXPLAINED, ""),   # trap: after flood window
    ("Chennai-Bangalore", "2025-03-17", UNEXPLAINED, ""),   # trap: N009 "returned to normal"
    ("Mumbai-Pune", "2025-09-15", UNEXPLAINED, ""),         # trap: N006 stable demand
    ("Mumbai-Pune", "2025-10-27", UNEXPLAINED, ""),         # trap: N010 no rate change
    ("Delhi-Jaipur", "2024-11-11", UNEXPLAINED, ""),
])
def test_verdicts_on_key_rows(weekly, peers, retriever, route, week, expected, note):
    rec = _rec(weekly, route, week)
    v = decide(rec, retriever.search(build_query(route, week), route=route), peers[route])
    assert (v.flagged, v.matched_note_id) == (expected, note)


def test_trap_notes_never_pass_the_gate_anywhere(weekly, peers, notes):
    for rec in weekly[weekly.rule_flagged].to_dict("records"):
        for nid in TRAP_NOTES:
            assert not check_note(notes[nid], rec, peers[rec["route"]]).passed


def test_wrong_route_rejected(weekly, peers, notes):
    rec = _rec(weekly, "Delhi-Jaipur", "2024-11-11")
    c = check_note(notes["N002"], rec, peers["Delhi-Jaipur"])
    assert (c.passed, c.code) == (False, "wrong_route")


def test_shifted_window_rejected(weekly, peers, notes):
    from dataclasses import replace
    from datetime import date
    rec = _rec(weekly, "Ahmedabad-Mumbai", "2025-01-20")
    shifted = replace(notes["N002"], window_start=date(2025, 1, 27), window_end=date(2025, 2, 2))
    assert check_note(shifted, rec, peers["Ahmedabad-Mumbai"]).code == "outside_window"


def test_all_routes_note_cannot_explain_peer_gap(weekly, peers, notes):
    rec = _rec(weekly, "Mumbai-Pune", "2025-06-23")  # flagged on peer gap only
    assert rec["peer_trigger"] and not rec["own_trigger"]
    assert check_note(notes["N003"], rec, peers["Mumbai-Pune"]).code == "applies_to_peers"


def test_note_size_cap(notes):
    """N003 says ~5-7%: with the 2-point tolerance it can explain at most 9%."""
    rec = {"route": "Mumbai-Delhi", "week_of": "2025-05-12", "vs_own_pct": 9.5, "vs_peer_pct": 0.0,
           "own_trigger": True, "peer_trigger": False, "rule_flagged": True}
    assert check_note(notes["N003"], rec, []).code == "implausible_size"
    rec["vs_own_pct"] = 8.5
    assert check_note(notes["N003"], rec, []).passed


def test_not_flagged_rows_skip_notes(weekly, peers):
    rec = _rec(weekly, "Mumbai-Delhi", "2024-07-29")
    assert decide(rec, [], peers["Mumbai-Delhi"]).flagged == NOT_FLAGGED


def test_retrieval_recall_for_every_flagged_row(weekly, peers, notes, retriever):
    """Any note that would pass the gate must appear in the top-k for that row."""
    for rec in weekly[weekly.rule_flagged].to_dict("records"):
        r = rec["route"]
        valid = {n for n, note in notes.items() if check_note(note, rec, peers[r]).passed}
        got = {h.note.note_id for h in retriever.search(build_query(r, rec["week_of"]), route=r)}
        assert valid <= got, (r, rec["week_of"], valid, got)


def test_retrieval_is_deterministic(retriever):
    q = build_query("Chennai-Bangalore", "2025-03-03")
    first = [(h.note.note_id, h.score) for h in retriever.search(q)]
    for _ in range(5):
        assert [(h.note.note_id, h.score) for h in retriever.search(q)] == first
