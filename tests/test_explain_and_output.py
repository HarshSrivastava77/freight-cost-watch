import csv

import pandas as pd
import pytest

from fcw import config
from fcw.costs import build_weekly_table, peer_routes
from fcw.explain import Explainer, validate
from fcw.justify import JUSTIFIED, UNEXPLAINED, decide
from fcw.llm import LLMError, LLMResult, TokenLedger
from fcw.notes import load_notes
from fcw.output import COLUMNS, FORMAT_PATTERNS, read_contract_rows, sample_header
from fcw.pipeline import run
from fcw.retrieval import NoteRetriever, build_query


@pytest.fixture(scope="module")
def ctx():
    weekly = build_weekly_table()
    notes = load_notes()
    return weekly, peer_routes(weekly), NoteRetriever(notes)


def _case(ctx, route, week):
    weekly, peers, retriever = ctx
    rec = weekly[(weekly.route == route) & (weekly.week_of == week)].iloc[0].to_dict()
    return rec, decide(rec, retriever.search(build_query(route, week), route=route), peers[route])


class FakeClient:
    provider, model = "groq", "fake"

    def __init__(self, text=None, fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def complete(self, system, user, max_tokens=120):
        self.calls += 1
        if self.fail:
            raise LLMError("boom")
        return LLMResult(self.text, 200, 40)


# ---------------------------------------------------------------- validator --
def test_validator_accepts_grounded_text(ctx):
    rec, v = _case(ctx, "Ahmedabad-Mumbai", "2025-01-20")
    assert v.flagged == JUSTIFIED
    ok, why = validate("Note N002 describes a festival-week surcharge from 2025-01-20; cost rose 29.5% "
                       "over its past average and sits 22.5% above similar routes.", rec, v)
    assert ok, why


@pytest.mark.parametrize("text,problem", [
    ("Note N003 shows diesel rose; cost is up 29.5% on its past average here.", "note ids"),
    ("Note N002 festival surcharge pushed cost up 41.0% on its usual level this week.", "number 41.0"),
    ("Line one about N002 and 29.5%.\nLine two.", "plain sentence"),
    ("Short.", "length"),
])
def test_validator_rejects_ungrounded_text(ctx, text, problem):
    rec, v = _case(ctx, "Ahmedabad-Mumbai", "2025-01-20")
    ok, why = validate(text, rec, v)
    assert not ok and problem in why


def test_validator_rejects_any_note_or_cause_on_unexplained_rows(ctx):
    rec, v = _case(ctx, "Chennai-Bangalore", "2025-03-17")
    assert v.flagged == UNEXPLAINED
    assert not validate("Cost is 3.47 per tonne-km; N009 says the route is back to normal.", rec, v)[0]
    assert not validate("Cost rose 13.7% on its own average, likely due to lingering flood damage.", rec, v)[0]
    assert validate("Cost was 3.47 INR per tonne-km: 13.7% above its own average and 33.5% above "
                    "similar routes, with no supporting note.", rec, v)[0]


# ------------------------------------------------------------ explainer/LLM --
def test_bad_llm_output_falls_back_to_template_and_is_logged(ctx):
    rec, v = _case(ctx, "Ahmedabad-Mumbai", "2025-01-20")
    ledger = TokenLedger("groq", "fake")
    reason = Explainer(FakeClient("Diesel note N003 explains a 41% jump."), ledger).reason(rec, v)
    assert "N003" not in reason and "N002" in reason
    log = ledger.to_dict()
    assert log["llm_calls"] == 1 and log["validator_rejections"] == 1 and log["template_fallbacks"] == 1
    assert log["total_input_tokens"] == 200 and log["total_output_tokens"] == 40
    assert log["cost_usd"] == pytest.approx((200 * config.PRICE_USD_PER_1M["groq"][0] + 40 * config.PRICE_USD_PER_1M["groq"][1]) / 1e6)


def test_good_llm_output_is_used_and_verdict_prefix_is_fixed(ctx):
    rec, v = _case(ctx, "Ahmedabad-Mumbai", "2025-01-20")
    text = "The N002 festival surcharge lines up with this week and the 29.5% rise."
    reason = Explainer(FakeClient(text), TokenLedger("groq", "fake")).reason(rec, v)
    assert reason.startswith("Justified by note N002")
    assert "festival surcharge lines up" in reason and "," not in reason


def test_llm_error_falls_back(ctx):
    rec, v = _case(ctx, "Delhi-Jaipur", "2024-11-11")
    ledger = TokenLedger("groq", "fake")
    reason = Explainer(FakeClient(fail=True), ledger).reason(rec, v)
    assert reason.startswith("Unexplained") and ledger.to_dict()["call_errors"] == 1


def test_not_flagged_rows_never_call_llm(ctx):
    rec, v = _case(ctx, "Mumbai-Delhi", "2024-07-29")
    client = FakeClient("x" * 50)
    Explainer(client, TokenLedger("groq", "fake")).reason(rec, v)
    assert client.calls == 0


# ----------------------------------------------------------- output contract --
@pytest.fixture(scope="module")
def output(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    run(out_path=d / "out.csv", llm_provider="none", token_log_path=d / "t.json", report_path=d / "r.json")
    return d / "out.csv"


def test_header_matches_sample_exactly(output):
    with open(output, newline="") as fh:
        assert next(csv.reader(fh)) == sample_header() == COLUMNS


def test_every_line_has_exactly_8_fields_even_with_naive_split(output):
    lines = output.read_text().splitlines()
    assert all(len(line.split(",")) == len(COLUMNS) for line in lines)


def test_formats_and_note_consistency(output):
    df = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert len(df) == 728
    for col, pat in FORMAT_PATTERNS.items():
        assert df[col].str.match(pat).all(), col
    justified = df.flagged == "No (justified)"
    assert (df.loc[justified, "matched_note_id"] != "").all()
    assert (df.loc[~justified, "matched_note_id"] == "").all()


def test_sample_rows_match_character_for_character(output):
    df = pd.read_csv(output, dtype=str, keep_default_na=False).set_index(["route", "week_of"])
    for s in read_contract_rows(config.SAMPLE_OUTPUT_CSV):
        got = df.loc[(s["route"], s["week_of"])]
        for col in COLUMNS[2:-1]:
            assert got[col] == s[col], (s["route"], col)


def test_no_llm_run_logs_zero_calls_and_a_projection(output):
    import json
    log = json.loads((output.parent / "t.json").read_text())
    assert log["llm_calls"] == 0 and log["cost_usd"] == 0
    assert log["projection_if_run_with_groq"]["llm_calls"] == 19



def test_reasons_are_ascii_safe(tmp_path):
    from fcw.output import comma_free, write_output
    text = comma_free("Cost rose \u2011 see \u201cN002\u201d \u2014 festival\u2026")
    assert text.isascii() and "-" in text
    row = dict.fromkeys(COLUMNS, "x")
    row["reason"] = text
    write_output([row], tmp_path / "o.csv")
    assert (tmp_path / "o.csv").read_bytes().isascii()
