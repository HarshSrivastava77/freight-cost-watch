import pandas as pd
import pytest

from fcw.llm import LLMResult
from fcw.pipeline import run
from fcw.qa import NOT_FOUND, answer, parse_question, validate_answer


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    d = tmp_path_factory.mktemp("ask")
    run(out_path=d / "o.csv", llm_provider="none", token_log_path=d / "t.json", report_path=d / "r.json")
    return pd.read_csv(d / "o.csv", dtype=str, keep_default_na=False)


ROUTES = ["Ahmedabad-Mumbai", "Chennai-Bangalore", "Delhi-Chennai", "Delhi-Jaipur",
          "Kolkata-Bhubaneswar", "Mumbai-Delhi", "Mumbai-Pune"]


def test_parse_route_month_year():
    f = parse_question("Why did Chennai to Bangalore get pricier in March 2025?", ROUTES)
    assert f.routes == ["Chennai-Bangalore"] and f.months == [3] and f.years == [2025] and f.wants_rise


def test_parse_may_is_only_a_month_when_clear():
    assert parse_question("Why may Pune costs rise?", ROUTES).months == []
    assert parse_question("What happened in May 2025 on Mumbai-Pune?", ROUTES).months == [5]


def test_march_question_returns_chennai_rows(df):
    text, usage = answer("why did this route get pricier in March?", df)
    assert "Chennai-Bangalore" in text and "N001" in text and usage["rows_used"] == 3


def test_unrelated_question_is_not_found(df):
    assert answer("What is the weather in Goa?", df)[0] == NOT_FOUND


class Fake:
    provider, model = "groq", "fake"

    def __init__(self, text):
        self.text = text

    def complete(self, system, user, max_tokens=250):
        assert "Result rows" in user
        return LLMResult(self.text, 300, 50)


def test_llm_answer_with_invented_note_is_replaced_by_template(df):
    text, usage = answer("why was Ahmedabad-Mumbai pricier in January 2025?", df,
                         Fake("Because of N003 diesel prices rising 12.4%."))
    assert not usage["used_llm_answer"] and "N003" not in text and "N002" in text


def test_grounded_llm_answer_is_used(df):
    good = "Ahmedabad-Mumbai in the week of 2025-01-20 was +29.5% vs its past average; justified by N002."
    text, usage = answer("why was Ahmedabad-Mumbai pricier in January 2025?", df, Fake(good))
    assert usage["used_llm_answer"] and text == good


def test_validate_answer_number_check(df):
    rows = df[(df.route == "Delhi-Jaipur") & (df.week_of == "2024-11-11")]
    ok = "Delhi-Jaipur in the week of 2024-11-11 rose 35.5% and is unexplained."
    assert validate_answer(ok, rows)
    assert not validate_answer(ok.replace("35.5", "52.0"), rows)  # invented number
    assert not validate_answer("It rose 35.5%.", rows)  # skips the week and the verdict



def test_llm_answer_that_skips_unexplained_weeks_is_rejected(df):
    partial = ("Chennai-Bangalore was pricier in March because of flooding (N001): "
               "the week of 2025-03-03 was +22.8% vs its past average.")
    text, usage = answer("why did this route get pricier in March?", df, Fake(partial))
    assert not usage["used_llm_answer"]
    assert "2025-03-10" in text and "2025-03-17" in text

