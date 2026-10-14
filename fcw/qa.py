"""Answer plain-English questions using ONLY rows from output.csv.

1. parse_question() pulls deterministic filters out of the question
   (route / city, month, year, verdict words).
2. select_rows() filters output.csv. No rows -> "Not found in results."
3. The LLM (if any) sees only those rows and must answer from them; its answer is
   checked so every note id and number appears in the rows. Otherwise a template
   answer built from the rows is returned.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field

import pandas as pd

from .explain import NOTE_ID_RE, NUMBER_RE
from .llm import LLMError
from .output import to_ascii

NOT_FOUND = "Not found in results."
MAX_ROWS = 12

# "may" is handled separately: it is also an everyday verb.
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m and m != "May"}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m and m != "May"})
_RISE_WORDS = re.compile(r"\b(pricier|expensive|costlier|rise|rose|increase|jump|spike|up|why)\b", re.I)

QA_SYSTEM = (
    "You answer questions about freight cost alerts using ONLY the result rows provided. "
    "Cover EVERY flagged row and give its week_of in YYYY-MM-DD form. State the verdict of each row exactly as given: Yes means unexplained; No (justified) means justified by its matched_note_id. Never use a note to explain a row whose matched_note_id is blank. Plain text with no markdown. Quote numbers and note ids exactly "
    "as they appear in the rows. Do not add causes, notes or numbers that are not in the rows. "
    f"If the rows do not answer the question, reply exactly: {NOT_FOUND}"
)


@dataclass
class Filters:
    routes: list[str] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    wants_rise: bool = False

    @property
    def empty(self) -> bool:
        return not (self.routes or self.months or self.years or self.verdicts)


def parse_question(question: str, routes: list[str]) -> Filters:
    q = question.lower()
    f = Filters()

    for r in routes:
        a, b = (c.lower() for c in r.split("-"))
        if re.search(rf"\b{a}\b.*\b{b}\b", q):
            f.routes.append(r)
    if not f.routes:  # a single city -> every route touching it
        cities = {c.lower() for r in routes for c in r.split("-")}
        named = {c for c in cities if re.search(rf"\b{c}\b", q)}
        f.routes = [r for r in routes if named & {c.lower() for c in r.split("-")}]

    for m in re.findall(r"\b(20\d{2})-(\d{2})\b", q):
        f.years.append(int(m[0]))
        f.months.append(int(m[1]))
    for word, num in _MONTHS.items():
        if re.search(rf"\b{word}\b", q):
            f.months.append(num)
    if re.search(r"\bmay\b\s+20\d{2}|\bin may\b", q):  # 'may' only when clearly the month
        f.months.append(5)
    f.years += [int(y) for y in re.findall(r"\b(20\d{2})\b(?!-)", q)]

    if re.search(r"\b(unexplained|suspicious|unjustified|review)\b", q):
        f.verdicts.append("Yes")
    if re.search(r"\b(justified|explained by|which note)\b", q):
        f.verdicts.append("No (justified)")
    if re.search(r"\bflag", q) and not f.verdicts:
        f.verdicts += ["Yes", "No (justified)"]
    f.wants_rise = bool(_RISE_WORDS.search(q))

    f.months = sorted(set(f.months))
    f.years = sorted(set(f.years))
    return f


def _own_pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.extract(r"^([+-]\d+\.\d)")[0], errors="coerce")


def select_rows(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if f.empty:
        return df.iloc[0:0]
    rows = df
    weeks = pd.to_datetime(rows["week_of"])
    if f.routes:
        rows = rows[rows["route"].isin(f.routes)]
    if f.months:
        rows = rows[weeks.loc[rows.index].dt.month.isin(f.months)]
    if f.years:
        rows = rows[weeks.loc[rows.index].dt.year.isin(f.years)]
    if f.verdicts:
        rows = rows[rows["flagged"].isin(f.verdicts)]
    if rows.empty:
        return rows
    flagged = rows[rows["flagged"] != "No"]
    if f.wants_rise and not flagged.empty:
        rows = flagged
    rank = rows.assign(_f=(rows["flagged"] == "No").astype(int), _own=_own_pct(rows["vs_own_history"]))
    rank = rank.sort_values(["_f", "_own"], ascending=[True, False]).head(MAX_ROWS)
    return rank.drop(columns=["_f", "_own"]).sort_values(["route", "week_of"])


def template_answer(rows: pd.DataFrame) -> str:
    if rows.empty:
        return NOT_FOUND
    lines = []
    for r in rows.itertuples(index=False):
        note = f" [{r.matched_note_id}]" if r.matched_note_id else ""
        lines.append(f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km} INR/t-km; "
                     f"{r.vs_own_history}; {r.vs_similar_routes}. Flagged: {r.flagged}{note}. {r.reason}")
    head = "From the results:" if (rows["flagged"] != "No").any() else \
        "No flagged week matches this question. Matching results:"
    return head + "\n" + "\n".join(lines)


def validate_answer(answer: str, rows: pd.DataFrame) -> bool:
    if answer.strip() == NOT_FOUND:
        return True
    flagged = rows[rows["flagged"] != "No"]
    if any(w not in answer for w in flagged["week_of"]):
        return False  # skipping a flagged week can make a note look like it covers that week
    if (rows["flagged"] == "Yes").any() and "unexplained" not in answer.lower():
        return False  # unexplained weeks must be called unexplained
    context = rows.to_csv(index=False)
    if set(NOTE_ID_RE.findall(answer)) - set(NOTE_ID_RE.findall(context)):
        return False
    allowed = {float(x) for x in NUMBER_RE.findall(NOTE_ID_RE.sub("", context))}
    return all(any(abs(float(t) - a) < 0.051 for a in allowed)
               for t in NUMBER_RE.findall(NOTE_ID_RE.sub("", answer)))


def answer(question: str, df: pd.DataFrame, client=None) -> tuple[str, dict]:
    """Returns (answer_text, usage)."""
    f = parse_question(question, sorted(df["route"].unique()))
    rows = select_rows(df, f)
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "rows_used": len(rows), "used_llm_answer": False}
    if rows.empty:
        return NOT_FOUND, usage
    if client is not None:
        prompt = f"Question: {question}\n\nResult rows (CSV):\n{rows.to_csv(index=False)}"
        try:
            res = client.complete(QA_SYSTEM, prompt, max_tokens=500)
            usage.update(llm_calls=1, input_tokens=res.input_tokens, output_tokens=res.output_tokens)
            text = to_ascii(res.text).replace('**', '')
            if validate_answer(text, rows):
                usage["used_llm_answer"] = True
                return text, usage
        except LLMError:
            pass
    return template_answer(rows), usage

