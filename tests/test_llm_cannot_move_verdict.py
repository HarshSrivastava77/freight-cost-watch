"""Whatever the LLM returns, flags, numbers and matched_note_id must not change."""
import random

import pandas as pd

from fcw.llm import LLMResult
from fcw.pipeline import run


class ChaoticLLM:
    provider, model = "groq", "chaos"

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def complete(self, system, user, max_tokens=120):
        options = [
            "This was caused by N003 diesel prices and N009 floods; up 57.3%.",
            "Justified.",
            f"Cost moved {self.rng.uniform(0, 99):.1f}% for reasons in N00{self.rng.randint(1, 9)}.",
            "The rise matches the numbers given, with no other detail available.",
        ]
        return LLMResult(self.rng.choice(options), 150, 30)


def _run(tmp_path, name, **kw):
    out = tmp_path / f"{name}.csv"
    run(out_path=out, token_log_path=tmp_path / f"{name}.json", report_path=tmp_path / f"{name}_r.json", **kw)
    return pd.read_csv(out, dtype=str, keep_default_na=False)


def test_llm_output_never_changes_verdicts_or_numbers(tmp_path):
    base = _run(tmp_path, "base", llm_provider="none")
    cols = [c for c in base.columns if c != "reason"]
    for seed in range(3):
        chaos = _run(tmp_path, f"chaos{seed}", client=ChaoticLLM(seed))
        pd.testing.assert_frame_equal(base[cols], chaos[cols])
        # an unexplained row's reason never names a note other than the deterministic suffix's
        bad = chaos[(chaos.flagged == "Yes") & chaos.reason.str.contains("57.3")]
        assert bad.empty
