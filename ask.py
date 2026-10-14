"""Ask questions about the results in plain English.

    python ask.py "Why did Chennai-Bangalore get pricier in March 2025?"
    python ask.py --no-llm "Which routes were unexplained in November?"

Answers come only from output.csv rows. If nothing matches: "Not found in results."
"""
import argparse
import sys

import pandas as pd

from fcw import config
from fcw.llm import make_client
from fcw.qa import answer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="+")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm", default="auto", choices=["auto", "groq", "ollama", "none"])
    p.add_argument("--results", default=str(config.OUTPUT_CSV))
    args = p.parse_args()

    df = pd.read_csv(args.results, dtype=str, keep_default_na=False)
    client = make_client("none" if args.no_llm else args.llm)
    text, usage = answer(" ".join(args.question), df, client)
    print(text)
    print(f"\n[rows used: {usage['rows_used']}, llm calls: {usage['llm_calls']}, "
          f"tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}, "
          f"llm answer accepted: {usage['used_llm_answer']}]", file=sys.stderr)


if __name__ == "__main__":
    main()
