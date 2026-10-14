"""CLI entry point.

    python run.py                 # LLM explanations if GROQ_API_KEY is set, else templates
    python run.py --no-llm        # templates only (zero LLM calls)
    python run.py --llm ollama    # local model via Ollama
"""
import argparse
import logging
from pathlib import Path

from fcw import config
from fcw.pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-llm", action="store_true", help="use template explanations only")
    p.add_argument("--llm", default="auto", choices=["auto", "groq", "ollama", "none"])
    p.add_argument("--retriever", default="auto", choices=["auto", "minilm", "tfidf"])
    p.add_argument("--out", type=Path, default=config.OUTPUT_CSV)
    p.add_argument("--token-log", type=Path, default=config.TOKEN_LOG_JSON)
    p.add_argument("--report", type=Path, default=config.RUN_REPORT_JSON)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = run(
        out_path=args.out,
        llm_provider="none" if args.no_llm else args.llm,
        retriever_backend=args.retriever,
        token_log_path=args.token_log,
        report_path=args.report,
    )
    counts = df["flagged"].value_counts().to_dict()
    print(f"wrote {args.out} ({len(df)} route-weeks): {counts}")


if __name__ == "__main__":
    main()
