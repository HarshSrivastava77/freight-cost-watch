"""End-to-end run: numbers -> retrieval -> verdict -> explanation -> output.csv."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import config
from .costs import build_weekly_table, peer_routes
from .explain import Explainer
from .justify import decide, valid_notes_full_scan
from .llm import TokenLedger, make_client
from .notes import load_notes
from .output import fmt_cost, fmt_vs_own, fmt_vs_peer, write_output
from .retrieval import NoteRetriever, build_query

log = logging.getLogger(__name__)


def run(
    out_path: Path = config.OUTPUT_CSV,
    llm_provider: str = "auto",
    retriever_backend: str = "auto",
    token_log_path: Path = config.TOKEN_LOG_JSON,
    report_path: Path = config.RUN_REPORT_JSON,
    client=None,
) -> pd.DataFrame:
    """Run everything and write output.csv, token_log.json and run_report.json.

    `client` lets tests inject a fake LLM; otherwise one is built from `llm_provider`.
    """
    weekly = build_weekly_table()
    notes = load_notes()
    peers = peer_routes(weekly)
    retriever = NoteRetriever(notes, backend=retriever_backend)

    if client is None:
        client = make_client(llm_provider)
    ledger = TokenLedger(
        provider=client.provider if client else "none",
        model=client.model if client else "templates",
    )
    explainer = Explainer(client, ledger)

    rows, audit = [], []
    for rec in weekly.to_dict("records"):
        hits = []
        if rec["rule_flagged"]:
            route_filter = rec["route"] if config.RETRIEVAL_ROUTE_FILTER else None
            hits = retriever.search(build_query(rec["route"], rec["week_of"]), route=route_filter)
        verdict = decide(rec, hits, peers[rec["route"]])
        if rec["rule_flagged"]:
            valid = valid_notes_full_scan(rec, notes, peers[rec["route"]])
            audit.append({
                "route": rec["route"],
                "week_of": rec["week_of"],
                "retrieved": [{"note_id": h.note.note_id, "score": h.score} for h in hits],
                "checks": [{"note_id": c.note_id, "passed": c.passed, "code": c.code} for c in verdict.checks],
                "verdict": verdict.flagged,
                "matched_note_id": verdict.matched_note_id,
                "valid_notes_full_scan": valid,
                "retrieval_missed_valid_note": sorted(set(valid) - {h.note.note_id for h in hits}),
            })
        rows.append({
            "route": rec["route"],
            "week_of": rec["week_of"],
            "cost_per_tonne_km": fmt_cost(rec["cost_per_tonne_km"]),
            "vs_own_history": fmt_vs_own(rec["vs_own_pct"]),
            "vs_similar_routes": fmt_vs_peer(rec["vs_peer_pct"]),
            "flagged": verdict.flagged,
            "matched_note_id": verdict.matched_note_id,
            "reason": explainer.reason(rec, verdict),
        })

    write_output(rows, out_path)
    ledger.write(token_log_path)

    misses = [a for a in audit if a["retrieval_missed_valid_note"]]
    if misses:
        log.warning("retrieval missed a valid note for %d flagged rows -- see %s", len(misses), report_path)
    report = {
        "retriever_backend": retriever.backend,
        "top_k": config.RETRIEVAL_TOP_K,
        "route_filter": config.RETRIEVAL_ROUTE_FILTER,
        "thresholds": {"own_rise_pct": config.OWN_RISE_FLAG_PCT, "peer_gap_pct": config.PEER_GAP_FLAG_PCT},
        "route_weeks": len(rows),
        "rule_flagged": len(audit),
        "justified": sum(a["verdict"] == "No (justified)" for a in audit),
        "unexplained": sum(a["verdict"] == "Yes" for a in audit),
        "rows_with_fewer_than_8_prior_weeks": int(((weekly["own_weeks_used"] > 0) & (weekly["own_weeks_used"] < 8)).sum()),
        "rows_with_no_history": int((weekly["own_weeks_used"] == 0).sum()),
        "retrieval_misses": len(misses),
        "flagged_rows": audit,
    }
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return pd.DataFrame(rows)
