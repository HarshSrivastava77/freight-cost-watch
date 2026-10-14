"""All tunable settings in one place. Nothing else in the package hard-codes these."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SHIPMENTS_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"
NOTES_STRUCTURED_JSON = DATA_DIR / "notes_structured.json"
SAMPLE_OUTPUT_CSV = DATA_DIR / "sample_output_format_v2.csv"
OUTPUT_CSV = ROOT / "output.csv"
TOKEN_LOG_JSON = ROOT / "token_log.json"
RUN_REPORT_JSON = ROOT / "run_report.json"

# --- Baselines (fixed by the brief) -----------------------------------------
OWN_HISTORY_WEEKS = 8  # trailing window, strictly before the current week

# --- Flag rule ---------------------------------------------------------------
# flagged  <=>  vs_own > 0  AND  (vs_own >= OWN_RISE_FLAG_PCT  OR  vs_peer >= PEER_GAP_FLAG_PCT)
OWN_RISE_FLAG_PCT = 15.0
PEER_GAP_FLAG_PCT = 20.0

# --- Retrieval ---------------------------------------------------------------
RETRIEVAL_TOP_K = 3
RETRIEVAL_ROUTE_FILTER = True  # metadata filter: only notes for this route or "All Routes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Justification guardrails ------------------------------------------------
# A note with max_effect_pct = M can explain a rise of at most M + tolerance points.
PLAUSIBILITY_TOLERANCE_PCT = 2.0
# Notes that give no figure (e.g. a flood) still get a sanity cap: an event note
# should not be able to "explain" an arbitrarily large jump.
UNBOUNDED_NOTE_CAP_PCT = 50.0

# --- LLM ---------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_MAX_TOKENS = 400
LLM_TIMEOUT_S = 30
LLM_RETRIES = 3
# USD per 1M tokens (input, output). Groq published rate for openai/gpt-oss-20b;
# Ollama runs locally so has no per-token charge.
PRICE_USD_PER_1M = {"groq": (0.075, 0.30), "ollama": (0.0, 0.0), "none": (0.0, 0.0)}
GROQ_REASONING_EFFORT = "low"  # gpt-oss is a reasoning model; keep its hidden reasoning short

