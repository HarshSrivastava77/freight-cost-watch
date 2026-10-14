#!/usr/bin/env bash
# Run the full pipeline 3 times in fresh processes and compare outputs.
# Every column except `reason` must be byte-identical (reason wording may vary
# when an LLM is used). Extra args are passed to run.py, e.g. ./repro_check.sh --no-llm
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p repro
for i in 1 2 3; do
  python run.py "$@" --out "repro/run_$i.csv" --token-log "repro/token_log_$i.json" \
    --report "repro/run_report_$i.json" 2>/dev/null
done

python - "$@" <<'PY' | tee repro/repro_result.txt
import hashlib, sys
import pandas as pd

runs = [pd.read_csv(f"repro/run_{i}.csv", dtype=str, keep_default_na=False) for i in (1, 2, 3)]
cols = [c for c in runs[0].columns if c != "reason"]
mode = " ".join(sys.argv[1:]) or "(default)"
print(f"repro check: 3 runs, run.py args: {mode}")
ok = True
for i in (2, 3):
    a, b = runs[0][cols], runs[i - 1][cols]
    if a.shape != b.shape or not a.equals(b):
        ok = False
        diff = (a != b).any(axis=1) if a.shape == b.shape else None
        print(f"run 1 vs run {i}: MISMATCH in non-reason columns"
              + (f" ({int(diff.sum())} rows)" if diff is not None else " (shape differs)"))
    else:
        print(f"run 1 vs run {i}: identical on {len(cols)} columns x {len(a)} rows")
    same_reason = runs[0]["reason"].equals(runs[i - 1]["reason"])
    print(f"run 1 vs run {i}: reason column {'identical' if same_reason else 'differs (allowed)'}")
for i in (1, 2, 3):
    h = hashlib.sha256(runs[i - 1][cols].to_csv(index=False, lineterminator="\n").encode()).hexdigest()[:16]
    print(f"run {i} sha256(non-reason columns) = {h}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
