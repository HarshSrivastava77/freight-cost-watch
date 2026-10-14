"""Thin LLM clients (stdlib HTTP only) plus a token/cost ledger.

Both clients run at temperature 0 with a fixed seed. The LLM never decides a
verdict or picks a note; it only words an explanation that is then validated.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode()
    last_exc: Exception | None = None
    for attempt in range(config.LLM_RETRIES):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "freight-cost-watch/1.0", **headers})
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
        time.sleep(2 ** attempt)
    raise LLMError(f"LLM request failed: {last_exc}")


class GroqClient:
    provider = "groq"

    def __init__(self, model: str = config.GROQ_MODEL):
        self.model = model
        self._key = os.environ.get("GROQ_API_KEY")
        if not self._key:
            raise LLMError("GROQ_API_KEY is not set")

    def complete(self, system: str, user: str, max_tokens: int = config.LLM_MAX_TOKENS) -> LLMResult:
        data = _post_json(
            config.GROQ_URL,
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": config.LLM_TEMPERATURE,
                "seed": config.LLM_SEED,
                "max_tokens": max_tokens, "reasoning_effort": config.GROQ_REASONING_EFFORT,
            },
            {"Authorization": f"Bearer {self._key}"},
        )
        usage = data.get("usage", {})
        return LLMResult(
            data["choices"][0]["message"]["content"].strip(),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


class OllamaClient:
    provider = "ollama"

    def __init__(self, model: str = config.OLLAMA_MODEL):
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = config.LLM_MAX_TOKENS) -> LLMResult:
        data = _post_json(
            config.OLLAMA_URL,
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": config.LLM_TEMPERATURE, "seed": config.LLM_SEED, "num_predict": max_tokens},
            },
            {},
        )
        return LLMResult(
            data["message"]["content"].strip(),
            int(data.get("prompt_eval_count", 0)),
            int(data.get("eval_count", 0)),
        )


def make_client(provider: str):
    """provider: 'groq' | 'ollama' | 'none' | 'auto' (groq if GROQ_API_KEY is set, else none)."""
    if provider == "auto":
        provider = "groq" if os.environ.get("GROQ_API_KEY") else "none"
    if provider == "none":
        return None
    if provider == "groq":
        return GroqClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"unknown LLM provider {provider!r}")


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) -- only used for the --no-llm cost projection."""
    return math.ceil(len(text) / 4)


@dataclass
class TokenLedger:
    provider: str
    model: str
    calls: list[dict] = field(default_factory=list)
    projected_prompts: list[tuple[str, str]] = field(default_factory=list)

    def record(self, label: str, result: LLMResult | None, accepted: bool,
               problem: str = "", error: str = "") -> None:
        self.calls.append({
            "row": label,
            "input_tokens": result.input_tokens if result else 0,
            "output_tokens": result.output_tokens if result else 0,
            "validator_accepted": accepted,
            "validator_problem": problem,
            "error": error,
        })

    def to_dict(self) -> dict:
        price_in, price_out = config.PRICE_USD_PER_1M.get(self.provider, (0.0, 0.0))
        tin = sum(c["input_tokens"] for c in self.calls)
        tout = sum(c["output_tokens"] for c in self.calls)
        out = {
            "provider": self.provider,
            "model": self.model,
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
            "max_output_tokens_per_call": config.LLM_MAX_TOKENS,
            "llm_calls": len(self.calls),
            "total_input_tokens": tin,
            "total_output_tokens": tout,
            "validator_rejections": sum(1 for c in self.calls if not c["validator_accepted"] and not c["error"]),
            "call_errors": sum(1 for c in self.calls if c["error"]),
            "template_fallbacks": sum(1 for c in self.calls if not c["validator_accepted"]),
            "price_usd_per_1m_tokens": {"input": price_in, "output": price_out},
            "cost_usd": round((tin * price_in + tout * price_out) / 1e6, 8),
            "calls": self.calls,
        }
        if self.projected_prompts:
            g_in, g_out = config.PRICE_USD_PER_1M["groq"]
            p_in = sum(approx_tokens(s) + approx_tokens(u) for s, u in self.projected_prompts)
            p_out = len(self.projected_prompts) * config.LLM_MAX_TOKENS
            out["projection_if_run_with_groq"] = {
                "note": "ESTIMATE, not measured: this run used templates. Input tokens ~ chars/4 of the "
                        "exact prompts that would have been sent; output assumes the max_tokens ceiling.",
                "llm_calls": len(self.projected_prompts),
                "approx_input_tokens": p_in,
                "max_output_tokens": p_out,
                "approx_cost_usd_upper_bound": round((p_in * g_in + p_out * g_out) / 1e6, 8),
            }
        return out

    def write(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")


