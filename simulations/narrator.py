"""
Optional: board-level incident brief from a LOCAL model (LM Studio or mlx_lm.server).

The simulator is deterministic by design; the LLM only *narrates* numbers the
engine already computed. It is never in the decision path, so a hallucination
cannot change a metric. If no local server is reachable, a deterministic
template is returned instead.

Both LM Studio and `mlx_lm.server` expose an OpenAI-compatible
/v1/chat/completions endpoint. Configure with env vars:
    WARROOM_LLM_BASE_URL   default http://localhost:1234/v1   (LM Studio)
                           e.g.    http://localhost:8080/v1   (mlx_lm.server)
    WARROOM_LLM_MODEL      default: first model the server lists
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE_URL = os.environ.get("WARROOM_LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
TIMEOUT_S = float(os.environ.get("WARROOM_LLM_TIMEOUT", "60"))

SYSTEM = (
    "You are the Head of AI writing a SEV-1 brief for the CFO, CISO and CTO. "
    "Use ONLY the numbers provided; never invent figures. Structure: 1) What happened (2 sentences), "
    "2) Business impact (cost, customers, regulatory), 3) Decision taken and trade-offs accepted, "
    "4) Next 72h actions with owners (Platform Eng, Security, FinOps). Max 220 words. No hedging."
)


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=3) as r:
        return json.loads(r.read())


def available() -> tuple[bool, str]:
    try:
        models = _get("/models").get("data", [])
        return (True, models[0]["id"]) if models else (False, "server up, no model loaded")
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
        return False, f"unreachable ({exc.__class__.__name__})"


def generate_brief(facts: list[str]) -> tuple[str, str]:
    """Returns (brief, source) where source is the model id or 'template'."""
    ok, model = available()
    model = os.environ.get("WARROOM_LLM_MODEL", model)
    if ok:
        body = json.dumps({
            "model": model, "temperature": 0.2, "max_tokens": 450,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Facts:\n- " + "\n- ".join(facts)}],
        }).encode()
        req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip(), model
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError):
            pass
    return "**Deterministic brief (no local LLM reachable)**\n\n- " + "\n- ".join(facts), "template"
