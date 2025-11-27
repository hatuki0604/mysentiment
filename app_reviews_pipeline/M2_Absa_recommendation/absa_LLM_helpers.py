"""
ABSA LLM Helpers
Organized with section comments for clarity and maintainability.
"""

# ===========================
# Imports & Configuration
# ===========================
import os
import json
import re
import logging
import openai
import math
from typing import Optional, List, Dict, Callable

logger = logging.getLogger(__name__)

# ===========================
# Chat Function Registration
# ===========================
# Optional shared chat function (OpenAI/Mistral/Llama2) injected from absa_recommendation.py
ChatFn = Callable[[List[Dict[str, str]], float, int], str]
_CHAT_FN: Optional[ChatFn] = None

def set_chat_fn(fn: ChatFn) -> None:
    """Register a shared chat function (e.g., llm_config.get_llm(...))."""
    global _CHAT_FN
    _CHAT_FN = fn

# ===========================
# LLM Call Logic
# ===========================
def call_llm(prompt: str, model: str = "gpt-4o-mini", temp: float = 0.2, max_tokens: int = 256,
             chat_fn: Optional[ChatFn] = None) -> str:
    """
    If chat_fn (or a previously set _CHAT_FN) is available, use that.
    Otherwise, fall back to OpenAI ChatCompletion for backwards compatibility.
    """
    fn = chat_fn or _CHAT_FN
    if fn:
        return fn([{"role": "user", "content": prompt}], temperature=temp, max_tokens=max_tokens)

    # Fallback: direct OpenAI (legacy)
    if not openai.api_key:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

# ===========================
# Parsing Utilities
# ===========================
def _find_json_snippet(s: str) -> str | None:
    """Extract the first JSON object/array; strips code fences and trailing commas."""
    if not s:
        return None
    txt = s.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(\w+)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    m = re.search(r"\{[\s\S]*\}", txt) or re.search(r"\[[\s\S]*\]", txt)
    if not m:
        return None
    cand = m.group(0)
    cand = re.sub(r",(\s*[}\]])", r"\1", cand)
    return cand

def safe_parse_list(raw: str) -> list:
    """Parse a JSON list; fallback to simple line/CSV parsing."""
    j = _find_json_snippet(raw)
    if j:
        try:
            val = json.loads(j)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            logger.warning("Failed JSON list parse; falling back. Raw: %r", raw)
    # fallback
    items = []
    for line in (raw or "").splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        if "," in line and "[" not in line and "]" not in line:
            parts = [p.strip() for p in line.split(",")]
            items.extend([p for p in parts if p])
        else:
            items.append(line)
    return items

def safe_parse_dict(raw: str) -> dict:
    """Parse a JSON dict; fallback to lines like 'battery -> Negative'."""
    j = _find_json_snippet(raw)
    if j:
        try:
            val = json.loads(j)
            if isinstance(val, dict):
                return {str(k): str(v) for k, v in val.items()}
        except Exception:
            logger.warning("Failed JSON dict parse; falling back. Raw: %r", raw)
    mapping = {}
    for line in (raw or "").splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        m = re.match(r'(.+?)(?:\s*[:\-–>]\s*)(Positive|Negative|Neutral)\b', line, re.I)
        if m:
            k = m.group(1).strip().strip('"').strip("'")
            v = m.group(2).strip().capitalize()
            mapping[k] = v
    return mapping

def safe_parse_list_of_lists(raw: str) -> list[list[str]]:
    """Parse a JSON array of arrays; fallback to bracket scanning."""
    j = _find_json_snippet(raw)
    if j:
        try:
            val = json.loads(j)
            if isinstance(val, list):
                return [
                    [str(x) for x in (row if isinstance(row, list) else [row])]
                    for row in val
                ]
        except Exception:
            logger.warning("Failed JSON list-of-lists parse; falling back. Raw: %r", raw)
    groups = re.findall(r'\[([^\[\]]+)\]', raw or "")
    return [re.findall(r'"([^"]+)"', g) for g in groups]

# ===========================
# Compute confidence from logprobs
# ===========================

SENTIMENT_LABELS = ["Positive", "Negative", "Neutral"]

def compute_confidence_from_logprobs(resp):
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()

    # Logprobs structure changed in API 2024
    logprob_items = choice.logprobs.content

    if not logprob_items:
        return content, 0.0, {}

    # Lấy token cuối cùng (thường là Positive/Negative/Neutral)
    last = logprob_items[-1]

    top = last.top_logprobs or []
    probs = {}

    for item in top:
        tok = item.token.strip()
        lp = item.logprob
        if tok in ["Positive", "Negative", "Neutral"]:
            probs[tok] = math.exp(lp)

    # ensure all 3 present
    for lbl in ["Positive", "Negative", "Neutral"]:
        probs.setdefault(lbl, 1e-12)

    total = sum(probs.values())
    conf = {k: v / total for k, v in probs.items()}

    return content, conf.get(content, 0.0), conf
