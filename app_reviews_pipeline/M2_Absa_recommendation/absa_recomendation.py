"""
M2 - ABSA Pipeline (Aspect + Sentiment Only)
Recommendation module has been completely removed.
"""

# ===========================
# Imports & Path Setup  
# ===========================

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Path setup
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
for p in (HERE.parent, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Local helpers
from absa_LLM_helpers import (
    call_llm,
    safe_parse_list,
    safe_parse_dict,
    set_chat_fn,
)

# Prompts (NO recommendation prompt anymore)
from absa_prompts import (
    ASPECT_PROMPT,
    SENTIMENT_PROMPT,
)

# Shared menus + LLM factory
from user_selection import (
    choose_dataset,
    choose_provider_and_model,
    choose_sample_size,
)
from llm_config import get_llm, ChatFn
from prompt_optimize import optimize_prompt

load_dotenv()

PROMPTS: Dict[str, str] = {
    "aspect": ASPECT_PROMPT,
    "sentiment": SENTIMENT_PROMPT,
}

# ===========================
# Aspect extraction
# ===========================
def extract_aspects(sentences: List[str], chat_fn: ChatFn, model: str = "gpt-4o-mini") -> pd.DataFrame:
    records = []
    for s in tqdm(sentences, desc="Aspects", unit="sent"):
        prompt = PROMPTS["aspect"].replace("{sentence}", str(s))
        raw = call_llm(prompt, model=model, max_tokens=256, chat_fn=chat_fn)
        aspects = safe_parse_list(raw)

        clean, seen = [], set()
        for term in aspects:
            t = str(term).lower().strip()
            t = re.sub(r'^[^\w]+|[^\w]+$', '', t)
            if t and t not in seen:
                clean.append(t)
                seen.add(t)

        records.append({"sentence": s, "aspects": clean})
    return pd.DataFrame(records)

# ===========================
# Sentiment classification
# ===========================
def classify_sentiments(df: pd.DataFrame, chat_fn: ChatFn, model: str = "gpt-4o-mini") -> pd.DataFrame:
    out = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Sentiments", unit="sent"):
        aspects = row.aspects
        if not aspects:
            out.append({})
            continue

        prompt = (
            PROMPTS["sentiment"]
            .replace("{sentence}", str(row.sentence))
            .replace("{aspects}", json.dumps(aspects, ensure_ascii=False))
        )
        raw = call_llm(prompt, model=model, max_tokens=256, chat_fn=chat_fn)
        out.append(safe_parse_dict(raw))

    df = df.copy()
    df["sentiments"] = out
    return df

# ===========================
# Pipeline helpers
# ===========================
# def split_into_sentences(text: str) -> List[str]:
#     s = str(text or "").strip()
#     if not s:
#         return []
#     try:
#         import nltk
#         from nltk.tokenize import sent_tokenize
#         try:
#             nltk.data.find("tokenizers/punkt")
#         except LookupError:
#             nltk.download("punkt", quiet=True)
#         return [t.strip() for t in sent_tokenize(s) if t.strip()]
#     except Exception:
#         parts = re.split(r'(?<=[.!?])\s+', s)
#         return [t.strip() for t in parts if t.strip()]

def split_into_sentences(text: str) -> List[str]:
    """Không tách câu — giữ nguyên toàn bộ review như một câu duy nhất."""
    s = str(text or "").strip()
    return [s] if s else []


def explode_reviews_to_sentences(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        sents = split_into_sentences(r["review_text"])
        if not sents:
            sents = [str(r["review_text"]).strip()]
        for i, sent in enumerate(sents):
            rows.append({"review_id": r["review_id"], "sent_idx": i, "sentence": sent})
    return pd.DataFrame(rows)

# ===========================
# Full ABSA
# ===========================
def full_analysis(sentences: List[str], chat_fn: ChatFn, model: str) -> pd.DataFrame:
    df = extract_aspects(sentences, chat_fn, model)
    df = classify_sentiments(df, chat_fn, model)
    return df

# ===========================
# Saving output
# ===========================
def save_csv_and_jsonl(df: pd.DataFrame, out_dir: Path, ds: str, model: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(":", "_")
    csv_path = out_dir / f"{ds}_absa_{safe_model}_{len(df)}.csv"
    jsonl_path = out_dir / f"{ds}_absa_{safe_model}_{len(df)}.jsonl"
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path

# ===========================
# Summary (NO recommendation)
# ===========================
def summarize_aspects(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in result_df.iterrows():
        aspects = r.aspects or []
        sentiments = r.sentiments or {}
        for a in aspects:
            s = sentiments.get(a)
            if isinstance(s, str):
                s = s.capitalize()
            if s not in {"Positive", "Negative", "Neutral"}:
                s = None
            rows.append({"aspect": a, "sentiment": s})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["aspect", "mentions", "Positive", "Negative", "Neutral", "positivity_pct"])

    freq = df.groupby("aspect").size().rename("mentions")
    sent = df.pivot_table(index="aspect", columns="sentiment", aggfunc="size", fill_value=0)

    for col in ["Positive", "Negative", "Neutral"]:
        if col not in sent.columns:
            sent[col] = 0

    out = pd.concat([freq, sent[["Positive", "Negative", "Neutral"]]], axis=1).reset_index()
    denom = out[["Positive", "Negative", "Neutral"]].sum(axis=1).replace(0, 1)
    out["positivity_pct"] = (out["Positive"] / denom).round(3)
    return out

# ===========================
# Entrypoint
# ===========================
def main():
    print("\n=== ABSA Pipeline (No Recommendation) ===")

    ds, csv_path, topic = choose_dataset()

    provider, model = choose_provider_and_model()
    chat_fn = get_llm(provider, model)
    set_chat_fn(chat_fn)

    base = pd.read_csv(csv_path, usecols=["review_id", "review_text"])
    sample_n = choose_sample_size(len(base))
    if sample_n:
        base = base.sample(n=sample_n, random_state=42).reset_index(drop=True)

    sample_texts = base["review_text"].astype(str).head(200).tolist()
    print("\nOptimizing prompts...")

    tuned_aspect, _ = optimize_prompt(
        dataset=ds,
        sample_texts=sample_texts,
        chat_fn=chat_fn,
        base_prompt=PROMPTS["aspect"],
        provider=provider,
        model=model,
        enabled=True,
        prompt_type="aspect"
    )

    PROMPTS.update({"aspect": tuned_aspect})

    sent_df = explode_reviews_to_sentences(base)
    sentences = sent_df["sentence"].tolist()

    result_df = full_analysis(sentences, chat_fn, model)

    result_df.insert(0, "review_id", sent_df["review_id"].values)
    result_df.insert(1, "sent_idx", sent_df["sent_idx"].values)

    out_dir = Path(f"outputs/absa/{ds}")
    
    csv_path_out, jsonl_path_out = save_csv_and_jsonl(result_df, out_dir, ds, model)

    aspects_summary = summarize_aspects(result_df)
    aspects_summary.to_csv(out_dir / f"{ds}_absa_aggregate_aspects.csv", index=False)

    total_aspects = int(result_df.aspects.map(lambda x: len(x) if isinstance(x, list) else 0).sum())

    print("\n=== Run Summary ===")
    print(f"Dataset:        {ds}")
    print(f"Provider/Model: {provider} / {model}")
    print(f"Sentences:      {len(result_df)}")
    print(f"Aspects:        {total_aspects}")
    print("\nSaved files:")
    print(f"  • Rows (CSV):      {csv_path_out}")
    print(f"  • Rows (JSONL):    {jsonl_path_out}")
    print(f"  • Aspects summary: {out_dir / f'{ds}_absa_aggregate_aspects.csv'}")


if __name__ == "__main__":
    main()
