"""
M2 - ABSA Recommendation Pipeline
Aspect-Based Sentiment Analysis with Recommendation Generation

This module performs comprehensive ABSA analysis on app reviews:
1. Aspect Extraction: Identify key aspects mentioned in reviews
2. Sentiment Classification: Determine sentiment for each aspect  
3. Recommendation Generation: Create actionable recommendations
4. Quality Evaluation: Assess output completeness and accuracy

Usage:
  python -m app_reviews_pipeline.M2_Absa_recommendation.absa_recommendation
  
  # Or directly:
  python app_reviews_pipeline/M2_Absa_recommendation/absa_recommendation.py

Requirements:
  - LLM API key (OpenAI/Mistral/LLaMA)
  - Preprocessed review data in data/processed/
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

# ── Path setup so imports work both as script and module ──────────────────────
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]  # .../app_reviews_pipeline
for p in (HERE.parent, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Local helpers (same folder)
from absa_LLM_helpers import (  # type: ignore
    call_llm,
    safe_parse_list,
    safe_parse_dict,
    set_chat_fn,
    compute_confidence_from_logprobs,
)

# Prompt templates (same folder)
from absa_prompts import (  # type: ignore
    RECO_PROMPT,
    ASPECT_PROMPT,
    SENTIMENT_PROMPT,
)

# Shared menus + LLM factory (package root)
from user_selection import (
    choose_dataset,
    choose_provider_and_model,
    choose_sample_size,
    choose_prompt_tuning,
)
from llm_config import get_llm, ChatFn
from prompt_optimize import optimize_prompt

load_dotenv()

PROMPTS: Dict[str, str] = {
    "aspect": ASPECT_PROMPT,
    "sentiment": SENTIMENT_PROMPT,
    "recommendation": RECO_PROMPT,
}

# ──────────────────────────────────────────────────────────────────────────────
# Aspect extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_aspects(sentences: List[str], topic: str, chat_fn: ChatFn, model: str = "gpt-4o-mini") -> pd.DataFrame:
    records = []
    for s in tqdm(sentences, desc="Aspects", unit="sent"):
        topic_prompt = f"Bạn là chuyên gia phân tích và đánh giá sản phẩm. {topic}."
        prompt = topic_prompt + PROMPTS["aspect"].replace("{sentence}", str(s))
        raw = call_llm(prompt, model=model, max_tokens=256, chat_fn=chat_fn)
        aspects = safe_parse_list(raw)

        # clean + dedupe
        clean, seen = [], set()
        for term in aspects:
            t = str(term).lower().strip()
            t = re.sub(r'^[^\w]+|[^\w]+$', '', t)
            if t and t not in seen:
                clean.append(t)
                seen.add(t)
        records.append({"sentence": s, "aspects": clean})
    return pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────────────────────
# Sentiment classification (per aspect)
# ──────────────────────────────────────────────────────────────────────────────
# def classify_sentiments(df: pd.DataFrame, chat_fn: ChatFn, model: str = "gpt-4o-mini") -> pd.DataFrame:
#     out = []
#     for row in tqdm(df.itertuples(index=False), total=len(df), desc="Sentiments", unit="sent"):
#         aspects = row.aspects
#         if not aspects:
#             out.append({})
#             continue
#         prompt = (
#             PROMPTS["sentiment"]
#             .replace("{sentence}", str(row.sentence))
#             .replace("{aspects}", json.dumps(aspects, ensure_ascii=False))
#         )
#         raw = call_llm(prompt, model=model, max_tokens=256, chat_fn=chat_fn)
#         out.append(safe_parse_dict(raw))
#     df = df.copy()
#     df["sentiments"] = out
#     return df

def classify_sentiments(df: pd.DataFrame, chat_fn: ChatFn, model: str = "gpt-4o-mini") -> pd.DataFrame:
    results = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Sentiments"):
        aspects = row.aspects
        if not aspects:
            results.append({})
            continue

        aspect_result = {}

        for aspect in aspects:

            ## Mini prompt for 1-token output (for logprobs)
            prompt = (
                f"Review: {row.sentence}\n"
                f"Aspect: {aspect}\n"
            )

            messages = [
                {"role": "system", "content": PROMPTS["sentiment"]},
                {"role": "user", "content": prompt},
            ]

            # Call LLM with logprobs
            resp = chat_fn(messages, with_logprobs=True)

            # Extract label + confidence
            label, conf, _ = compute_confidence_from_logprobs(resp)

            aspect_result[aspect] = {
                "label": label,
                "confidence": round(conf, 4)
            }

        results.append(aspect_result)

    df = df.copy()
    df["sentiments"] = results
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Recommendation mining
# ──────────────────────────────────────────────────────────────────────────────
def extract_recommendations(sentences: List[str], chat_fn: ChatFn, model: str = "gpt-4o-mini") -> List[List[str]]:
    recos = []
    for s in tqdm(sentences, desc="Recommendations", unit="sent"):
        prompt = PROMPTS["recommendation"].replace("{sentence}", str(s))
        raw = call_llm(prompt, model=model, max_tokens=120, chat_fn=chat_fn)
        items = safe_parse_list(raw)
        recos.append([i.strip().rstrip(".") for i in items if str(i).strip()])
    return recos

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# # ──────────────────────────────────────────────────────────────────────────────
# def split_into_sentences(text: str) -> List[str]:
#     """Best-effort splitter: NLTK if available; else a simple regex fallback."""
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
    """Turn review-level rows into sentence-level rows with review_id + sent_idx."""
    rows = []
    for _, r in df.iterrows():
        sents = split_into_sentences(r["review_text"])
        if not sents:  # if no punctuation, keep whole review as one "sentence"
            sents = [str(r["review_text"]).strip()]
        for i, sent in enumerate(sents):
            if sent:
                rows.append({"review_id": r["review_id"], "sent_idx": i, "sentence": sent})
    return pd.DataFrame(rows)

def full_analysis(sentences: List[str], topic: str, chat_fn: ChatFn, model: str = "gpt-4o-mini") -> pd.DataFrame:
    df = extract_aspects(sentences, topic, chat_fn=chat_fn, model=model)
    df = classify_sentiments(df, chat_fn=chat_fn, model=model)
    df["recommendations"] = extract_recommendations(df["sentence"].tolist(), chat_fn=chat_fn, model=model)
    return df

def save_csv_and_jsonl(df: pd.DataFrame, out_dir: Path, ds: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{ds}_absa_{len(df)}.csv"
    jsonl_path = out_dir / f"{ds}_absa_{len(df)}.jsonl"

    df.to_csv(csv_path, index=False)  # lists/dicts become strings; fine for CSV
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path

def summarize_aggregates(result_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two tables:
      1) aspect-level counts with sentiment breakdown
      2) recommendation phrase counts
    """
    # aspects + sentiments
    rows = []
    for _, r in result_df.iterrows():
        aspects = r.get("aspects", []) or []
        sentiments = r.get("sentiments", {}) or {}
        if not isinstance(aspects, list):
            aspects = []
        if not isinstance(sentiments, dict):
            sentiments = {}
        for a in aspects:
            a_str = str(a).strip()
            if not a_str:
                continue
            # s = sentiments.get(a) or sentiments.get(a_str)
            # if isinstance(s, str):
            #     s = s.strip().capitalize()
            #     if s not in {"Positive", "Negative", "Neutral"}:
            #         s = None

            s = sentiments.get(a) or sentiments.get(a_str)
            # CASE 1: bản cũ (s là string)
            if isinstance(s, str):
                s = s.strip().capitalize()

            # CASE 2: bản mới (s là dict với {"label": "...", "confidence": ...})
            elif isinstance(s, dict):
                label = s.get("label", "")
                if isinstance(label, str):
                    s = label.strip().capitalize()
                else:
                    s = None
            else:
                s = None

            if s not in {"Positive", "Negative", "Neutral"}:
                s = None

            rows.append({"aspect": a_str, "sentiment": s})
    aspects_df = pd.DataFrame(rows)

    if aspects_df.empty:
        aspects_summary = pd.DataFrame(
            columns=["aspect", "mentions", "Positive", "Negative", "Neutral", "positivity_pct"]
        )
    else:
        freq = aspects_df.groupby("aspect").size().rename("mentions")
        sent = aspects_df.pivot_table(index="aspect", columns="sentiment", aggfunc="size", fill_value=0)
        for col in ["Positive", "Negative", "Neutral"]:
            if col not in sent.columns:
                sent[col] = 0
        sent = sent[["Positive", "Negative", "Neutral"]]
        aspects_summary = (
            pd.concat([freq, sent], axis=1)
            .reset_index()
            .sort_values(["mentions", "Positive", "Negative"], ascending=[False, False, True])
        )
        denom = (aspects_summary[["Positive", "Negative", "Neutral"]].sum(axis=1)).replace(0, 1)
        aspects_summary["positivity_pct"] = (aspects_summary["Positive"] / denom).round(3)

    # recommendations
    reco_rows = []
    for _, r in result_df.iterrows():
        recs = r.get("recommendations", []) or []
        if isinstance(recs, list):
            for rec in recs:
                rec_str = str(rec).strip()
                if rec_str:
                    reco_rows.append({"recommendation": rec_str})

    if reco_rows:
        recos_df = (
            pd.DataFrame(reco_rows)
            .value_counts("recommendation")
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
    else:
        recos_df = pd.DataFrame(columns=["recommendation", "count"])

    return aspects_summary, recos_df

def save_aggregate_tables(aspects_summary: pd.DataFrame, recos_summary: pd.DataFrame, out_dir: Path, ds: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    a_path = out_dir / f"{ds}_absa_aggregate_aspects.csv"
    r_path = out_dir / f"{ds}_absa_aggregate_recommendations.csv"
    aspects_summary.to_csv(a_path, index=False)
    recos_summary.to_csv(r_path, index=False)
    return a_path, r_path

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== ABSA + Recommendations ===")

    # 1) Dataset (clean CSV) - Chọn dataset
    ds, csv_path, topic = choose_dataset()

    # 2) Provider + model - Chọn model ?
    provider, model = choose_provider_and_model()
    chat_fn = get_llm(provider, model)
    set_chat_fn(chat_fn)  # optional; we still pass chat_fn explicitly

    # 3) Load + optional sample - Chọn sample size
    base = pd.read_csv(csv_path, usecols=["review_id", "review_text"])
    sample_n = choose_sample_size(len(base))
    if sample_n:
        base = base.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # 4) Prompt optimization with automatic evaluation - Tối ưu prompt
    sample_texts = base["review_text"].astype(str).head(500).tolist()

    # Optimize and evaluate each prompt type - Tối ưu và đánh giá từng loại prompt
    print("\nOptimizing and evaluating prompts...")
    
    #
    tuned_aspect, aspect_eval = optimize_prompt( 
        dataset=ds,
        sample_texts=sample_texts,
        chat_fn=chat_fn,
        base_prompt=PROMPTS["aspect"],
        provider=provider,
        model=model,
        enabled=True,
        prompt_type="aspect"
    )
    print(f"\n[Aspect Extraction] Evaluation:")
    print(f"Selected: {'Optimized' if aspect_eval['recommendation'] == 'b' else 'Base'} prompt")
    print(f"Confidence: {aspect_eval.get('confidence', 0)*100:.1f}%")
    print(f"Reason: {aspect_eval.get('explanation', 'N/A')}")
    
    # Skip optimization for sentiment - base prompt works better
    tuned_sent = PROMPTS["sentiment"]
    sent_eval = {
        "recommendation": "base", 
        "confidence": 1.0,
        "explanation": "Using base sentiment prompt - known to work reliably"
    }
    print(f"\n[Sentiment Analysis] Evaluation:")
    print(f"Selected: Base prompt (optimization disabled)")
    print(f"Confidence: 100.0%")
    print(f"Reason: {sent_eval.get('explanation', 'N/A')}")
    
    # tuned_reco, reco_eval = optimize_prompt(
    #     dataset=ds,
    #     sample_texts=sample_texts,
    #     chat_fn=chat_fn,
    #     base_prompt=PROMPTS["recommendation"],
    #     provider=provider,
    #     model=model,
    #     enabled=True,
    #     prompt_type="recommendation"
    # )
    # print(f"\n[Recommendation Mining] Evaluation:")
    # print(f"Selected: {'Optimized' if reco_eval['recommendation'] == 'b' else 'Base'} prompt")
    # print(f"Confidence: {reco_eval.get('confidence', 0)*100:.1f}%")
    # print(f"Reason: {reco_eval.get('explanation', 'N/A')}")
    
    # Cập nhật prompt đã điều chỉnh (tối ưu)
    PROMPTS.update({
        "aspect": tuned_aspect,
        "sentiment": tuned_sent,
        # "recommendation": tuned_reco
    })


    # 5) Sentence-level expansion (review_id preserved)
    sent_df = explode_reviews_to_sentences(base)
    sentences = sent_df["sentence"].tolist()

    # 6) Run pipeline
    result_df = full_analysis(sentences, topic, chat_fn=chat_fn, model=model)

    # 7) Attach identifiers (review_id + sent_idx)
    result_df.insert(0, "review_id", sent_df["review_id"].values)
    if "sent_idx" in sent_df.columns and "sent_idx" not in result_df.columns:
        result_df.insert(1, "sent_idx", sent_df["sent_idx"].values)

    # 8) Save rows
    out_dir = Path(f"outputs/absa/{provider}/{ds}")
    csv_path_out, jsonl_path_out = save_csv_and_jsonl(result_df, out_dir, ds)

    # 9) Aggregates
    aspects_summary, recos_summary = summarize_aggregates(result_df)
    a_path, r_path = save_aggregate_tables(aspects_summary, recos_summary, out_dir, ds)

    # 10) Console summary
    total_aspects = int(result_df["aspects"].map(lambda x: len(x) if isinstance(x, list) else 0).sum())
    # total_recos   = int(result_df["recommendations"].map(lambda x: len(x) if isinstance(x, list) else 0).sum())

    print("\n=== Run Summary ===")
    print(f"Dataset:        {ds}")
    print(f"Provider/Model: {provider} / {model}")
    print(f"Sentences:      {len(result_df)}")
    print(f"Aspects:        {total_aspects}")
    # print(f"Recommendations:{total_recos}")
    print("\nSaved files:")
    print(f"  • Rows (CSV):      {csv_path_out}")
    print(f"  • Rows (JSONL):    {jsonl_path_out}")
    print(f"  • Aspects summary: {a_path}")
    print(f"  • Recos summary:   {r_path}")

if __name__ == "__main__":
    main()
