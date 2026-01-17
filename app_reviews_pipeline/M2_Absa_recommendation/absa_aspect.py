"""
ABSA – Step 1: Aspect Extraction Only (with prompt optimization)
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# PATH setup
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
for p in (HERE.parent, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# internal imports
from absa_LLM_helpers import call_llm, safe_parse_list, set_chat_fn
from absa_prompts import ASPECT_PROMPT
from user_selection import (
    choose_dataset,
    choose_provider_and_model,
    choose_sample_size,
    choose_prompt_tuning,
)
from llm_config import get_llm
from prompt_optimize import optimize_prompt

load_dotenv()

PROMPTS = {"aspect": ASPECT_PROMPT}


# ============================
# Aspect extraction
# ============================
def extract_aspects(sentences, topic, chat_fn, model="gpt-4o-mini"):
    records = []
    for s in tqdm(sentences, desc="Extracting aspects", unit="sent"):
        topic_prompt = f"Bạn là chuyên gia phân tích và đánh giá sản phẩm. {topic}."
        prompt = topic_prompt + PROMPTS["aspect"].replace("{sentence}", str(s))

        raw = call_llm(prompt, model=model, max_tokens=256, chat_fn=chat_fn)
        aspects = safe_parse_list(raw)

        clean, seen = [], set()
        for term in aspects:
            t = str(term).lower().strip()
            t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
            if t and t not in seen:
                clean.append(t)
                seen.add(t)

        records.append({"sentence": s, "aspects": clean})
    return pd.DataFrame(records)


def split_into_sentences(text):
    text = str(text or "").strip()
    return [text] if text else []


def explode_reviews(df):
    rows = []
    for _, r in df.iterrows():
        sents = split_into_sentences(r["review_text"])
        for i, sent in enumerate(sents):
            rows.append({
                "review_id": r["review_id"],
                "sent_idx": i,
                "sentence": sent,
            })
    return pd.DataFrame(rows)

def save_aspect_csv_and_jsonl(df: pd.DataFrame, out_dir: Path, ds: str):
    """
    Lưu kết quả aspect extraction ra 2 file:
    - CSV  : {ds}_absa_aspect_{n}.csv
    - JSONL: {ds}_absa_aspect_{n}.jsonl
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{ds}_absa_aspect_{len(df)}.csv"
    jsonl_path = out_dir / f"{ds}_absa_aspect_{len(df)}.jsonl"

    # CSV: bình thường
    df.to_csv(csv_path, index=False)

    # JSONL: mỗi dòng là 1 record JSON
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path



# ============================
# Main
# ============================
def main():
    print("\n=== ABSA – STEP 1: ASPECT EXTRACTION (with optimization) ===")

    # 1) Dataset
    ds, csv_path, topic = choose_dataset()

    # 2) Provider/model
    provider, model = choose_provider_and_model()
    chat_fn = get_llm(provider, model)
    set_chat_fn(chat_fn)

    # 3) Load sample
    base = pd.read_csv(csv_path, usecols=["review_id", "review_text"])
    sample_n = choose_sample_size(len(base))
    if sample_n:
        base = base.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # 4) Prompt optimization
    print("\nOptimizing ASPECT prompt…")

    sample_texts = base["review_text"].astype(str).head(500).tolist()

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
    print(f"Selected: {'Optimized' if aspect_eval['recommendation']=='b' else 'Base'} prompt")
    print(f"Confidence: {aspect_eval.get('confidence', 0)*100:.1f}%")
    print(f"Reason: {aspect_eval.get('explanation', 'N/A')}")

    # Update prompt
    PROMPTS["aspect"] = tuned_aspect

    # 5) Explode → sentence-level rows
    sent_df = explode_reviews(base)
    sentences = sent_df["sentence"].tolist()

    # 6) Run aspect extraction
    result_df = extract_aspects(sentences, topic, chat_fn, model)

    # attach ID fields
    result_df.insert(0, "review_id", sent_df["review_id"].values)
    result_df.insert(1, "sent_idx", sent_df["sent_idx"].values)

    # 7) Save output
    out_dir = Path(f"outputs/absa/{provider}/{ds}")
    csv_path_out, jsonl_path_out = save_aspect_csv_and_jsonl(result_df, out_dir, ds)

    print("\nDone!")
    print(f"Saved aspect CSV   : {csv_path_out}")
    print(f"Saved aspect JSONL : {jsonl_path_out}")

    csv_path_out = out_dir / f"{ds}_absa_aspect_{len(result_df)}.csv"
    result_df.to_csv(csv_path_out, index=False)

    print("\nDone!")
    print(f"Saved aspect CSV: {csv_path_out}")


if __name__ == "__main__":
    main()
