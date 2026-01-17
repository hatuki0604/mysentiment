"""
ABSA – SENTIMENT ONLY
----------------------
Input  : CSV đã có các cột: review_id, sent_idx, sentence, aspects
Output : Thêm sentiment cho từng aspect + SSS + WSSS
"""

import os
import json
import sys
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

# helpers
from absa_LLM_helpers import call_llm, safe_parse_list, set_chat_fn
from absa_prompts import SENTIMENT_PROMPT
from user_selection import choose_provider_and_model
from llm_config import get_llm, ChatFn

load_dotenv()

# ============================
# CONFIG
# ============================

MODEL_DEFAULT = "gpt-4o-mini"

def _normalize_label(text: str) -> str:
    """Map noisy model output → Positive / Negative / Neutral."""
    if not text:
        return "Neutral"
    t = str(text).lower()

    first = t.strip().split()[0]

    if first.startswith("pos"):
        return "Positive"
    if first.startswith("neg"):
        return "Negative"
    if first.startswith("neu"):
        return "Neutral"

    if "positive" in t:
        return "Positive"
    if "negative" in t:
        return "Negative"
    if "neutral" in t:
        return "Neutral"

    return "Neutral"

# ============================
# SENTIMENT ENGINE
# ============================
def classify_sentiments(df: pd.DataFrame, chat_fn: ChatFn, model: str):
    results = []
    SSS = []
    WSSS = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Sentiment"):
        aspects = row.aspects
        if not isinstance(aspects, list):
            aspects = safe_parse_list(aspects)

        if not aspects:
            results.append({})
            SSS.append(0.0)
            WSSS.append(0.0)
            continue

        aspect_result = {}
        aspect_values = []
        conf_list = []

        for aspect in aspects:

            prompt = f"""
        {SENTIMENT_PROMPT}
        Bây giờ đến lượt bạn:
        Review: "{row.sentence}"
        Aspect: "{aspect}"
        """


            raw = call_llm(prompt, model=model, chat_fn=chat_fn, max_tokens=20)

            # Parse JSON
            try:
                obj = json.loads(raw)
                label = _normalize_label(obj.get("label", ""))
                confidence = float(obj.get("confidence", 0))
            except:
                label = _normalize_label(raw)
                confidence = 0.0

            # Polarity
            if label == "Positive":
                polarity = 1
            elif label == "Negative":
                polarity = -1
            else:
                polarity = 0

            score = polarity * confidence

            aspect_result[aspect] = {
                "label": label,
                "confidence": round(confidence, 4),
                "score": round(score, 4),
            }

            aspect_values.append(score)
            conf_list.append(confidence)

        # Sentence-level scores
        sss = sum(aspect_values) / len(aspect_values) if aspect_values else 0.0
        wsss = sum(aspect_values) / sum(conf_list) if sum(conf_list) > 0 else 0.0

        SSS.append(round(sss, 4))
        WSSS.append(round(wsss, 4))
        results.append(aspect_result)

    df = df.copy()
    df["sentiments"] = results
    df["sentiment_score_mean"] = SSS
    df["sentiment_score_weighted"] = WSSS

    return df


# ============================
# SAVE OUTPUT
# ============================

def save_csv_and_jsonl(df: pd.DataFrame, out_dir: Path, ds: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{ds}_sentiment_{len(df)}.csv"
    json_path = out_dir / f"{ds}_sentiment_{len(df)}.jsonl"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return csv_path, json_path

# ============================
# MAIN
# ============================

def main():
    print("\n=== ABSA SENTIMENT ONLY ===")

    # ========== INPUT CSV ==========
    ds = "uit_sentiment"
    csv_path = "/Users/hatrungkien/my-sentiment/outputs/absa/openai/uit/uit_absa_aspect_2224.csv"
    print(f"\nLoading: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"review_id", "sent_idx", "sentence", "aspects"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}")

    # Parse aspects
    df["aspects"] = df["aspects"].apply(lambda x: safe_parse_list(x))

    # ========== LLM MODEL ==========
    provider, model = choose_provider_and_model()
    chat_fn = get_llm(provider, model)
    set_chat_fn(chat_fn)

    # ========== RUN SENTIMENT ==========

    result_df = classify_sentiments(df, chat_fn=chat_fn, model=model)

    # ========== SAVE ==========
    out_dir = Path(f"outputs/absa/{provider}/{ds}")
    csv_path_out, json_path_out = save_csv_and_jsonl(result_df, out_dir, ds)

    print("\n=== DONE ===")
    print(f"Saved CSV:   {csv_path_out}")
    print(f"Saved JSONL: {json_path_out}")

if __name__ == "__main__":
    main()
