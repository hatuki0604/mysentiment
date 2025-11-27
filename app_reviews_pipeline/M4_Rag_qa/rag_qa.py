"""
RAG QA Interactive Pipeline
Organized with section comments for clarity and maintainability.
"""

# ===========================
# Imports & Configuration
# ===========================
import os
import re
import json
from typing import List, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

# ===========================
# Path Setup (Script/Module)
# ===========================
import sys
from pathlib import Path
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]  # .../app_reviews_pipeline
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# ===========================
# User Selection & Prompts
# ===========================
from user_selection import (
    choose_dataset,
    choose_provider_and_model,
    choose_sample_size,
)
from prompt_optimize import optimize_prompt
from llm_config import get_llm, ChatFn
from rag_prompt import PROMPT_TEMPLATE
from tqdm import tqdm

# ===========================
# Config
# ===========================
load_dotenv()
CACHE_DIR = str(PKG_ROOT.parent / "outputs" / "rag_cache")  # outputs/rag_cache
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ===========================
# Helper Functions
# ===========================
def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s))

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]

def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    """Token-based chunking via tiktoken; falls back to char-based if missing."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(str(text))
        chunks, start, n = [], 0, len(ids)
        while start < n:
            end = min(start + max_tokens, n)
            chunks.append(enc.decode(ids[start:end]))
            if end == n:
                break
            start = end - overlap
        return chunks
    except Exception:
        s = str(text); max_chars, overlap_chars = 1200, 200
        chunks, start, n = [], 0, len(s)
        while start < n:
            end = min(start + max_chars, n)
            chunks.append(s[start:end])
            if end == n:
                break
            start = end - overlap_chars
        return chunks

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(query: str, embed_model: SentenceTransformer, index, chunks: List[dict], k: int = 8):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

def generate_answer(
    query: str,
    retrieved: List[Tuple[dict, float]],
    dataset_name: str,
    request_type: str,
    chat_fn: ChatFn,
    max_tokens: int = 512,
) -> str:
    """
    Build the RAG message using the (possibly tuned) global PROMPT_TEMPLATE and call the selected LLM.
    """
    # Build compact CONTEXT (≤15) with stable ids
    context_objs = []
    for chunk, _score in retrieved[:30]:
        context_objs.append({
            "idx": str(chunk.get("review_id", "")),
            "text": str(chunk.get("text", "")).replace("\n", " ").strip(),
        })

    filled = PROMPT_TEMPLATE.format(
        question=query.strip(),
        context_json=json.dumps(context_objs, ensure_ascii=False, indent=2),
        dataset_name=dataset_name,
        request_type=request_type,
    )

    messages = [
        {"role": "system", "content": "Follow the instructions and answer concisely using only the provided CONTEXT."},
        {"role": "user", "content": filled},
    ]
    return chat_fn(messages, temperature=0.0, max_tokens=max_tokens).strip()

# ===========================
# Main Pipeline
# ===========================
def main():
    print("\n=== RAG over Preprocessed Reviews ===")

    # ===========================
    # Dataset Selection & Loading
    # ===========================
    ds = "processed_tgdd_reviews"
    csv_path = "/Users/hatrungkien/my-sentiment/data/processed/processed_tgdd_reviews.csv"
    topic = "Đây là các reviews về điện thoại di động của Thế Giới Di Động"

    base = pd.read_csv(csv_path, usecols=["review_id", "review_text"])

    # ===========================
    # Provider & Model Selection
    # ===========================
    provider = "openai"
    model = "gpt-4o-mini"
    chat_fn = get_llm(provider, model)

    # ===========================
    # Chunking & Embedding (Cache-Aware)
    # ===========================
    key = f"{ds}__{MODEL_NAME.split('/')[-1]}__{provider}__{sanitize(model)}"
    bundle_dir = os.path.join(CACHE_DIR, key)
    os.makedirs(bundle_dir, exist_ok=True)
    chunks_path = os.path.join(bundle_dir, "chunks.csv")
    emb_path    = os.path.join(bundle_dir, "embeddings.npy")
    idx_path    = os.path.join(bundle_dir, "index.faiss")

    # Helper to chunk one review
    def _chunk_row(row) -> List[dict]:
        ck = []
        for sub in chunk_text(str(row["review_text"])):
            if sub.strip():
                ck.append({"review_id": str(row["review_id"]), "text": sub})
        return ck

    chunks: List[dict] = []
    embed_model = None
    if os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(idx_path):
        print(f"\nLoading cache for '{key}'…")
        chunks_df = pd.read_csv(chunks_path)
        chunks = chunks_df.to_dict("records")
        embeddings = np.load(emb_path)
        index = faiss.read_index(idx_path)
        embed_model = SentenceTransformer(MODEL_NAME)  # for queries
    else:
        print("\nChunking…")
        for _idx, row in tqdm(base.iterrows(), total=len(base), unit="row"):
            chunks.extend(_chunk_row(row))
        if not chunks:
            raise RuntimeError("No chunks produced. Check your input CSV columns.")

        print("Embedding & indexing…")
        texts = [c["text"] for c in chunks]
        embed_model = SentenceTransformer(MODEL_NAME)

        # Batch encode for speed/memory
        batch_size = 512
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), unit="batch"):
            batch = embed_model.encode(texts[i:i+batch_size], convert_to_numpy=True, show_progress_bar=False)
            embs.append(batch)
        embeddings = np.vstack(embs)
        faiss.normalize_L2(embeddings)
        index = build_faiss_index(embeddings)

        # Persist bundle
        pd.DataFrame(chunks).to_csv(chunks_path, index=False)
        np.save(emb_path, embeddings)
        faiss.write_index(index, idx_path)
        print(f"Cached under '{key}'.")

    # ===========================
    # Prompt Tuning (DISABLED - causing template corruption)
    # ===========================
    # global PROMPT_TEMPLATE  # overwrite module-level template
    # sample_texts = [c["text"] for c in chunks][:500]

    # base_len = len(PROMPT_TEMPLATE)
    # base_sha = _sha1(PROMPT_TEMPLATE)
    # print(f"[PromptTuner] ENABLED — BEFORE len={base_len} sha={base_sha}")

    # tuned, eval_details = optimize_prompt(
    #     base_prompt=PROMPT_TEMPLATE,
    #     sample_texts=sample_texts,
    #     chat_fn=chat_fn,
    #     dataset=ds,
    #     provider=provider,
    #     model=model or "",
    #     must_have_tags=["ANSWER:", "EVIDENCE:", "INSIGHTS:", "SUMMARY:", "TOP_EXAMPLES:"],
    # )
    # PROMPT_TEMPLATE = tuned  # Only use the prompt, not the evaluation details

    # tuned_len = len(PROMPT_TEMPLATE)
    # tuned_sha = _sha1(PROMPT_TEMPLATE)
    # delta = tuned_len - base_len
    # print(f"[PromptTuner] AFTER  len={tuned_len} sha={tuned_sha} (Δ={delta:+d})")

    # # Optional preview via env flag
    # if os.getenv("PROMPT_TUNE_VERBOSE", "0") == "1":
    #     preview = PROMPT_TEMPLATE[:320]
    #     print("[PromptTuner] Preview:\n" + preview + ("..." if len(PROMPT_TEMPLATE) > 320 else ""))

    # # Dump tuned prompt for reproducibility
    # dumps_dir = PKG_ROOT.parent / "outputs" / "prompt_dumps"
    # dumps_dir.mkdir(parents=True, exist_ok=True)
    # dump_path = dumps_dir / f"{ds}__{provider}__{sanitize(model)}.txt"
    # dump_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    # print(f"[PromptTuner] Saved tuned prompt → {dump_path}")
    
    print("[PromptTuner] DISABLED - Using original template to avoid corruption")

    # ===========================
    # Ready: Interactive Q&A Loop
    # ===========================
    print(f"\nIndex ready. Provider='{provider}' Model='{model}'.")
    print("Ask me questions about the reviews! (type 'quit' to exit)")

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Good-bye!")
            break

        ql = query.lower()
        if ql.startswith("summary:") or "overview" in ql or "general view" in ql:
            request_type = "summary"
            query_clean = re.sub(r"^(summary:|overview of|general view of)\s*", "", query, flags=re.I).strip()
            k = 50
        else:
            request_type = "answer"
            query_clean = query
            k = 15

        topk = retrieve_top_k(query_clean, embed_model, index, chunks, k=k)
        response = generate_answer(
            query=query_clean,
            retrieved=topk,
            dataset_name=ds,
            request_type=request_type,
            chat_fn=chat_fn,
        )
        print("\n" + response + "\n")

if __name__ == "__main__":
    main()