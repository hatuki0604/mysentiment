import streamlit as st
import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ================== PATH & IMPORT RAG ==================
FILE_DIR = Path(__file__).resolve()
MODULE_DIR = FILE_DIR.parent        # M4_Rag_qa/
PROJECT_ROOT = FILE_DIR.parents[1]  # app_reviews_pipeline/
sys.path.append(str(MODULE_DIR))
sys.path.append(str(PROJECT_ROOT))

from rag_qa import (
    chunk_text,
    retrieve_top_k,
    generate_answer,
    sanitize,
    MODEL_NAME,
)
from llm_config import get_llm

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="RAG QA Demo", page_icon="🔍", layout="centered")

# ================== SESSION STATE INIT & CLEAR HANDLING ==================

# Khởi tạo cờ clear nếu chưa có
if "clear_triggered" not in st.session_state:
    st.session_state["clear_triggered"] = False

# Nếu ở vòng rerun này đang có yêu cầu clear → reset query_box TRƯỚC KHI TẠO text_input
if st.session_state["clear_triggered"]:
    st.session_state["query_box"] = ""
    st.session_state["clear_triggered"] = False

# ================== CONFIG ==================
DATA_CSV = "/Users/hatrungkien/my-sentiment/data/processed/processed_tgdd_reviews.csv"
DS_NAME = "processed_tgdd_reviews"
PROVIDER = "openai"
MODEL = "gpt-4o-mini"
chat_fn = get_llm(PROVIDER, MODEL)

# ================== LOAD RAG (cache) ==================
@st.cache_resource(show_spinner=True)
def load_rag_system():
    df = pd.read_csv(DATA_CSV, usecols=["review_id", "review_text"])

    key = f"{DS_NAME}__{MODEL_NAME.split('/')[-1]}__{PROVIDER}__{sanitize(MODEL)}"
    cache_dir = Path(PROJECT_ROOT).parent / "outputs" / "rag_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = cache_dir / "chunks.csv"
    emb_path = cache_dir / "embeddings.npy"
    index_path = cache_dir / "index.faiss"

    embed_model = SentenceTransformer(MODEL_NAME)

    if chunks_path.exists() and emb_path.exists() and index_path.exists():
        chunks_df = pd.read_csv(chunks_path)
        chunks = chunks_df.to_dict("records")
        embeddings = np.load(emb_path)
        index = faiss.read_index(str(index_path))
        return df, chunks, embeddings, index, embed_model

    # build mới
    chunks = []
    for _, row in df.iterrows():
        for sub in chunk_text(str(row["review_text"])):
            if sub.strip():
                chunks.append({"review_id": row["review_id"], "text": sub})

    texts = [c["text"] for c in chunks]
    embs = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        b = embed_model.encode(texts[i:i+batch_size], convert_to_numpy=True)
        embs.append(b)
    embeddings = np.vstack(embs)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    pd.DataFrame(chunks).to_csv(chunks_path, index=False)
    np.save(emb_path, embeddings)
    faiss.write_index(index, str(index_path))

    return df, chunks, embeddings, index, embed_model

# ================== CSS CHO NÚT ==================
st.markdown("""
    <style>
        .big-btn {
            height: 48px !important;
            width: 100% !important;
            font-size: 16px !important;
            border-radius: 8px !important;
        }
        .btn-submit {
            background-color: #ff4b4b !important;
            color: red !important;
            border: none !important;
        }
        .btn-clear {
            background-color: #f1f1f1 !important;
            color: #333 !important;
            border: 1px solid #dcdcdc !important;
        }
        .btn-submit:hover {
        background-color: #e84343 !important;
}

    </style>
""", unsafe_allow_html=True)

# ================== UI ==================
st.title("Demo RAG QA from User's Reviews")
st.caption("Hệ thống truy vấn")

with st.spinner("Đang load hệ thống..."):
    df, chunks, embeddings, index, embed_model = load_rag_system()

st.divider()

# ---- FORM: Enter = Submit ----
with st.form("rag_form"):
    query = st.text_input("Nhập truy vấn:", "", key="query_box")

    # hàng nút – 2 nút + 1 cột trống để nút nhỏ lại
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        submit = st.form_submit_button("Submit", use_container_width=True)
    with col2:
        clear = st.form_submit_button("Clear", use_container_width=True)

# style cho nút sau khi render
st.markdown("""
<script>
const submitBtn = window.parent.document.querySelector('button[data-testid="baseButton-rag_form-Submit"]');
if (submitBtn) submitBtn.classList.add('big-btn', 'btn-submit');
const clearBtn = window.parent.document.querySelector('button[data-testid="baseButton-rag_form-Clear"]');
if (clearBtn) clearBtn.classList.add('big-btn', 'btn-clear');
</script>
""", unsafe_allow_html=True)

# ================== CLEAR: set cờ & RERUN ==================
if clear:
    st.session_state["clear_triggered"] = True
    st.rerun()

# ================== SUBMIT: xử lý truy vấn ==================
if submit and query.strip():
    q = query.strip()

    if q.lower().startswith("summary:") or "overview" in q.lower():
        request_type = "summary"
        q_clean = re.sub(r"^(summary:|overview of|tóm tắt)\s*", "", q, flags=re.I).strip()
        k = 50
    else:
        request_type = "answer"
        q_clean = q
        k = 15

    with st.spinner("Đang retrieve top-K..."):
        topk = retrieve_top_k(q_clean, embed_model, index, chunks, k=k)

    with st.spinner("Đang truy vấn..."):
        answer = generate_answer(
            query=q_clean,
            retrieved=topk,
            dataset_name=DS_NAME,
            request_type=request_type,
            chat_fn=chat_fn,
        )

    st.subheader("Kết quả:")
    st.write(answer)
