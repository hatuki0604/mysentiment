import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ===========================
# Fix import path
# ===========================
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[0]

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


# ===========================
# Load modules nội bộ
# ===========================
from app_reviews_pipeline.user_selection import choose_dataset
from app_reviews_pipeline.llm_config import get_llm
from app_reviews_pipeline.M4_Rag_qa.rag_prompt import PROMPT_TEMPLATE
from app_reviews_pipeline.M4_Rag_qa.rag_qa import (
    retrieve_top_k,
    generate_answer,
    sanitize,
)


# ===========================
# Setup
# ===========================
load_dotenv()
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = "outputs/rag_cache"


# ===========================
# UI
# ===========================
st.set_page_config(page_title="RAG QA Demo", layout="wide")
st.title("📘 RAG QA – Q&A Interface")


# ===========================
# Step 1: Load Dataset Info
# ===========================
st.header("1️⃣ Chọn dataset")

ds, csv_path, topic = choose_dataset()

chunks_path = None
emb_path = None
index_path = None

key_prefix = None


# ===========================
# Step 2: Chọn provider/model
# ===========================
st.header("2️⃣ Chọn Provider & Model")

provider = st.selectbox("Provider", ["openai", "azure"], index=0)
model = st.text_input("Model name", "gpt-4o-mini")

chat_fn = get_llm(provider, model)


# ===========================
# Step 3: Load RAG Cache
# ===========================
st.header("3️⃣ Load FAISS Index (từ lần chạy RAG ở terminal)")

key_prefix = f"{ds}__all-mpnet-base-v2__{provider}__{sanitize(model)}"
bundle_dir = os.path.join(CACHE_DIR, key_prefix)

chunks_path = os.path.join(bundle_dir, "chunks.csv")
emb_path = os.path.join(bundle_dir, "embeddings.npy")
index_path = os.path.join(bundle_dir, "index.faiss")

st.info(f"📁 Cache folder: `{bundle_dir}`")

# Kiểm tra cache
if not (os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(index_path)):
    st.error(
        "❌ Không tìm thấy cache RAG.\n\n"
        "➡️ Vui lòng chạy **rag_qa.py** trước bằng lệnh:\n\n"
        "```bash\npython3 rag_qa.py\n```"
    )
    st.stop()

st.success("✔ Đã tìm thấy cache! Streamlit sẽ dùng lại index này.")

# Load cache
chunks_df = pd.read_csv(chunks_path)
chunks = chunks_df.to_dict("records")
embeddings = np.load(emb_path)
index = faiss.read_index(index_path)

embed_model = SentenceTransformer(MODEL_NAME)


# ===========================
# Step 4: Q&A Chat UI
# ===========================
st.header("4️⃣ Hỏi đáp RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Nhập câu hỏi...")

if query:

    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Determine request type
    ql = query.lower()
    if ql.startswith("summary:") or "overview" in ql:
        request_type = "summary"
        query_clean = query.replace("summary:", "").strip()
        k = 50
    else:
        request_type = "answer"
        query_clean = query
        k = 15

    # Retrieve
    with st.spinner("🔍 Đang truy xuất..."):
        topk = retrieve_top_k(query_clean, embed_model, index, chunks, k=k)

    # Answer
    with st.spinner("🤖 Đang tạo câu trả lời..."):
        response = generate_answer(
            query=query_clean,
            retrieved=topk,
            dataset_name=ds,
            request_type=request_type,
            chat_fn=chat_fn,
        )

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
