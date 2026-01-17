"""
Streamlit Demo App for Phone Review RAG System
"""

import streamlit as st
import base64
from io import BytesIO
import os

# Import RAG system
from source import SimpleRAG

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Products Review Analyzer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .example-query {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        cursor: pointer;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load RAG system (cached)"""
    api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', None)
    
    if not api_key:
        return None
    
    try:
        rag = SimpleRAG("clean_reviews.csv", openai_api_key=api_key)
        return rag
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        return None


def display_base64_image(img_base64: str, caption: str = ""):
    """Display base64 encoded image"""
    img_bytes = base64.b64decode(img_base64)
    st.image(img_bytes, caption=caption, use_container_width=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    
    # API Key input (if not set in environment)
    if not os.getenv('OPENAI_API_KEY'):
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Nhập OpenAI API Key của bạn"
        )
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.success("✅ API Key đã được thiết lập!")
    
    st.markdown("---")
    
    show_charts = st.checkbox("📊 Hiển thị biểu đồ", value=True)
    show_debug = st.checkbox("🐛 Hiển thị debug info", value=False)
    
    st.markdown("---")
    
    st.markdown("### 📝 Hướng dẫn")
    st.markdown("""
    **Câu hỏi mẫu:**
    - Pin Xiaomi 15T có tốt không?
    - So sánh camera Xiaomi 15T và 15T Pro
    - Đánh giá chung về Xiaomi 15T Pro
    - Nhược điểm của Samsung Galaxy S24?
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Khía cạnh hỗ trợ")
    aspects = [
        "🔋 Pin (battery)",
        "📷 Camera",
        "⚡ Hiệu năng (performance)",
        "📱 Màn hình (screen)",
        "🎨 Thiết kế (design)",
        "💰 Giá cả (price)",
        "💾 Bộ nhớ (storage)",
        "✨ Tính năng (features)",
        "🛠️ Dịch vụ (ser&acc)"
    ]
    for asp in aspects:
        st.markdown(f"- {asp}")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<div class="main-header">📱 Phone Review Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống phân tích đánh giá điện thoại thông minh sử dụng RAG + LLM</div>', unsafe_allow_html=True)

# Load RAG system
rag = load_rag_system()

if rag is None:
    st.warning("⚠️ Vui lòng nhập OpenAI API Key trong sidebar để sử dụng hệ thống.")
    st.stop()

# Display dataset stats
col1, col2, col3 = st.columns(3)

dataset_stats = rag.get_dataset_stats()

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{dataset_stats['total_reviews']:,}</div>
        <div class="stat-label">Tổng số đánh giá</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <div class="stat-number">{dataset_stats['total_products']:,}</div>
        <div class="stat-label">Sản phẩm</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="stat-number">9</div>
        <div class="stat-label">Khía cạnh phân tích</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Example queries
st.markdown("### 💡 Câu hỏi gợi ý")
example_queries = [
    "Pin Xiaomi 15T có tốt không?",
    "So sánh camera Xiaomi 15T và 15T Pro",
    "Đánh giá chung về Xiaomi 15T Pro",
    "Nhược điểm của Xiaomi 15T?",
    "Xiaomi 15T Pro màn hình thế nào?"
]

cols = st.columns(len(example_queries))
selected_example = None

for i, query in enumerate(example_queries):
    with cols[i]:
        if st.button(query, key=f"example_{i}", use_container_width=True):
            selected_example = query

st.markdown("---")

# Query input
st.markdown("### 🔍 Đặt câu hỏi")

# Use selected example or empty string
default_query = selected_example if selected_example else ""
query = st.text_input(
    "Nhập câu hỏi của bạn:",
    value=default_query,
    placeholder="VD: Pin Xiaomi 15T có tốt không?",
    key="query_input"
)

# Process button
if st.button("🚀 Phân tích", type="primary", use_container_width=True) or selected_example:
    if query:
        with st.spinner("🔄 Đang phân tích..."):
            try:
                result = rag.answer(query, show_charts=show_charts)
                
                # Display answer
                st.markdown("### 🤖 Câu trả lời")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Display charts
                if show_charts and result.get('charts'):
                    st.markdown("### 📊 Biểu đồ phân tích")
                    
                    chart_titles = {
                        'sentiment_pie': '🥧 Phân bố cảm xúc tổng thể',
                        'product_comparison': '📱 So sánh sản phẩm',
                        'aspect_breakdown': '🔍 Phân tích theo khía cạnh'
                    }
                    
                    # Display charts in columns
                    charts = result['charts']
                    
                    if 'sentiment_pie' in charts:
                        with st.container():
                            st.markdown(f"#### {chart_titles['sentiment_pie']}")
                            display_base64_image(charts['sentiment_pie'])
                    
                    if 'product_comparison' in charts:
                        with st.container():
                            st.markdown(f"#### {chart_titles['product_comparison']}")
                            display_base64_image(charts['product_comparison'])
                    
                    if 'aspect_breakdown' in charts:
                        with st.container():
                            st.markdown(f"#### {chart_titles['aspect_breakdown']}")
                            display_base64_image(charts['aspect_breakdown'])
                
                # Display debug info
                if show_debug and result.get('stats'):
                    st.markdown("### 🐛 Debug Info")
                    with st.expander("Xem thống kê chi tiết"):
                        st.json(result['stats'])
                
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
    else:
        st.warning("⚠️ Vui lòng nhập câu hỏi!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    📱 Phone Review Analyzer | Powered by OpenAI GPT-4o-mini + RAG System
</div>
""", unsafe_allow_html=True)