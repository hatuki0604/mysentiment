# ABSA (Aspect-Based Sentiment Analysis) cho Review tiếng Việt

ABSA là dự án phân tích cảm xúc theo **khía cạnh (ABSA)** cho review. Mục tiêu là biến review thành insight có cấu trúc như:

- **Khía cạnh (aspect)** được nhắc tới (pin, camera, giao diện, giá, dịch vụ, ...)
- **Cảm xúc** cho từng khía cạnh (positive/neutral/negative hoặc thang điểm)
- (Tuỳ chọn) **Gợi ý/cải thiện** trích xuất từ review

Repo cũng có các thành phần hỗ trợ pipeline xử lý dữ liệu, baseline so sánh, lưu cache prompt và một demo giao diện chat bằng Streamlit.

---

## Tính năng chính

- Tiền xử lý dữ liệu review (làm sạch, chuẩn hoá)
- ABSA: trích xuất (aspect, sentiment, recommendation) theo dạng cấu trúc
- Baseline (so sánh nhanh)
- Lưu kết quả ra thư mục `outputs/`
- Demo UI: chatbot/interactive analysis với Streamlit

---

## Cấu trúc thư mục

```
mysentiment/
├── app_reviews_pipeline/        # Pipeline chính
├── baseline/                    # Baseline/so sánh
├── crawl-data/                  # (Tuỳ chọn) crawl/thu thập dữ liệu
├── data/                        # Dữ liệu thô/đã xử lý
├── outputs/                     # Kết quả chạy
├── prompt_cache/                # Cache prompt / kết quả trung gian
├── streamlit_chatbot.py         # Demo UI bằng Streamlit
├── requirements.txt
└── ...
```

---

## Cài đặt

### 1) Clone repo & tạo môi trường
```bash
git clone https://github.com/hatuki0604/mysentiment.git
cd mysentiment

python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Cài dependencies
```bash
pip install -r requirements.txt
```

### 3) (Tuỳ chọn) Thiết lập API key
Nếu bạn dùng LLM provider (OpenAI/Mistral/...), tạo file `.env` ở root:

```env
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
# các biến khác nếu code của bạn có dùng
```

---

## Chạy nhanh

### A. Tiền xử lý dữ liệu
Đặt dữ liệu thô trong `data/` (hoặc theo cấu trúc bạn đang dùng) rồi chạy bước preprocessing:

```bash
python app_reviews_pipeline/preprocessing.py
```

Kết quả (cleaned/processed) sẽ được ghi lại vào `data/` và/hoặc các file thống kê tuỳ theo pipeline.

### B. Chạy pipeline
```bash
python app_reviews_pipeline/run_pipeline.py
```

Pipeline thường được thiết kế theo dạng chọn module/chọn dataset/chọn cấu hình khi chạy (tuỳ implement).

### C. Chạy demo Streamlit (Chatbot/UI)
```bash
streamlit run streamlit_chatbot.py
```

---

## Input / Output (khuyến nghị)

### Input dữ liệu (CSV)
Khuyến nghị dữ liệu review ở dạng CSV, mỗi dòng là một review. Tối thiểu nên có:
- `text`/`review`: nội dung review
- (tuỳ chọn) `rating`: số sao (nếu có) để so sánh với sentiment từ text
- (tuỳ chọn) metadata khác: thời gian, sản phẩm/app, user, ...

> Nếu dataset của bạn đang dùng tên cột khác, bạn chỉ cần sửa mapping trong phần preprocessing/pipeline.

### Output
Kết quả chạy được ghi vào `outputs/` (hoặc thư mục con theo từng module).

Ví dụ output ABSA thường gồm các trường:
- `review_id`
- `aspect`
- `sentiment`
- `recommendation` (nếu có)
- `evidence` (trích dẫn câu/đoạn liên quan — nếu pipeline hỗ trợ)

---