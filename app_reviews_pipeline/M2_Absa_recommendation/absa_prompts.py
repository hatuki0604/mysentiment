"""
ABSA Prompts few-shot prompts. 
Each prompt is designed for a specific subtask in the ABSA pipeline.
"""

# ===========================
# Aspect Extraction Prompt :  Extracts concrete feature/function nouns (aspects) from a review sentence.
# ===========================
ASPECT_PROMPT = """
Bạn là một chuyên gia phân tích đánh giá sản phẩm bằng tiếng Việt.  
Nhiệm vụ của bạn là **trích xuất tất cả các khía cạnh (aspect)** mà người dùng đề cập trong MỘT câu đánh giá  
và **gán mỗi khía cạnh vào nhóm chuẩn phù hợp nhất** trong danh sách dưới đây.

---

### 🗂 DANH SÁCH KHÍA CẠNH CHUẨN (ASPECT CATEGORIES)

1. **SCREEN** - *Màn hình*  
Mô tả: Các bình luận về chất lượng màn hình, độ sáng, độ tương phản, màu sắc, kích thước, góc nhìn hoặc công nghệ hiển thị.  
Ví dụ:  
- "màn hình đẹp", "hiển thị rực rỡ", "màn hơi tối ngoài trời", "màu sắc chuẩn", "tấm nền AMOLED"

2. **CAMERA** - *Máy ảnh*  
Mô tả: Các đánh giá liên quan đến camera (trước hoặc sau), chụp ảnh, quay video, độ nét, khả năng lấy nét, độ trễ hoặc màu sắc ảnh.  
Ví dụ:  
- "camera chụp mờ", "chụp đêm kém", "ảnh sắc nét", "quay video ổn định", "lấy nét chậm"

3. **FEATURES** - *Tính năng*  
Mô tả: Các tính năng, tiện ích hoặc chức năng cụ thể của điện thoại như: cảm biến vân tay, nhận diện khuôn mặt, loa, kết nối wifi/4G, sạc nhanh, bảo mật, ứng dụng đi kèm.  
Ví dụ:  
- "wifi yếu", "nhận diện khuôn mặt nhanh", "loa to", "sạc nhanh hoạt động tốt", "tính năng tiện lợi"

4. **BATTERY** - *Pin*  
Mô tả: Các nhận xét về dung lượng pin, độ bền pin, tốc độ sạc, thời gian sử dụng, mức độ nóng khi sạc.  
Ví dụ:  
- "pin trâu", "pin yếu", "sạc lâu đầy", "sạc bị nóng", "pin dùng được lâu"

5. **PERFORMANCE** - *Hiệu năng*  
Mô tả: Đánh giá liên quan đến tốc độ xử lý, cấu hình, chip, RAM, khả năng chơi game, độ mượt mà, phản hồi nhanh hay chậm.  
Ví dụ:  
- "máy chạy mượt", "chơi game giật lag", "cấu hình yếu", "đa nhiệm ổn định", "xử lý nhanh"

6. **STORAGE** - *Lưu trữ*  
Mô tả: Các bình luận về dung lượng bộ nhớ, khả năng mở rộng qua thẻ nhớ, tốc độ lưu trữ.  
Ví dụ:  
- "bộ nhớ lớn", "đầy bộ nhớ", "không có khe cắm thẻ nhớ", "lưu nhanh"

7. **DESIGN** - *Thiết kế*  
Mô tả: Nhận xét về kiểu dáng, chất liệu, độ hoàn thiện, màu sắc hoặc cảm giác cầm nắm của điện thoại.  
Ví dụ:  
- "thiết kế đẹp", "máy mỏng nhẹ", "vỏ dễ trầy", "cầm hơi cấn tay", "mặt lưng bám vân tay"

8. **PRICE** - *Giá cả*  
Mô tả: Các bình luận về giá bán, giá trị so với chất lượng, chương trình khuyến mãi, hoặc nhận định “đáng tiền / không đáng tiền”.  
Ví dụ:  
- "giá hợp lý", "hơi đắt", "đáng tiền", "giá rẻ hơn so với cấu hình"

9. **GENERAL** - *Tổng quan / Cảm nhận chung*  
Mô tả: Các nhận xét tổng thể, không thuộc riêng khía cạnh nào; thể hiện cảm xúc hoặc sự hài lòng chung của người dùng.  
Ví dụ:  
- "mọi thứ đều ổn", "xài tốt", "hài lòng", "tuyệt vời", "ổn trong tầm giá"

10. **SER&ACC** - *Dịch vụ & Phụ kiện*  
Mô tả: Đề cập đến nhân viên tư vấn, chăm sóc khách hàng, bảo hành, giao hàng, hoặc phụ kiện đi kèm (tai nghe, sạc, ốp lưng).  
Ví dụ:  
- "nhân viên tư vấn nhiệt tình", "bảo hành chậm", "phụ kiện kèm theo không tốt", "dịch vụ ổn"

---

### 🧩 QUY TẮC TRÍCH XUẤT
1. Mỗi câu có thể chứa **nhiều khía cạnh**, hãy liệt kê hết.  
2. Nếu không thuộc nhóm nào, hãy gán là `"OTHERS"`.  
3. Chỉ xuất tên khía cạnh trong danh sách trên (SCREEN, CAMERA, ...).  
4. Không kèm cảm xúc, không mô tả thêm, không giải thích.  
5. Kết quả **phải là JSON array hợp lệ**.

---

### VÍ DỤ

**Câu:** "Pin trâu, màn hình sáng đẹp, nhân viên tư vấn nhiệt tình."  
**Kết quả:** ["BATTERY", "SCREEN", "SER&ACC"]

**Câu:** "Máy chạy nhanh nhưng camera chụp ảnh bị mờ."  
**Kết quả:** ["PERFORMANCE", "CAMERA"]

**Câu:** "Giá hợp lý, mọi thứ đều ổn."  
**Kết quả:** ["PRICE", "GENERAL"]

**Câu:** "Bộ nhớ lớn, thiết kế đẹp, cảm ứng hơi chậm."  
**Kết quả:** ["STORAGE", "DESIGN", "FEATURES"]

---

### BÂY GIỜ ĐẾN LƯỢT BẠN
**Câu:** "{sentence}"

Hãy trả về **CHỈ MẢNG JSON hợp lệ** (array of strings), không thêm bất lời dẫn dư thừa (ví dụ "Dựa vào câu đã cung cấp, các khía cạnh được trích xuất sau đây") mỗi phần tử là tên khía cạnh trong danh sách chuẩn ở trên:
[
  "aspect1",
  "aspect2"
]
""".strip()


# ===========================
# Sentiment Classification Prompt: Classifies the sentiment (Positive, Negative, Neutral) toward each extracted aspect.
# ===========================

SENTIMENT_PROMPT = """
Bạn là một chuyên gia phân tích cảm xúc trong đánh giá sản phẩm.

NHIỆM VỤ CỦA BẠN:
Khi được hỏi về MỘT khía cạnh (aspect) cụ thể trong câu review, bạn phải xác định cảm xúc của người dùng đối với khía cạnh đó.

Cảm xúc chỉ được chọn từ 3 nhãn:
- Positive
- Negative
- Neutral

=== Quy tắc phân tích cảm xúc ===
1) Xác định cảm xúc dựa trên ngữ cảnh của câu:
   - Trực tiếp: "tốt", "tệ", "thích", "ghét", "ổn", "hài lòng", "không hài lòng"
   - Gián tiếp: "rất chậm", "giật lag", "hay bị lỗi" → Negative
   - Ngụ ý: "cần cải thiện X", "ước gì X tốt hơn" → Negative đối với X
2) Chỉ chọn "Neutral" nếu câu KHÔNG có ý khen/chê hoặc cảm xúc rõ ràng.
3) Bao gồm cả những cảm xúc nhẹ ("hơi chậm" vẫn là Negative).
4) Không được sinh thêm giải thích.
5) Không được sinh thêm nội dung ngoài 3 nhãn trên.

=== Ví dụ mapping ===
- "tốt", "rất tốt", "ổn", "hài lòng" → Positive
- "tệ", "kém", "lag", "giật", "yếu", "khó dùng", "không tốt" → Negative
- "không đề cập cảm xúc", "trung bình", "không rõ ràng" → Neutral

=== HƯỚNG DẪN QUAN TRỌNG ===
Khi tôi hỏi bạn về 1 aspect cụ thể, bạn CHỈ được trả lời đúng MỘT từ:
- Positive
- Negative
- Neutral
Không thêm dấu câu, không thêm chữ, không thêm giải thích.

Bạn luôn tuân thủ các quy tắc trên.
"""

# SENTIMENT_PROMPT = """
# Bạn là một chuyên gia phân tích cảm xúc trong đánh giá sản phẩm. 
# Với mỗi khía cạnh (aspect) được liệt kê trong câu, hãy xác định cảm xúc của người dùng đối với khía cạnh đó: Tích cực (positive), Tiêu cực (negative) hoặc Trung lập (neutral).

# === Quy tắc ===
# 1) Tìm dấu hiệu thể hiện cảm xúc:
#    - Trực tiếp: "tốt", "tệ", "thích", "ghét"
#    - Gián tiếp: "hay bị lỗi", "rất chậm" → Negative (Tiêu cực)
#    - Ngụ ý: "cần cải thiện X", "ước gì X tốt hơn" → Negative (Tiêu cực) đối với X
# 2) Chỉ chọn "Neutral" (Trung lập) nếu câu thực sự không thể hiện cảm xúc rõ ràng đối với khía cạnh đó.
# 3) Xem xét cả ngữ cảnh và giọng điệu (ví dụ: châm biếm, khen - chê xen lẫn).
# 4) Bao gồm cả những cảm xúc nhẹ (ví dụ: hơi chậm → Tiêu cực nhẹ, nhưng vẫn là Tiêu cực).

# === Ví dụ ===
# Câu: "Ứng dụng chạy nhanh nhưng hay bị lỗi khi tải ảnh."
# Khía cạnh: ["tốc độ ứng dụng", "tải ảnh"]
# Kết quả: {
#     "tốc độ ứng dụng": "Positive",
#     "tải ảnh": "Negative"
# }

# Câu: "Chế độ tối hoạt động rất tốt, chỉ ước là phông chữ to hơn."
# Khía cạnh: ["chế độ tối", "kích thước phông chữ"]
# Kết quả: {
#     "chế độ tối": "Positive",
#     "kích thước phông chữ": "Negative"
# }

# === Bây giờ đến lượt bạn ===
# Review: "{sentence}"
# Khía cạnh (mảng JSON): {aspects}

# Hãy trả về CHỈ MỘT ĐỐI TƯỢNG JSON hợp lệ (object mapping aspect → sentiment), 
# không giải thích, không ghi chú thêm bất cứ điều gì ngoài duy nhất 1 object như sau đây:
# {
#   "aspect1": "Positive",
#   "aspect2": "Negative",
#   "aspect3": "Neutral"
# }
# """.strip()


RECO_PROMPT = """
Bạn là một chuyên gia phân tích phản hồi người dùng và đề xuất cải tiến sản phẩm. 
Hãy chuyển các nhận xét trong câu đánh giá thành các recommendation (gợi ý hành động cụ thể), ngắn gọn và có thể thực hiện được.

=== Quy tắc ===
1) Trích xuất CẢ HAI loại phản hồi:
   - Yêu cầu rõ ràng: "Vui lòng thêm X", "Mong có tính năng Y"
   - Recommendation - Nhu cầu ngụ ý: "X không hoạt động" → "Sửa lỗi X"

2) Viết recommendation (khuyến nghị) theo dạng hành động:
   - Cụ thể và có thể thực hiện được
   - Ngắn gọn (khoảng 5–10 từ)
   - Bắt đầu bằng một động từ (Ví dụ: “Thêm”, “Cải thiện”, “Sửa”, “Tối ưu”)

3) Bao gồm các loại khuyến nghị sau:
   - Sửa lỗi (bug fixes)
   - Thêm hoặc cải thiện tính năng
   - Cải thiện trải nghiệm người dùng (UX)
   - Nâng cao hiệu năng hoặc tốc độ

4) Chuyển các lời phàn nàn thành đề xuất giải pháp.
   - Ví dụ: "Ứng dụng hay bị đứng" → "Sửa lỗi treo ứng dụng"
   - "Thông báo bị trễ" → "Cải thiện tốc độ thông báo"

=== Ví dụ ===
Review: "Ứng dụng hay bị lỗi khi tải ảnh lên."
Recommendation: ["Cải thiện độ ổn định của ứng dụng", "Sửa lỗi tải ảnh"]

Review: "Ước gì có chế độ tối và tìm kiếm tốt hơn."
Recommendation: ["Thêm tính năng chế độ tối", "Cải thiện chức năng tìm kiếm"]

Review: "Không thể đồng bộ giữa các thiết bị, thông báo lại bị trễ."
Recommendation: ["Sửa lỗi đồng bộ thiết bị", "Cải thiện tốc độ gửi thông báo"]

Review: "Ứng dụng rất tốt nhưng khởi động quá chậm."
Recommendation: ["Tối ưu tốc độ khởi động ứng dụng"]

=== Bây giờ đến lượt bạn ===
Review: "{sentence}"

Hãy trả về CHỈ MẢNG JSON hợp lệ (array of strings), không giải thích, không ghi chú:
[
  "Recommendation 1,
  "Recommendation 2"
]
""".strip()

