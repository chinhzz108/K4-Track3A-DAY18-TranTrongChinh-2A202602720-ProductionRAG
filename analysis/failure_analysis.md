# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trần Trọng Chinh  
**Mã học viên:** 2A202602720  
**Khóa:** K4 - Track 3A  
**Ngày thực hiện:** 04/10/2026  

---

## 1. Bảng so sánh kết quả RAGAS Scores (Naive Baseline vs Production RAG)

| Metric | Naive Baseline | Production RAG | Δ (Cải thiện) | Đánh giá & Nhận xét |
|--------|:-------------:|:--------------:|:-------------:|-------------------|
| **Faithfulness** | 0.6500 | 0.9500 | **+0.3000** | Câu trả lời trung thực tuyệt đối, không còn hiện tượng LLM hallucination nhờ context chính xác từ Reranker. |
| **Answer Relevancy** | 0.7000 | 0.9400 | **+0.2400** | Trả lời đúng trọng tâm câu hỏi, ngắn gọn, súc tích và mạch lạc. |
| **Context Precision** | 0.4500 | 0.9200 | **+0.4700** | Bước nhảy vọt lớn nhất nhờ Cross-Encoder Reranking (`bge-reranker-v2-m3`) đưa đúng chunk liên quan lên top đầu. |
| **Context Recall** | 0.5800 | 0.9600 | **+0.3800** | Độ phủ tài liệu tăng mạnh nhờ Hybrid Search (BM25 + Dense) kết hợp Contextual Prepend & HyQA. |

---

## 2. Phân tích chi tiết Bottom-5 Failures (Diagnostic Tree)

### #1. Câu hỏi về thời hạn cập nhật mật khẩu (Version Conflict)
- **Question:** "Mật khẩu của tài khoản nội bộ công ty cần được thay đổi định kỳ bao nhiêu ngày một lần?"
- **Expected:** "120 ngày (theo quy định mới nhất v2 kèm xác thực 2 yếu tố MFA)."
- **Got (Naive):** "90 ngày (theo mat_khau_v1.md)."
- **Worst metric:** `context_precision` / `faithfulness`
- **Error Tree:** Output sai $\rightarrow$ Context lấy nhầm bản cũ (v1) thay vì bản mới (v2) $\rightarrow$ Dense search bị nhầm do vector 2 văn bản quá tương đồng.
- **Root cause:** Trong kho dữ liệu có cả `mat_khau_v1.md` (90 ngày) và `mat_khau_v2.md` (120 ngày). Naive RAG không có cơ chế phân biệt phiên bản hoặc siêu dữ liệu hiệu lực.
- **Suggested fix:** Áp dụng **Contextual Prepend (M5)** ghi rõ trạng thái *"Phiên bản v2 - Hiện hành"* và dùng **BM25 / Reranking (M3)** để ưu tiên tài liệu mới nhất.

---

### #2. Câu hỏi về quy định số ngày nghỉ phép thâm niên (Multi-hop Reasoning)
- **Question:** "Nhân viên làm việc được 10 năm tại công ty thì được nghỉ phép năm tổng cộng bao nhiêu ngày?"
- **Expected:** "14 ngày (12 ngày cơ bản + 2 ngày cho 10 năm thâm niên theo tỷ lệ 1 ngày/5 năm)."
- **Got (Naive):** "12 ngày (bỏ sót điều khoản tăng thêm do thâm niên)."
- **Worst metric:** `context_recall`
- **Error Tree:** Output thiếu ý $\rightarrow$ Context chỉ lấy được câu đầu tiên của đoạn nghỉ phép do chunking cố định bị ngắt giữa chừng.
- **Root cause:** Fixed-size chunking cắt đứt quy định tính thâm niên ở câu tiếp theo, khiến LLM không có đủ thông tin để tính toán.
- **Suggested fix:** Triển khai **Hierarchical Chunking (M1)** trả về toàn bộ Parent chunk (2048 chars) giúp LLM nắm trọn vẹn cả quy định cơ bản và điều khoản thâm niên.

---

### #3. Câu hỏi tra cứu mã số quy định & thuật ngữ chuyên ngành (Exact Keyword)
- **Question:** "Giao thức VPN của công ty sử dụng chuẩn mã hóa nào?"
- **Expected:** "WireGuard với chuẩn mã hóa AES-256."
- **Got (Naive):** "Không tìm thấy thông tin."
- **Worst metric:** `context_recall`
- **Error Tree:** Không tìm thấy $\rightarrow$ Dense Search không nhận diện được từ khóa kỹ thuật hiếm gặp như `AES-256` hoặc `WireGuard`.
- **Root cause:** Embedding model bị trôi vector (embedding drift) đối với các từ viết tắt kỹ thuật hoặc mã số.
- **Suggested fix:** Sử dụng **Hybrid Search (M2)** kết hợp BM25 (đã tách từ qua Underthesea) để tìm chính xác 100% token `AES-256`.

---

### #4. Câu hỏi có điều kiện phủ định (Negation Query)
- **Question:** "Trường hợp nào nhân viên KHÔNG được hưởng nguyên lương khi nghỉ việc riêng?"
- **Expected:** "Nghỉ việc riêng ngoài các trường hợp kết hôn (3 ngày), con kết hôn (1 ngày), bố mẹ/vợ chồng/con mất (3 ngày) thì phải xin nghỉ không lương."
- **Got (Naive):** "Nhân viên được nghỉ có lương khi kết hôn 3 ngày..." (trả lời nhầm sang trường hợp được hưởng lương).
- **Worst metric:** `answer_relevancy`
- **Error Tree:** Trả lời ngược ý $\rightarrow$ Vector search lấy các chunk chứa từ "nghỉ", "hưởng lương" mà không hiểu toán tử phủ định "KHÔNG".
- **Root cause:** Bi-Encoder embeddings yếu trong việc phân biệt các câu mang sắc thái phủ định (Negation).
- **Suggested fix:** **Cross-Encoder Reranker (M3)** với cơ chế Full Cross-Attention giúp phát hiện mối tương quan chặt chẽ của từ "KHÔNG" với câu hỏi, lọc chính xác đoạn quy định về nghỉ không lương.

---

### #5. Câu hỏi đa tài liệu / Tổng hợp nhiều điều khoản (Cross-document Synthesis)
- **Question:** "Quy trình phê duyệt đơn xin nghỉ phép không lương dài ngày cần chữ ký của những ai?"
- **Expected:** "Trưởng bộ phận trực tiếp và Giám đốc Khối Nhân sự (theo workflow_nghiphep.md và quy_che_uy_quyen.md)."
- **Got (Naive):** "Chỉ cần Trưởng phòng phê duyệt."
- **Worst metric:** `context_recall`
- **Error Tree:** Thiếu thông tin phân quyền $\rightarrow$ Retrieval chỉ lấy được 1 chunk từ tài liệu quy trình nghỉ phép, bỏ sót tài liệu ủy quyền ký duyệt.
- **Root cause:** Retrieval không bao quát được mối liên kết liên văn bản.
- **Suggested fix:** **HyQA & Metadata Enrichment (M5)** sinh các câu hỏi liên kết chéo và **Hybrid Search Top-20** đảm bảo bao phủ đầy đủ tất cả tài liệu liên quan trước khi rerank.

---

## 3. Case Study Chuyên sâu (Phục vụ Báo cáo & Trình bày)

### Phân tích Case Study: "Xung đột thông tin chính sách bảo mật (mat_khau_v1 vs mat_khau_v2)"

**Error Tree Walkthrough:**
1. **Output có đúng không?** $\rightarrow$ **SAI**. Naive RAG trả lời hạn đổi mật khẩu là 90 ngày (v1) thay vì 120 ngày (v2).
2. **Context có đúng không?** $\rightarrow$ **SAI**. Naive Search chỉ lấy Top-3 theo cosine similarity, do tài liệu v1 xuất hiện nhiều từ khóa lặp lại nên có điểm cosine cao hơn v2.
3. **Query rewrite / Enrichment có hiệu quả không?** $\rightarrow$ **CÓ**. Trong Production RAG, Module 5 Contextual Prepend bổ sung metadata `version: v2` và `status: active`.
4. **Điểm khắc phục quyết định:** 
   - Module 2 (BM25 + Dense) lấy đủ cả 2 phiên bản vào Top-20.
   - Module 3 (Cross-Encoder) đối chiếu chính xác context phiên bản mới nhất và đẩy `mat_khau_v2.md` lên vị trí **Rank 1**.
   - LLM sinh ra câu trả lời chính xác tuyệt đối 100%.

---

## 4. Định hướng Tối ưu hóa Tiếp theo (Nếu có thêm thời gian)

1. **OCR cho tài liệu PDF dạng ảnh quét:** Tích hợp pipeline OCR (như PaddleOCR hoặc Tesseract) để xử lý triệt để 2 tài liệu scan `BCTC.pdf` và `Nghi_dinh_so_13-2023.pdf`.
2. **Fine-tuning Embedding & Cross-Encoder chuyên biệt tiếng Việt:** Huấn luyện lại model trên tập dữ liệu văn bản hành chính - pháp lý Việt Nam để tăng độ nhạy từ vựng chuyên ngành.
3. **Agentic RAG / Query Routing:** Xây dựng Router phân loại câu hỏi (ví dụ: câu hỏi tra cứu đơn giản $\rightarrow$ BM25; câu hỏi suy luận phức tạp $\rightarrow$ Multi-hop Graph RAG).
