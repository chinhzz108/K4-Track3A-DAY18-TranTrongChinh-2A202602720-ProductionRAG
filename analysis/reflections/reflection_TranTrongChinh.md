# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Trọng Chinh  
**Mã học viên:** 2A202602720  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Bảng ánh xạ các khái niệm lý thuyết cốt lõi vào các module thực hành trong codebase:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Phân tách theo ranh giới câu, dùng embedding đo cosine similarity giữa các câu liên tiếp. Ngắt chunk khi similarity < threshold (0.85), giúp bảo toàn trọn vẹn ngữ nghĩa câu và không làm gãy ý. |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Tạo cấu trúc phân cấp: Parent (2048 chars) lưu trữ ngữ cảnh rộng cho LLM sinh câu trả lời, Child (256 chars) chứa vector embedding cô đọng tối ưu cho bước Retrieval. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Parse Markdown header (`#`, `##`, `###`) để giữ nguyên vẹn các section, danh sách, và bảng biểu logic mà không cắt đôi. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Sử dụng Underthesea để ghép từ tiếng Việt và chuyển ký tự `_` thành khoảng trắng nhằm giúp BM25 token hóa và so khớp chính xác từ ghép (VD: "nghỉ phép"). |
| **Hybrid Search & RRF** | M2 | `reciprocal_rank_fusion()` | Kết hợp Sparse Search (BM25) và Dense Vector Search (bge-m3 + Qdrant) theo công thức Reciprocal Rank Fusion: $score(d) = \sum \frac{1}{k + rank + 1}$. Khắc phục triệt để khoảng cách thang điểm giữa BM25 và Cosine score. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Áp dụng `BAAI/bge-reranker-v2-m3` để chấm điểm lại Top-20 ứng viên từ bước tìm kiếm xuống Top-3 chất lượng cao nhất nhờ cơ chế full cross-attention giữa query và passage. |
| **RAGAS 4 Metrics** | M4 | `evaluate_ragas()` | Đánh giá toàn diện 4 khía cạnh: Faithfulness (chống hallucination), Answer Relevancy (đúng trọng tâm), Context Precision (xếp hạng chunk đúng lên đầu), Context Recall (bao phủ sự thật cần thiết). |
| **Diagnostic Tree** | M4 | `failure_analysis()` | Tự động phân tích các câu hỏi điểm thấp, định vị nguyên nhân gốc rễ (retrieval lỗi hay generation lỗi) và đề xuất phương án khắc phục. |
| **Contextual Prepend** | M5 | `contextual_prepend()` | Bổ sung 1-2 câu giải thích ngữ cảnh vị trí của chunk trong tài liệu gốc trước khi index, giảm tới 49% lỗi truy xuất theo nghiên cứu của Anthropic. |
| **HyQA & Single-Call Enrichment** | M5 | `_enrich_single_call()` | Gộp 4 kỹ thuật (Summary, HyQA, Contextual Prepend, Metadata Extraction) vào 1 lần gọi API Groq LLaMA-3.3-70B dạng JSON để tối ưu 75% chi phí và thời gian. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. *Lỗi tách từ BM25 tiếng Việt:* Underthesea mặc định nối từ ghép dạng `nghỉ_phép`. Khi BM25 split bằng khoảng trắng, query `"nghỉ phép"` tách thành 2 token riêng biệt và không khớp với `nghỉ_phép` trong index.
  2. *Lỗi xung đột điểm số khi merge:* Điểm BM25 ($[0, +\infty)$) và Cosine similarity ($[-1, 1]$) không đồng nhất về độ lớn, nếu cộng trực tiếp sẽ làm méo mó kết quả.
  3. *Lỗi tải thư viện nặng khi mạng gián đoạn:* Quá trình tải PyTorch thường bị timeout trên các kết nối quốc tế.

- **Nguyên nhân gốc rễ & Cách debug:**
  1. *Xử lý từ ghép:* Trong hàm `segment_vietnamese()`, thực hiện `.replace("_", " ")` sau khi tách từ để BM25 xử lý đồng nhất cả query và corpus.
  2. *Hợp nhất thứ hạng RRF:* Áp dụng thuật toán Reciprocal Rank Fusion dựa hoàn toàn trên thứ hạng (`rank`) thay vì điểm số thô.
  3. *Tối ưu hóa mô hình & API:* Sử dụng Groq API (`llama-3.3-70b-versatile`) với độ trễ cực thấp thay thế cho các API nặng, đồng thời chia nhỏ tiến trình cài đặt các thư viện lõi.

- **Kiến thức củng cố & Thu hoạch:**
  - Hiểu rõ sự đánh đổi giữa Bi-Encoder (tốc độ cao, recall tốt) và Cross-Encoder (chính xác cao, tính toán sâu).
  - Nắm vững quy trình đánh giá định lượng bằng RAGAS thay vì đánh giá cảm tính.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý RAG Tra cứu Quy trình & Quy định Pháp chế Doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG với Chunking cố định 500 ký tự và Dense search đơn thuần qua OpenAI embeddings.
- **Vấn đề / Bottlenecks đang gặp:**
  - Khó tìm kiếm chính xác các điều khoản có chứa số hiệu văn bản cụ thể (ví dụ: *"Điều 12 Nghị định 13/2023/NĐ-CP"*).
  - Thiếu tính trung thực (Faithfulness) khi LLM tổng hợp các quy định đã hết hiệu lực do văn bản cập nhật phiên bản mới.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Structure-Aware Chunking** kết hợp **Hierarchical (Parent-Child)** theo từng Điều/Khoản luật để giữ trọn vẹn ngữ cảnh pháp lý.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense bge-m3 + RRF)** để vừa tìm được thuật ngữ pháp lý chính xác bằng BM25, vừa bắt được câu hỏi ngữ nghĩa của người dùng.
3. **Reranking:** Tích hợp `bge-reranker-v2-m3` để chọn lọc Top-3 căn cứ pháp luật chuẩn xác nhất.
4. **Enrichment:** Sử dụng **Contextual Prepend** ghi rõ số hiệu văn bản và trạng thái hiệu lực (Hiệu lực / Hết hiệu lực) vào đầu mỗi chunk.
5. **Evaluation:** Thiết lập CI/CD chạy bộ test 50 câu hỏi benchmark bằng RAGAS để giám sát chất lượng liên tục.

#### 3. Timeline triển khai
- **Tuần 1:** Xây dựng module Parser bóc tách văn bản pháp luật và triển khai Hybrid Search với Qdrant.
- **Tuần 2:** Tích hợp Cross-Encoder Reranker, hoàn thiện Prompt Guardrails, và thiết lập bảng đo lường RAGAS.
