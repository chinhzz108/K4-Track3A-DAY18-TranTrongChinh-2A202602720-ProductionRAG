# Rubric — Lab 18: Production RAG Pipeline

**Tổng: 100 điểm + 10 bonus · Cá nhân**

---

## Implementation (60 điểm)

| # | Tiêu chí | Điểm | Cách chấm |
|---|----------|------|-----------|
| 1 | M1 Chunking — 3 strategies đúng logic | 12 | Code review + `pytest tests/test_m1.py` |
| 2 | M2 Search — BM25 + Dense + RRF | 12 | Code review + `pytest tests/test_m2.py` |
| 3 | M3 Rerank — CrossEncoder load + rerank | 12 | Code review + `pytest tests/test_m3.py` |
| 4 | M4 Eval — RAGAS + failure analysis | 12 | Code review + `pytest tests/test_m4.py` |
| 5 | M5 Enrichment — ít nhất 1 technique | 12 | Code review + `pytest tests/test_m5.py` |

### Thang điểm test (mỗi module)

| Tests pass | Điểm |
|-----------|------|
| 100% | 12 |
| ≥ 75% | 9 |
| ≥ 50% | 6 |
| < 50% | 3 |

---

## Pipeline & Evaluation (25 điểm)

| # | Tiêu chí | Điểm | Cách chấm |
|---|----------|------|-----------|
| 6 | Pipeline chạy end-to-end | 10 | `python src/pipeline.py` exit code 0 |
| 7 | RAGAS scores hợp lý | 10 | Check `reports/ragas_report.json` |
| 8 | Failure analysis có insight | 5 | Review `analysis/failure_analysis.md` |

### Thang điểm RAGAS (#7)

| Điều kiện | Điểm |
|-----------|------|
| ≥ 3 metrics đạt 0.70 | 10 |
| ≥ 2 metrics đạt 0.70 | 8 |
| ≥ 1 metric đạt 0.70 | 5 |
| Pipeline chạy nhưng scores thấp | 3 |

### Thang điểm Failure Analysis (#8)

| Điều kiện | Điểm |
|-----------|------|
| Bottom-5 có diagnosis + fix + Error Tree | 5 |
| Có diagnosis nhưng thiếu Error Tree | 3 |
| Liệt kê failures không phân tích | 1 |

---

## Reflection (15 điểm)

| # | Tiêu chí | Điểm | Cách chấm |
|---|----------|------|-----------|
| 9 | Lecture mapping — concept → code cụ thể | 5 | Bảng mapping đầy đủ 5 modules |
| 10 | Khó khăn — mô tả + cách giải quyết | 5 | Có exact error + debug process |
| 11 | Action plan — áp dụng vào project cá nhân | 5 | Plan cụ thể, có timeline |

---

## Bonus (+10 max)

| Bonus | Điểm | Kiểm tra |
|-------|------|----------|
| RAGAS Faithfulness ≥ 0.85 | +3 | `reports/ragas_report.json` |
| RAGAS tất cả metrics ≥ 0.75 | +3 | `reports/ragas_report.json` |
| Enrichment combined mode (1 call/chunk) | +2 | Code review: `_enrich_single_call()` |
| Latency breakdown report | +2 | Có bảng thời gian từng bước |

---

## Auto-grading

**1. Kiểm tra toàn diện trước khi nộp (khuyến nghị cho mọi hệ điều hành Windows / Linux / macOS):**
```bash
python check_lab.py
```

**2. Các câu lệnh kiểm tra từng phần:**
```bash
# Unit Tests
pytest tests/ -v

# Lint (tùy chọn)
ruff check src/

# Đếm số TODO còn lại (yêu cầu = 0):
# Linux / macOS:
grep -r "# TODO" src/m*.py | wc -l

# Windows (PowerShell):
(Select-String -Path src/*.py -Pattern "# TODO").Count

# Chạy Pipeline
python src/pipeline.py
```

---

## Quy trình nộp

1. Implement tất cả TODOs (đảm bảo 100% tests pass và 0 TODOs còn lại).
2. Chạy `python main.py` → sinh báo cáo bắt buộc tại `reports/ragas_report.json`.
3. Điền kết quả phân tích vào `analysis/failure_analysis.md`.
4. Viết bài suy ngẫm tại `analysis/reflections/reflection_[HọTên].md` (theo mẫu `reflection_TEMPLATE.md`).
5. Kiểm tra tính hợp lệ của bài làm: `python check_lab.py`.
6. Đặt tên repository cá nhân theo đúng quy chuẩn duy nhất:  
   `K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
   *(Ví dụ: `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*.
7. Push toàn bộ lên GitHub (Public) và nộp link repository lên VLearn LMS trước **23h59 ngày diễn ra bài lab**.
