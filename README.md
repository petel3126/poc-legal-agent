# PoC — Legal Q&A Retrieval Pipeline (AI Legal & Compliance Assistant)

PoC cho Bước 1–4 trong lộ trình xây dựng (xem TDD v1.0 mục 4).
Dữ liệu: Bộ luật Lao động 45/2019/QH14 — Chương I–VIII (Điều 1–129, một phần,
đủ dùng cho PoC). Nguồn: thuvienphapluat.vn.

## Cấu trúc
- `evaluation.py` — Pipeline 1: Chạy tuần tự Chunking -> Embedding -> Evaluate 30 câu -> Xuất báo cáo.
- `main.py` — Pipeline 2: Chạy tuần tự Chunking -> Embedding -> Mở khung Hỏi & Đáp Tương tác cho người dùng.
- `data/raw/` — văn bản luật gốc dạng text đã làm sạch.
- `data/processed/` — chunk đã tách theo Chương→Điều→Khoản, kèm metadata và vector embeddings (`.npy`).
- `data/eval/legal_qa_eval_30.json` — Bộ 30 câu hỏi benchmark gán nhãn ground truth.
- `src/chunk_legal_text.py` — Bước tiền xử lý: Hierarchical Legal Chunking.
- `src/build_embeddings.py` — Bước tạo dense vector embeddings (`vietnamese-bi-encoder`).
- `src/retrieve_baseline.py` — Retrieval Baseline (BM25-only).
- `src/retrieve_hybrid.py` — Hybrid Retrieval (BM25 + Dense Search + RRF Fusion + Reranker).
- `src/evaluate_retrieval.py` — Script đánh giá định lượng tự động (Recall@k, MRR@5, MAP@5).
- `src/evaluate_reranker.py` — Script đánh giá so sánh chi tiết Baseline vs Hybrid Reranker.

## Chạy chương trình

```bash
# 1. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 2. Pipeline 1: Đánh giá bộ 30 câu hỏi và xuất báo cáo tự động
python evaluation.py

# 3. Pipeline 2: Dành cho người dùng (Chunking -> Embedding -> Hỏi đáp tương tác trực tiếp)
python main.py
```



## Việc còn thiếu để thành Hybrid Retrieval đúng thiết kế TDD
Môi trường tạo PoC này không tải được embedding model (BGE-M3) từ
HuggingFace, nên `retrieve_baseline.py` hiện chỉ là **lexical-only (BM25)**,
chưa phải Hybrid (BM25 + Dense Vector) như TDD mục 1/4 mô tả. Để hoàn
thiện, chạy trên máy có internet:

1. `pip install sentence-transformers qdrant-client`
2. Embed toàn bộ `content` trong file chunk bằng `BAAI/bge-m3`, đẩy vào Qdrant
   (payload = toàn bộ metadata chunk, để filter status/effective_date ngay
   trong query — đúng TDD mục 3.2).
3. Kết hợp điểm BM25 (đã có) với điểm cosine similarity từ Qdrant theo
   công thức `score = w_v*vector_score + w_l*lexical_score` (TDD mục 4,
   bước 4).
4. Bổ sung Bước 6 (Cross-Encoder Reranking) bằng `bge-reranker-v2-m3`.

## Bước tiếp theo (theo lộ trình đã thống nhất)
- Soạn 50–100 câu hỏi + gán nhãn chunk_id đúng → đo Recall@5/MRR (SRS mục 12).
- Thêm bước 9–10 (LLM Generation + Grounding Verification) bằng Claude API.
