# Báo cáo So sánh Hiệu năng Pipeline: Pretrained vs Fine-tuned Legal Embedding

Báo cáo so sánh chi tiết giữa **Mô hình Gốc chưa fine-tune** (`bkai-foundation-models/vietnamese-bi-encoder`) và **Mô hình Đã Fine-tune** (`finetune/fine_tuned_model`) trên **Bộ 30 câu hỏi thử nghiệm pháp lý (Benchmark 30 QA)**. Tất cả các phương pháp đều kết hợp đầy đủ các kỹ thuật **Hierarchical Legal Chunking**, **BM25 + Dense Hybrid RRF**, và **Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)**.

---

## 1. Bảng So sánh Tổng thể Pipeline (Overall Performance)

### A. So sánh riêng Mô hình Dense Search (Bi-Encoder Only)

| Metric | Mô hình Gốc (Pretrained) | Mô hình Đã Fine-tune | Mức Tăng Trưởng |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 0.6000 (60.0%) | **0.7000 (70.0%)** | **+10.00%** 🚀 |
| **Recall@3** | 0.7726 (77.3%) | **0.8333 (83.3%)** | **+6.07%** |
| **Recall@5** | 0.8571 (85.7%) | **0.9000 (90.0%)** | **+4.29%** |
| **MRR@5** | 0.7250 | **0.7917** | **+6.67%** |
| **MAP@5** | 0.6933 | **0.7750** | **+8.17%** |

---

### B. So sánh Full Pipeline Hoàn chỉnh (BM25 + Dense RRF + Reranker)

| Metric | Full Pipeline (Chưa Fine-tune) | Full Pipeline (Đã Fine-tune) | Mức Tăng Trưởng |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 0.7000 (70.0%) | **0.7667 (76.7%)** | **+6.67%** 🚀 |
| **Recall@3** | 0.8310 (83.1%) | **0.8643 (86.4%)** | **+3.33%** 🚀 |
| **Recall@5** | 0.8655 (86.6%) | **0.9024 (90.2%)** | **+3.70%** 🚀 |
| **MRR@5** | 0.7944 | **0.8361** | **+4.17%** 🚀 |
| **MAP@5** | 0.7766 | **0.8254** | **+4.88%** 🚀 |
| **Latency** | 301.76 ms | 302.00 ms | ~ |

---

## 2. Thư mục `data/eval/` hiện tại

Thư mục `data/eval/` đã được tinh chỉnh đúng chuẩn chỉ bao gồm:
1. **[legal_qa_eval_30.json](file:///c:/Users/Lam/Downloads/poc-legal-rag/poc-legal-rag/data/eval/legal_qa_eval_30.json)**: Bộ 30 câu hỏi thử nghiệm pháp lý gán nhãn ground truth.
2. **[eval_results_pretrained.json](file:///c:/Users/Lam/Downloads/poc-legal-rag/poc-legal-rag/data/eval/eval_results_pretrained.json)**: Kết quả đánh giá full 4-stage pipeline của Mô hình Gốc.
3. **[eval_results_finetuned.json](file:///c:/Users/Lam/Downloads/poc-legal-rag/poc-legal-rag/data/eval/eval_results_finetuned.json)**: Kết quả đánh giá full 4-stage pipeline của Mô hình Đã Fine-tune.
4. **[finetuned_vs_pretrained_report.md](file:///c:/Users/Lam/Downloads/poc-legal-rag/poc-legal-rag/data/eval/finetuned_vs_pretrained_report.md)**: Báo cáo tổng hợp so sánh trực quan định dạng Markdown.
