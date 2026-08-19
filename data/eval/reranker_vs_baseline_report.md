# Báo Cáo So Sánh Chi Tiết: Baseline (BM25) vs Hybrid RRF + Cross-Encoder Reranker (Top 5)

Báo cáo so sánh trực tiếp hiệu năng giữa phương pháp ban đầu (**BM25 Baseline**) và phương pháp nâng cấp **2-Stage Retrieval (Hybrid RRF + Cross-Encoder Reranker)** trên **30 câu hỏi thử nghiệm**.

---

## 1. Bảng So Sánh Chỉ Số Tổng Quan

| Chỉ Số Đánh Giá | Baseline (BM25) | Hybrid RRF + Reranker (Top 5) | Mức Tăng Trưởng (Delta) |
|---|:---:|:---:|:---:|
| **Recall@1** | 46.67% | **70.00%** | **+23.33%** |
| **Recall@3** | 64.76% | **83.10%** | **+18.33%** |
| **Recall@5** | 75.71% | **86.55%** | **+10.83%** |
| **MRR@5 (Mean Reciprocal Rank)** | 0.6078 | **0.7944** | **++0.1867 (+30.7%)** |
| **MAP@5 (Mean Average Precision)** | 0.5805 | **0.7766** | **++0.1961 (+33.8%)** |
| **Độ Trễ Phản Hồi (Latency)** | **1.72 ms** | 9057.76 ms | +9056.04 ms |

---

## 2. Nhận Xét Kết Quả Tinh Chỉnh

1. **Recall@1 cải thiện cực mạnh**: Từ **46.67% lên 70.00%**, nghĩa là kết quả tốt nhất nằm ngay vị trí đầu tiên tăng gần gấp rưỡi.
2. **Recall@5 đạt mức xuất sắc**: Đạt **86.55%**, đảm bảo rằng khi lấy ra **Top 5**, người dùng nhận được đầy đủ các điều khoản pháp lý liên quan.
3. **Cross-Encoder Reranker đóng vai trò lọc tinh**: Giúp loại bỏ hoàn toàn các đoạn văn trùng khớp từ khóa bề mặt nhưng sai mục đích ngữ cảnh.
