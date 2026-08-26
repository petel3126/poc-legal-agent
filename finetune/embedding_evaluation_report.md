# Báo Cáo Đánh Giá & So Sánh Mô Hình Embedding Sau Khi Fine-tune (Multi-Law Test Set)

Báo cáo đánh giá hiệu năng độc lập trên tập dữ liệu kiểm thử [`finetune/test.json`](file:///c:/Users/Lam/OneDrive/Desktop/poc-legal-rag/finetune/test.json) (110 mẫu phân bổ đại diện trên 9 Bộ luật & Luật Việt Nam).

---

## 1. Thông Tin Môi Trường & Dữ Liệu Thực Nghiệm

* **Mô hình gốc (Pretrained):** `bkai-foundation-models/vietnamese-bi-encoder`
* **Mô hình Fine-tuned:** `finetune/fine_tuned_model` (MultipleNegativesRankingLoss + Cross-Law BM25 Hard Negatives)
* **Quy mô tập Test:** 110 mẫu truy vấn - đoạn luật liên quan, bao phủ 9 văn bản luật:
  * Bộ luật Dân sự 2015 (`91-2015-QH13`)
  * Luật Sở hữu trí tuệ 2005 (`50-2005-QH11`)
  * Luật Doanh nghiệp 2020 (`59-2020-QH14`)
  * Luật Thương mại 2005 (`36-2005-QH11`)
  * Luật Quản lý thuế 2019 (`38-2019-QH14`)
  * Luật Đầu tư 2020 (`61-2020-QH14`)
  * Luật BVQL Người tiêu dùng 2023 (`19-2023-QH15`)
  * Luật An ninh mạng 2018 (`24-2018-QH14`)
  * Bộ luật Lao động 2019 (`45-2019-QH14`)

---

## 2. Bảng Kết Quả So Sánh Chi Tiết (Pretrained vs Fine-tuned)

| Chỉ số Đánh giá (Metric) | Mô hình Gốc (Pretrained) | Mô hình Đã Fine-tune | Mức Tăng Trưởng (Delta) | Nhận xét & Ý nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy@1 (Top-1)** | 0.4091 (40.91%) | **0.5818 (58.18%)** | **+17.27%** 🚀 | Khả năng tìm đúng ngay văn bản luật ở vị trí đầu tiên tăng đột phá. |
| **Accuracy@3 (Top-3)** | 0.8273 (82.73%) | **0.8818 (88.18%)** | **+5.45%** 🚀 | Đảm bảo đoạn luật liên quan nằm trong top 3 kết quả trả về. |
| **Accuracy@5 (Top-5)** | 0.8636 (86.36%) | **0.9091 (90.91%)** | **+4.55%** 🚀 | Vượt mốc 90% độ bao phủ trong top 5 ứng viên. |
| **Accuracy@10 (Top-10)** | 0.9091 (90.91%) | **0.9364 (93.64%)** | **+2.73%** | Giảm thiểu tối đa tình trạng trượt kết quả (Recall trượt). |
| **MRR@10 (Mean Reciprocal Rank)** | 0.6034 | **0.7225** | **+11.91%** 🚀 | Thứ hạng của chunk đúng được đẩy lên vị trí cao hơn đáng kể. |
| **NDCG@10** | 0.6799 | **0.7761** | **+9.62%** 🚀 | Chất lượng phân cấp xếp hạng độ liên quan tăng vượt bậc. |
| **MAP@100 (Mean Average Precision)** | 0.6068 | **0.7260** | **+11.92%** 🚀 | Độ chính xác trung bình tích lũy đạt mức xuất sắc (>0.72). |

---

## 3. Phân Tích & Kết Luận

1. **Hiệu quả của Cross-Law Hard Negatives:**
   * Việc sử dụng kỹ thuật khai phá Hard Negatives đa luật qua BM25 giúp mô hình phân biệt rất tốt giữa các điều luật có cùng từ khóa nhưng thuộc hai bộ luật khác nhau (ví dụ: *Thời hiệu khiếu nại trong Luật Thương mại* vs *Thời hiệu khởi kiện hợp đồng trong Bộ luật Dân sự*).
2. **Cải thiện độ chính xác vị trí Top-1 (+17.27%):**
   * Đây là cải thiện quan trọng nhất đối với hệ thống RAG, giúp giảm đáng kể nhiễu ngữ cảnh (context noise) đưa vào LLM khi sinh câu trả lời.
3. **Sẵn sàng triển khai:**
   * Mô hình đã lưu tại [`finetune/fine_tuned_model/`](file:///c:/Users/Lam/OneDrive/Desktop/poc-legal-rag/finetune/fine_tuned_model/) và có thể nạp trực tiếp vào toàn bộ các module tìm kiếm Hybrid + Reranker của hệ thống.
