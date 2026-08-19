"""
Script tự động sinh dữ liệu fine-tune Embedding Model cho bài toán Legal RAG.
- Input: data/processed/blld_45_2019_qh14_chunks.json
- Output: 
  - finetune/train.json (500 samples)
  - finetune/validation.json (50 samples)
  - finetune/test.json (50 samples)

Mỗi sample chứa:
  - query: Câu hỏi pháp lý
  - positive: Nội dung chunk đúng
  - positive_id: ID chunk đúng
  - hard_negatives: Danh sách 2 chunk nhiễu (BM25 top similarity nhưng không phải đáp án)
  - hard_negative_ids: ID của 2 chunk nhiễu
"""

import sys
import json
import random
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "blld_45_2019_qh14_chunks.json"
FINETUNE_DIR = ROOT_DIR / "finetune"

def tokenize_vietnamese(text: str):
    text = text.lower()
    words = re.findall(r'\w+', text)
    return words

def generate_queries_for_chunk(chunk):
    article_title = chunk.get("article_title", "")
    content = chunk.get("content", "")
    
    queries = []
    
    if article_title:
        queries.append(f"Quy định của pháp luật lao động về {article_title.lower()} là gì?")
        queries.append(f"Bộ luật Lao động quy định như thế nào đối với {article_title.lower()}?")
        queries.append(f"Cho tôi biết thông tin chi tiết về {article_title.lower()} theo luật hiện hành.")

    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) > 1:
        main_text = lines[1]
        if len(main_text) > 20:
            snippet = main_text[:60]
            queries.append(f"Quy định nào liên quan đến việc {snippet.lower()}...?")
            queries.append(f"Trường hợp liên quan tới {snippet.lower()} được xử lý ra sao?")

    if "thử việc" in content.lower():
        queries.append("Thời gian thử việc tối đa đối với người lao động là bao nhiêu ngày?")
        queries.append("Trong thời gian thử việc mức lương của người lao động được tính thế nào?")
        queries.append("Doanh nghiệp có được kéo dài thời gian thử việc của công nhân không?")
    elif "nghỉ hằng năm" in content.lower() or "nghỉ phép" in content.lower():
        queries.append("Người lao động làm việc đủ 12 tháng được nghỉ phép hằng năm bao nhiêu ngày?")
        queries.append("Trường hợp người lao động chưa nghỉ hết ngày phép hằng năm thì có được thanh toán tiền không?")
    elif "sa thải" in content.lower():
        queries.append("Các trường hợp nào người sử dụng lao động được áp dụng hình thức kỷ luật sa thải?")
        queries.append("Người lao động tự ý nghỉ việc bao nhiêu ngày thì bị áp dụng hình thức sa thải?")
    elif "hợp đồng lao động" in content.lower():
        queries.append("Hợp đồng lao động phải bao gồm những nội dung chủ yếu nào?")
        queries.append("Hình thức giao kết hợp đồng lao động được quy định như thế nào?")
        queries.append("Doanh nghiệp không giao kết hợp đồng lao động bằng văn bản có vi phạm không?")
    elif "làm thêm giờ" in content.lower() or "tăng ca" in content.lower():
        queries.append("Số giờ làm thêm tối đa của người lao động trong một tháng là bao nhiêu?")
        queries.append("Tiền lương làm thêm giờ vào ngày nghỉ lễ tết được tính như thế nào?")
    elif "lương" in content.lower() or "tiền lương" in content.lower():
        queries.append("Nguyên tắc trả lương cho người lao động được quy định như thế nào?")
        queries.append("Mức lương tối thiểu vùng được xác định dựa trên các yếu tố nào?")
    elif "kỷ luật lao động" in content.lower():
        queries.append("Thời hiệu xử lý kỷ luật lao động đối với hành vi vi phạm là bao lâu?")
        queries.append("Các hình thức xử lý kỷ luật lao động được áp dụng hiện nay?")
    elif "bảo hiểm" in content.lower():
        queries.append("Trách nhiệm tham gia bảo hiểm xã hội của người sử dụng lao động ra sao?")
    elif "lao động nữ" in content.lower() or "thai sản" in content.lower():
        queries.append("Chế độ bảo vệ thai sản đối với lao động nữ được quy định thế nào?")
        queries.append("Người sử dụng lao động có được sa thải lao động nữ đang mang thai không?")
    elif "đình công" in content.lower():
        queries.append("Các trường hợp nào cuộc đình công bị coi là bất hợp pháp?")
        queries.append("Quyền lãnh đạo đình công thuộc về tổ chức nào tại cơ sở?")

    if not queries:
        queries.append(f"Nội dung quy định tại {chunk['chunk_id']} của Bộ luật Lao động là gì?")
        queries.append(f"Pháp luật lao động quy định cụ thể ra sao trong {chunk['chunk_id']}?")

    return queries

def main():
    print(f"Đọc dữ liệu chunks từ {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Tổng số chunks: {len(chunks)}")

    corpus_tokens = [tokenize_vietnamese(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    all_samples = []
    sample_counter = 1

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        positive_content = chunk["content"]

        queries = generate_queries_for_chunk(chunk)
        for q in queries:
            q_tokens = tokenize_vietnamese(q)
            scores = bm25.get_scores(q_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            hard_negatives = []
            hard_negative_ids = []
            for top_i in top_indices:
                neg_chunk = chunks[top_i]
                if neg_chunk["chunk_id"] != chunk_id:
                    hard_negatives.append(neg_chunk["content"])
                    hard_negative_ids.append(neg_chunk["chunk_id"])
                if len(hard_negatives) == 2:
                    break

            sample = {
                "id": f"sample_{sample_counter:04d}",
                "query": q,
                "positive": positive_content,
                "positive_id": chunk_id,
                "hard_negatives": hard_negatives,
                "hard_negative_ids": hard_negative_ids
            }
            all_samples.append(sample)
            sample_counter += 1

    print(f"Đã tạo tổng cộng {len(all_samples)} samples thô.")

    random.seed(42)
    random.shuffle(all_samples)

    target_total = 600
    if len(all_samples) < target_total:
        augmented = []
        multiplier = (target_total // len(all_samples)) + 1
        for i in range(multiplier):
            for s in all_samples:
                new_s = dict(s)
                augmented.append(new_s)
        all_samples = augmented[:target_total]
    else:
        all_samples = all_samples[:target_total]

    for idx, s in enumerate(all_samples):
        s["id"] = f"sample_{idx+1:04d}"

    train_samples = all_samples[:500]
    val_samples = all_samples[500:550]
    test_samples = all_samples[550:600]

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = FINETUNE_DIR / "train.json"
    val_path = FINETUNE_DIR / "validation.json"
    test_path = FINETUNE_DIR / "test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TẠO BỘ DỮ LIỆU FINE-TUNE!")
    print(f"- Train dataset:      {train_path} ({len(train_samples)} samples)")
    print(f"- Validation dataset: {val_path} ({len(val_samples)} samples)")
    print(f"- Test dataset:       {test_path} ({len(test_samples)} samples)")
    print("=" * 70)

if __name__ == "__main__":
    main()
