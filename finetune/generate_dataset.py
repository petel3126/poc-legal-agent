"""
Script tự động sinh dữ liệu fine-tune Embedding Model cho Hệ thống RAG Đa Luật (Multi-Law Legal RAG).
- Kho chunks nguồn: data/processed/legal_chunks.json (19,120 chunks từ 9 Bộ luật & Luật)
- Phân bổ mẫu theo tỷ lệ quy mô (Proportional Sampling):
  - Train: ~2,100 samples
  - Validation: 110 samples
  - Test: 110 samples

Tự động trích xuất Cross-Law Hard Negatives thông qua BM25 Okapi trên toàn bộ 19,120 chunks.
Output:
  - finetune/train.json
  - finetune/validation.json
  - finetune/test.json
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
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "legal_chunks.json"
FINETUNE_DIR = ROOT_DIR / "finetune"
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

# Tỷ lệ mẫu mong muốn theo từng bộ luật trong tập Train (~2,100 mẫu tổng cộng)
PROPORTIONAL_TRAIN_TARGETS = {
    "91-2015-QH13": 700,   # Bộ luật Dân sự 2015
    "50-2005-QH11": 500,   # Luật Sở hữu trí tuệ 2005
    "59-2020-QH14": 300,   # Luật Doanh nghiệp 2020
    "36-2005-QH11": 200,   # Luật Thương mại 2005
    "38-2019-QH14": 150,   # Luật Quản lý thuế 2019
    "61-2020-QH14": 100,   # Luật Đầu tư 2020
    "19-2023-QH15": 60,    # Luật BVQL Người tiêu dùng 2023
    "24-2018-QH14": 50,    # Luật An ninh mạng 2018
    "45-2019-QH14": 40,    # Bộ luật Lao động 2019
}

VALIDATION_TARGET = 110
TEST_TARGET = 110

def tokenize_vietnamese(text: str):
    text = text.lower()
    words = re.findall(r'\w+', text)
    return words

def generate_law_specific_queries(chunk):
    doc_id = chunk.get("document_id", "")
    doc_title = chunk.get("title", "")
    art_title = chunk.get("article_title", "") or ""
    content = chunk.get("content", "")
    art_num = chunk.get("article", "")
    
    queries = []
    art_clean = art_title.strip().lower()

    # 1. Mẫu câu hỏi tổng quát theo tên luật & điều
    if art_clean:
        queries.append(f"Quy định của {doc_title} về {art_clean} là gì?")
        queries.append(f"Theo {doc_title}, {art_clean} được quy định như thế nào?")
        queries.append(f"Cho tôi biết thông tin chi tiết về {art_clean} theo pháp luật hiện hành?")

    # 2. Mẫu câu hỏi theo từ khóa miền chuyên sâu
    content_lower = content.lower()

    if doc_id == "91-2015-QH13": # Dân sự
        if "bồi thường thiệt hại" in content_lower:
            queries.append("Trách nhiệm bồi thường thiệt hại ngoài hợp đồng được xác định ra sao?")
            queries.append("Căn cứ phát sinh trách nhiệm bồi thường thiệt hại theo Bộ luật Dân sự?")
        elif "thừa kế" in content_lower:
            queries.append("Quy định về di thừa kế và phân chia di sản thừa kế theo pháp luật?")
            queries.append("Thời hiệu chia di sản thừa kế được quy định là bao lâu?")
        elif "hợp đồng" in content_lower:
            queries.append("Các điều kiện để hợp đồng dân sự có hiệu lực pháp luật?")
            queries.append("Trường hợp hợp đồng dân sự bị vô hiệu được giải quyết thế nào?")
    
    elif doc_id == "50-2005-QH11": # Sở hữu trí tuệ
        if "quyền tác giả" in content_lower or "tác phẩm" in content_lower:
            queries.append("Quyền tác giả đối với tác phẩm văn học nghệ thuật được bảo hộ bao lâu?")
            queries.append("Các hành vi nào bị xem là xâm phạm quyền tác giả?")
        elif "sáng chế" in content_lower or "bằng bảo hộ" in content_lower:
            queries.append("Điều kiện để một giải pháp kỹ thuật được bảo hộ dưới hình thức bằng sáng chế?")
        elif "nhãn hiệu" in content_lower:
            queries.append("Quyền sở hữu công nghiệp đối với nhãn hiệu được xác lập như thế nào?")

    elif doc_id == "59-2020-QH14": # Doanh nghiệp
        if "công ty cổ phần" in content_lower:
            queries.append("Quyền hạn và nghĩa vụ của Đại hội đồng cổ đông trong công ty cổ phần?")
            queries.append("Điều kiện thành lập và cơ cấu tổ chức quản lý của công ty cổ phần?")
        elif "vốn điều lệ" in content_lower:
            queries.append("Thời hạn góp vốn điều lệ khi đăng ký thành lập doanh nghiệp?")
        elif "người đại diện" in content_lower:
            queries.append("Quy định về người đại diện theo pháp luật của doanh nghiệp?")

    elif doc_id == "36-2005-QH11": # Thương mại
        if "mua bán hàng hóa" in content_lower:
            queries.append("Trách nhiệm chuyển giao rủi ro đối với hàng hóa trong hợp đồng thương mại?")
        elif "phạt vi phạm" in content_lower or "bồi thường" in content_lower:
            queries.append("Mức phạt vi phạm tối đa trong hợp đồng thương mại được quy định bao nhiêu %?")
        elif "đại lý" in content_lower or "khuyến mại" in content_lower:
            queries.append("Các hình thức xúc tiến thương mại và khuyến mại bị cấm?")

    elif doc_id == "38-2019-QH14": # Quản lý thuế
        if "kê khai" in content_lower or "nộp thuế" in content_lower:
            queries.append("Thời hạn nộp hồ sơ khai thuế và thời hạn nộp tiền thuế?")
        elif "hoàn thuế" in content_lower:
            queries.append("Hồ sơ và thủ tục hoàn thuế được xử lý như thế nào?")

    elif doc_id == "61-2020-QH14": # Đầu tư
        if "ưu đãi" in content_lower:
            queries.append("Các hình thức và đối tượng được hưởng ưu đãi đầu tư?")
        elif "ngành nghề" in content_lower:
            queries.append("Danh mục ngành nghề cấm đầu tư kinh doanh bao gồm những gì?")

    elif doc_id == "19-2023-QH15": # BVQL Người tiêu dùng
        if "quyền của người tiêu dùng" in content_lower or "bảo vệ" in content_lower:
            queries.append("Người tiêu dùng có những quyền cơ bản nào theo Luật Bảo vệ người tiêu dùng?")
            queries.append("Trách nhiệm của tổ chức kinh doanh khi cung cấp sản phẩm có khuyết tật?")

    elif doc_id == "24-2018-QH14": # An ninh mạng
        if "không gian mạng" in content_lower or "hành vi bị cấm" in content_lower:
            queries.append("Các hành vi bị nghiêm cấm thực hiện trên không gian mạng theo Luật An ninh mạng?")
            queries.append("Trách nhiệm của doanh nghiệp cung cấp dịch vụ mạng trong bảo vệ dữ liệu người dùng?")

    elif doc_id == "45-2019-QH14": # Lao động
        if "thử việc" in content_lower:
            queries.append("Thời gian thử việc tối đa đối với từng loại công việc là bao nhiêu ngày?")
        elif "sa thải" in content_lower:
            queries.append("Trường hợp nào doanh nghiệp được quyền sa thải người lao động?")

    # 3. Trích đoạn ngắn dòng nội dung chính
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) > 1:
        snippet = lines[1][:70]
        if len(snippet) > 25:
            queries.append(f"Pháp luật quy định như thế nào về việc '{snippet.lower()}'?")

    # Trường hợp fallback nếu không tạo được query nào
    if not queries:
        queries.append(f"Nội dung quy định tại Điều {art_num} của {doc_title} là gì?")

    return queries


def main():
    print("=== BẮT ĐẦU SINH TẬP DỮ LIỆU FINE-TUNE EMBEDDING ĐA LUẬT (MULTI-LAW FINETUNING) ===")
    print(f"Đang đọc kho chunks từ {CHUNKS_PATH}...")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chưa có file {CHUNKS_PATH}. Hãy chạy `python src/chunk_legal_text.py` trước!")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Tổng số chunks: {len(chunks):,}")

    # Gom chunks theo document_id
    doc_chunks_map = {}
    for c in chunks:
        doc_id = c["document_id"]
        doc_chunks_map.setdefault(doc_id, []).append(c)

    print("\nKhởi tạo BM25 Okapi Index trên toàn bộ 19,120 chunks để khai thác Cross-Law Hard Negatives...")
    corpus_tokens = [tokenize_vietnamese(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    raw_samples = []
    sample_id_counter = 1

    print("\nTiến hành sinh queries & khai thác Hard Negatives theo tỷ lệ từng luật...")

    for doc_id, target_count in PROPORTIONAL_TRAIN_TARGETS.items():
        doc_chunks = doc_chunks_map.get(doc_id, [])
        if not doc_chunks:
            continue

        generated_for_doc = 0
        random.shuffle(doc_chunks)

        while generated_for_doc < target_count + 30: # Sinh dư 30 mẫu cho valid/test
            for chunk in doc_chunks:
                if generated_for_doc >= target_count + 30:
                    break

                queries = generate_law_specific_queries(chunk)
                query = random.choice(queries)

                # Khai thác BM25 Hard Negatives (Cross-Law negatives)
                q_tokens = tokenize_vietnamese(query)
                scores = bm25.get_scores(q_tokens)
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:40]

                pos_chunk_id = chunk["chunk_id"]
                hard_negs = []
                hard_neg_ids = []

                for idx in top_indices:
                    cand_chunk = chunks[idx]
                    cand_id = cand_chunk["chunk_id"]
                    if cand_id != pos_chunk_id and cand_id not in hard_neg_ids:
                        hard_negs.append(cand_chunk["content"])
                        hard_neg_ids.append(cand_id)
                        if len(hard_negs) == 2: # Lấy 2 hard negatives cho mỗi query
                            break

                if len(hard_negs) < 2:
                    continue

                raw_samples.append({
                    "id": f"multi_law_qa_{sample_id_counter:05d}",
                    "document_id": doc_id,
                    "query": query,
                    "positive": chunk["content"],
                    "positive_id": pos_chunk_id,
                    "hard_negatives": hard_negs,
                    "hard_negative_ids": hard_neg_ids
                })
                sample_id_counter += 1
                generated_for_doc += 1

        print(f"  ✓ {doc_id}: Đã sinh {generated_for_doc} samples")

    # Xáo trộn toàn bộ mẫu
    random.seed(42)
    random.shuffle(raw_samples)

    # Phân chia Train / Validation / Test
    val_samples = raw_samples[:VALIDATION_TARGET]
    test_samples = raw_samples[VALIDATION_TARGET : VALIDATION_TARGET + TEST_TARGET]
    train_samples = raw_samples[VALIDATION_TARGET + TEST_TARGET :]

    # Ghi file JSON
    train_file = FINETUNE_DIR / "train.json"
    val_file = FINETUNE_DIR / "validation.json"
    test_file = FINETUNE_DIR / "test.json"

    train_file.write_text(json.dumps(train_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    val_file.write_text(json.dumps(val_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    test_file.write_text(json.dumps(test_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("HOÀN THÀNH TẠO BỘ DỮ LIỆU FINETUNE ĐA LUẬT!")
    print(f"  - Train dataset      ({len(train_samples):,} mẫu) -> {train_file}")
    print(f"  - Validation dataset ({len(val_samples):,} mẫu)  -> {val_file}")
    print(f"  - Test dataset       ({len(test_samples):,} mẫu)  -> {test_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
