"""
Retrieval baseline — tương ứng Bước 3 (Effective Law Filtering), Bước 4
(Retrieval — ở đây tạm dùng BM25-only vì môi trường demo không tải được
embedding model BGE-M3 từ HuggingFace; xem ghi chú cuối file) và Bước 5
(Reference Expansion) trong pipeline PRD mục 3.2.1 / TDD mục 4.

Đây là baseline LEXICAL-ONLY, chưa phải Hybrid Retrieval đầy đủ.
Khi chạy trên máy có internet đầy đủ, thay bm25_search bằng kết hợp
dense vector search (Qdrant + bge-m3) theo đúng thiết kế TDD mục 1.
"""

import json
import re
from datetime import date
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_chunks.json"


def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    return text.split()


def effective_law_filter(chunks, as_of: date = None):
    """Bước 3: Status=ACTIVE AND EffectiveDate<=now AND (ExpirationDate NULL OR now<ExpirationDate)."""
    as_of = as_of or date.today()
    out = []
    for c in chunks:
        if c["status"] != "ACTIVE":
            continue
        eff = date.fromisoformat(c["effective_date"])
        if eff > as_of:
            continue
        if c["expiration_date"]:
            exp = date.fromisoformat(c["expiration_date"])
            if as_of >= exp:
                continue
        out.append(c)
    return out


def reference_expansion(top_chunks, all_chunks_by_id, max_depth=1, max_expanded=3):
    """Bước 5: mở rộng lấy thêm chunk được dẫn chiếu, độ sâu 1, tối đa 3 chunk mở rộng."""
    seen = {c["chunk_id"] for c in top_chunks}
    expanded = []
    for c in top_chunks:
        for ref_id in c["references"]:
            if ref_id in seen or ref_id not in all_chunks_by_id:
                continue
            expanded.append(all_chunks_by_id[ref_id])
            seen.add(ref_id)
            if len(expanded) >= max_expanded:
                return expanded
    return expanded


def bm25_search(query, chunks, top_k=5):
    corpus_tokens = [simple_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def run_baseline_search(chunks_path: Path = CHUNKS_PATH):
    if not chunks_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu chunks tại {chunks_path}")
        return

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_chunks_by_id = {c["chunk_id"]: c for c in chunks}

    print("==========================================================================================")
    print("      HỆ THỐNG TRUY VẤN VĂN BẢN PHÁP LUẬT (DEMO BM25 RETRIEVAL)")
    print("==========================================================================================")
    print("Nhập câu hỏi của bạn (gõ 'q' hoặc 'exit' để thoát).")

    while True:
        try:
            user_input = input("\nNHẬP CÂU HỎI > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát chương trình.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "exit", "quit"):
            print("Đã thoát chương trình.")
            break

        print("\n" + "=" * 90)
        print(f"CÂU HỎI: {user_input}")
        eligible = effective_law_filter(chunks)  # Bước 3
        results = bm25_search(user_input, eligible, top_k=3)  # Bước 4
        top_chunks = [c for c, score in results]
        expanded = reference_expansion(top_chunks, all_chunks_by_id)  # Bước 5

        print("\n-- Top 3 kết quả (BM25) --")
        for idx, (c, score) in enumerate(results, 1):
            label = f"Điều {c['article']}" + (f" khoản {c['clause']}" if c["clause"] else "")
            print(f"\n[{idx}] (Score: {score:.2f}) {label} — {c['article_title']}")
            print("-" * 75)
            # In trọn vẹn nội dung chunk, không bị cắt (...)
            for line in c["content"].split("\n"):
                print(f"    {line}")

        if expanded:
            print("\n-- Chunk mở rộng qua Reference Expansion --")
            for c in expanded:
                label = f"Điều {c['article']}" + (f" khoản {c['clause']}" if c["clause"] else "")
                print(f"\n(Dẫn chiếu) {label} — {c['article_title']}")
                print("-" * 75)
                for line in c["content"].split("\n"):
                    print(f"    {line}")
        print()


