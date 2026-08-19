"""
Pipeline Người Dùng (User Application Pipeline) — Dùng cho người dùng cuối tương tác hỏi đáp.

Luồng thực hiện tuần tự:
  Bước 1: Hierarchical Legal Chunking (src/chunk_legal_text.py)
  Bước 2: Build Vector Embeddings (src/build_embeddings.py)
  Bước 3: Hybrid Retrieval & Cross-Encoder Reranking (Tương tác hỏi đáp trực tiếp)
"""

import sys
from pathlib import Path

# Đảm bảo UTF-8 encoding trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunk_legal_text import run_chunking
from src.build_embeddings import run_embedding
from src.retrieve_hybrid import run_hybrid_search


def run_user_pipeline():
    print("\n" + "=" * 95)
    print("      PIPELINE 2: HỆ THỐNG PHÁP LUẬT HỎI ĐÁP DÀNH CHO NGUỜI DÙNG (HYBRID RRF + RERANKER)      ")
    print("=" * 95 + "\n")

    chunks_file = ROOT_DIR / "data" / "processed" / "blld_45_2019_qh14_chunks.json"
    embeddings_file = ROOT_DIR / "data" / "processed" / "blld_45_2019_qh14_embeddings.npy"

    if not chunks_file.exists():
        print("\n[BƯỚC 1/3] HIERARCHICAL LEGAL CHUNKING (TÁCH ĐOẠN VĂN BẢN LUẬT)")
        print("-" * 75)
        run_chunking()
    else:
        print("[INFO] Đã tìm thấy dữ liệu chunking sẵn có.")

    if not embeddings_file.exists():
        print("\n[BƯỚC 2/3] BUILD VECTOR EMBEDDINGS (TẠO MẢNG VECTOR DENSE)")
        print("-" * 75)
        run_embedding()
    else:
        print("[INFO] Đã tìm thấy dữ liệu vector embeddings sẵn có.")

    print("\n[BƯỚC 3/3] TRUY VẤN VÀ XẾP HẠNG (HYBRID RETRIEVAL & RERANKER TOP 5)")
    print("-" * 75)
    run_hybrid_search(interactive=True)


if __name__ == "__main__":
    run_user_pipeline()
