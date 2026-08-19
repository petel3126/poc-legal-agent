"""
Pipeline Đánh giá (Evaluation Pipeline) — Đánh giá bộ 30 câu hỏi thử nghiệm và xuất báo cáo.

Luồng thực hiện tuần tự:
  Bước 1: Hierarchical Legal Chunking (src/chunk_legal_text.py)
  Bước 2: Build Vector Embeddings (src/build_embeddings.py)
  Bước 3: Retrieval Benchmark Evaluation (src/evaluate_retrieval.py)
  Bước 4: Reranker vs Baseline Comparison & Report Generation (src/evaluate_reranker.py)
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
from src.evaluate_retrieval import run_evaluate_retrieval
from src.evaluate_reranker import run_evaluate_reranker


def run_evaluation_pipeline():
    print("\n" + "=" * 95)
    print("      PIPELINE 1: ĐÁNH GIÁ VÀ XUẤT BÁO CÁO HỆ THỐNG RAG (BENCHMARK 30 CÂU HỎI)      ")
    print("=" * 95 + "\n")

    print("\n[BƯỚC 1/4] HIERARCHICAL LEGAL CHUNKING (TÁCH ĐOẠN VĂN BẢN LUẬT)")
    print("-" * 75)
    run_chunking()

    print("\n[BƯỚC 2/4] BUILD VECTOR EMBEDDINGS (TẠO MẢNG VECTOR DENSE)")
    print("-" * 75)
    run_embedding()

    print("\n[BƯỚC 3/4] ĐÁNH GIÁ ĐỊNH LƯỢNG RETRIEVAL (BENCHMARK 30 CÂU HỎI)")
    print("-" * 75)
    run_evaluate_retrieval()

    print("\n[BƯỚC 4/4] CROSS-ENCODER RERANKING & XUẤT BÁO CÁO SO SÁNH")
    print("-" * 75)
    run_evaluate_reranker()

    print("\n" + "=" * 95)
    print("     HOÀN THÀNH PIPELINE ĐÁNH GIÁ VÀ XUẤT BÁO CÁO THÀNH CÔNG!     ")
    print("==========================================================================================\n")


if __name__ == "__main__":
    run_evaluation_pipeline()
