"""
Hybrid Retrieval System 2-Stage — kết hợp Lexical Search (BM25) và Dense Vector Search (SentenceTransformer)
sử dụng thuật toán Reciprocal Rank Fusion (RRF) kết hợp với Cross-Encoder Reranker (BAAI/bge-reranker-base).

Các bước xử lý:
  Bước 3: Effective Law Filtering (Lọc văn bản còn hiệu lực)
  Bước 4: Hybrid Candidate Retrieval (BM25 + Dense Search -> Top 15 RRF Candidates)
  Bước 5: Cross-Encoder Reranking (Xếp hạng lại bằng BAAI/bge-reranker-base -> Top 5 Final Results)
  Bước 6: Reference Expansion (Mở rộng văn bản được dẫn chiếu)
"""

import sys
import json
import re
from datetime import date
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.llm_generation import generate_legal_answer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_chunks.json"

EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_embeddings.npy"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_embedding_meta.json"

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"



def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    return text.split()


def effective_law_filter(chunks, embeddings_matrix, as_of: date = None):
    """Bước 3: Lọc các chunk còn hiệu lực."""
    as_of = as_of or date.today()
    eligible_chunks = []
    eligible_indices = []

    for idx, c in enumerate(chunks):
        if c["status"] != "ACTIVE":
            continue
        eff = date.fromisoformat(c["effective_date"])
        if eff > as_of:
            continue
        if c["expiration_date"]:
            exp = date.fromisoformat(c["expiration_date"])
            if as_of >= exp:
                continue
        eligible_chunks.append(c)
        eligible_indices.append(idx)

    eligible_embeddings = embeddings_matrix[eligible_indices] if embeddings_matrix is not None else None
    return eligible_chunks, eligible_embeddings


def bm25_search(query: str, chunks: list, top_k=15):
    """BM25 Lexical Search."""
    corpus_tokens = [simple_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def hybrid_rrf_search(query: str, query_vec: np.ndarray, chunks: list, embeddings_matrix: np.ndarray, bm25_model: BM25Okapi = None, k=60, top_k=20):
    """
    Giai đoạn 1 (Retrieval): Reciprocal Rank Fusion (RRF) Search lấy Top Candidates.
    Công thức RRF: Score = 1/(k + rank_bm25) + 1/(k + rank_dense)
    Tính điểm Cosine Similarity trực tiếp cho tất cả candidate để luôn hiển thị đầy đủ.
    """
    # 1. BM25 Search (Sử dụng chỉ mục BM25 đã cache)
    if bm25_model is not None:
        bm25 = bm25_model
    else:
        corpus_tokens = [simple_tokenize(c["content"]) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)

    query_tokens = simple_tokenize(query)
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    # 2. Dense Vector Search (Cosine Similarity)
    dense_scores = np.dot(embeddings_matrix, query_vec)

    # 3. Tính Xếp hạng (Rank) cho toàn bộ chunks
    bm25_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-bm25_scores), 1)}
    dense_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-dense_scores), 1)}

    # 4. Tính điểm RRF
    candidates = []
    for idx, c in enumerate(chunks):
        r_bm25 = bm25_ranks[idx]
        r_dense = dense_ranks[idx]
        rrf_score = (1.0 / (k + r_bm25)) + (1.0 / (k + r_dense))
        b_raw = bm25_scores[idx]
        d_raw = dense_scores[idx]
        candidates.append((c, rrf_score, b_raw, d_raw))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def rerank_candidates(query: str, candidates: list, reranker_model: CrossEncoder, top_k=5):
    """
    Giai đoạn 2 (Reranking): Dùng Cross-Encoder lọc Top 20 Candidates từ RRF ra Top 5 Kết Quả Cuối Cùng.
    candidates: list of (chunk, rrf_score, bm25_score, cosine_sim)
    """
    if not candidates:
        return []

    pairs = [(query, c["content"]) for c, rrf_score, b_score, d_score in candidates]
    rerank_scores = reranker_model.predict(pairs, batch_size=32, show_progress_bar=False)

    reranked = []
    for idx, (c, rrf_score, b_score, d_score) in enumerate(candidates):
        r_score = float(rerank_scores[idx])
        reranked.append((c, r_score, rrf_score, b_score, d_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]




def reference_expansion(top_chunks, all_chunks_by_id, max_expanded=3):
    """Bước mở rộng chunk dựa trên dẫn chiếu pháp lý."""
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


def run_hybrid_search(chunks_path: Path = CHUNKS_PATH, embeddings_path: Path = EMBEDDINGS_PATH, meta_path: Path = META_PATH, sample_query: str = None, interactive: bool = True):
    print("==========================================================================================")
    print("      HỆ THỐNG TRUY VẤN PHÁP LUẬT HYBRID 2-STAGE (BM25 + DENSE + RRF + RERANKER TOP 5)")
    print("==========================================================================================")

    if not chunks_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu chunks tại {chunks_path}")
        return

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_chunks_by_id = {c["chunk_id"]: c for c in chunks}

    if not embeddings_path.exists():
        print(f"CẢNH BÁO: Chưa tìm thấy file embeddings {embeddings_path}.")
        print("Vui lòng chạy bước 'build_embeddings' trước để tạo vector embeddings.")
        return

    print("Đang tải vector embeddings...")
    embeddings_matrix = np.load(embeddings_path)

    model_name = MODEL_NAME
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = meta.get("model_name", MODEL_NAME)

    print(f"Đang khởi tạo Bi-Encoder embedding model '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Đang khởi tạo Cross-Encoder Reranker model '{RERANKER_MODEL_NAME}'...")
    try:
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception as e:
        print(f"Lỗi tải reranker {RERANKER_MODEL_NAME}: {e}")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Pre-filter các văn bản còn hiệu lực và tạo sẵn BM25 index một lần duy nhất
    print("Đang chuẩn bị chỉ mục BM25 cho các văn bản...")
    eligible_chunks, eligible_embeddings = effective_law_filter(chunks, embeddings_matrix)
    corpus_tokens = [simple_tokenize(c["content"]) for c in eligible_chunks]
    bm25_model = BM25Okapi(corpus_tokens)

    def execute_query(user_input: str):
        print("\n" + "=" * 95)
        print(f"CÂU HỎI: {user_input}")

        # Bước 4: Hybrid Candidate Retrieval (Reciprocal Rank Fusion k=60 -> Top Candidates)
        query_vec = model.encode(user_input, normalize_embeddings=True)
        candidate_results = hybrid_rrf_search(
            user_input, query_vec, eligible_chunks, eligible_embeddings, bm25_model=bm25_model, k=60, top_k=15
        )

        # Bước 5: Cross-Encoder Reranking -> Top 5 Final Results
        final_top5 = rerank_candidates(user_input, candidate_results, reranker, top_k=5)

        top_chunks = [c for c, r_score, rrf_score, b_score, d_score in final_top5]
        expanded = reference_expansion(top_chunks, all_chunks_by_id)

        

        # Bước 6: Gemini LLM Answer Generation
        answer = generate_legal_answer(user_input, top_chunks, expanded)
        print("\n" + "=" * 95)
        print("          🤖 CÂU TRẢ LỜI TỰ ĐỘNG")
        print("=" * 95)
        print(answer)
        print("=" * 95 + "\n")


    if sample_query:
        execute_query(sample_query)
        if not interactive:
            return

    print("\nHệ thống sẵn sàng! Nhập câu hỏi của bạn (gõ 'q' hoặc 'exit' để thoát).")

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

        execute_query(user_input)


