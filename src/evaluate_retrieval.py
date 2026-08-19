"""
Evaluation Script cho RAG Retrieval Pipeline (2-Stage với Cross-Encoder Reranker).
So sánh 4 phương pháp:
  1. BM25 Lexical Search
  2. Dense Vector Search (vietnamese-bi-encoder)
  3. Hybrid RRF (BM25 + Dense Search + RRF)
  4. Hybrid RRF + Reranker (BM25 + Dense + RRF + BAAI/bge-reranker-base)

Các chỉ số đánh giá:
  - Recall@1, Recall@3, Recall@5
  - MRR@5 (Mean Reciprocal Rank)
  - MAP@5 (Mean Average Precision)
  - Latency (ms)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_QA_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "legal_qa_eval_30.json"
CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_chunks.json"
EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_embeddings.npy"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_embedding_meta.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_results.json"


MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"



def simple_tokenize(text: str):
    import re
    text = text.lower()
    text = re.sub(r"[^\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    return text.split()


def bm25_search(query: str, chunks: list, bm25: BM25Okapi, top_k=15):
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, score in ranked[:top_k]]


def dense_search(query_vec: np.ndarray, chunks: list, embeddings_matrix: np.ndarray, top_k=15):
    scores = np.dot(embeddings_matrix, query_vec)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, score in ranked[:top_k]]


def hybrid_rrf_search(query: str, query_vec: np.ndarray, chunks: list, embeddings_matrix: np.ndarray, bm25: BM25Okapi, k=60, top_k=20):
    bm25_raw = np.array(bm25.get_scores(simple_tokenize(query)))
    dense_raw = np.dot(embeddings_matrix, query_vec)

    bm25_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-bm25_raw), 1)}
    dense_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-dense_raw), 1)}

    rrf_scores = []
    for idx in range(len(chunks)):
        r_bm25 = bm25_ranks[idx]
        r_dense = dense_ranks[idx]
        rrf = (1.0 / (k + r_bm25)) + (1.0 / (k + r_dense))
        rrf_scores.append(rrf)

    ranked = sorted(zip(chunks, rrf_scores), key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, score in ranked[:top_k]]


def reranker_search(query: str, candidate_ids: list, chunk_dict: dict, reranker_model: CrossEncoder, top_k=5):
    if not candidate_ids:
        return []
    candidates = [chunk_dict[cid] for cid in candidate_ids if cid in chunk_dict]
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker_model.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, score in ranked[:top_k]]




def compute_metrics(retrieved_ids, ground_truth_ids):
    gt_set = set(ground_truth_ids)
    
    rec_1 = 1.0 if any(cid in gt_set for cid in retrieved_ids[:1]) else 0.0
    rec_3 = sum(1 for cid in retrieved_ids[:3] if cid in gt_set) / len(gt_set)
    rec_5 = sum(1 for cid in retrieved_ids[:5] if cid in gt_set) / len(gt_set)

    rr_5 = 0.0
    for rank, cid in enumerate(retrieved_ids[:5], 1):
        if cid in gt_set:
            rr_5 = 1.0 / rank
            break

    hits = 0
    sum_precisions = 0.0
    for rank, cid in enumerate(retrieved_ids[:5], 1):
        if cid in gt_set:
            hits += 1
            sum_precisions += hits / rank
    ap_5 = sum_precisions / len(gt_set) if gt_set else 0.0

    return {
        "recall_1": rec_1,
        "recall_3": min(rec_3, 1.0),
        "recall_5": min(rec_5, 1.0),
        "mrr_5": rr_5,
        "map_5": ap_5
    }


def run_evaluate_retrieval(eval_qa_path=EVAL_QA_PATH, chunks_path=CHUNKS_PATH, embeddings_path=EMBEDDINGS_PATH, meta_path=META_PATH, results_path=RESULTS_PATH):
    print("==========================================================================================")
    print("      ĐÁNH GIÁ HIỆU NĂNG RETRIEVAL 2-STAGE: BM25 vs DENSE vs HYBRID RRF vs RERANKER")
    print("==========================================================================================")

    if not chunks_path.exists() or not embeddings_path.exists() or not eval_qa_path.exists():
        print(f"Lỗi: Thiếu dữ liệu để chạy đánh giá. Vui lòng kiểm tra:\n  - Chunks: {chunks_path}\n  - Embeddings: {embeddings_path}\n  - QA Eval: {eval_qa_path}")
        return

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_dict = {c["chunk_id"]: c for c in chunks}
    embeddings_matrix = np.load(embeddings_path)
    qa_list = json.loads(eval_qa_path.read_text(encoding="utf-8"))

    model_name = MODEL_NAME
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        model_name = meta.get("model_name", MODEL_NAME)

    print("Tạo chỉ mục BM25...")
    corpus_tokens = [simple_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"Loading reranker model '{RERANKER_MODEL_NAME}'...")
    try:
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    methods = ["BM25", "Dense Search", "Hybrid RRF (Top 20)", "Hybrid RRF + Reranker"]
    results_by_method = {m: [] for m in methods}
    latencies = {m: [] for m in methods}

    print(f"\nBắt đầu đánh giá trên bộ {len(qa_list)} câu hỏi benchmark...\n")

    for item in qa_list:
        qid = item["id"]
        qtext = item["question"]
        gt_ids = item["ground_truth_chunk_ids"]
        category = item["category"]

        # 1. BM25
        t0 = time.perf_counter()
        bm25_res = bm25_search(qtext, chunks, bm25, top_k=5)
        lat_bm25 = (time.perf_counter() - t0) * 1000
        m_bm25 = compute_metrics(bm25_res, gt_ids)
        m_bm25.update({"id": qid, "category": category, "latency_ms": lat_bm25})
        results_by_method["BM25"].append(m_bm25)
        latencies["BM25"].append(lat_bm25)

        # Encode query
        t0_enc = time.perf_counter()
        q_vec = model.encode(qtext, normalize_embeddings=True)
        t_enc = (time.perf_counter() - t0_enc) * 1000

        # 2. Dense
        t0 = time.perf_counter()
        dense_res = dense_search(q_vec, chunks, embeddings_matrix, top_k=5)
        lat_dense = (time.perf_counter() - t0) * 1000 + t_enc
        m_dense = compute_metrics(dense_res, gt_ids)
        m_dense.update({"id": qid, "category": category, "latency_ms": lat_dense})
        results_by_method["Dense Search"].append(m_dense)
        latencies["Dense Search"].append(lat_dense)

        # 3. Hybrid RRF (Top 20 Candidates)
        t0 = time.perf_counter()
        hybrid_candidates = hybrid_rrf_search(qtext, q_vec, chunks, embeddings_matrix, bm25, k=60, top_k=20)
        hybrid_res = hybrid_candidates[:5]
        lat_hybrid = (time.perf_counter() - t0) * 1000 + t_enc
        m_hybrid = compute_metrics(hybrid_res, gt_ids)
        m_hybrid.update({"id": qid, "category": category, "latency_ms": lat_hybrid})
        results_by_method["Hybrid RRF (Top 20)"].append(m_hybrid)
        latencies["Hybrid RRF (Top 20)"].append(lat_hybrid)



        # 4. Hybrid RRF + Reranker
        t0 = time.perf_counter()
        rerank_res = reranker_search(qtext, hybrid_candidates, chunk_dict, reranker, top_k=5)
        lat_rerank = (time.perf_counter() - t0) * 1000 + lat_hybrid
        m_rerank = compute_metrics(rerank_res, gt_ids)
        m_rerank.update({"id": qid, "category": category, "latency_ms": lat_rerank})
        results_by_method["Hybrid RRF + Reranker"].append(m_rerank)
        latencies["Hybrid RRF + Reranker"].append(lat_rerank)



    # Tính chỉ số trung bình
    summary = {}
    for m in methods:
        res_list = results_by_method[m]
        summary[m] = {
            "Recall@1": np.mean([r["recall_1"] for r in res_list]),
            "Recall@3": np.mean([r["recall_3"] for r in res_list]),
            "Recall@5": np.mean([r["recall_5"] for r in res_list]),
            "MRR@5": np.mean([r["mrr_5"] for r in res_list]),
            "MAP@5": np.mean([r["map_5"] for r in res_list]),
            "Avg_Latency_ms": np.mean(latencies[m])
        }

    # Bảng tổng hợp tổng quan
    print("=" * 105)
    print(f"{'Phương Pháp':<24} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR@5':<10} | {'MAP@5':<10} | {'Latency (ms)':<12}")
    print("-" * 105)
    for m in methods:
        s = summary[m]
        print(f"{m:<24} | {s['Recall@1']:<10.4f} | {s['Recall@3']:<10.4f} | {s['Recall@5']:<10.4f} | {s['MRR@5']:<10.4f} | {s['MAP@5']:<10.4f} | {s['Avg_Latency_ms']:<12.2f}")
    print("=" * 105)

    # Category Breakdown
    categories = ["keyword_match", "semantic_paraphrase", "complex_legal"]
    cat_names = {
        "keyword_match": "Từ Khóa Chính Xác (10 câu)",
        "semantic_paraphrase": "Đồng Nghĩa / Diễn Đạt Tự Nhiên (10 câu)",
        "complex_legal": "Tình Huống / Phức Tạp (10 câu)"
    }

    print("\n\n" + "=" * 105)
    print(" PHÂN TÍCH HIỆU NĂNG THEO TỪNG NHÓM CÂU HỎI (RECALL@1 & MRR@5)")
    print("=" * 105)
    for cat in categories:
        print(f"\n>>> Nhóm: {cat_names[cat]}")
        print(f"{'Phương Pháp':<24} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR@5':<10}")
        print("-" * 80)
        for m in methods:
            cat_items = [r for r in results_by_method[m] if r["category"] == cat]
            r1 = np.mean([r["recall_1"] for r in cat_items])
            r3 = np.mean([r["recall_3"] for r in cat_items])
            r5 = np.mean([r["recall_5"] for r in cat_items])
            mrr = np.mean([r["mrr_5"] for r in cat_items])
            print(f"{m:<24} | {r1:<10.4f} | {r3:<10.4f} | {r5:<10.4f} | {mrr:<10.4f}")

    output_data = {
        "summary": summary,
        "detail_by_method": results_by_method
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã xuất kết quả chi tiết ra {results_path}\n")
    return output_data

