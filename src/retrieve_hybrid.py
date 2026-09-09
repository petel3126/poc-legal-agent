"""
Hybrid Retrieval System 2-Stage — kết hợp Lexical Search (BM25) và Dense Vector Search (SentenceTransformer)
sử dụng thuật toán Reciprocal Rank Fusion (RRF) kết hợp với Cross-Encoder Reranker (BAAI/bge-reranker-base).

Các bước xử lý:
  Bước 3: Effective Law Filtering (Lọc văn bản còn hiệu lực)
  Bước 4: Hybrid Candidate Retrieval (BM25 + Dense Search -> Top 15 RRF Candidates)
  Bước 5: Cross-Encoder Reranking (Xếp hạng lại bằng BAAI/bge-reranker-base -> Top 5 Final Results)
  Bước 6: Reference Expansion (Mở rộng văn bản được dẫn chiếu)
"""

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import date
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.llm_generation import generate_legal_answer, get_gemini_client

try:
    from google.genai import types
except ImportError:
    types = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_chunks.json"

EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_embeddings.npy"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_embedding_meta.json"

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Cấu hình tham số Top K mặc định cho toàn bộ hệ thống RAG (đồng bộ Web & CLI)
DEFAULT_CANDIDATE_TOP_K = 35  # Giai đoạn 1: Hybrid BM25 + Dense FAISS lấy Top 35 ứng viên
DEFAULT_RERANK_TOP_K = 7      # Giai đoạn 2: Cross-Encoder lọc ra Top 7 kết quả cao nhất
DEFAULT_MAX_EXPANDED = 3      # Giai đoạn 3: Mở rộng tối đa 3 dẫn chiếu đồ thị (Graph expansion)

DECOMPOSE_SYSTEM_PROMPT = """Bạn là một chuyên gia phân tích và xử lý truy vấn Pháp lý (Legal Query Decomposition Expert).
Nhiệm vụ của bạn: Nhận một câu hỏi pháp lý phức tạp của người dùng và phân tích, chia nhỏ (decompose) thành từ 2 đến 4 câu truy vấn con (sub-queries) độc lập, ngắn gọn và giàu thuật ngữ pháp lý để phục vụ tìm kiếm văn bản quy phạm pháp luật.

Quy tắc phân rã:
1. Tách từng quan hệ pháp lý, tình huống hoặc câu hỏi riêng biệt thành 1 sub-query độc lập.
2. Chuẩn hóa ngôn ngữ đời thường thành thuật ngữ pháp lý chuẩn của Việt Nam (ví dụ: "chết trước" -> "thừa kế thế vị", "bố tôi có được hưởng không" -> "quy định về hàng thừa kế và thừa kế thế vị", "cho lại đất" -> "tặng cho quyền sử dụng đất hoặc thỏa thuận phân chia di sản", "nghỉ việc đột ngột" -> "đơn phương chấm dứt hợp đồng lao động trái pháp luật").
3. Không thêm các suy đoán sai lệch, chỉ tập trung vào các vấn đề pháp lý then chốt cần tra cứu.
4. Trả về DUY NHẤT một JSON array chứa danh sách các chuỗi sub-queries: ["sub_query_1", "sub_query_2", ...]
"""
# =====================================================================
# TRỤ CỘT 4: SUBSUMPTIVE QUERY ENRICHMENT (QUY NẠP TỪ VỰNG ĐỜI THƯỜNG -> PHÁP LÝ)
# =====================================================================
LEGAL_SUBSUMPTION_DICTIONARY = [
    # Nhóm hồ sơ, văn bằng, học vấn, bằng cấp
    (r"\b(bằng\s+cấp|bằng\s+cấp\s+ba|bằng\s+cấp\s+3|mượn\s+bằng|khai\s+bằng|học\s+hết\s+lớp\s+\d+|không\s+có\s+bằng|gian\s+dối\s+bằng|khai\s+man)\b",
     "trình độ học vấn văn bằng chứng chỉ nghĩa vụ cung cấp thông tin trung thực khi giao kết hợp đồng lao động"),
    
    # Nhóm giữ giấy tờ tùy thân, văn bằng chứng chỉ, bản chính (Điều 17 Khoản 1 BLLĐ 2019)
    (r"(giấy\s+tờ\s+tùy\s+thân|bản\s+chính\s+giấy\s+tờ|giữ\s+.*(giấy\s+tờ|cccd|căn\s+cước|chứng\s+minh|hộ\s+chiếu|bằng\s+gốc|văn\s+bằng|chứng\s+chỉ)|(giấy\s+tờ|cccd|căn\s+cước|chứng\s+minh|hộ\s+chiếu|bằng\s+gốc|văn\s+bằng|chứng\s+chỉ).*giữ|thu\s+giữ\s+giấy\s+tờ|tịch\s+thu\s+giấy\s+tờ|nộp\s+giấy\s+tờ|đưa\s+giấy\s+tờ|giữ\s+bằng|giữ\s+cccd)",
     "giữ bản chính giấy tờ tùy thân văn bằng chứng chỉ hành vi người sử dụng lao động không được làm khi giao kết thực hiện hợp đồng lao động"),

    # Nhóm yêu cầu đặt cọc tiền, giữ tài sản khi xin việc (Điều 17 Khoản 2 BLLĐ 2019)
    (r"(đặt\s+cọc\s+tiền|thế\s+chấp\s+tiền|ký\s+quỹ\s+xin\s+việc|bảo\s+đảm\s+bằng\s+tiền|nộp\s+tiền\s+cọc|giữ\s+tiền\s+cọc|đặt\s+tiền\s+mới\s+cho\s+làm)",
     "yêu cầu người lao động thực hiện biện pháp bảo đảm bằng tiền hoặc tài sản khác cho việc thực hiện hợp đồng lao động hành vi không được làm"),
    
    # Nhóm chấm dứt hợp đồng, nghỉ việc
    (r"\b(nghỉ\s+việc\s+đột\s+ngột|tự\s+ý\s+bỏ\s+việc|nghỉ\s+ngang|bỏ\s+việc\s+không\s+phép|tự\s+ý\s+nghỉ)\b",
     "đơn phương chấm dứt hợp đồng lao động tự ý bỏ việc mà không có lý do chính đáng thời hạn báo trước"),
    
    # Nhóm tiền lương, nợ lương
    (r"\b(quỵt\s+lương|chậm\s+trả\s+lương|nợ\s+lương|không\s+trả\s+lương|giam\s+lương)\b",
     "tiền lương nguyên tắc trả lương chậm trả lương đền bù tiền lãi"),
     
    # Nhóm kỷ luật, phạt tiền
    (r"\b(phạt\s+tiền\s+trừ\s+lương|trừ\s+tiền\s+lương|cắt\s+lương\s+phạt|phạt\s+tiền\s+nhân\s+viên)\b",
     "kỷ luật lao động các hành vi bị nghiêm cấm khi xử lý kỷ luật lao động phạt tiền cắt lương thay xử lý kỷ luật"),
     
    # Nhóm thử việc
    (r"\b(thử\s+việc\s+kéo\s+dài|thử\s+việc\s+\d+\s+tháng|thử\s+việc\s+lại)\b",
     "thời gian thử việc hợp đồng thử việc tiền lương trong thời gian thử việc"),
     
    # Nhóm dân sự: thừa kế, sổ đỏ
    (r"\b(chết\s+không\s+để\s+lại\s+di\s+chúc|mất\s+không\s+có\s+di\s+chúc|không\s+di\s+chúc)\b",
     "thừa kế theo pháp luật hàng thừa kế người thừa kế di sản thừa kế"),
    (r"\b(đất\s+không\s+sổ\s+đỏ|mua\s+bán\s+giấy\s+tay|chưa\s+có\s+sổ\s+đỏ)\b",
     "giấy chứng nhận quyền sử dụng đất điều kiện cấp giấy chứng nhận hợp đồng chuyển nhượng quyền sử dụng đất"),

    # Nhóm lao động nữ mang thai, dưỡng thai, tạm hoãn hợp đồng lao động (Điều 138 BLLĐ 2019)
    (r"(giữ\s+thai|dưỡng\s+thai|dễ\s+sẩy\s+thai|dễ\s+hư\s+thai|động\s+thai|ảnh\s+hưởng\s+thai|bác\s+s[ĩỹ].*(tư|nghỉ)|nghỉ\s+không\s+lương.*thai)",
     "lao động nữ mang thai tạm hoãn thực hiện hợp đồng lao động ảnh hưởng xấu tới thai nhi cơ sở khám bệnh chữa bệnh có thẩm quyền")
]


def enrich_legal_query(query: str) -> str:
    """
    Subsumptive Query Expansion: Tự động quy nạp và làm giàu truy vấn từ ngôn ngữ đời thường
    sang các thuật ngữ quy phạm pháp luật tương ứng để BM25 và Dense Search quét trúng luật.
    """
    expanded_terms = []
    q_lower = query.lower()
    for pattern, legal_terms in LEGAL_SUBSUMPTION_DICTIONARY:
        if re.search(pattern, q_lower, re.IGNORECASE):
            expanded_terms.append(legal_terms)

    if expanded_terms:
        # Nối các thuật ngữ pháp lý bổ sung vào câu truy vấn tìm kiếm
        return query + " " + " ".join(expanded_terms)
    return query


def is_complex_query(query: str) -> bool:
    """
    Kiểm tra xem câu hỏi có phải là câu hỏi tình huống phức tạp nhiều ý (Multi-intent / Multi-issue)
    cần dùng Query Decomposition hay không.

    Không áp dụng cho:
    - Câu hỏi tra cứu luật trực tiếp (định nghĩa, con số, mức phạt, điều kiện đơn lẻ).
    - Câu hỏi tình huống đơn giản chỉ chứa 1 vấn đề pháp lý.

    Chỉ áp dụng cho:
    - Câu hỏi có cấu trúc chia mục rõ ràng (1. ... 2. ... hoặc thứ nhất ... thứ hai ...).
    - Câu hỏi chứa từ 2 dấu '?' trở lên ở các mệnh đề riêng biệt.
    - Câu hỏi tình huống dài kết hợp nhiều vấn đề pháp lý và nhiều yêu cầu giải quyết khác nhau (đồng thời, ngoài ra, hướng giải quyết nào,...).
    """
    q = query.strip()
    words = q.split()
    word_count = len(words)

    # 1. Nếu có đánh số câu hỏi rõ ràng (1. ... 2. ... hoặc thứ nhất ... thứ hai ...)
    if re.search(r"(?:^|\s)(?:1[\.\:\)]|thứ\s+nhất).*(?:2[\.\:\)]|thứ\s+hai)", q, re.IGNORECASE):
        return True

    # 2. Nếu có từ 2 dấu hỏi chấm '?' trở lên ở các câu riêng biệt
    q_marks = q.count("?")
    if q_marks >= 2 and word_count >= 20:
        return True

    # 3. Câu hỏi tra cứu định nghĩa / con số / mức phạt / thủ tục đơn lẻ -> ĐƠN GIẢN (dù câu có dài)
    simple_lookup_patterns = [
        r"^thời\s+giờ\s+làm\s+việc",
        r"^thời\s+gian\s+thử\s+việc",
        r"^mức\s+lương",
        r"^mức\s+phạt",
        r"^hồ\s+sơ\s+gồm",
        r"^điều\s+kiện\s+(để|thành\s+lập|được)",
        r"^thời\s+hạn\s+(là|bao\s+lâu|góp\s+vốn)",
        r"bao\s+nhiêu\s+(giờ|ngày|tháng|năm|lần|triệu|đồng|phần\s+trăm|%)"
    ]
    is_direct_lookup = any(re.search(p, q, re.IGNORECASE) for p in simple_lookup_patterns)
    if is_direct_lookup and not re.search(r"(đồng\s+thời|ngoài\s+ra|hướng\s+giải\s+quyết|mặt\s+khác)", q, re.IGNORECASE):
        return False

    # 4. Kiểm tra liên từ ghép chuyển tiếp đa ý (chỉ kích hoạt khi có tình huống nhiều yêu cầu độc lập)
    multi_topic_markers = [
        r"\bđồng\s+thời\b",
        r"\bngoài\s+ra\b",
        r"\bhướng\s+giải\s+quyết\s+nào\b",
        r"\bcần\s+xử\s+lý\s+theo\s+những\s+quy\s+định\s+nào\b",
        r"\bvừa\s+.*\bvừa\s+.*(?:không|\?)",
        r"\btrong\s+trường\s+hợp\s+.*(?:thì\s+.*)?(?:đồng\s+thời|mặt\s+khác)\b"
    ]
    has_multi_topic_marker = any(re.search(p, q, re.IGNORECASE) for p in multi_topic_markers)

    # 5. Câu hỏi tình huống phức tạp thực sự:
    # - Rất dài (>= 40 từ) VÀ có chứa liên từ chuyển tiếp đa yêu cầu
    if word_count >= 40 and has_multi_topic_marker:
        return True

    # - HOẶC dài (>= 25 từ) VÀ có từ 2 yêu cầu hỏi khác nhau kết hợp liên từ
    if word_count >= 25:
        question_clauses = re.findall(r"(?:có\s+[^,\.\?]+(?:không|k|chăng)|hướng\s+giải\s+quyết|như\s+thế\s+nào|xử\s+lý\s+ra\s+sao|quy\s+định\s+nào)", q, re.IGNORECASE)
        if len(question_clauses) >= 2 and has_multi_topic_marker:
            return True

    return False


def decompose_query(query: str, client=None) -> list:
    """Phân rã câu hỏi phức tạp thành danh sách 2-4 sub-queries qua Gemini API."""
    if not is_complex_query(query):
        return [query]

    gemini_client = client or get_gemini_client()
    if not gemini_client or types is None:
        return [query]

    prompt = f"Phân tích và tách câu hỏi pháp lý phức tạp sau thành 2-4 sub-queries ngắn gọn, chuẩn thuật ngữ pháp luật:\n\n{query}"

    candidate_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
    for model_name in candidate_models:
        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=DECOMPOSE_SYSTEM_PROMPT,
                response_mime_type="application/json"
            )
            resp = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            text = resp.text.strip()
            sub_queries = json.loads(text)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                cleaned = [str(sq).strip() for sq in sub_queries if str(sq).strip()]
                if cleaned:
                    return cleaned
        except Exception:
            continue

    return [query]



try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

FAISS_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_index.faiss"


def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    return text.split()


def effective_law_filter(chunks, embeddings_matrix, as_of: date = None):
    """Lọc các chunk còn hiệu lực (Status=ACTIVE)."""
    as_of = as_of or date.today()
    eligible_chunks = []
    eligible_indices = []

    for idx, c in enumerate(chunks):
        status = c.get("status", "ACTIVE")
        if status and status in ("EXPIRED", "HET_HIEU_LUC", "Hết hiệu lực"):
            continue
        if c.get("effective_date"):
            eff = date.fromisoformat(c["effective_date"])
            if eff > as_of:
                continue
        if c.get("expiration_date"):
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


def hybrid_rrf_search(
    query: str,
    query_vec: np.ndarray,
    chunks: list,
    embeddings_matrix: np.ndarray,
    bm25_model: BM25Okapi = None,
    faiss_index=None,
    k: int = 60,
    top_k: int = DEFAULT_CANDIDATE_TOP_K
):
    """
    Giai đoạn 1 (Candidate Retrieval): Kết hợp BM25 và Dense Search qua thuật toán RRF.
    Hỗ trợ tăng tốc bằng FAISS IndexFlatIP.
    """
    # 1. BM25 Search (Sử dụng chỉ mục BM25 đã cache)
    if bm25_model is not None:
        bm25 = bm25_model
    else:
        corpus_tokens = [simple_tokenize(c["content"]) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)

    query_tokens = simple_tokenize(query)
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    # 2. Dense Vector Search (Dùng FAISS nếu có, fallback về NumPy Dot Product)
    if faiss_index is not None:
        q_vec = query_vec.reshape(1, -1).astype("float32")
        D, I = faiss_index.search(q_vec, len(chunks))
        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        dense_scores[I[0]] = D[0]
        dense_ranks = {int(idx): rank for rank, idx in enumerate(I[0], 1)}
    else:
        dense_scores = np.dot(embeddings_matrix, query_vec)
        dense_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-dense_scores), 1)}

    # 3. Tính Xếp hạng BM25
    bm25_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-bm25_scores), 1)}

    # 4. Tính điểm RRF
    candidates = []
    for idx, c in enumerate(chunks):
        r_bm25 = bm25_ranks[idx]
        r_dense = dense_ranks[idx]
        rrf_score = (1.0 / (k + r_bm25)) + (1.0 / (k + r_dense))
        b_raw = bm25_scores[idx]
        d_raw = float(dense_scores[idx])
        candidates.append((c, rrf_score, b_raw, d_raw))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def rerank_candidates(query: str, candidates: list, reranker_model: CrossEncoder, top_k: int = DEFAULT_RERANK_TOP_K):
    """
    Giai đoạn 2 (Reranking): Dùng Cross-Encoder lọc Candidates từ RRF ra Top Kết Quả Cuối Cùng.
    candidates: list of (chunk, rrf_score, bm25_score, cosine_sim)
    """
    if not candidates:
        return []

    pairs = [(query, c["content"][:800]) for c, rrf_score, b_score, d_score in candidates]
    rerank_scores = reranker_model.predict(pairs, batch_size=16, show_progress_bar=False)

    reranked = []
    for idx, (c, rrf_score, b_score, d_score) in enumerate(candidates):
        r_score = float(rerank_scores[idx])
        reranked.append((c, r_score, rrf_score, b_score, d_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]




# =====================================================================
# TRỤ CỘT 3: DYNAMIC TOP-K & SCORE GAP CUTOFF (CẮT TỈA NHIỄU THEO KHOẢNG CÁCH ĐIỂM)
# =====================================================================
def filter_by_score_gap(
    candidates: list,
    min_score: float = 0.0001,
    max_gap: float = 0.80,
    min_keep: int = 2,
    max_keep: int = DEFAULT_RERANK_TOP_K
) -> list:
    """
    Cắt tỉa động các chunk ứng viên sau khi xếp hạng:
    1. Giữ ít nhất min_keep chunks (nếu danh sách có đủ).
    2. Cắt tối đa max_keep chunks (tránh nhồi nhét làm ngợp LLM).
    3. Loại bỏ mọi chunk có điểm tuyệt đối < min_score (trừ khi để đảm bảo min_keep).
    4. Loại bỏ các chunk có khoảng cách tương đối so với Top-1 rơi vào vùng nhiễu (relative gap > max_gap).
    """
    if not candidates:
        return []

    def get_score(item):
        if isinstance(item, dict):
            return item.get("rerank_score", 0.0)
        elif isinstance(item, (list, tuple)) and len(item) > 1:
            return float(item[1])
        return 0.0

    top_1_score = get_score(candidates[0])
    filtered = []

    for idx, item in enumerate(candidates):
        if idx >= max_keep:
            break
        score = get_score(item)

        # Luôn giữ tối thiểu min_keep chunks đầu tiên
        if idx < min_keep:
            filtered.append(item)
            continue

        # Tính khoảng cách tương đối (relative score drop)
        rel_drop = (top_1_score - score) / (top_1_score + 1e-9) if top_1_score > 0 else 0.0

        if score >= min_score and rel_drop <= max_gap:
            filtered.append(item)
        else:
            # Điểm bắt đầu rơi xuống vùng nhiễu -> Cắt bỏ ngay
            break

    return filtered


def reference_expansion(
    top_chunks,
    all_chunks_by_id,
    graph_retriever=None,
    reranker_model: Optional[CrossEncoder] = None,
    max_expanded: int = DEFAULT_MAX_EXPANDED,
    missing_elements: Optional[List[str]] = None,
    query_text: str = "",
    min_score_threshold: float = 0.85
):
    """
    Bước mở rộng chunk dựa trên đồ thị dẫn chiếu pháp lý (Scored Directional Graph Expansion).
    Tận dụng Article Bundling, lọc ngưỡng sàn và hậu kiểm qua Cross-Encoder.
    """
    if not top_chunks:
        return []

    seen = {c["chunk_id"] for c in top_chunks}
    seed_ids = [c["chunk_id"] for c in top_chunks]

    if graph_retriever and hasattr(graph_retriever, "expand_with_ranking"):
        ranked_candidates = graph_retriever.expand_with_ranking(
            seed_chunk_ids=seed_ids,
            missing_elements=missing_elements,
            query_text=query_text,
            top_n=max_expanded * 2,
            min_score_threshold=min_score_threshold,
            all_chunks_by_id=all_chunks_by_id
        )

        # Hậu kiểm bằng Cross-Encoder nếu có reranker_model và query_text
        if reranker_model and query_text and ranked_candidates:
            pairs = [(query_text, cand.get("content", "")[:800]) for cand in ranked_candidates]
            scores = reranker_model.predict(pairs, batch_size=16, show_progress_bar=False)
            valid_candidates = []
            for idx, cand in enumerate(ranked_candidates):
                r_score = float(scores[idx])
                cand["post_rerank_score"] = r_score
                # Ngưỡng hậu kiểm tương đối: chỉ loại bỏ các node có điểm quá thấp (< 0.00005)
                if r_score >= 0.00005:
                    valid_candidates.append(cand)
            valid_candidates.sort(key=lambda x: (x.get("post_rerank_score", 0), x.get("graph_score", 0)), reverse=True)
            ranked_candidates = valid_candidates

        expanded = []
        for cand in ranked_candidates:
            if cand["chunk_id"] not in seen:
                expanded.append(cand)
                seen.add(cand["chunk_id"])
                if len(expanded) >= max_expanded:
                    break
        return expanded

    # Fallback in-memory
    expanded = []
    for c in top_chunks:
        for ref_id in c.get("references", []):
            if ref_id in seen or ref_id not in all_chunks_by_id:
                continue
            expanded.append(all_chunks_by_id[ref_id])
            seen.add(ref_id)
            if len(expanded) >= max_expanded:
                return expanded

    return expanded


def execute_hybrid_rag_pipeline(
    query_str: str,
    model: SentenceTransformer,
    reranker: CrossEncoder,
    eligible_chunks: list,
    eligible_embeddings: np.ndarray,
    bm25_model: BM25Okapi,
    faiss_index,
    all_chunks_by_id: dict,
    graph_retriever=None,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    max_expanded: int = DEFAULT_MAX_EXPANDED,
    gemini_client=None
):
    """
    Quy trình Hybrid Retrieval + GraphRAG + Query Decomposition tích hợp:
    1. Kiểm tra câu hỏi:
       - Nếu đơn giản: Chạy 1 lượt Hybrid + Graph + Rerank.
       - Nếu phức tạp: Phân rã thành 2-4 sub-queries -> Retrieval đa luồng -> Merge -> Global Rerank -> Graph.
    """
    is_complex = is_complex_query(query_str)

    # TRỤ CỘT 4: Mở rộng truy vấn quy nạp từ vựng đời thường sang pháp lý
    search_query = enrich_legal_query(query_str)

    if not is_complex:
        query_vec = model.encode(search_query, normalize_embeddings=True)
        candidate_results = hybrid_rrf_search(
            search_query, query_vec, eligible_chunks, eligible_embeddings,
            bm25_model=bm25_model, faiss_index=faiss_index, k=60, top_k=candidate_top_k
        )
        # Rerank bằng câu hỏi đã được enrich để Cross-Encoder nhận diện được cả ngữ cảnh lẫn thuật ngữ luật
        final_top = rerank_candidates(search_query, candidate_results, reranker, top_k=rerank_top_k)
        top_chunks = [c for c, r_score, rrf_score, b_score, d_score in final_top]
        expanded = reference_expansion(top_chunks, all_chunks_by_id, graph_retriever=graph_retriever, max_expanded=max_expanded)
        return top_chunks, expanded

    print("\n🔍 [Query Decomposition] Phát hiện câu hỏi phức tạp! Đang phân rã câu hỏi...")
    sub_queries = decompose_query(query_str, client=gemini_client)
    print(f"-> Đã tách thành {len(sub_queries)} sub-queries:")
    for idx, sq in enumerate(sub_queries, 1):
        print(f"   [{idx}] {sq}")

    all_sub_candidates = {}
    all_queries_to_search = [search_query] + [enrich_legal_query(sq) for sq in sub_queries]

    for q_item in all_queries_to_search:
        q_vec = model.encode(q_item, normalize_embeddings=True)

        # 1. Hybrid Search cho subquery
        candidates = hybrid_rrf_search(
            q_item, q_vec, eligible_chunks, eligible_embeddings,
            bm25_model=bm25_model, faiss_index=faiss_index, k=60, top_k=20
        )
        cand_chunks = [c for c, rrf, b, d in candidates]

        # 2. Neo4j Graph Expansion cho subquery
        sub_expanded = reference_expansion(
            cand_chunks[:4], all_chunks_by_id,
            graph_retriever=graph_retriever,
            reranker_model=reranker,
            max_expanded=2,
            query_text=query_str,
            min_score_threshold=0.85
        )

        # Gộp candidates + graph expanded
        combined_cand = list(candidates)
        seen_cand_ids = {c["chunk_id"] for c, rrf, b, d in candidates}
        for exp_c in sub_expanded:
            if exp_c["chunk_id"] not in seen_cand_ids:
                combined_cand.append((exp_c, 0.0, 0.0, 0.0))
                seen_cand_ids.add(exp_c["chunk_id"])

        # 3. Rerank cục bộ theo từng subquery
        sub_reranked = rerank_candidates(q_item, combined_cand, reranker, top_k=8)
        for c, r_score, rrf_score, b_score, d_score in sub_reranked:
            cid = c["chunk_id"]
            if cid not in all_sub_candidates or r_score > all_sub_candidates[cid][1]:
                all_sub_candidates[cid] = (c, r_score, rrf_score, b_score, d_score)

    merged_candidates = list(all_sub_candidates.values())
    print(f"\n📦 [Merge] Tổng hợp được {len(merged_candidates)} chunk ứng viên độc nhất từ các sub-queries.")

    # 4. Rerank toàn cục lại toàn bộ ứng viên đối chiếu với CÂU HỎI ĐÃ ENRICH
    print(f"🎯 [Global Reranking] Xếp hạng lại toàn bộ ứng viên theo câu hỏi đã enrich -> Chọn Top {rerank_top_k}...")
    pairs = [(search_query, item[0]["content"][:800]) for item in merged_candidates]
    rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)

    final_ranked = []
    for idx, item in enumerate(merged_candidates):
        c = item[0]
        r_score = float(rerank_scores[idx])
        final_ranked.append((c, r_score))

    final_ranked.sort(key=lambda x: x[1], reverse=True)
    top_chunks = []
    for c, score in final_ranked[:rerank_top_k]:
        c_dict = dict(c)
        c_dict["rerank_score"] = float(score)
        top_chunks.append(c_dict)

    # 5. Mở rộng dẫn chiếu đồ thị lần cuối
    expanded = reference_expansion(top_chunks, all_chunks_by_id, graph_retriever=graph_retriever, max_expanded=max_expanded, query_text=search_query)

    return top_chunks, expanded


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

    # Nếu là đường dẫn thư mục cục bộ (như finetune/fine_tuned_model)
    local_model_path = (Path(__file__).resolve().parent.parent / model_name).resolve()
    if local_model_path.exists():
        model_name = str(local_model_path)

    print(f"Đang khởi tạo Bi-Encoder embedding model '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"⚠️ Không thể tải model '{model_name}': {e}. Dùng fallback.")
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Đang khởi tạo Cross-Encoder Reranker model '{RERANKER_MODEL_NAME}'...")
    try:
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception as e:
        print(f"Lỗi tải reranker {RERANKER_MODEL_NAME}: {e}")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Pre-filter các văn bản còn hiệu lực và tạo sẵn BM25 & FAISS index một lần duy nhất
    print("Đang chuẩn bị chỉ mục BM25 và FAISS cho các văn bản...")
    eligible_chunks, eligible_embeddings = effective_law_filter(chunks, embeddings_matrix)
    corpus_tokens = [simple_tokenize(c["content"]) for c in eligible_chunks]
    bm25_model = BM25Okapi(corpus_tokens)

    faiss_index = None
    if FAISS_AVAILABLE and eligible_embeddings is not None:
        faiss_index = faiss.IndexFlatIP(eligible_embeddings.shape[1])
        faiss_index.add(eligible_embeddings.astype("float32"))
        print(f"Đã kích hoạt FAISS IndexFlatIP ({faiss_index.ntotal:,} vectors).")

    graph_retriever = None
    try:
        from src.retrieve_graph import LegalGraphRetriever
        graph_retriever = LegalGraphRetriever()
    except Exception as e:
        pass

    def rag_retrieve(query_str: str):
        return execute_hybrid_rag_pipeline(
            query_str=query_str,
            model=model,
            reranker=reranker,
            eligible_chunks=eligible_chunks,
            eligible_embeddings=eligible_embeddings,
            bm25_model=bm25_model,
            faiss_index=faiss_index,
            all_chunks_by_id=all_chunks_by_id,
            graph_retriever=graph_retriever,
            candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
            rerank_top_k=DEFAULT_RERANK_TOP_K,
            max_expanded=DEFAULT_MAX_EXPANDED
        )

    cli_session_id = "cli_interactive_session"
    from src.conversational import GLOBAL_SESSION_MANAGER
    GLOBAL_SESSION_MANAGER.clear_session(cli_session_id)

    def execute_query(user_input: str):
        print("\n" + "=" * 95)
        print(f"CÂU HỎI: {user_input}")

        answer_user_query(user_input, rag_retriever_func=rag_retrieve, stream=True, session_id=cli_session_id)


    if sample_query:
        execute_query(sample_query)
        if not interactive:
            return

    print("\n" + "-" * 95)
    print("💡 HỆ THỐNG SẴN SÀNG! Bạn có thể:")
    print("   1. Nhập câu hỏi Pháp luật hoặc Nhân sự nội bộ VNTech (hỗ trợ hỏi nối tiếp đa lượt).")
    print("   2. Dán nội dung hợp đồng để Thẩm định Rủi ro tự động.")
    print("   3. Nhập đường dẫn file ảnh/text hợp đồng (ví dụ: data/hop_dong.png hoặc hop_dong.txt).")
    print("   (Gõ 'clear' hoặc 'new' để bắt đầu hội thoại mới, 'q' hoặc 'exit' để thoát)")
    print("-" * 95)

    while True:
        try:
            user_input = input("\nNHẬP CÂU HỎI HOẶC ĐƯỜNG DẪN HỢP ĐỒNG > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát chương trình.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "exit", "quit"):
            print("Đã thoát chương trình.")
            break
        if user_input.lower() in ("clear", "new", "reset"):
            GLOBAL_SESSION_MANAGER.clear_session(cli_session_id)
            print("\n🧹 Đã xóa lịch sử và bắt đầu phiên hội thoại mới!")
            continue

        execute_query(user_input)


