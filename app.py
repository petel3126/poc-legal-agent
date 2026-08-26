"""
FastAPI Server — Backend cho Hệ thống Legal AI Chatbot & Thẩm định Hợp đồng.
Cung cấp REST API, Server-Sent Events (SSE) Streaming và tiếp nhận Upload file.
"""

import sys
import os
import json
import asyncio
import queue
import threading
from pathlib import Path
from typing import Optional
import numpy as np

# Đảm bảo UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.router import classify_query, answer_user_query
from src.contract_analyzer import analyze_contract
from src.retrieve_hybrid import (
    simple_tokenize,
    effective_law_filter,
    hybrid_rrf_search,
    rerank_candidates,
    reference_expansion,
    execute_hybrid_rag_pipeline,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    META_PATH,
    MODEL_NAME,
    RERANKER_MODEL_NAME,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_MAX_EXPANDED,
    FAISS_AVAILABLE
)

if FAISS_AVAILABLE:
    import faiss

# Thư mục gốc & Thư mục static/uploads
ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Bộ nhớ RAG toàn cục
RAG_STATE = {
    "model": None,
    "reranker": None,
    "eligible_chunks": None,
    "eligible_embeddings": None,
    "bm25_model": None,
    "faiss_index": None,
    "all_chunks_by_id": None,
    "graph_retriever": None,
    "ready": False
}


def load_rag_pipeline():
    """Tải và khởi tạo sẵn toàn bộ mô hình và chỉ mục RAG."""
    print("🚀 [FASTAPI LIFESPAN] Đang nạp hệ thống RAG & Mô hình AI...")

    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        print("⚠️ [CẢNH BÁO] Chưa tìm thấy file chunks hoặc embeddings! Vui lòng kiểm tra thư mục data/processed.")
        return

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings_matrix = np.load(EMBEDDINGS_PATH)

    model_name = MODEL_NAME
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        model_name = meta.get("model_name", MODEL_NAME)

    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    try:
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    eligible_chunks, eligible_embeddings = effective_law_filter(chunks, embeddings_matrix)
    corpus_tokens = [simple_tokenize(c["content"]) for c in eligible_chunks]
    bm25_model = BM25Okapi(corpus_tokens)

    faiss_index = None
    if FAISS_AVAILABLE and eligible_embeddings is not None:
        faiss_index = faiss.IndexFlatIP(eligible_embeddings.shape[1])
        faiss_index.add(eligible_embeddings.astype("float32"))

    graph_retriever = None
    try:
        from src.retrieve_graph import LegalGraphRetriever
        graph_retriever = LegalGraphRetriever()
    except Exception:
        pass

    RAG_STATE["model"] = model
    RAG_STATE["reranker"] = reranker
    RAG_STATE["eligible_chunks"] = eligible_chunks
    RAG_STATE["eligible_embeddings"] = eligible_embeddings
    RAG_STATE["bm25_model"] = bm25_model
    RAG_STATE["faiss_index"] = faiss_index
    RAG_STATE["all_chunks_by_id"] = {c["chunk_id"]: c for c in chunks}
    RAG_STATE["graph_retriever"] = graph_retriever
    RAG_STATE["ready"] = True
    print("✅ [FASTAPI LIFESPAN] Đã tải hoàn tất RAG Pipeline & Mô hình AI sẵn sàng phục vụ!")


def get_rag_retriever():
    """Hàm truy vấn RAG dùng chung cho API."""
    def rag_retrieve(query_str: str):
        if not RAG_STATE["ready"]:
            return [], []

        model = RAG_STATE["model"]
        reranker = RAG_STATE["reranker"]
        eligible_chunks = RAG_STATE["eligible_chunks"]
        eligible_embeddings = RAG_STATE["eligible_embeddings"]
        bm25_model = RAG_STATE["bm25_model"]
        faiss_index = RAG_STATE["faiss_index"]
        all_chunks_by_id = RAG_STATE["all_chunks_by_id"]
        graph_retriever = RAG_STATE["graph_retriever"]

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

    return rag_retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo background thread để nạp models không chặn event loop
    threading.Thread(target=load_rag_pipeline, daemon=True).start()
    yield


app = FastAPI(
    title="Legal AI & Contract Review Assistant",
    description="Hệ thống Trợ lý Pháp luật, Nhân sự VNTech & Thẩm định Rủi ro Hợp đồng",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_ready": RAG_STATE["ready"],
        "models": {
            "bi_encoder": MODEL_NAME,
            "reranker": RERANKER_MODEL_NAME
        }
    }


@app.post("/api/chat")
async def api_chat_stream(request: ChatRequest):
    """
    API Chatbot hỗ trợ Server-Sent Events (SSE) Streaming tốc độ cao (Zero Delay).
    Tự động phân loại ý định (Pháp luật, Nhân sự, Hợp đồng) và stream câu trả lời tức thì.
    """
    user_message = request.message.strip()
    session_id = request.session_id or "default"
    if not user_message:
        raise HTTPException(status_code=400, detail="Tin nhắn rỗng.")

    intent = classify_query(user_message)
    rag_retriever = get_rag_retriever()

    async def event_generator():
        # 1. Gửi thông tin Intent ngay lập tức
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent}, ensure_ascii=False)}\n\n"

        loop = asyncio.get_running_loop()
        token_queue = asyncio.Queue()

        def stream_callback(token: str):
            loop.call_soon_threadsafe(token_queue.put_nowait, token)

        def worker():
            try:
                answer_user_query(
                    query=user_message,
                    rag_retriever_func=rag_retriever,
                    stream=True,
                    stream_callback=stream_callback,
                    session_id=session_id
                )
            except Exception as e:
                loop.call_soon_threadsafe(token_queue.put_nowait, f"\n[Lỗi xử lý]: {str(e)}")
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        # 2. Đẩy token tức thì ra client ngay khi Gemini sinh ra (Zero polling delay)
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/upload-contract")
async def api_upload_contract(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None)
):
    """
    API tiếp nhận file ảnh (.png, .jpg) hoặc file text (.txt, .md) hợp đồng để thẩm định rủi ro tốc độ cao.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Không có file nào được tải lên.")

    filename = file.filename
    file_path = UPLOADS_DIR / filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    rag_retriever = get_rag_retriever()
    intent = "CONTRACT_RISK"

    async def event_generator():
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent, 'filename': filename}, ensure_ascii=False)}\n\n"

        loop = asyncio.get_running_loop()
        token_queue = asyncio.Queue()

        def stream_callback(token: str):
            loop.call_soon_threadsafe(token_queue.put_nowait, token)

        def worker():
            try:
                analyze_contract(
                    input_source=str(file_path),
                    rag_retriever_func=rag_retriever,
                    stream=True,
                    stream_callback=stream_callback
                )
            except Exception as e:
                loop.call_soon_threadsafe(token_queue.put_nowait, f"\n[Lỗi thẩm định file]: {str(e)}")
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Mount Static Files & Serve Frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Trợ lý Legal AI đang khởi động... Vui lòng refresh lại sau giây lát.</h1>")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 90)
    print("      🌐 KHỞI CHẠY WEBSERVER FASTAPI — LEGAL AI & CONTRACT REVIEW CHATBOT")
    print("      👉 Truy cập giao diện tại: http://127.0.0.1:8000")
    print("=" * 90 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
