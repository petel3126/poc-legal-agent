"""
Tạo Dense Vector Embeddings và Chỉ mục FAISS cho toàn bộ các chunk pháp lý.
Mô hình mặc định: finetune/fine_tuned_model (hoặc bkai-foundation-models/vietnamese-bi-encoder)

Output:
  - data/processed/legal_embeddings.npy (Mảng vector NumPy)
  - data/processed/legal_index.faiss   (Chỉ mục FAISS IndexFlatIP tối ưu tốc độ tìm kiếm)
  - data/processed/legal_embedding_meta.json
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_chunks.json"
EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_embeddings.npy"
FAISS_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_index.faiss"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_embedding_meta.json"
FINETUNED_MODEL_DIR = Path(__file__).resolve().parent.parent / "finetune" / "fine_tuned_model"
DEFAULT_MODEL_NAME = str(FINETUNED_MODEL_DIR) if FINETUNED_MODEL_DIR.exists() else "bkai-foundation-models/vietnamese-bi-encoder"


def build_faiss_index(embeddings: np.ndarray, index_path: Path = FAISS_INDEX_PATH):
    """Tạo và lưu chỉ mục FAISS IndexFlatIP."""
    if not FAISS_AVAILABLE:
        print("[WARNING] Thư viện FAISS chưa được cài đặt. Bỏ qua bước tạo legal_index.faiss.")
        return None

    dim = embeddings.shape[1]
    print(f"\nĐang xây dựng chỉ mục FAISS (IndexFlatIP, Dimension={dim}, Vectors={len(embeddings):,})...")
    
    vectors = embeddings.astype("float32")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Đã lưu chỉ mục FAISS thành công tại: {index_path} (Total indexed: {index.ntotal:,} vectors)")
    return index


def run_embedding(
    chunks_path: Path = CHUNKS_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    meta_path: Path = META_PATH,
    force_reembed: bool = False,
    model_name: str = None,
    batch_size: int = 64
):
    chosen_model = model_name or DEFAULT_MODEL_NAME

    if embeddings_path.exists() and meta_path.exists() and not force_reembed:
        print(f"[INFO] File vector embeddings đã tồn tại tại {embeddings_path}.")
        print("[INFO] Để ép buộc tạo lại vector mới bằng mô hình fine-tune, hãy chạy: `python src/build_embeddings.py --force`")
        embeddings = np.load(embeddings_path)
        
        # Nếu đã có file npy nhưng chưa có file faiss, tự động build faiss
        if not faiss_index_path.exists() and FAISS_AVAILABLE:
            build_faiss_index(embeddings, faiss_index_path)
            
        return embeddings

    print(f"Loading chunks từ {chunks_path}...")

    if not chunks_path.exists():
        raise FileNotFoundError(f"File chunks không tồn tại tại {chunks_path}. Hãy chạy bước chunking trước!")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Tổng số chunk: {len(chunks):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nĐang tải embedding model '{chosen_model}' trên thiết bị: {device.upper()}...")
    try:
        model = SentenceTransformer(chosen_model, device=device)
    except Exception as e:
        print(f"Lỗi khi tải '{chosen_model}': {e}")
        fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"Chuyển sang mô hình fallback '{fallback_model}'...")
        model = SentenceTransformer(fallback_model, device=device)

    print(f"\nĐang sinh vector embeddings cho {len(chunks):,} chunk (Batch size: {batch_size})...")
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    # Save vector npy
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"\nĐã lưu vector embeddings vào {embeddings_path} (Shape: {embeddings.shape})")

    # Save FAISS index
    if FAISS_AVAILABLE:
        build_faiss_index(embeddings, faiss_index_path)

    # Save meta json
    meta = {
        "model_name": chosen_model,
        "num_chunks": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "embedding_dim": int(embeddings.shape[1]),
        "has_faiss_index": FAISS_AVAILABLE
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu metadata vào {meta_path}")
    print("Hoàn tất sinh vector embeddings và chỉ mục FAISS!\n")
    return embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Tạo Vector Embeddings và Chỉ mục FAISS cho Legal Chunks")
    parser.add_argument("--force", action="store_true", help="Ép buộc tính toán lại và ghi đè vector embeddings")
    parser.add_argument("--batch_size", type=int, default=64 if torch.cuda.is_available() else 32, help="Batch size khi sinh vector")
    parser.add_argument("--model_name", type=str, default=None, help="Tên hoặc đường dẫn mô hình embedding tùy chỉnh")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_embedding(force_reembed=args.force, batch_size=args.batch_size, model_name=args.model_name)
