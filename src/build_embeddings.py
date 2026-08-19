"""
Tạo Dense Vector Embeddings cho toàn bộ các chunk pháp lý.
Mô hình mặc định: bkai-foundation-models/vietnamese-bi-encoder (hoặc paraphrase-multilingual-MiniLM-L12-v2)
Output:
  - data/processed/blld_45_2019_qh14_embeddings.npy
  - data/processed/blld_45_2019_qh14_embedding_meta.json
"""

import sys
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_chunks.json"
EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_embeddings.npy"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_embedding_meta.json"

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


def run_embedding(chunks_path: Path = CHUNKS_PATH, embeddings_path: Path = EMBEDDINGS_PATH, meta_path: Path = META_PATH, force_reembed: bool = False):
    if embeddings_path.exists() and meta_path.exists() and not force_reembed:
        print(f"[INFO] File vector embeddings đã tồn tại tại {embeddings_path}. Sử dụng dữ liệu đã lưu (Bỏ qua bước sinh embeddings).")
        return np.load(embeddings_path)

    print(f"Loading chunks từ {chunks_path}...")

    if not chunks_path.exists():
        raise FileNotFoundError(f"File chunks không tồn tại tại {chunks_path}. Hãy chạy bước chunking trước!")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Tổng số chunk: {len(chunks)}")

    print(f"\nĐang tải embedding model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Lỗi khi tải '{MODEL_NAME}': {e}")
        fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"Chuyển sang mô hình fallback '{fallback_model}'...")
        model = SentenceTransformer(fallback_model)

    print("\nĐang sinh vector embeddings cho các chunk...")
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Save vector npy
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Đã lưu vector embeddings vào {embeddings_path} (Shape: {embeddings.shape})")

    # Save meta json
    meta = {
        "model_name": MODEL_NAME,
        "num_chunks": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "embedding_dim": int(embeddings.shape[1])
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu metadata vào {meta_path}")
    print("Hoàn tất sinh vector embeddings!\n")
    return embeddings


if __name__ == "__main__":
    run_embedding(force_reembed=True)

