"""
Script huấn luyện fine-tune mô hình Vietnamese Embedding (Bi-Encoder)
sử dụng SentenceTransformers và MultipleNegativesRankingLoss.

Input datasets:
  - finetune/train.json (500 samples)
  - finetune/validation.json (50 samples)
  - finetune/test.json (50 samples)
"""

import json
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FINETUNE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
OUTPUT_MODEL_DIR = FINETUNE_DIR / "fine_tuned_model"


def load_triplet_examples(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        query = item["query"]
        positive = item["positive"]
        hard_negs = item.get("hard_negatives", [])
        # InputExample dạng [query, positive, neg1, neg2] hỗ trợ MultipleNegativesRankingLoss
        texts = [query, positive] + hard_negs
        examples.append(InputExample(texts=texts))
    return examples, data


def main():
    print("=" * 80)
    print("      BẮT ĐẦU QUY TRÌNH FINE-TUNE EMBEDDING MODEL (BI-ENCODER)      ")
    print("=" * 80)

    train_path = FINETUNE_DIR / "train.json"
    val_path = FINETUNE_DIR / "validation.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Chưa thấy bộ dataset fine-tune. Vui lòng chạy `python finetune/generate_dataset.py` trước!")

    print(f"\n1. Đang tải bộ dữ liệu fine-tune từ {FINETUNE_DIR}...")
    train_examples, _ = load_triplet_examples(train_path)
    val_examples, val_data = load_triplet_examples(val_path)
    print(f"   - Số mẫu huấn luyện (Train): {len(train_examples)}")
    print(f"   - Số mẫu kiểm định (Validation): {len(val_examples)}")

    print(f"\n2. Khởi tạo mô hình gốc '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - Sử dụng thiết bị tính toán: {device.upper()}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Đưa các mẫu train vào DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    # Sử dụng MultipleNegativesRankingLoss (tối ưu cho Triplet/Hard Negatives)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Khởi tạo Evaluator cho tập Validation
    queries = {item["id"]: item["query"] for item in val_data}
    corpus = {}
    relevant_docs = {}

    for item in val_data:
        pos_id = item["positive_id"]
        corpus[pos_id] = item["positive"]
        relevant_docs[item["id"]] = {pos_id}
        for neg_id, neg_text in zip(item["hard_negative_ids"], item["hard_negatives"]):
            corpus[neg_id] = neg_text

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val_eval",
        show_progress_bar=True
    )

    epochs = 3
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    print(f"\n3. Tiến hành huấn luyện fine-tune ({epochs} epochs)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=20,
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_MODEL_DIR),
        save_best_model=True
    )

    print("\n" + "=" * 80)
    print(f"HOÀN THÀNH FINE-TUNE! Mô hình đã được lưu tại: {OUTPUT_MODEL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
