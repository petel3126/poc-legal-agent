"""
Script huấn luyện fine-tune mô hình Vietnamese Embedding (Bi-Encoder)
cho Hệ thống Multi-Law Legal RAG (Bao phủ 9 Bộ luật & Luật Việt Nam).

Mô hình nền: bkai-foundation-models/vietnamese-bi-encoder
Hàm mất mát: MultipleNegativesRankingLoss (Hỗ trợ In-batch Negatives + Cross-Law Hard Negatives)
Input datasets:
  - finetune/train.json (~2,150 samples)
  - finetune/validation.json (110 samples)
  - finetune/test.json (110 samples)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
FINETUNE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
DEFAULT_OUTPUT_DIR = FINETUNE_DIR / "fine_tuned_model"


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


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Fine-tune Vietnamese Bi-Encoder cho Legal RAG")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Tên hoặc đường dẫn mô hình nền HuggingFace")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu mô hình sau khi fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện (mặc định: 3)")
    parser.add_argument("--batch_size", type=int, default=16 if torch.cuda.is_available() else 8, help="Batch size (mặc định: 16 cho GPU, 8 cho CPU)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (mặc định: 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (mặc định: 0.01)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Tỷ lệ warmup steps (mặc định: 0.1)")
    parser.add_argument("--eval_steps", type=int, default=50, help="Số bước đánh giá và checkpoint (mặc định: 50)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("      BẮT ĐẦU QUY TRÌNH FINE-TUNE EMBEDDING MODEL (MULTI-LAW LEGAL RAG)      ")
    print("=" * 85)

    train_path = FINETUNE_DIR / "train.json"
    val_path = FINETUNE_DIR / "validation.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Chưa thấy bộ dataset fine-tune tại {FINETUNE_DIR}. Vui lòng chạy `python finetune/generate_dataset.py` trước!"
        )

    print(f"\n1. Đang tải bộ dữ liệu fine-tune từ {FINETUNE_DIR}...")
    train_examples, train_data = load_triplet_examples(train_path)
    val_examples, val_data = load_triplet_examples(val_path)

    # Thống kê luật bao phủ
    train_docs = set(item.get("document_id", "") for item in train_data)
    val_docs = set(item.get("document_id", "") for item in val_data)

    print(f"   - Số mẫu huấn luyện (Train): {len(train_examples):,} mẫu (Bao phủ {len(train_docs)} văn bản luật)")
    print(f"   - Số mẫu kiểm định (Validation): {len(val_examples):,} mẫu (Bao phủ {len(val_docs)} văn bản luật)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()

    print(f"\n2. Khởi tạo mô hình gốc '{args.model_name}'...")
    print(f"   - Thiết bị tính toán: {device.upper()}")
    print(f"   - Tăng tốc Mixed Precision (FP16 / AMP): {'BẬT' if use_amp else 'TẮT'}")
    print(f"   - Siêu tham số: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Weight Decay={args.weight_decay}")

    model = SentenceTransformer(args.model_name, device=device)

    # Đưa các mẫu train vào DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Sử dụng MultipleNegativesRankingLoss (tối ưu cho Triplet/Hard Negatives & In-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Khởi tạo Evaluator cho tập Validation
    queries = {item["id"]: item["query"] for item in val_data}
    corpus = {}
    relevant_docs = {}

    for item in val_data:
        pos_id = item["positive_id"]
        corpus[pos_id] = item["positive"]
        relevant_docs[item["id"]] = {pos_id}
        for neg_id, neg_text in zip(item.get("hard_negative_ids", []), item.get("hard_negatives", [])):
            corpus[neg_id] = neg_text

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val_eval",
        show_progress_bar=True
    )

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"   - Tổng số training steps: {total_steps} (Warmup steps: {warmup_steps})")

    print(f"\n3. Tiến hành huấn luyện fine-tune ({args.epochs} epochs)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=args.eval_steps,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        weight_decay=args.weight_decay,
        output_path=str(output_dir),
        save_best_model=True,
        use_amp=use_amp,
        show_progress_bar=True
    )

    print("\n" + "=" * 85)
    print(f"HOÀN THÀNH FINE-TUNE! Mô hình tốt nhất đã được lưu tại: {output_dir}")
    print("=" * 85)
    print("\nBước tiếp theo: Hãy chạy đánh giá so sánh mô hình trên tập test độc lập:")
    print("   python finetune/test_embedding.py")


if __name__ == "__main__":
    main()
