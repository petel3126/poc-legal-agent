"""
Script đánh giá độc lập mô hình Embedding sau fine-tune trên tập test.json.
So sánh hiệu năng giữa Mô hình Gốc (Pretrained) và Mô hình Đã Fine-tune.
"""

import json
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.evaluation import InformationRetrievalEvaluator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FINETUNE_DIR = Path(__file__).resolve().parent
TEST_PATH = FINETUNE_DIR / "test.json"
BASE_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
FINETUNED_MODEL_PATH = FINETUNE_DIR / "fine_tuned_model"


def load_evaluator(test_json_path: Path, name: str):
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    queries = {item["id"]: item["query"] for item in test_data}
    corpus = {}
    relevant_docs = {}

    for item in test_data:
        pos_id = item["positive_id"]
        corpus[pos_id] = item["positive"]
        relevant_docs[item["id"]] = {pos_id}
        for neg_id, neg_text in zip(item.get("hard_negative_ids", []), item.get("hard_negatives", [])):
            corpus[neg_id] = neg_text

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True
    )
    return evaluator, len(test_data)


def main():
    print("=" * 85)
    print("          ĐÁNH GIÁ SO SÁNH MÔ HÌNH EMBEDDING TRÊN TẬP TEST (TEST.JSON)          ")
    print("=" * 85)

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file {TEST_PATH}. Vui lòng chạy `generate_dataset.py` trước!")

    evaluator, n_samples = load_evaluator(TEST_PATH, "test_eval")
    print(f"Tổng số mẫu test: {n_samples}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị tính toán: {device.upper()}\n")

    # 1. Đánh giá Mô hình Gốc
    print(f"[1/2] Đánh giá Mô hình Gốc ('{BASE_MODEL_NAME}')...")
    base_model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    base_results = evaluator(base_model)

    # 2. Đánh giá Mô hình Fine-tuned
    model_to_test = FINETUNED_MODEL_PATH if FINETUNED_MODEL_PATH.exists() else BASE_MODEL_NAME
    print(f"\n[2/2] Đánh giá Mô hình Fine-tuned ('{model_to_test}')...")
    finetuned_model = SentenceTransformer(str(model_to_test), device=device)
    ft_results = evaluator(finetuned_model)

    print("\n" + "=" * 85)
    print("                    BẢNG KẾT QUẢ SO SÁNH TRÊN TẬP TEST                      ")
    print("=" * 85)
    print(f"{'Metric':<38} | {'Mô hình Gốc':<14} | {'Fine-tuned':<14} | {'Mức Tăng':<10}")
    print("-" * 85)

    # In các chỉ số quan trọng
    target_metrics = [
        "test_eval_cosine_accuracy@1",
        "test_eval_cosine_accuracy@3",
        "test_eval_cosine_accuracy@5",
        "test_eval_cosine_accuracy@10",
        "test_eval_cosine_mrr@10",
        "test_eval_cosine_ndcg@10",
        "test_eval_cosine_map@100"
    ]

    for m in target_metrics:
        if m in ft_results:
            b_val = base_results[m]
            f_val = ft_results[m]
            diff = f_val - b_val
            short_name = m.replace("test_eval_cosine_", "")
            print(f"{short_name:<38} | {b_val:<14.4f} | {f_val:<14.4f} | {diff:+10.4f}")

    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
