"""
LLM Answer Generation Module sử dụng Gemini API (google-genai SDK).
Đọc prompt mẫu từ src/promt.txt và sinh câu trả lời dựa trên Căn cứ Pháp lý (Grounding Context).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Đảm bảo UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "promt.txt"
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()  # Load từ working directory nếu có


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Nếu chưa tìm thấy trong os.getenv, thử đọc trực tiếp file .env
        if ENV_PATH.exists():
            content = ENV_PATH.read_text(encoding="utf-8").strip()
            if content and not content.startswith("#"):
                for line in content.split("\n"):
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                            api_key = v.strip()
                            break
                    elif len(line.strip()) > 20:
                        api_key = line.strip()
                        break

    if not api_key:
        print("CẢNH BÁO: Không tìm thấy GEMINI_API_KEY trong file .env hoặc biến môi trường!")
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Lỗi khởi tạo Gemini Client: {e}")
        return None


def format_context(top_chunks: list, expanded_chunks: list = None) -> str:
    context_blocks = []
    
    for idx, c in enumerate(top_chunks, 1):
        label = f"Điều {c['article']}" + (f" khoản {c['clause']}" if c.get("clause") else "")
        doc_info = f"{c.get('document_type', '')} {c.get('title', '')} ({c.get('document_number', '')})"
        context_blocks.append(f"[{idx}] {label} — {c.get('article_title', '')}\nNguồn: {doc_info}\nNội dung:\n{c.get('content', '')}")

    if expanded_chunks:
        start_idx = len(top_chunks) + 1
        for idx, c in enumerate(expanded_chunks, start_idx):
            label = f"Điều {c['article']}" + (f" khoản {c['clause']}" if c.get("clause") else "")
            context_blocks.append(f"[{idx}] (Đoạn dẫn chiếu) {label} — {c.get('article_title', '')}\nNội dung:\n{c.get('content', '')}")

    return "\n\n".join(context_blocks)


def generate_legal_answer(query: str, top_chunks: list, expanded_chunks: list = None, prompt_path: Path = PROMPT_PATH, model_name: str = "gemini-2.5-flash") -> str:
    client = get_gemini_client()
    if not client:
        return "Không thể sinh câu trả lời LLM do chưa cấu hình GEMINI_API_KEY."

    if not prompt_path.exists():
        print(f"Lỗi: Không tìm thấy file prompt mẫu tại {prompt_path}")
        prompt_template = "CĂN CỨ PHÁP LÝ:\n{context}\n\nCÂU HỎI:\n{query}\n\nCÂU TRẢ LỜI:"
    else:
        prompt_template = prompt_path.read_text(encoding="utf-8")

    context_str = format_context(top_chunks, expanded_chunks)
    full_prompt = prompt_template.format(context=context_str, query=query)

    print(f"\n[GEMINI LLM] Đang tổng hợp câu trả lời dựa trên Căn cứ Pháp lý bằng mô hình '{model_name}'...")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API ({model_name}): {e}")
        # Fallback thử gemini-2.0-flash nếu model_name thất bại
        fallback_model = "gemini-2.0-flash"
        if model_name != fallback_model:
            print(f"Thử lại với mô hình fallback '{fallback_model}'...")
            try:
                response = client.models.generate_content(
                    model=fallback_model,
                    contents=full_prompt
                )
                return response.text
            except Exception as e2:
                print(f"Lỗi khi gọi fallback Gemini API: {e2}")

        return f"Rất tiếc, xảy ra lỗi khi tạo câu trả lời tự động: {e}"
