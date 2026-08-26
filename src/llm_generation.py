"""
LLM Generation Module sử dụng Gemini API (google-genai SDK).
Đọc các module prompt chuyên biệt từ src/prompts/ và sinh câu trả lời cho:
1. Legal RAG (Căn cứ Pháp lý)
2. HRM Assistant (Thông tin Nhân sự & Doanh nghiệp)
3. Hybrid QA (Kết hợp Nhân sự và Pháp luật)
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Đảm bảo UTF-8 stdout trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()


def get_gemini_client() -> Optional[genai.Client]:
    """Khởi tạo và trả về Gemini Client."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and ENV_PATH.exists():
        content = ENV_PATH.read_text(encoding="utf-8").strip()
        for line in content.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    api_key = v.strip().strip("'").strip('"')
                    break

    if not api_key:
        print("⚠️ CẢNH BÁO: Chưa cấu hình GEMINI_API_KEY trong file .env!")
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Gemini Client: {e}")
        return None


def read_prompt(filename: str, default_text: str = "") -> str:
    """Đọc nội dung một file prompt từ thư mục src/prompts/."""
    file_path = PROMPTS_DIR / filename
    if file_path.exists():
        return file_path.read_text(encoding="utf-8").strip()
    return default_text


def format_legal_context(top_chunks: list, expanded_chunks: list = None) -> str:
    """Format các đoạn trích dẫn văn bản luật."""
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


def call_gemini(
    user_prompt: str,
    system_instruction: Optional[str] = None,
    preferred_model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    stream: bool = True,
    stream_callback=None
) -> str:
    """Gọi Gemini API với cơ chế dự phòng nhiều model, hỗ trợ Streaming và tự động retry khi gặp tải cao."""
    client = get_gemini_client()
    if not client:
        err_msg = "Không thể sinh câu trả lời do chưa cấu hình GEMINI_API_KEY trong file .env."
        if stream and not stream_callback:
            print(err_msg)
        return err_msg

    candidate_models = [preferred_model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-pro"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction if system_instruction else None,
    )

    last_error = None
    for current_model in models_to_try:
        for attempt in range(2):
            try:
                if stream:
                    response_stream = client.models.generate_content_stream(
                        model=current_model,
                        contents=user_prompt,
                        config=config
                    )
                    full_text = []
                    for chunk in response_stream:
                        if chunk.text:
                            full_text.append(chunk.text)
                            if stream_callback:
                                stream_callback(chunk.text)
                            else:
                                print(chunk.text, end="", flush=True)
                    return "".join(full_text).strip()
                else:
                    response = client.models.generate_content(
                        model=current_model,
                        contents=user_prompt,
                        config=config
                    )
                    return response.text.strip()
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(1.5)
                    continue
                else:
                    break

    err_msg = f"Rất tiếc, đã xảy ra lỗi khi tạo câu trả lời tự động: {last_error}"
    if stream and not stream_callback:
        print(err_msg)
    return err_msg


def generate_legal_answer(
    query: str,
    top_chunks: list,
    expanded_chunks: list = None,
    model_name: str = "gemini-2.5-flash",
    stream: bool = True,
    stream_callback=None
) -> str:
    """Sinh câu trả lời Pháp luật (Legal RAG) sử dụng Prompt chuyên biệt."""
    sys_inst = read_prompt("system_instruction.txt") + "\n\n" + read_prompt("citation_rules.txt")
    template = read_prompt("task_legal_template.txt", "CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:")

    context_str = format_legal_context(top_chunks, expanded_chunks)
    user_prompt = template.format(context=context_str, query=query)

    print(f"\n[GEMINI LLM] Đang tổng hợp câu trả lời dựa trên Căn cứ Pháp lý...\n")
    return call_gemini(
        user_prompt=user_prompt,
        system_instruction=sys_inst,
        preferred_model=model_name,
        stream=stream,
        stream_callback=stream_callback
    )


def generate_explanation_answer(
    query: str,
    previous_query: str,
    previous_answer: str,
    top_chunks: list,
    expanded_chunks: list = None,
    model_name: str = "gemini-2.5-flash",
    stream: bool = True,
    stream_callback=None
) -> str:
    """Sinh câu trả lời giải thích sâu / làm rõ căn cứ dựa trên ngữ cảnh và chunks đã trích xuất từ lượt trước."""
    sys_inst = read_prompt("system_instruction.txt") + "\n\n" + read_prompt("citation_rules.txt")
    context_str = format_legal_context(top_chunks, expanded_chunks)

    prompt = f"""CĂN CỨ PHÁP LÝ ĐÃ SỬ DỤNG:
{context_str}

LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
- Câu hỏi của người dùng: {previous_query}
- Kết luận của bạn ở lượt trước: {previous_answer}

YÊU CẦU GIẢI THÍCH / LÀM RÕ CỦA NGƯỜI DÙNG:
"{query}"

NHIỆM VỤ CỦA BẠN:
1. Giải thích cặn kẽ, chi tiết lý do và phân tích logic pháp lý vì sao lại đưa ra kết luận như ở lượt trước.
2. Trích dẫn chính xác và phân tích rõ từng Điều, Khoản cụ thể trong phần CĂN CỨ PHÁP LÝ ở trên đã dẫn tới kết luận đó.
3. Giữ giọng văn khách quan, chuẩn mực, giải thích dễ hiểu cho người dùng."""

    print(f"\n[GEMINI LLM] Đang giải thích chuyên sâu dựa trên Căn cứ pháp lý đã trích xuất...\n")
    return call_gemini(
        user_prompt=prompt,
        system_instruction=sys_inst,
        preferred_model=model_name,
        stream=stream,
        stream_callback=stream_callback
    )


def generate_hrm_answer(
    query: str,
    db_context: str,
    model_name: str = "gemini-2.5-flash",
    stream: bool = True,
    stream_callback=None
) -> str:
    """Sinh câu trả lời Nhân sự & Doanh nghiệp từ dữ liệu Supabase Database."""
    sys_inst = read_prompt("hrm_system_instruction.txt")
    template = read_prompt("task_hrm_template.txt", "DATABASE:\n{db_context}\n\nQUESTION:\n{query}\n\nANSWER:")

    user_prompt = template.format(db_context=db_context, query=query)

    print(f"\n[GEMINI LLM] Đang tổng hợp thông tin nhân sự từ Cơ sở dữ liệu...\n")
    return call_gemini(
        user_prompt=user_prompt,
        system_instruction=sys_inst,
        preferred_model=model_name,
        temperature=0.0,
        stream=stream,
        stream_callback=stream_callback
    )


def generate_hybrid_answer(
    query: str,
    db_context: str,
    top_chunks: list,
    expanded_chunks: list = None,
    model_name: str = "gemini-2.5-flash",
    stream: bool = True,
    stream_callback=None
) -> str:
    """Sinh câu trả lời kết hợp cả Dữ liệu Nhân sự và Điều khoản Pháp luật."""
    sys_inst = read_prompt("system_instruction.txt") + "\n\n" + read_prompt("hrm_system_instruction.txt") + "\n\n" + read_prompt("citation_rules.txt")
    template = read_prompt("task_hybrid_template.txt")

    legal_context_str = format_legal_context(top_chunks, expanded_chunks)
    user_prompt = template.format(db_context=db_context, legal_context=legal_context_str, query=query)

    print(f"\n[GEMINI LLM] Đang tổng hợp câu trả lời Kết hợp (Nhân sự + Pháp luật)...\n")
    return call_gemini(
        user_prompt=user_prompt,
        system_instruction=sys_inst,
        preferred_model=model_name,
        stream=stream,
        stream_callback=stream_callback
    )


def generate_contract_risk_analysis(
    contract_text: str,
    top_chunks: list,
    expanded_chunks: list = None,
    model_name: str = "gemini-2.5-flash",
    stream: bool = True,
    stream_callback=None
) -> str:
    """Sinh báo cáo Thẩm định & Đánh giá Rủi ro Hợp đồng (Contract Risk Assessment)."""
    sys_inst = read_prompt("contract_risk_system_instruction.txt") + "\n\n" + read_prompt("citation_rules.txt")
    template = read_prompt("task_contract_risk_template.txt")

    legal_context_str = format_legal_context(top_chunks, expanded_chunks)
    user_prompt = template.format(contract_text=contract_text, legal_context=legal_context_str)

    print(f"\n[GEMINI LLM] Đang thẩm định rủi ro hợp đồng và đối chiếu căn cứ pháp lý...\n")
    return call_gemini(
        user_prompt=user_prompt,
        system_instruction=sys_inst,
        preferred_model=model_name,
        temperature=0.1,
        stream=stream,
        stream_callback=stream_callback
    )

