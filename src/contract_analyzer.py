"""
Contract Analyzer Module — Thẩm định và Đánh giá Rủi ro Hợp đồng (Legal Contract Risk Assessment).
Kết hợp tách điều khoản hợp đồng, đối chiếu cơ sở dữ liệu Luật qua Legal RAG và sinh báo cáo thẩm định qua Gemini LLM.
"""

import sys
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.llm_generation import generate_contract_risk_analysis, get_gemini_client
from PIL import Image

# Các chủ đề pháp lý rủi ro cao thường gặp trong hợp đồng tại Việt Nam
COMMON_RISK_TOPICS = [
    # Nhóm Hợp đồng Lao động (BLLĐ 2019)
    ("thử việc", "thời gian thử việc và tiền lương thử việc quy định như thế nào"),
    ("giữ bằng", "người sử dụng lao động có được giữ bản chính văn bằng chứng chỉ không"),
    ("chấm dứt", "quy định về đơn phương chấm dứt hợp đồng lao động và thời hạn báo trước"),
    ("không cạnh tranh", "thỏa thuận bảo mật và cam kết không làm việc cho đối thủ cạnh tranh"),
    ("chi phí đào tạo", "nghĩa vụ hoàn trả chi phí đào tạo nghề khi nghỉ việc"),
    ("làm thêm giờ", "quy định về thời giờ làm việc và giới hạn giờ làm thêm"),
    ("kỷ luật", "các hình thức xử lý kỷ luật lao động và bồi thường thiệt hại"),

    # Nhóm Hợp đồng Thương mại & Kinh tế (LTM 2005, BLDS 2015)
    ("phạt vi phạm", "mức phạt vi phạm hợp đồng tối đa trong thương mại là bao nhiêu phần trăm"),
    ("bồi thường", "quy định về bồi thường thiệt hại và căn cứ phát sinh trách nhiệm bồi thường"),
    ("chậm thanh toán", "lãi suất phạt chậm thanh toán trong hợp đồng thương mại"),
    ("bất khả kháng", "sự kiện bất khả kháng và miễn trừ trách nhiệm theo Bộ luật Dân sự"),
    ("tranh chấp", "thẩm quyền giải quyết tranh chấp hợp đồng tòa án hoặc trọng tài thương mại"),
    ("sở hữu trí tuệ", "quyền sở hữu trí tuệ đối với sản phẩm sáng tạo phát sinh trong hợp đồng"),
]


def extract_text_from_image(image_path: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Sử dụng Gemini Vision Multimodal API để OCR và trích xuất toàn bộ văn bản tiếng Việt từ file ảnh.
    """
    client = get_gemini_client()
    if not client:
        return "Lỗi: Không thể khởi tạo Gemini Client do chưa có API KEY."

    p = Path(image_path)
    if not p.exists():
        return f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'."

    try:
        img = Image.open(p)
        prompt = (
            "Bạn là chuyên gia trích xuất tài liệu pháp lý. "
            "Hãy đọc và trích xuất toàn bộ nội dung văn bản trong ảnh hợp đồng/tài liệu này một cách chính xác từng câu chữ, "
            "giữ nguyên cấu trúc các Điều, Khoản và các bên tham gia. Không thêm bớt hay bình luận gì ngoài nội dung văn bản."
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[img, prompt]
        )
        return response.text.strip()
    except Exception as e:
        return f"Lỗi khi nhận diện hình ảnh qua Gemini Vision: {e}"


def extract_text_from_pdf(pdf_path: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Trích xuất văn bản từ file PDF (Hỗ trợ cả PDF dạng văn bản số hóa lẫn PDF scan ảnh tài liệu).
    1. Ưu tiên đọc nhanh bằng pypdf nếu là Digital PDF có layer text.
    2. Tự động chuyển sang Gemini Multimodal Document AI nếu là Scanned PDF hoặc pypdf không trích xuất đủ văn bản.
    """
    p = Path(pdf_path)
    if not p.exists():
        return f"Lỗi: Không tìm thấy file PDF tại '{pdf_path}'."

    # 1. Thử đọc nhanh qua pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                extracted_pages.append(f"--- TRANG {i+1} ---\n{page_text.strip()}")

        full_pdf_text = "\n\n".join(extracted_pages).strip()
        if len(full_pdf_text) >= 80:
            return full_pdf_text
    except Exception as e:
        print(f"[PYPDF INFO] Không thể đọc text trực tiếp ({e}), chuyển sang Gemini Document Vision...")

    # 2. Xử lý Scanned PDF / Image PDF qua Gemini Multimodal
    client = get_gemini_client()
    if not client:
        return "Lỗi: Không thể khởi tạo Gemini Client để xử lý PDF scan."

    try:
        from google.genai import types
        pdf_bytes = p.read_bytes()
        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        prompt = (
            "Bạn là chuyên gia số hóa và thẩm định hợp đồng pháp lý. "
            "Hãy đọc và trích xuất toàn bộ nội dung văn bản trong tài liệu PDF này một cách chính xác từng câu chữ, "
            "giữ nguyên cấu trúc các Điều, Khoản, Tiêu đề, Bảng biểu và các bên tham gia. "
            "Không thêm bớt hoặc tóm tắt bất kỳ nội dung nào."
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[pdf_part, prompt]
        )
        return response.text.strip()
    except Exception as e:
        return f"Lỗi khi đọc file PDF qua Gemini Multimodal: {e}"


def split_contract_into_clauses(contract_text: str) -> List[Dict[str, Any]]:
    """
    Tách văn bản hợp đồng thành danh sách các điều khoản có cấu trúc.
    Hỗ trợ nhận diện các định dạng: 'Điều 1.', 'Điều 1:', 'Điều 1 -', 'Mục 1.', 'Khoản 1.'
    """
    if not contract_text or not contract_text.strip():
        return []

    lines = contract_text.strip().splitlines()
    clauses = []
    current_title = "Mở đầu / Thông tin chung"
    current_lines = []

    article_pattern = re.compile(r"^\s*(Điều\s+\d+[\.:\s\-][^\n]*)", re.IGNORECASE)

    for line in lines:
        match = article_pattern.match(line)
        if match:
            if current_lines:
                clauses.append({
                    "title": current_title.strip(),
                    "content": "\n".join(current_lines).strip()
                })
                current_lines = []
            current_title = match.group(1)
            # Thêm phần còn lại của dòng nếu có
            rest_of_line = line[len(match.group(1)):].strip()
            if rest_of_line:
                current_lines.append(rest_of_line)
        else:
            current_lines.append(line)

    if current_lines:
        clauses.append({
            "title": current_title.strip(),
            "content": "\n".join(current_lines).strip()
        })

    return clauses


def extract_risk_queries_from_contract(contract_text: str, clauses: List[Dict[str, Any]] = None) -> List[str]:
    """
    Tự động trích xuất danh sách các câu truy vấn pháp lý cần đối chiếu từ nội dung hợp đồng.
    """
    text_lower = contract_text.lower()
    queries = []

    for keyword, query_prompt in COMMON_RISK_TOPICS:
        if keyword in text_lower:
            queries.append(query_prompt)

    # Nếu có danh sách điều khoản cụ thể, bổ sung truy vấn từ tên điều khoản
    if clauses:
        for c in clauses:
            title = c.get("title", "")
            if "điều" in title.lower() and len(title) < 80:
                queries.append(f"Quy định pháp luật liên quan đến: {title}")

    # Loại bỏ truy vấn trùng lặp
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped[:8]  # Lấy tối đa 8 truy vấn trọng tâm nhất


def retrieve_contract_legal_context(
    contract_text: str,
    rag_retriever_func,
    max_total_chunks: int = 8
) -> Tuple[List[Dict], List[Dict]]:
    """
    Truy xuất các điều luật đối chiếu cho hợp đồng từ hệ thống Legal RAG.
    """
    if not rag_retriever_func:
        return [], []

    clauses = split_contract_into_clauses(contract_text)
    risk_queries = extract_risk_queries_from_contract(contract_text, clauses)

    all_top_chunks = []
    all_expanded_chunks = []
    seen_chunk_ids = set()

    # Truy vấn với toàn bộ hợp đồng hoặc các câu hỏi trọng tâm
    if not risk_queries:
        risk_queries = [contract_text[:300]]

    for query in risk_queries:
        try:
            top_chunks, expanded_chunks = rag_retriever_func(query)
            for c in top_chunks:
                cid = c.get("chunk_id", c.get("content", "")[:30])
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_top_chunks.append(c)

            for ec in expanded_chunks:
                cid = ec.get("chunk_id", ec.get("content", "")[:30])
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_expanded_chunks.append(ec)
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi truy vấn RAG cho '{query[:40]}...': {e}")

        if len(all_top_chunks) >= max_total_chunks:
            break

    return all_top_chunks[:max_total_chunks], all_expanded_chunks[:4]


def analyze_contract(
    contract_text: Optional[str] = None,
    input_source: Optional[str] = None,
    rag_retriever_func=None,
    model_name: str = "gemini-2.5-flash",
    stream: bool = True,
    stream_callback=None
) -> Tuple[str, List[Dict]]:
    """
    Hàm thực thi chính: Tiếp nhận văn bản / đường dẫn file ảnh / text hợp đồng -> Tách điều khoản -> RAG đối chiếu luật -> Sinh Báo cáo Thẩm định.
    Hỗ trợ cả tham số `contract_text` và `input_source`.
    """
    raw_input = (contract_text or input_source or "").strip().strip("'").strip('"')
    if not raw_input:
        err = "Lỗi: Nội dung hoặc đường dẫn hợp đồng rỗng."
        if stream and not stream_callback:
            print(err)
        return err, []

    p = Path(raw_input)
    contract_text = raw_input

    # 1. Tự động nhận diện nếu người dùng truyền đường dẫn file ảnh
    if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        print(f"\n🖼️ [GEMINI VISION] Đang nhận diện và đọc văn bản từ file ảnh: {p.name} ...")
        extracted_text = extract_text_from_image(str(p), model_name=model_name)
        if extracted_text.startswith("Lỗi:"):
            if stream and not stream_callback:
                print(extracted_text)
            return extracted_text, []
        contract_text = extracted_text
        print("✅ Đã trích xuất thành công nội dung văn bản từ hình ảnh!")

    # 2. Tự động nhận diện nếu người dùng truyền file PDF (.pdf)
    elif p.exists() and p.suffix.lower() == ".pdf":
        print(f"\n📑 [PDF EXTRACTOR] Đang trích xuất nội dung từ tài liệu PDF: {p.name} ...")
        extracted_text = extract_text_from_pdf(str(p), model_name=model_name)
        if extracted_text.startswith("Lỗi:"):
            if stream and not stream_callback:
                print(extracted_text)
            return extracted_text, []
        contract_text = extracted_text
        print("✅ Đã trích xuất thành công nội dung văn bản từ tài liệu PDF!")

    # 3. Tự động nhận diện nếu người dùng truyền file text (.txt, .md, .json)
    elif p.exists() and p.suffix.lower() in (".txt", ".md", ".json"):
        print(f"\n📄 [FILE LOADER] Đang nạp nội dung hợp đồng từ file: {p.name} ...")
        contract_text = p.read_text(encoding="utf-8")

    clauses = split_contract_into_clauses(contract_text)
    print(f"\n📑 [CONTRACT ANALYZER] Đã nhận diện được {len(clauses)} điều khoản / phân đoạn trong hợp đồng.")

    # 3. RAG Retrieve: Tìm các căn cứ pháp lý liên quan
    top_chunks, expanded_chunks = [], []
    if rag_retriever_func:
        print("🔍 [CONTRACT ANALYZER] Đang đối chiếu các điều khoản với Kho Dữ liệu Pháp luật...")
        top_chunks, expanded_chunks = retrieve_contract_legal_context(
            contract_text=contract_text,
            rag_retriever_func=rag_retriever_func,
            max_total_chunks=8
        )
        print(f"✅ Đã tìm thấy {len(top_chunks)} căn cứ pháp lý trực tiếp và {len(expanded_chunks)} điều khoản dẫn chiếu liên quan.")

    # 4. LLM Risk Assessment
    report = generate_contract_risk_analysis(
        contract_text=contract_text,
        top_chunks=top_chunks,
        expanded_chunks=expanded_chunks,
        model_name=model_name,
        stream=stream,
        stream_callback=stream_callback
    )

    return report, top_chunks
