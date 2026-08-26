"""
Query Router Module — Điều phối câu hỏi thông minh giữa:
1. Cơ sở dữ liệu Nhân sự & Doanh nghiệp (Supabase PostgreSQL)
2. Hệ thống Legal RAG (BM25 + Vector Search + Reranker)
3. Hybrid QA (Kết hợp cả hai)
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.database import (
    get_company_info,
    search_employees,
    get_department_summary,
)
from src.llm_generation import (
    generate_hrm_answer,
    generate_legal_answer,
    generate_hybrid_answer,
    generate_explanation_answer,
)
from src.contract_analyzer import analyze_contract
from src.conversational import (
    ConversationTurn,
    GLOBAL_SESSION_MANAGER,
    classify_followup_intent,
    rewrite_conversational_query,
    extract_cited_articles,
)

# Danh sách từ khóa đặc trưng cho dữ liệu Nhân sự & Doanh nghiệp nội bộ VNTech
HR_EXPLICIT_KEYWORDS = [
    "vntech", "công ty vntech", "toàn bộ thông tin công ty", "thông tin công ty vntech",
    "mã số thuế công ty", "địa chỉ công ty", "lương trung bình", "thống kê phòng ban",
    "danh sách nhân viên", "danh sách nhân sự", "tổng số nhân viên", "ai là giám đốc",
    "ai là tổng giám đốc", "ai là ceo", "ai là coo", "ai là trưởng phòng", "ai là kỹ sư",
    "lương của", "thu nhập của", "sđt của", "số điện thoại của", "email của",
    "ngày vào làm của", "chức vụ của"
]

# Từ khóa thành phần bổ trợ nhân sự nội bộ
HR_GENERAL_KEYWORDS = [
    "nhân viên", "nhân sự", "phòng ban", "kế toán", "kinh doanh", "marketing",
    "kỹ thuật", "tech lead", "developer", "tester", "qa/qc", "devops", "designer",
    "c&b", "tuyển dụng", "pháp chế"
]

# Danh sách từ khóa đặc trưng cho 10 nguồn Pháp luật (BLLĐ, BLDS, LDN, LĐTT, LSHTT, LTM, LANM, BVQLNTD, LQLT, LATVSLĐ)
LEGAL_KEYWORDS = [
    # Căn cứ pháp lý & quy định chung
    "điều", "khoản", "điểm", "bộ luật", "luật", "quy định", "nghị định", "thông tư",
    "căn cứ pháp lý", "theo quy định", "theo luật", "hợp pháp", "trái pháp luật",
    "vi phạm", "bị cấm", "nghiêm cấm", "bao nhiêu ngày", "bao nhiêu năm", "bao nhiêu phần trăm",
    "mức phạt", "xử phạt", "thời hiệu", "thời hạn",

    # Luật Doanh nghiệp & Luật Đầu tư
    "tnhh", "công ty tnhh", "công ty cổ phần", "cổ phần", "cổ đông", "hội đồng quản trị",
    "doanh nghiệp tư nhân", "công ty hợp danh", "thành lập công ty", "thành lập doanh nghiệp",
    "góp vốn", "thời hạn góp vốn", "đăng ký doanh nghiệp", "giấy chứng nhận đăng ký",
    "vốn điều lệ", "người đại diện theo pháp luật", "giải thể", "phá sản",
    "ngành nghề cấm đầu tư", "ưu đãi đầu tư", "chuyển nhượng vốn", "chuyển nhượng cổ phần",

    # Bộ luật Lao động & An toàn vệ sinh lao động
    "hợp đồng lao động", "thử việc", "thời gian thử việc", "lương thử việc",
    "đơn phương chấm dứt", "chấm dứt hợp đồng", "sa thải", "kỷ luật lao động",
    "thời giờ làm việc", "thời giờ nghỉ ngơi", "nghỉ phép", "nghỉ hàng năm", "thai sản",
    "trợ cấp thôi việc", "trợ cấp mất việc", "bảo hiểm xã hội", "bhxh", "bảo hiểm y tế", "bhyt",
    "bảo hiểm thất nghiệp", "bhtn", "tai nạn lao động", "bệnh nghề nghiệp",
    "an toàn lao động", "vệ sinh lao động", "người lao động", "người sử dụng lao động",
    "giữ bản chính", "văn bằng chứng chỉ",

    # Bộ luật Dân sự
    "bồi thường thiệt hại", "ngoài hợp đồng", "hợp đồng dân sự", "giao dịch dân sự",
    "tài sản bảo đảm", "thế chấp", "cầm cố", "bảo lãnh", "đại diện theo ủy quyền",
    "thời hiệu khởi kiện", "thừa kế", "di chúc", "vô hiệu", "hợp đồng vô hiệu",

    # Luật Sở hữu trí tuệ
    "sở hữu trí tuệ", "quyền tác giả", "tác giả", "bảo hộ quyền", "sáng chế",
    "bằng độc quyền", "nhãn hiệu", "kiểu dáng công nghiệp", "chỉ dẫn địa lý",
    "bí mật kinh doanh", "xâm phạm quyền", "chuyển giao quyền", "tác phẩm điện ảnh", "nhiếp ảnh",

    # Luật Thương mại
    "thương mại", "phạt vi phạm", "phạt hợp đồng", "mua bán hàng hóa", "cung ứng dịch vụ",
    "miễn trách nhiệm", "khiếu nại", "trung gian thương mại",

    # Luật Đất đai
    "đất đai", "thửa đất", "quyền sử dụng đất", "sổ đỏ", "sổ hồng", "tách thửa", "hợp thửa",
    "chuyển nhượng đất", "tặng cho đất", "thừa kế quyền sử dụng đất", "thu hồi đất", "bồi thường đất",
    "tái định cư", "giá đất", "bảng giá đất", "chuyển mục đích sử dụng đất", "cấp sổ đỏ",

    # Luật An ninh mạng & Bảo vệ quyền lợi người tiêu dùng & Quản lý thuế
    "an ninh mạng", "không gian mạng", "lưu trữ dữ liệu", "an toàn thông tin",
    "người tiêu dùng", "hàng hóa có khuyết tật", "đổi trả sản phẩm", "thu hồi sản phẩm",
    "khai thuế", "nộp thuế", "thuế thu nhập doanh nghiệp", "quyết toán thuế"
]


# Danh sách từ khóa & regex nhận diện Lời chào & Yêu cầu giới thiệu
GREETING_PATTERNS = [
    r"^(xin\s+)?chào(\s+(bạn|bot|em|anh|chị|ad|admin|mọi\s+người|cả\s+nhà|nhé|nha|ạ))?[\s!.,?~:)]*$",
    r"^(hello|hi|hey|helo|hế\s*lô|halo|hallo|alo|alô)(\s+(bot|bạn|admin|ad|mọi\s+người|nhé|nha|ạ))?[\s!.,?~:)]*$",
    r"^(good\s+(morning|afternoon|evening|day))[\s!.,?~:)]*$",
    r"^(bạn\s+là\s+ai|mày\s+là\s+ai|bạn\s+tên\s+gì|bot\s+là\s+ai)[\s!.,?~:)]*$",
    r"^(bạn\s+(có\s+thể\s+)?(làm|giúp|hỗ\s+trợ)\s+được\s+gì|giới\s+thiệu\s+về\s+bạn|giới\s+thiệu\s+bản\s+thân|chức\s+năng\s+của\s+bạn|hướng\s+dẫn\s+sử\s+dụng|help|trợ\s+giúp)[\s!.,?~:)]*$"
]

GREETING_KEYWORDS = [
    "xin chào", "chào bạn", "chào bot", "chào em", "chào anh", "chào chị", "chào ad",
    "hello", "hi bot", "hello bot", "hey bot", "hế lô", "alo", "alô", "bạn là ai",
    "bạn có thể làm gì", "giới thiệu bản thân", "giới thiệu về bạn"
]

GREETING_RESPONSE = """👋 **Xin chào bạn! Tôi là Trợ lý AI Pháp luật, Thẩm định Hợp đồng & Quản trị Nhân sự VNTech.**

Tôi có thể hỗ trợ bạn giải quyết ngay lập tức các nhu cầu sau:

1. ⚖️ **Tra cứu & Tư vấn Pháp luật chuyên sâu:**
   - Trích dẫn chính xác Điều, Khoản và nội dung từ **11 bộ luật Việt Nam** hiện hành: *Bộ luật Lao động 2019, Bộ luật Dân sự 2015, Luật Đất đai 2024, Luật Doanh nghiệp 2020, Luật Đầu tư 2020, Luật Sở hữu trí tuệ, Luật Thương mại, Luật An ninh mạng, Luật Bảo vệ quyền lợi người tiêu dùng, Luật Quản lý thuế, Luật ATVS Lao động*.
   - Hướng dẫn thủ tục, quy định về thử việc, nghỉ phép, sa thải, tranh chấp, bồi thường, thừa kế đất đai, thành lập doanh nghiệp, xử phạt vi phạm,...

2. 📄 **Thẩm định & Rà soát Rủi ro Hợp đồng (Zero-Risk Review):**
   - Tải lên tài liệu hợp đồng hoặc ảnh chụp (*PNG, JPG, PDF/Text*) hoặc dán trực tiếp điều khoản.
   - Hệ thống tự động quét và cảnh báo các **bẫy pháp lý**, điều khoản vô hiệu, chế tài phạt vi phạm vượt mức trần hoặc các nghĩa vụ bất lợi.

3. 👥 **Tra cứu Cơ sở dữ liệu Nhân sự & Doanh nghiệp VNTech:**
   - Tra cứu tức thì danh sách nhân viên, phòng ban, chức vụ, mức lương, ngày vào làm, thông tin công ty và đối chiếu chính sách nội bộ với luật định.

---
💡 **Gợi ý bắt đầu nhanh:** Bạn có thể nhập câu hỏi trực tiếp (ví dụ: *"Thời giờ làm việc bình thường của người lao động không quá bao nhiêu giờ trong một ngày và bao nhiêu giờ trong một tuần?"*,*
"Thống kê các phòng ban của VNTech"*), hoặc bấm vào biểu tượng 📎 đính kèm để tải file hợp đồng lên thẩm định!"""


def is_greeting(query: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là lời chào hoặc yêu cầu giới thiệu bot không."""
    q = query.strip().lower()
    q_clean = re.sub(r"[!.,?~:;\-_+*#\n\t]+", " ", q).strip()

    # 1. Khớp regex chính xác cho các câu ngắn
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE) or re.search(pattern, q_clean, re.IGNORECASE):
            return True

    # 2. Khớp từ khóa nếu câu ngắn (< 7 từ) và không chứa từ khóa nghiệp vụ chuyên sâu
    words = q_clean.split()
    if len(words) <= 6:
        for kw in GREETING_KEYWORDS:
            if kw in q_clean:
                # Kiểm tra nếu không có từ khóa pháp lý hoặc nhân sự cụ thể
                if not any(lk in q_clean for lk in ["điều", "khoản", "bộ luật", "luật", "sa thải", "hợp đồng", "lương", "thuế"]):
                    return True
    return False


_CACHED_EMPLOYEE_NAMES: List[str] = None


def get_cached_employee_names() -> List[str]:
    """Lấy và cache danh sách họ tên nhân viên để phân loại tức thì (<1ms)."""
    global _CACHED_EMPLOYEE_NAMES
    if _CACHED_EMPLOYEE_NAMES is None:
        try:
            emps = search_employees(limit=100)
            _CACHED_EMPLOYEE_NAMES = [e["full_name"].lower() for e in emps if e.get("full_name")]
        except Exception:
            _CACHED_EMPLOYEE_NAMES = []
    return _CACHED_EMPLOYEE_NAMES


def classify_query(query: str) -> str:
    """
    Phân loại câu hỏi thành một trong các loại:
    - 'GREETING': Lời chào, hỏi thăm, giới thiệu bản thân trợ lý
    - 'CONTRACT_RISK': Yêu cầu thẩm định rủi ro hợp đồng (file ảnh / file text / văn bản dán trực tiếp)
    - 'HR_DATABASE': Câu hỏi tra cứu dữ liệu nội bộ công ty/nhân viên VNTech
    - 'LEGAL_RAG': Câu hỏi tra cứu điều luật pháp lý (10 nguồn luật)
    - 'HYBRID': Câu hỏi kết hợp (nhân sự nội bộ + đối chiếu quy định pháp luật)
    """
    cleaned_input = query.strip().strip("'").strip('"')
    p = Path(cleaned_input)

    # 1. Kiểm tra nếu là lời chào hỏi / giới thiệu bản thân
    if is_greeting(query):
        return "GREETING"

    # 2. Kiểm tra nếu người dùng truyền đường dẫn file hợp đồng (ảnh hoặc text)
    if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt", ".md", ".json"):
        return "CONTRACT_RISK"

    q_lower = query.lower()

    # 3. Kiểm tra các từ khóa yêu cầu thẩm định / rà soát rủi ro hợp đồng
    contract_risk_keywords = [
        "thẩm định hợp đồng", "rà soát hợp đồng", "đánh giá rủi ro hợp đồng",
        "soi hợp đồng", "kiểm tra hợp đồng", "phân tích hợp đồng", "bẫy hợp đồng",
        "báo cáo thẩm định", "rủi ro hợp đồng", "review hợp đồng", "thẩm định rủi ro"
    ]
    if any(kw in q_lower for kw in contract_risk_keywords):
        return "CONTRACT_RISK"

    # 4. Kiểm tra nếu nội dung là một văn bản hợp đồng được dán trực tiếp
    if "cộng hòa xã hội chủ nghĩa việt nam" in q_lower or (
        q_lower.count("điều ") >= 2 and ("bên a" in q_lower or "bên b" in q_lower or "hợp đồng" in q_lower)
    ):
        return "CONTRACT_RISK"

    # 5. Kiểm tra xem có nhắc đến tên nhân viên cụ thể trong CSDL VNTech không
    emp_names = get_cached_employee_names()
    has_specific_name = any(name in q_lower for name in emp_names) if emp_names else False

    # 6. Tính điểm phân loại
    has_explicit_hr = any(kw in q_lower for kw in HR_EXPLICIT_KEYWORDS)
    hr_general_hits = sum(1 for kw in HR_GENERAL_KEYWORDS if kw in q_lower)
    
    hr_score = (5 if has_specific_name else 0) + (4 if has_explicit_hr else 0) + min(hr_general_hits, 2)
    legal_score = sum(1 for kw in LEGAL_KEYWORDS if kw in q_lower)

    # 7. Phân luồng điều phối
    if has_specific_name and legal_score > 0:
        # Nhắc đến nhân viên cụ thể và hỏi luật áp dụng
        return "HYBRID"
    elif hr_score > 0 and legal_score > 0:
        # Có cả yếu tố hỏi nhân sự và hỏi điều khoản luật
        if has_explicit_hr or has_specific_name:
            return "HYBRID"
        else:
            return "LEGAL_RAG"
    elif hr_score >= 3 and legal_score == 0:
        # Rõ ràng là hỏi nhân sự nội bộ (tên người, lương của ai, thông tin cty vntech)
        return "HR_DATABASE"
    elif legal_score > 0:
        # Rõ ràng là câu hỏi pháp luật
        return "LEGAL_RAG"
    else:
        # Mặc định ưu tiên Legal RAG nếu câu hỏi chung chung
        return "LEGAL_RAG"




def retrieve_hr_data(query: str) -> str:
    """
    Truy vấn và tổng hợp dữ liệu từ PostgreSQL dựa trên nội dung câu hỏi.
    """
    q_lower = query.lower()
    data_parts = []

    # 1. Nếu hỏi về toàn bộ thông tin / tổng quan công ty
    is_full_company_overview = any(k in q_lower for k in [
        "toàn bộ thông tin", "tất cả thông tin", "thông tin công ty", "thông tin doanh nghiệp",
        "tổng quan công ty", "báo cáo công ty", "giới thiệu công ty", "overview", "tổng thể công ty"
    ])

    if is_full_company_overview:
        company = get_company_info(1)
        if company:
            data_parts.append(
                f"🏢 THÔNG TIN DOANH NGHIỆP:\n"
                f"- Tên công ty: {company['company_name']}\n"
                f"- Mã số thuế : {company['tax_code']}\n"
                f"- Địa chỉ    : {company['address']}"
            )

        dept_stats = get_department_summary()
        if dept_stats:
            dept_text = "📊 THỐNG KÊ PHÒNG BAN & LƯƠNG TRUNG BÌNH:\n"
            for d in dept_stats:
                avg_sal = f"{d['avg_salary']:,.0f} VNĐ" if d.get('avg_salary') else "N/A"
                dept_text += f"- {d['department']:<16}: {d['employee_count']:>2} người | Lương TB: {avg_sal}\n"
            data_parts.append(dept_text)

        all_emps = search_employees(limit=25)
        if all_emps:
            emp_text = "👥 DANH SÁCH TOÀN BỘ NHÂN SỰ:\n"
            for idx, emp in enumerate(all_emps, 1):
                sal_str = f"{emp['salary']:,.0f} VNĐ" if emp.get("salary") else "Chưa cập nhật"
                phone_str = emp.get("phone", "N/A")
                emp_text += f"[{idx}] {emp['full_name']} | Phòng: {emp['department']} | Chức vụ: {emp['position']} | Lương: {sal_str} | SĐT: {phone_str} | Email: {emp['email']} | Ngày vào: {emp['hire_date']}\n"
            data_parts.append(emp_text)

        return "\n\n".join(data_parts) if data_parts else "Không tìm thấy dữ liệu nhân sự."

    # 2. Kiểm tra thông tin công ty đơn lẻ
    if any(k in q_lower for k in ["công ty", "mã số thuế", "mst", "địa chỉ", "doanh nghiệp"]):
        company = get_company_info(1)
        if company:
            data_parts.append(
                f"🏢 THÔNG TIN DOANH NGHIỆP:\n"
                f"- Tên công ty: {company['company_name']}\n"
                f"- Mã số thuế : {company['tax_code']}\n"
                f"- Địa chỉ    : {company['address']}"
            )

    # 3. Kiểm tra thống kê phòng ban
    if any(k in q_lower for k in ["phòng ban", "thống kê", "bao nhiêu phòng", "lương trung bình"]):
        dept_stats = get_department_summary()
        if dept_stats:
            dept_text = "📊 THỐNG KÊ PHÒNG BAN & LƯƠNG TRUNG BÌNH:\n"
            for d in dept_stats:
                avg_sal = f"{d['avg_salary']:,.0f} VNĐ" if d.get('avg_salary') else "N/A"
                dept_text += f"- {d['department']:<16}: {d['employee_count']:>2} người | Lương TB: {avg_sal}\n"
            data_parts.append(dept_text)

    # 4. Tìm kiếm nhân sự theo tên / chức vụ / phòng ban / từ khóa
    employees = []
    words = query.split()
    for i in range(len(words)):
        for j in range(i + 2, min(i + 5, len(words) + 1)):
            candidate_name = " ".join(words[i:j])
            matches = search_employees(keyword=candidate_name, limit=5)
            for m in matches:
                if m not in employees:
                    employees.append(m)

    common_depts = ["kỹ thuật", "công nghệ", "marketing", "sales", "kinh doanh", "nhân sự", "kế toán", "tài chính"]
    for d in common_depts:
        if d in q_lower:
            dept_matches = search_employees(department=d, limit=10)
            for m in dept_matches:
                if m not in employees:
                    employees.append(m)

    if not employees and not data_parts and any(k in q_lower for k in (HR_EXPLICIT_KEYWORDS + HR_GENERAL_KEYWORDS)):
        employees = search_employees(limit=20)

    if employees:
        emp_text = "👥 DANH SÁCH NHÂN SỰ LIÊN QUAN:\n"
        for idx, emp in enumerate(employees, 1):
            sal_str = f"{emp['salary']:,.0f} VNĐ" if emp.get("salary") else "Chưa cập nhật"
            phone_str = emp.get("phone", "N/A")
            emp_text += f"[{idx}] {emp['full_name']} | Phòng: {emp['department']} | Chức vụ: {emp['position']} | Lương: {sal_str} | SĐT: {phone_str} | Email: {emp['email']} | Ngày vào làm: {emp['hire_date']}\n"
        data_parts.append(emp_text)

    return "\n\n".join(data_parts) if data_parts else "Không tìm thấy dữ liệu nhân sự phù hợp trong cơ sở dữ liệu."


def answer_user_query(
    query: str,
    rag_retriever_func=None,
    stream: bool = True,
    stream_callback=None,
    session_id: str = "default"
) -> Tuple[str, str]:
    """
    Hàm xử lý chính: Nhận câu hỏi -> Quản lý Hội thoại Đa lượt (Conversational State) -> Phân loại -> Gọi DB/RAG -> Trả về (intent, câu_trả_lời).
    Hỗ trợ streaming token thời gian thực.
    """
    # 1. Kiểm tra Lịch sử Hội thoại để phát hiện câu hỏi nối tiếp (Follow-up)
    history = GLOBAL_SESSION_MANAGER.get_history(session_id)
    query_type, is_followup = classify_followup_intent(query, history)

    rewritten_query = query
    retrieved_chunks = []
    expanded_chunks = []

    # XỬ LÝ TYPE A: Contextual Explanation (Giải thích sâu dựa trên luật cũ đã trích xuất)
    if query_type == "TYPE_A_EXPLANATION" and history:
        last_turn = history[-1]
        print(f"\n🧠 [CONVERSATIONAL RAG] Phát hiện câu hỏi giải thích/làm rõ (Type A). Tái sử dụng Căn cứ Pháp lý lượt trước!")
        intent = "LEGAL_RAG"
        retrieved_chunks = last_turn.retrieved_chunks
        expanded_chunks = last_turn.expanded_chunks

        answer = generate_explanation_answer(
            query=query,
            previous_query=last_turn.user_query,
            previous_answer=last_turn.answer,
            top_chunks=retrieved_chunks,
            expanded_chunks=expanded_chunks,
            stream=stream,
            stream_callback=stream_callback
        )

        turn = ConversationTurn(
            user_query=query,
            rewritten_query=query,
            query_type="TYPE_A_EXPLANATION",
            intent=intent,
            retrieved_chunks=retrieved_chunks,
            expanded_chunks=expanded_chunks,
            answer=answer
        )
        GLOBAL_SESSION_MANAGER.add_turn(session_id, turn)
        return intent, answer

    # XỬ LÝ TYPE B: Contextual Follow-up (Phát sinh tình tiết/đối tượng mới -> Tái cấu trúc truy vấn)
    if query_type == "TYPE_B_FOLLOWUP" and history:
        rewritten_query = rewrite_conversational_query(query, history)
        print(f"\n🧠 [CONVERSATIONAL RAG] Câu hỏi phụ thuộc (Type B).")
        print(f"   -> Câu gốc: '{query}'")
        print(f"   -> Viết lại: '{rewritten_query}'")

    # 2. Phân loại Ý định dựa trên câu hỏi (hoặc câu hỏi đã được viết lại)
    query_to_route = rewritten_query if is_followup else query
    intent = classify_query(query_to_route)
    print(f"\n🔍 [QUERY ROUTER] Đã nhận diện ý định câu hỏi: 👉 [{intent}]")

    if stream and not stream_callback:
        print("\n" + "=" * 95)
        print(f"          🤖 CÂU TRẢ LỜI TỰ ĐỘNG ({intent})")
        print("=" * 95)

    if intent == "GREETING":
        answer = GREETING_RESPONSE
        if stream and stream_callback:
            words = answer.split(" ")
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i+3])
                if i + 3 < len(words):
                    chunk += " "
                stream_callback(chunk)
        elif stream and not stream_callback:
            print(answer)
    elif intent == "CONTRACT_RISK":
        report, _ = analyze_contract(
            input_source=query,
            rag_retriever_func=rag_retriever_func,
            stream=stream,
            stream_callback=stream_callback
        )
        answer = report
    elif intent == "HR_DATABASE":
        db_context = retrieve_hr_data(query_to_route)
        answer = generate_hrm_answer(query_to_route, db_context, stream=stream, stream_callback=stream_callback)
    elif intent == "LEGAL_RAG":
        if rag_retriever_func:
            retrieved_chunks, expanded_chunks = rag_retriever_func(query_to_route)

            # Tự động bảo toàn các chunk nền tảng cốt lõi của turn trước cho câu hỏi Type B (nếu chưa có)
            if query_type == "TYPE_B_FOLLOWUP" and history:
                last_turn = history[-1]
                existing_ids = {c.get("chunk_id") for c in retrieved_chunks if isinstance(c, dict)}
                for old_c in last_turn.retrieved_chunks[:2]:
                    if isinstance(old_c, dict) and old_c.get("chunk_id") not in existing_ids:
                        retrieved_chunks.append(old_c)
                        existing_ids.add(old_c.get("chunk_id"))

            answer = generate_legal_answer(query_to_route, retrieved_chunks, expanded_chunks, stream=stream, stream_callback=stream_callback)
        else:
            answer = "Lỗi: Chưa khởi tạo hàm truy vấn RAG."
            if stream and not stream_callback:
                print(answer)
    else:  # HYBRID
        db_context = retrieve_hr_data(query_to_route)
        if rag_retriever_func:
            retrieved_chunks, expanded_chunks = rag_retriever_func(query_to_route)
            answer = generate_hybrid_answer(query_to_route, db_context, retrieved_chunks, expanded_chunks, stream=stream, stream_callback=stream_callback)
        else:
            answer = generate_hrm_answer(query_to_route, db_context, stream=stream, stream_callback=stream_callback)

    if stream and not stream_callback:
        print("\n" + "=" * 95 + "\n")

    # Lưu lượt hội thoại và danh sách điều luật trích dẫn vào SessionManager
    citations = extract_cited_articles(answer)
    turn = ConversationTurn(
        user_query=query,
        rewritten_query=rewritten_query,
        query_type=query_type,
        intent=intent,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        expanded_chunks=expanded_chunks,
        answer=answer
    )
    GLOBAL_SESSION_MANAGER.add_turn(session_id, turn)

    return intent, answer

