"""
Kiểm thử chức năng Phân tích & Đánh giá Rủi ro Hợp đồng (Contract Risk Assessment)
"""

import sys
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.contract_analyzer import (
    split_contract_into_clauses,
    extract_risk_queries_from_contract,
    analyze_contract
)

SAMPLE_RISKY_CONTRACT = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
---------------------------

HỢP ĐỒNG LAO ĐỘNG VÀ THƯƠNG MẠI NỘI BỘ
Số: 01/2024/HĐLĐ-VNT

Hôm nay, ngày 15 tháng 01 năm 2024, chúng tôi gồm:
Bên A (Người sử dụng lao động): Công ty Cổ phần Công nghệ VNTech
Bên B (Người lao động): Ông Nguyễn Văn X

Hai bên thống nhất ký kết hợp đồng với các điều khoản sau:

Điều 1. Công việc và Thời hạn hợp đồng
- Vị trí công việc: Kỹ sư Phần mềm.
- Thời gian thử việc là 180 ngày kể từ ngày ký hợp đồng.
- Lương trong thời gian thử việc bằng 70% mức lương chính thức.

Điều 2. Giữ giấy tờ tùy thân và văn bằng
- Để đảm bảo trách nhiệm làm việc và không nghỉ việc đột xuất, Bên B đồng ý nộp lại bản chính Bằng tốt nghiệp Đại học và Căn cước công dân cho Bên A giữ trong suốt thời hạn hợp đồng.

Điều 3. Chế độ bảo mật và cấm làm việc cho đối thủ
- Sau khi chấm dứt hợp đồng lao động, Bên B không được làm việc cho bất kỳ công ty nào cùng ngành công nghệ thông tin trong vòng 05 năm. Nếu vi phạm, Bên B phải bồi thường 500 triệu đồng.

Điều 4. Phạt vi phạm nghĩa vụ hợp đồng
- Trường hợp Bên B đơn phương chấm dứt hợp đồng trước thời hạn mà không được Bên A đồng ý bằng văn bản, Bên B phải chịu mức phạt vi phạm bằng 25% tổng thu nhập hàng năm và không được nhận tháng lương cuối cùng.

Điều 5. Giải quyết tranh chấp
- Mọi tranh chấp phát sinh sẽ do Giám đốc Công ty Bên A toàn quyền phán quyết cuối cùng, các bên không được khởi kiện ra Tòa án.
"""


def dummy_rag_retriever(query: str):
    """Giả lập hàm RAG retriever trả về các căn cứ luật liên quan."""
    q_lower = query.lower()
    chunks = []
    if "thử việc" in q_lower:
        chunks.append({
            "chunk_id": "45-2019-QH14-đ25",
            "article": "25",
            "clause": "1",
            "article_title": "Thời gian thử việc",
            "document_type": "Bộ luật",
            "title": "Bộ luật Lao động 2019",
            "document_number": "45/2019/QH14",
            "content": "Thời gian thử việc do hai bên thỏa thuận căn cứ vào tính chất và mức độ phức tạp của công việc nhưng chỉ được thử việc một lần đối với một công việc và bảo đảm điều kiện sau đây:\n1. Không quá 180 ngày đối với công việc của người quản lý doanh nghiệp...\n2. Không quá 60 ngày đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng trở lên;\n3. Tiền lương của người lao động trong thời gian thử việc do hai bên thỏa thuận nhưng ít nhất phải bằng 85% mức lương của công việc đó."
        })
    if "giữ" in q_lower or "bằng" in q_lower:
        chunks.append({
            "chunk_id": "45-2019-QH14-đ17",
            "article": "17",
            "clause": "1",
            "article_title": "Hành vi người sử dụng lao động không được làm khi giao kết, thực hiện hợp đồng lao động",
            "document_type": "Bộ luật",
            "title": "Bộ luật Lao động 2019",
            "document_number": "45/2019/QH14",
            "content": "1. Giữ bản chính giấy tờ tùy thân, văn bằng, chứng chỉ của người lao động.\n2. Yêu cầu người lao động phải thực hiện biện pháp bảo đảm bằng tiền hoặc tài sản khác cho việc thực hiện hợp đồng lao động."
        })
    if "phạt" in q_lower or "vi phạm" in q_lower:
        chunks.append({
            "chunk_id": "36-2005-QH11-đ301",
            "article": "301",
            "clause": "",
            "article_title": "Mức phạt vi phạm",
            "document_type": "Luật",
            "title": "Luật Thương mại 2005",
            "document_number": "36/2005/QH11",
            "content": "Mức phạt đối với vi phạm nghĩa vụ hợp đồng hoặc tổng mức phạt đối với nhiều vi phạm do các bên thoả thuận trong hợp đồng, nhưng không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm, trừ trường hợp quy định tại Điều 266 của Luật này."
        })
    return chunks, []


def test_contract_analyzer():
    print("=" * 90)
    print("      KIỂM THỬ MODULE PHÂN TÍCH & ĐÁNH GIÁ RỦI RO HỢP ĐỒNG")
    print("=" * 90)

    # 1. Test split_contract_into_clauses
    clauses = split_contract_into_clauses(SAMPLE_RISKY_CONTRACT)
    print(f"\n1. Tách điều khoản hợp đồng: Tìm thấy {len(clauses)} phân đoạn.")
    assert len(clauses) >= 5, f"Số điều khoản tách ra không đủ: {len(clauses)}"
    for idx, c in enumerate(clauses, 1):
        print(f"   [{idx}] {c['title']}")

    # 2. Test extract_risk_queries_from_contract
    risk_queries = extract_risk_queries_from_contract(SAMPLE_RISKY_CONTRACT, clauses)
    print(f"\n2. Trích xuất truy vấn rủi ro ({len(risk_queries)} truy vấn):")
    for q in risk_queries:
        print(f"   - {q}")
    assert len(risk_queries) > 0, "Không trích xuất được truy vấn rủi ro nào!"

    # 3. Test analyze_contract
    print("\n3. Chạy thử nghiệm phân tích và sinh Báo cáo Thẩm định:")
    report, top_chunks = analyze_contract(
        contract_text=SAMPLE_RISKY_CONTRACT,
        rag_retriever_func=dummy_rag_retriever,
        stream=True
    )

    print("\n" + "=" * 90)
    print("✅ KIỂM THỬ HOÀN TẤT THÀNH CÔNG!")
    print("=" * 90)


if __name__ == "__main__":
    test_contract_analyzer()
