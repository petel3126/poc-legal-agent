"""
Script kiểm thử toàn diện Query Router:
1. Hỏi thông tin lương cụ thể của người A, B
2. Hỏi danh sách phòng ban & chức vụ
3. Hỏi điều khoản pháp luật lao động
"""

import sys
from pathlib import Path

# UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.router import classify_query, answer_user_query

def dummy_rag_retriever(query: str):
    # Dummy RAG retriever phục vụ test router
    return [
        {
            "article": "35",
            "clause": "1",
            "article_title": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
            "document_type": "Bộ luật",
            "title": "Bộ luật Lao động 2019",
            "document_number": "45/2019/QH14",
            "content": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 30 ngày nếu làm việc theo hợp đồng lao động xác định thời hạn có thời hạn từ 12 tháng đến 36 tháng..."
        }
    ], []

def run_router_tests():
    test_queries = [
        ("Thời hạn góp vốn thành lập công ty TNHH kể từ ngày được cấp Giấy chứng nhận đăng ký doanh nghiệp là bao nhiêu ngày?", "LEGAL_RAG"),
        ("Số lượng thành viên tối đa trong công ty trách nhiệm hữu hạn hai thành viên trở lên là bao nhiêu người?", "LEGAL_RAG"),
        ("Mức phạt vi phạm hợp đồng tối đa trong thương mại theo Luật Thương mại là bao nhiêu phần trăm?", "LEGAL_RAG"),
        ("Thời hạn bảo hộ quyền tác giả đối với tác phẩm điện ảnh là bao nhiêu năm?", "LEGAL_RAG"),
        ("Thời hiệu khởi kiện yêu cầu bồi thường thiệt hại ngoài hợp đồng theo Bộ luật Dân sự là bao nhiêu năm?", "LEGAL_RAG"),
        ("Hãy cho tôi biết mức lương của Nguyễn Văn An và Trần Minh Đức?", "HR_DATABASE"),
        ("Phòng Kỹ thuật có những nhân sự nào và ai là Trưởng phòng?", "HR_DATABASE"),
        ("Công ty VNTech có mã số thuế và địa chỉ ở đâu?", "HR_DATABASE"),
        ("Người lao động có quyền đơn phương chấm dứt hợp đồng không?", "LEGAL_RAG"),
        ("Anh Phạm Quốc Bảo ở phòng Kỹ thuật đang thử việc với mức lương 32 triệu thì mức lương này có đúng quy định thử việc của Bộ luật Lao động không?", "HYBRID"),
        ("Nếu anh Lê Hoàng Nam muốn đơn phương chấm dứt hợp đồng lao động thì theo quy định của luật cần phải báo trước bao nhiêu ngày?", "HYBRID"),
        ("Hãy thẩm định rủi ro hợp đồng lao động này giúp tôi", "CONTRACT_RISK"),
        ("Rà soát hợp đồng mua bán và chỉ ra các bẫy pháp lý", "CONTRACT_RISK"),
        ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐiều 1. Công việc\nĐiều 2. Tiền lương\nBên A và Bên B thống nhất...", "CONTRACT_RISK"),
    ]

    print("=" * 80)
    print("      KIỂM THỬ QUERY ROUTER (HR DATABASE & LEGAL RAG)")
    print("=" * 80)

    passed = 0
    for idx, (q, expected) in enumerate(test_queries, 1):
        intent = classify_query(q)
        ok = (intent == expected)
        if ok: passed += 1
        status = "✅ PASS" if ok else f"❌ FAIL (Cần: {expected}, Ra: {intent})"
        print(f"[{idx:>2}] {status:<28} | INTENT: [{intent:<11}] | \"{q[:60]}...\"")

    print(f"\n📊 KẾT QUẢ: {passed}/{len(test_queries)} bài test PASS!")


if __name__ == "__main__":
    run_router_tests()
