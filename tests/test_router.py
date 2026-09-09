"""
Script kiểm thử toàn diện Query Router (50 câu hỏi):
- Nhóm 1 (10 câu): HR_DATABASE (Truy xuất dữ liệu nội bộ VNTech)
- Nhóm 2 (10 câu): LEGAL_RAG (Học thuật / Căn cứ Điều luật)
- Nhóm 3 (10 câu): LEGAL_RAG (Tình huống thực tế / Đời sống công sở / Chứa từ ngữ HR)
- Nhóm 4 (10 câu): HYBRID (Nhân sự cụ thể trong CSDL VNTech + Đối chiếu Pháp luật)
- Nhóm 5 (5 câu) : CONTRACT_RISK (Thẩm định rủi ro hợp đồng)
- Nhóm 6 (5 câu) : GREETING (Lời chào, giới thiệu trợ lý)
"""

import sys
from pathlib import Path

# UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.router import classify_query


def run_router_tests():
    test_suite = [
        # =========================================================================
        # NHÓM 1: HR_DATABASE (10 câu tra cứu dữ liệu nội bộ công ty VNTech)
        # =========================================================================
        ("Hãy cho tôi biết mức lương của Nguyễn Văn An và Trần Minh Đức?", "HR_DATABASE"),
        ("Phòng Kỹ thuật có những nhân sự nào và ai là Trưởng phòng?", "HR_DATABASE"),
        ("Công ty VNTech có mã số thuế và địa chỉ ở đâu?", "HR_DATABASE"),
        ("Cho tôi xem danh sách nhân viên công ty VNTech", "HR_DATABASE"),
        ("Thống kê các phòng ban và lương trung bình của VNTech", "HR_DATABASE"),
        ("Ai là Tổng Giám đốc của công ty VNTech?", "HR_DATABASE"),
        ("Thông tin liên hệ sđt và email của Bùi Thị Bích là gì?", "HR_DATABASE"),
        ("Chức vụ của nhân viên Đỗ Tiến Đạt là gì?", "HR_DATABASE"),
        ("Tổng số nhân viên hiện tại của công ty VNTech là bao nhiêu?", "HR_DATABASE"),
        ("Ngày vào làm của nhân viên Phan Mai Hương là ngày nào?", "HR_DATABASE"),

        # =========================================================================
        # NHÓM 2: LEGAL_RAG - Học thuật / Căn cứ điều luật (10 câu)
        # =========================================================================
        ("Thời hạn góp vốn thành lập công ty TNHH kể từ ngày được cấp Giấy chứng nhận đăng ký doanh nghiệp là bao nhiêu ngày?", "LEGAL_RAG"),
        ("Số lượng thành viên tối đa trong công ty trách nhiệm hữu hạn hai thành viên trở lên là bao nhiêu người?", "LEGAL_RAG"),
        ("Mức phạt vi phạm hợp đồng tối đa trong thương mại theo Luật Thương mại là bao nhiêu phần trăm?", "LEGAL_RAG"),
        ("Thời hạn bảo hộ quyền tác giả đối với tác phẩm điện ảnh là bao nhiêu năm?", "LEGAL_RAG"),
        ("Thời hiệu khởi kiện yêu cầu bồi thường thiệt hại ngoài hợp đồng theo Bộ luật Dân sự là bao nhiêu năm?", "LEGAL_RAG"),
        ("Thời giờ làm việc bình thường của người lao động không quá bao nhiêu giờ trong một ngày và bao nhiêu giờ trong một tuần?", "LEGAL_RAG"),
        ("Hành vi nào bị nghiêm cấm trong an toàn vệ sinh lao động theo Luật ATVSLĐ 2015?", "LEGAL_RAG"),
        ("Điều kiện để người tiêu dùng được đổi trả sản phẩm có khuyết tật theo Luật Bảo vệ quyền lợi người tiêu dùng?", "LEGAL_RAG"),
        ("Các trường hợp người sử dụng lao động được quyền đơn phương chấm dứt hợp đồng lao động theo Bộ luật Lao động 2019?", "LEGAL_RAG"),
        ("Hành vi phát tán thông tin sai sự thật trên mạng bị xử phạt theo Luật An ninh mạng như thế nào?", "LEGAL_RAG"),

        # =========================================================================
        # NHÓM 3: LEGAL_RAG - Tình huống thực tế / Ngôn ngữ đời sống / Chứa từ HR (10 câu)
        # =========================================================================
        # Câu hỏi trong ảnh của người dùng:
        ("Người làm thủ kho của công ty tôi đột ngột bỏ việc, giám đốc yêu cầu tôi là nhân viên phòng kế toán xuống làm thay một thời gian cho đến khi tìm được người thay thế. Thực lòng tôi không thích lắm vì lương thủ kho thấp hơn lương của tôi, và cũng không biết tôi phải làm thay bao lâu. Nhưng hiện nay, tìm được công việc không dễ nên tôi không có ý định bỏ việc dù được giao việc không đúng nội dung hợp đồng. Trường hợp như của tôi, được bảo đảm quyền lợi gì khi chuyển việc không?", "LEGAL_RAG"),
        ("Người lao động có quyền đơn phương chấm dứt hợp đồng không?", "LEGAL_RAG"),
        ("Tôi là kế toán, công ty chậm trả lương 2 tháng thì tôi có quyền đơn phương chấm dứt hợp đồng không và có được trợ cấp thôi việc không?", "LEGAL_RAG"),
        ("Giám đốc yêu cầu nhân viên phải làm thêm giờ liên tục vào cuối tuần, nếu tôi từ chối thì có bị sa thải không?", "LEGAL_RAG"),
        ("Công ty có được giữ bằng đại học bản chính của người lao động khi ký hợp đồng làm việc không?", "LEGAL_RAG"),
        ("Khi chuyển người lao động sang làm công việc khác so với hợp đồng thì công ty được phép chuyển tối đa bao lâu?", "LEGAL_RAG"),
        ("Lương của tôi bị công ty hạ xuống thấp hơn so với hợp đồng lao động đã ký thì công ty làm vậy có đúng pháp luật không?", "LEGAL_RAG"),
        ("Nhân viên thử việc có bắt buộc phải đóng bảo hiểm xã hội không và lương thử việc tối thiểu là bao nhiêu phần trăm?", "LEGAL_RAG"),
        ("Nếu tôi bị công ty đuổi việc trái pháp luật thì công ty có phải bồi thường những khoản tiền nào?", "LEGAL_RAG"),
        ("Công ty ép nhân viên nghỉ việc không lý do thì người lao động có quyền khiếu nại hoặc khởi kiện ở đâu?", "LEGAL_RAG"),

        # =========================================================================
        # NHÓM 4: HYBRID - Nhân sự cụ thể CSDL VNTech + Đối chiếu Pháp luật (10 câu)
        # =========================================================================
        ("Anh Phạm Quốc Bảo ở phòng Kỹ thuật đang thử việc với mức lương 32 triệu thì mức lương này có đúng quy định thử việc của Bộ luật Lao động không?", "HYBRID"),
        ("Nếu anh Lê Hoàng Nam muốn đơn phương chấm dứt hợp đồng lao động thì theo quy định của luật cần phải báo trước bao nhiêu ngày?", "HYBRID"),
        ("Chị Bùi Thị Bích là kế toán trưởng của VNTech, nếu chị ấy sinh con thì chế độ thai sản theo luật BHXH được hưởng bao nhiêu tháng lương?", "HYBRID"),
        ("Anh Nguyễn Văn An là Tổng Giám đốc VNTech, nếu công ty giải thể thì quyền lợi và nghĩa vụ của người đại diện theo pháp luật được luật quy định ra sao?", "HYBRID"),
        ("Mức lương 70 triệu của anh Nguyễn Văn An phải đóng thuế thu nhập cá nhân theo biểu thuế lũy tiến như thế nào?", "HYBRID"),
        ("Công ty VNTech muốn chuyển bạn Đỗ Tiến Đạt từ Frontend sang làm việc khác thì theo Điều 29 Bộ luật Lao động cần đáp ứng điều kiện gì?", "HYBRID"),
        ("Trần Minh Đức hiện là Giám đốc Vận hành, nếu có tranh chấp hợp đồng lao động thì thời hiệu yêu cầu giải quyết tranh chấp là bao lâu?", "HYBRID"),
        ("Lương của chị Hoàng Thùy Linh là 30 triệu, nếu chị ấy làm thêm giờ vào ngày nghỉ lễ thì tiền lương làm thêm được tính theo luật lao động là bao nhiêu?", "HYBRID"),
        ("Nhân viên Ngô Gia Huy có thời hạn hợp đồng xác định thời hạn, công ty VNTech muốn sa thải thì phải tuân theo trình tự kỷ luật lao động nào?", "HYBRID"),
        ("Chị Vũ Hải Yến muốn thôi việc tại VNTech, tiền trợ cấp thôi việc của chị ấy theo quy định pháp luật được tính như thế nào?", "HYBRID"),

        # =========================================================================
        # NHÓM 5: CONTRACT_RISK - Thẩm định rủi ro hợp đồng (5 câu)
        # =========================================================================
        ("Hãy thẩm định rủi ro hợp đồng lao động này giúp tôi", "CONTRACT_RISK"),
        ("Rà soát hợp đồng mua bán và chỉ ra các bẫy pháp lý", "CONTRACT_RISK"),
        ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐiều 1. Công việc\nĐiều 2. Tiền lương\nBên A và Bên B thống nhất...", "CONTRACT_RISK"),
        ("Soi hợp đồng dịch vụ công nghệ thông tin xem có điều khoản phạt vi phạm trái luật không", "CONTRACT_RISK"),
        ("Báo cáo thẩm định rủi ro hợp đồng thuê mặt bằng kinh doanh", "CONTRACT_RISK"),

        # =========================================================================
        # NHÓM 6: GREETING - Lời chào & Giới thiệu trợ lý (5 câu)
        # =========================================================================
        ("Xin chào bạn!", "GREETING"),
        ("Hello bot, bạn là ai?", "GREETING"),
        ("Chào bạn, bạn có thể giúp được gì cho tôi?", "GREETING"),
        ("Good morning bot!", "GREETING"),
        ("hướng dẫn sử dụng", "GREETING"),
    ]

    print("=" * 85)
    print("           BỘ KIỂM THỬ QUERY ROUTER TOÀN DIỆN (50 CÂU HỎI THỰC TẾ)")
    print("=" * 85)

    passed = 0
    failed_cases = []

    for idx, (q, expected) in enumerate(test_suite, 1):
        intent = classify_query(q)
        ok = (intent == expected)
        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            status = f"❌ FAIL (Cần: {expected}, Ra: {intent})"
            failed_cases.append((idx, q, expected, intent))

        preview = q.replace("\n", " ")[:55]
        print(f"[{idx:>2}/50] {status:<30} | {intent:<13} | \"{preview}...\"")

    print("=" * 85)
    print(f"📊 KẾT QUẢ CUỐI CÙNG: {passed}/{len(test_suite)} bài test PASS ({passed/len(test_suite)*100:.1f}%)")

    if failed_cases:
        print("\n⚠️ CÁC TRƯỜNG HỢP CẦN ĐIỀU CHỈNH:")
        for f_idx, f_q, f_exp, f_got in failed_cases:
            print(f"  - Câu #{f_idx}: Kỳ vọng [{f_exp}], Thực tế [{f_got}] -> '{f_q}'")
    else:
        print("🎉 TẤT CẢ 50 CÂU HỎI ĐỀU ĐÃ ĐỊNH TUYẾN CHÍNH XÁC 100%!")


if __name__ == "__main__":
    run_router_tests()
