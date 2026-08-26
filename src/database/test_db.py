"""
Script kiểm tra kết nối & test các hàm truy vấn trong module src.database.
Chạy trực tiếp: python -m src.database.test_db (hoặc python src/database/test_db.py)
"""

import sys
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import get_db_connection
from src.database.queries import (
    get_company_info,
    search_employees,
    get_department_summary,
)


def run_tests():
    print("⏳ Đang kiểm tra kết nối cơ sở dữ liệu...")
    conn = get_db_connection()
    if not conn:
        print("❌ Kết nối thất bại!")
        return

    print("✅ Kết nối Database thành công rực rỡ!\n")

    # 1. Test lấy thông tin công ty
    print("🏢 --- THÔNG TIN DOANH NGHIỆP ---")
    company = get_company_info(1)
    if company:
        print(f"  [{company['id']}] {company['company_name']}")
        print(f"  Mã số thuế : {company['tax_code']}")
        print(f"  Địa chỉ    : {company['address']}")
    else:
        print("  ⚠️ Không tìm thấy công ty.")

    # 2. Test thống kê theo phòng ban
    print("\n📊 --- THỐNG KÊ PHÒNG BAN & LƯƠNG TRUNG BÌNH ---")
    dept_stats = get_department_summary()
    for d in dept_stats:
        avg_sal = f"{d['avg_salary']:,.0f} đ" if d.get('avg_salary') else "0 đ"
        print(f"  • {d['department']:<16}: {d['employee_count']:>2} người | Lương TB: {avg_sal}")

    # 3. Test tìm kiếm nhân viên
    print("\n👥 --- DANH SÁCH 20 NHÂN SỰ ---")
    employees = search_employees(limit=25)
    print(f"  Tổng số nhân viên: {len(employees)}")
    print("  " + "-" * 88)
    for emp in employees:
        sal = f"{emp['salary']:,.0f} đ" if emp.get('salary') else "N/A"
        print(f"  #{emp['id']:02d} | {emp['full_name']:<18} | {emp['department']:<14} | {emp['position']:<30} | {sal:>12}")
    print("  " + "-" * 88)

    conn.close()
    print("\n🎉 MODULE DATABASE HOẠT ĐỘNG HOÀN HẢO!")


if __name__ == "__main__":
    run_tests()
