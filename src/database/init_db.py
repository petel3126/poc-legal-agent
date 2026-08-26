"""
Script tự động tạo bảng companies, employees và nạp sẵn 20 nhân sự lên Supabase.
Chạy trực tiếp: python -m src.database.init_db (hoặc python src/database/init_db.py)
"""

import sys
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm root vào sys.path để chạy standalone
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import get_db_connection

SQL_INIT = """
-- 1. Tạo bảng Doanh nghiệp
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    tax_code VARCHAR(50) UNIQUE,
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Tạo bảng Nhân viên
CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id),
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    department VARCHAR(100),
    position VARCHAR(100),
    salary NUMERIC(15, 2),
    hire_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Xóa dữ liệu cũ nếu có
TRUNCATE TABLE employees, companies RESTART IDENTITY CASCADE;

-- 3. Thêm 1 Doanh nghiệp
INSERT INTO companies (id, company_name, tax_code, address) VALUES
(1, 'Công ty Cổ phần Giải pháp Công nghệ VNTech', '0108889999', 'Tầng 12, Tòa nhà Keangnam Landmark 72, Nam Từ Liêm, Hà Nội');

-- 4. Thêm 20 Nhân sự
INSERT INTO employees (company_id, full_name, email, phone, department, position, salary, hire_date, is_active) VALUES
-- Ban Giám đốc
(1, 'Nguyễn Văn An', 'an.nv@vntech.vn', '0901234501', 'Ban Giám đốc', 'Tổng Giám đốc (CEO)', 70000000, '2019-01-02', TRUE),
(1, 'Trần Minh Đức', 'duc.tm@vntech.vn', '0901234502', 'Ban Giám đốc', 'Giám đốc Vận hành (COO)', 55000000, '2019-03-15', TRUE),

-- Phòng Kỹ thuật (IT & AI)
(1, 'Lê Hoàng Nam', 'nam.lh@vntech.vn', '0901234503', 'Kỹ thuật', 'Trưởng phòng Kỹ thuật (Tech Lead)', 45000000, '2019-06-01', TRUE),
(1, 'Phạm Quốc Bảo', 'bao.pq@vntech.vn', '0901234504', 'Kỹ thuật', 'Kỹ sư AI & NLP', 32000000, '2021-02-10', TRUE),
(1, 'Vũ Hải Yến', 'yen.vh@vntech.vn', '0901234505', 'Kỹ thuật', 'Lập trình viên Backend Senior', 28000000, '2020-08-20', TRUE),
(1, 'Đỗ Tiến Đạt', 'dat.dt@vntech.vn', '0901234506', 'Kỹ thuật', 'Lập trình viên Frontend', 20000000, '2022-04-15', TRUE),
(1, 'Ngô Gia Huy', 'huy.ng@vntech.vn', '0901234507', 'Kỹ thuật', 'Kỹ sư Mobile App (Flutter)', 22000000, '2022-09-01', TRUE),
(1, 'Trịnh Văn Cường', 'cuong.tv@vntech.vn', '0901234508', 'Kỹ thuật', 'Kỹ sư DevOps / Cloud', 30000000, '2021-11-15', TRUE),
(1, 'Lý Phương Thảo', 'thao.lp@vntech.vn', '0901234509', 'Kỹ thuật', 'Chuyên viên Kiểm thử (QA/QC)', 17000000, '2023-03-01', TRUE),

-- Phòng Nhân sự (HR)
(1, 'Hoàng Thùy Linh', 'linh.ht@vntech.vn', '0901234510', 'Nhân sự', 'Trưởng phòng Nhân sự', 30000000, '2019-10-01', TRUE),
(1, 'Nguyễn Kiều Trang', 'trang.nk@vntech.vn', '0901234511', 'Nhân sự', 'Chuyên viên Tuyển dụng (Talent Acquisition)', 16000000, '2022-05-10', TRUE),
(1, 'Tạ Thị Thanh', 'thanh.tt@vntech.vn', '0901234512', 'Nhân sự', 'Chuyên viên C&B (Lương & Phúc lợi)', 18000000, '2021-08-01', TRUE),

-- Phòng Kế toán & Tài chính
(1, 'Bùi Thị Bích', 'bich.bt@vntech.vn', '0901234513', 'Kế toán', 'Kế toán trưởng', 32000000, '2019-05-15', TRUE),
(1, 'Đặng Thái Sơn', 'son.dt@vntech.vn', '0901234514', 'Kế toán', 'Kế toán viên Tổng hợp', 15000000, '2023-01-10', TRUE),

-- Phòng Kinh doanh (Sales)
(1, 'Dương Đình Trí', 'tri.dd@vntech.vn', '0901234515', 'Kinh doanh', 'Trưởng phòng Kinh doanh B2B', 35000000, '2020-03-01', TRUE),
(1, 'Võ Minh Thắng', 'thang.vm@vntech.vn', '0901234516', 'Kinh doanh', 'Chuyên viên Kinh doanh Doanh nghiệp', 18000000, '2022-07-15', TRUE),

-- Phòng Marketing & Thiết kế
(1, 'Phan Mai Hương', 'huong.pm@vntech.vn', '0901234517', 'Marketing', 'Trưởng phòng Marketing', 28000000, '2021-04-01', TRUE),
(1, 'Trịnh Kim Oanh', 'oanh.tk@vntech.vn', '0901234518', 'Marketing', 'Senior UI/UX Designer', 24000000, '2021-10-20', TRUE),
(1, 'Lưu Hải Đăng', 'dang.lh@vntech.vn', '0901234519', 'Marketing', 'Chuyên viên Digital Marketing & SEO', 15000000, '2023-06-01', TRUE),

-- Phòng Pháp chế & Hành chính
(1, 'Hồ Ngọc Hà', 'ha.hn@vntech.vn', '0901234520', 'Pháp chế', 'Chuyên viên Pháp chế Doanh nghiệp', 22000000, '2022-01-15', TRUE);
"""


def init_database():
    conn = get_db_connection()
    if not conn:
        print("❌ Không thể kết nối tới Database để khởi tạo!")
        return

    try:
        cursor = conn.cursor()
        print("⏳ Đang tạo bảng và nạp 20 nhân viên lên Database...")
        cursor.execute(SQL_INIT)
        conn.commit()
        cursor.close()
        conn.close()
        print("🎉 NẠP DỮ LIỆU THÀNH CÔNG LÊN DATABASE!")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo dữ liệu: {e}")


if __name__ == "__main__":
    init_database()
