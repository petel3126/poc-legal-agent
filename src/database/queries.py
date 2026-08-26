"""
Các hàm nghiệp vụ truy vấn thông tin Doanh nghiệp & Nhân viên phục vụ Chatbot.
"""

from typing import List, Dict, Optional, Any
from .connection import get_db_connection


def get_company_info(company_id: int = 1) -> Optional[Dict[str, Any]]:
    """Lấy thông tin chung của doanh nghiệp."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, company_name, tax_code, address FROM companies WHERE id = %s;", (company_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
        return dict(res) if res else None
    except Exception as e:
        print(f"Lỗi khi lấy thông tin công ty: {e}")
        return None


def search_employees(
    keyword: Optional[str] = None,
    department: Optional[str] = None,
    position: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Tìm kiếm nhân viên theo từ khóa (tên/email/sđt), phòng ban hoặc chức vụ.
    """
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        query = """
            SELECT e.id, e.full_name, e.email, e.phone, e.department, e.position, 
                   e.salary, e.hire_date, e.is_active, c.company_name
            FROM employees e
            LEFT JOIN companies c ON e.company_id = c.id
            WHERE (%s IS NULL OR e.full_name ILIKE %s OR e.email ILIKE %s OR e.phone ILIKE %s)
              AND (%s IS NULL OR e.department ILIKE %s)
              AND (%s IS NULL OR e.position ILIKE %s)
            ORDER BY e.id ASC
            LIMIT %s;
        """
        kw = f"%{keyword}%" if keyword else None
        dept = f"%{department}%" if department else None
        pos = f"%{position}%" if position else None

        cursor.execute(query, (kw, kw, kw, kw, dept, dept, pos, pos, limit))
        results = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Lỗi khi tìm kiếm nhân viên: {e}")
        return []


def get_employee_by_id(employee_id: int) -> Optional[Dict[str, Any]]:
    """Lấy thông tin chi tiết một nhân viên theo ID."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.*, c.company_name
            FROM employees e
            LEFT JOIN companies c ON e.company_id = c.id
            WHERE e.id = %s;
        """, (employee_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
        return dict(res) if res else None
    except Exception as e:
        print(f"Lỗi khi lấy thông tin nhân viên {employee_id}: {e}")
        return None


def get_department_summary() -> List[Dict[str, Any]]:
    """Thống kê tổng số lượng nhân sự và mức lương trung bình theo từng phòng ban."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT department, COUNT(*) as employee_count, AVG(salary) as avg_salary,
                   MIN(salary) as min_salary, MAX(salary) as max_salary
            FROM employees
            WHERE is_active = TRUE
            GROUP BY department
            ORDER BY employee_count DESC;
        """)
        results = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Lỗi khi thống kê phòng ban: {e}")
        return []


def execute_custom_query(sql_query: str) -> List[Dict[str, Any]]:
    """
    Thực thi câu truy vấn SQL an toàn (chỉ cho phép SELECT có LIMIT).
    """
    clean_sql = sql_query.strip()
    if not clean_sql.upper().startswith("SELECT"):
        raise ValueError("Chỉ cho phép thực thi các câu lệnh SELECT (Read-only)!")

    # Chặn các từ khóa phá hoại
    forbidden = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "--", ";"]
    for kw in forbidden:
        if kw in clean_sql.upper():
            raise ValueError(f"Câu lệnh chứa từ khóa không an toàn: '{kw}'")

    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(clean_sql)
        results = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Lỗi khi thực thi custom query: {e}")
        return []
