"""
Database Module for HRM / Company Information (PostgreSQL / Supabase).
"""

from .connection import get_db_connection
from .queries import (
    get_company_info,
    search_employees,
    get_employee_by_id,
    get_department_summary,
    execute_custom_query,
)

__all__ = [
    "get_db_connection",
    "get_company_info",
    "search_employees",
    "get_employee_by_id",
    "get_department_summary",
    "execute_custom_query",
]
