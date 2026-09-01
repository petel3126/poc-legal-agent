"""
Quản lý kết nối tới PostgreSQL Database (Supabase) với tự động xử lý IPv4 Pooler.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None


def get_connection_url() -> str:
    """Tự động tìm kiếm Database URL từ file .env hoặc biến môi trường."""
    candidate_keys = [
        "SUPABASE_DB_URL",
        "DATABASE_URL",
        "POSTGRES_URL",
        "DB_URL",
        "SUPABASE_URL",
        "POSTGRESQL_URL",
    ]
    for key in candidate_keys:
        val = os.getenv(key)
        if val and ("postgres://" in val or "postgresql://" in val):
            return val

    # Quét trực tiếp file .env nếu biến môi trường chưa load kịp
    if ENV_PATH.exists():
        content = ENV_PATH.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip("'").strip('"')
                if "postgresql://" in v or "postgres://" in v:
                    return v
    return None


def get_db_connection():
    """
    Trả về đối tượng connection tới PostgreSQL Supabase.
    Tự động fallback sang IPv4 Connection Pooler nếu URL trực tiếp bị lỗi DNS IPv6.
    """
    db_url = get_connection_url()
    if not db_url:
        print("❌ Lỗi: Không tìm thấy chuỗi kết nối Database trong file .env!")
        return None

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    urls_to_try = [db_url]
    if "db.xukkwmnxpwjjeuyoacrx.supabase.co" in db_url:
        pooler_url = db_url.replace(
            "db.xukkwmnxpwjjeuyoacrx.supabase.co",
            "aws-0-ap-northeast-2.pooler.supabase.com"
        ).replace("postgres:", "postgres.xukkwmnxpwjjeuyoacrx:")
        urls_to_try.append(pooler_url)

    last_error = None
    for url in urls_to_try:
        try:
            conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=10)
            return conn
        except Exception as e:
            last_error = e

    print(f"❌ Không thể kết nối tới cơ sở dữ liệu: {last_error}")
    return None
