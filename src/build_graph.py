import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Reconfigure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    print("Lỗi: Thư viện 'neo4j' chưa được cài đặt. Vui lòng chạy: pip install neo4j")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "legal_chunks.json"


def get_neo4j_driver(uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> Driver:
    """Khởi tạo kết nối Neo4j từ tham số hoặc biến môi trường."""
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
    password = password or os.getenv("NEO4J_PASSWORD", "password")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Kiểm tra kết nối
        driver.verify_connectivity()
        print(f"✅ Kết nối Neo4j thành công tại: {uri} (User: {user})")
        return driver
    except Exception as e:
        print(f"\n❌ Không thể kết nối tới Neo4j tại '{uri}': {e}")
        print("\n💡 Hướng dẫn khởi động Neo4j:")
        print("  1. Dùng Docker (khuyên dùng):")
        print("     docker run -d --name neo4j-legal -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        print("  2. Hoặc Neo4j Desktop: Khởi động Local DBMS.")
        print("  3. Cấu hình biến môi trường trong .env nếu cần:")
        print("     NEO4J_URI=bolt://localhost:7687")
        print("     NEO4J_USERNAME=neo4j")
        print("     NEO4J_PASSWORD=your_password\n")
        sys.exit(1)


def setup_constraints_and_indexes(driver: Driver, database: str = "neo4j"):
    """Tạo Constraints và Indexes để tăng tốc tối đa tốc độ tra cứu và nạp dữ liệu."""
    queries = [
        "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE;",
        "CREATE CONSTRAINT article_id_unique IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE;",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;",
        "CREATE INDEX doc_number_idx IF NOT EXISTS FOR (d:Document) ON (d.document_number);",
        "CREATE INDEX article_doc_idx IF NOT EXISTS FOR (a:Article) ON (a.document_id);",
        "CREATE INDEX chunk_status_idx IF NOT EXISTS FOR (c:Chunk) ON (c.status);"
    ]
    with driver.session(database=database) as session:
        for q in queries:
            session.run(q)
    print("✅ Đã thiết lập Constraints & Indexes trên Neo4j.")


def clear_existing_graph(driver: Driver, database: str = "neo4j"):
    """Xóa toàn bộ đồ thị pháp luật cũ nếu được yêu cầu."""
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n;")
    print("🧹 Đã làm sạch toàn bộ dữ liệu đồ thị cũ.")


def build_legal_graph(
    driver: Driver,
    chunks_path: Path = PROCESSED_CHUNKS_PATH,
    batch_size: int = 500,
    database: str = "neo4j"
):
    """Đọc dữ liệu chunks và nạp đồ thị thứ bậc + liên kết dẫn chiếu vào Neo4j."""
    if not chunks_path.exists():
        print(f"❌ Không tìm thấy file: {chunks_path}")
        return

    print(f"Đang đọc dữ liệu từ {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_chunks = len(chunks)
    print(f"Tổng số chunks cần nạp: {total_chunks:,}")

    # 1. Trích xuất danh sách Documents duy nhất
    documents_dict = {}
    articles_dict = {}
    chunks_list = []
    references_list = []

    for c in chunks:
        doc_id = c.get("document_id", "UNKNOWN")
        if doc_id not in documents_dict:
            documents_dict[doc_id] = {
                "document_id": doc_id,
                "document_number": c.get("document_number", ""),
                "document_type": c.get("document_type", ""),
                "title": c.get("title", ""),
                "issue_date": c.get("issue_date", ""),
                "effective_date": c.get("effective_date", ""),
                "status": c.get("status", "ACTIVE")
            }

        # Article ID quy ước: {doc_id}-đ{article_number}
        art_num = str(c.get("article", "")).strip()
        art_id = f"{doc_id}-đ{art_num}" if art_num else f"{doc_id}-art-unknown"
        if art_id not in articles_dict:
            articles_dict[art_id] = {
                "article_id": art_id,
                "document_id": doc_id,
                "article_number": art_num,
                "article_title": c.get("article_title", ""),
                "chapter": c.get("chapter", ""),
                "chapter_name": c.get("chapter_name", "")
            }

        # Chunk node
        chunk_id = c["chunk_id"]
        chunks_list.append({
            "chunk_id": chunk_id,
            "article_id": art_id,
            "document_id": doc_id,
            "clause": str(c.get("clause", "")),
            "point": str(c.get("point", "")),
            "content": c.get("content", ""),
            "status": c.get("status", "ACTIVE"),
            "token_count": c.get("token_count", 0)
        })

        # Dẫn chiếu pháp lý (References)
        for ref in c.get("references", []):
            if ref and ref != chunk_id:
                references_list.append({
                    "source_chunk_id": chunk_id,
                    "target_ref": ref
                })

    print(f"-> Thống kê đối tượng:")
    print(f"   • Documents: {len(documents_dict):,} nodes")
    print(f"   • Articles:  {len(articles_dict):,} nodes")
    print(f"   • Chunks:    {len(chunks_list):,} nodes")
    print(f"   • Quan hệ dẫn chiếu (References): {len(references_list):,} edges")

    with driver.session(database=database) as session:
        # 1. Ingest Documents
        print("\n[1/4] Đang nạp Documents vào Neo4j...")
        doc_query = """
        UNWIND $batch AS doc
        MERGE (d:Document {document_id: doc.document_id})
        SET d.document_number = doc.document_number,
            d.document_type = doc.document_type,
            d.title = doc.title,
            d.issue_date = doc.issue_date,
            d.effective_date = doc.effective_date,
            d.status = doc.status;
        """
        session.run(doc_query, batch=list(documents_dict.values()))

        # 2. Ingest Articles và liên kết (:Document)-[:CONTAINS]->(:Article)
        print("[2/4] Đang nạp Articles và liên kết với Documents...")
        art_query = """
        UNWIND $batch AS art
        MERGE (a:Article {article_id: art.article_id})
        SET a.document_id = art.document_id,
            a.article_number = art.article_number,
            a.article_title = art.article_title,
            a.chapter = art.chapter,
            a.chapter_name = art.chapter_name
        WITH a, art
        MATCH (d:Document {document_id: art.document_id})
        MERGE (d)-[:CONTAINS]->(a);
        """
        articles_list = list(articles_dict.values())
        for i in range(0, len(articles_list), batch_size):
            session.run(art_query, batch=articles_list[i:i + batch_size])

        # 3. Ingest Chunks và liên kết (:Article)-[:CONTAINS]->(:Chunk)
        print("[3/4] Đang nạp Chunks và liên kết với Articles...")
        chunk_query = """
        UNWIND $batch AS chk
        MERGE (c:Chunk {chunk_id: chk.chunk_id})
        SET c.clause = chk.clause,
            c.point = chk.point,
            c.content = chk.content,
            c.status = chk.status,
            c.token_count = chk.token_count
        WITH c, chk
        MATCH (a:Article {article_id: chk.article_id})
        MERGE (a)-[:CONTAINS]->(c);
        """
        for i in range(0, len(chunks_list), batch_size):
            session.run(chunk_query, batch=chunks_list[i:i + batch_size])
            print(f"   Đã nạp {min(i + batch_size, len(chunks_list)):,}/{len(chunks_list):,} chunks...", end="\r")
        print()

        # 4. Ingest Dẫn chiếu (:Chunk)-[:REFERENCES]->(:Chunk / :Article)
        print("[4/4] Đang tạo quan hệ dẫn chiếu (:REFERENCES)...")
        ref_query = """
        UNWIND $batch AS ref
        MATCH (source:Chunk {chunk_id: ref.source_chunk_id})
        OPTIONAL MATCH (target_chunk:Chunk {chunk_id: ref.target_ref})
        OPTIONAL MATCH (target_art:Article {article_id: ref.target_ref})
        WITH source, COALESCE(target_chunk, target_art) AS target
        WHERE target IS NOT NULL
        MERGE (source)-[:REFERENCES]->(target);
        """
        for i in range(0, len(references_list), batch_size):
            session.run(ref_query, batch=references_list[i:i + batch_size])
            print(f"   Đã nạp {min(i + batch_size, len(references_list)):,}/{len(references_list):,} liên kết...", end="\r")
        print()

    print("\n🎉 HOÀN TẤT XÂY DỰNG ĐỒ THỊ PHÁP LUẬT TRÊN NEO4J!")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest dữ liệu Pháp luật vào Neo4j Knowledge Graph")
    parser.add_argument("--uri", type=str, default=None, help="Neo4j Connection URI (vd: bolt://localhost:7687)")
    parser.add_argument("--user", type=str, default=None, help="Neo4j Username (mặc định: neo4j)")
    parser.add_argument("--password", type=str, default=None, help="Neo4j Password")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size khi nạp Cypher (mặc định: 500)")
    parser.add_argument("--clear", action="store_true", help="Xóa sạch dữ liệu đồ thị cũ trước khi nạp mới")
    parser.add_argument("--database", type=str, default="neo4j", help="Tên database (mặc định: neo4j)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    driver = get_neo4j_driver(uri=args.uri, user=args.user, password=args.password)
    
    if args.clear:
        clear_existing_graph(driver, database=args.database)

    setup_constraints_and_indexes(driver, database=args.database)
    build_legal_graph(driver, batch_size=args.batch_size, database=args.database)
    driver.close()
