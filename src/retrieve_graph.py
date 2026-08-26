import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class LegalGraphRetriever:
    """Module truy vấn và duyệt đồ thị tri thức pháp luật Neo4j (Graph Traversal)."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j"
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self.database = database
        self.driver: Optional[Driver] = None
        self._connected = False
        self._init_connection()

    def _init_connection(self):
        if not NEO4J_AVAILABLE:
            print("⚠️  Thư viện 'neo4j' chưa cài đặt, Graph Retriever sẽ hoạt động ở chế độ fallback.")
            return

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._connected = True
            print(f"🌲 [GraphRAG] Kết nối Neo4j Graph Retriever thành công ({self.uri})")
        except Exception as e:
            print(f"⚠️  Không thể kết nối Neo4j ({e}). GraphRAG sẽ fallback sang tra cứu cục bộ.")
            self._connected = False

    def is_available(self) -> bool:
        return self._connected and self.driver is not None

    def expand_references(
        self,
        seed_chunk_ids: List[str],
        max_hops: int = 2,
        limit_per_seed: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Duyệt đồ thị mở rộng đa chặng (Multi-hop Graph Traversal) từ các Seed Chunks.
        Tìm các Chunks/Articles được dẫn chiếu trực tiếp hoặc gián tiếp.
        """
        if not self.is_available() or not seed_chunk_ids:
            return []

        cypher_query = f"""
        UNWIND $seed_ids AS seed_id
        MATCH (seed:Chunk {{chunk_id: seed_id}})
        OPTIONAL MATCH (seed)-[:REFERENCES*1..{max_hops}]->(target)
        OPTIONAL MATCH (target)-[:CONTAINS*0..1]->(target_chunk:Chunk)
        WITH COALESCE(target_chunk, CASE WHEN target:Chunk THEN target ELSE NULL END) AS res_node, seed
        WHERE res_node IS NOT NULL AND NOT res_node.chunk_id IN $seed_ids
        RETURN DISTINCT
            res_node.chunk_id AS chunk_id,
            res_node.content AS content,
            res_node.clause AS clause,
            res_node.point AS point,
            res_node.status AS status,
            1 AS distance
        LIMIT $total_limit
        """
        total_limit = len(seed_chunk_ids) * limit_per_seed
        expanded_chunks = []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    cypher_query,
                    seed_ids=seed_chunk_ids,
                    total_limit=total_limit
                )
                for record in result:
                    expanded_chunks.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "clause": record["clause"],
                        "point": record["point"],
                        "status": record["status"],
                        "graph_distance": record["distance"]
                    })
        except Exception as e:
            print(f"Lỗi khi thực thi Cypher query: {e}")

        return expanded_chunks

    def get_document_hierarchy(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Lấy bối cảnh thứ bậc hoàn chỉnh từ Chunk ngược lên Article -> Document."""
        if not self.is_available():
            return None

        cypher_query = """
        MATCH (d:Document)-[:CONTAINS]->(a:Article)-[:CONTAINS]->(c:Chunk {chunk_id: $chunk_id})
        RETURN
            d.document_id AS doc_id,
            d.document_number AS doc_number,
            d.document_type AS doc_type,
            d.title AS doc_title,
            d.effective_date AS doc_effective_date,
            a.article_number AS article_number,
            a.article_title AS article_title,
            a.chapter AS chapter_number,
            a.chapter_name AS chapter_name,
            c.clause AS clause,
            c.content AS content
        """
        try:
            with self.driver.session(database=self.database) as session:
                record = session.run(cypher_query, chunk_id=chunk_id).single()
                if record:
                    return dict(record)
        except Exception as e:
            print(f"Lỗi truy vấn bối cảnh thứ bậc: {e}")
        return None

    def close(self):
        if self.driver:
            self.driver.close()
