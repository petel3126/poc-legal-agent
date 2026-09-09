import os
import sys
import json
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

    def expand_with_ranking(
        self,
        seed_chunk_ids: List[str],
        missing_elements: Optional[List[str]] = None,
        query_text: str = "",
        top_n: int = 3,
        min_score_threshold: float = 0.85,
        all_chunks_by_id: Optional[Dict[str, dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Duyệt đồ thị có hướng kết hợp chấm điểm ứng viên (Scored Directional Graph Expansion):
        1. Hierarchical Article Bundling: Mở rộng hạt giống sang các Khoản anh em trong cùng một Điều.
        2. Phân loại trọng số:
           - OUTBOUND: (seed)-[:REFERENCES]->(target) -> Trọng số 1.0
           - GUIDED_BY: (seed)-[:GUIDED_BY*1..2]-(target) -> Trọng số 0.8
           - INBOUND: (seed)<-[:REFERENCES]-(target) -> Trọng số 0.7
           - SIBLING: Cùng Article -> Trọng số 0.7 (nếu có từ khóa) / 0.3 (fallback)
        3. Semantic Relevance Filter: Bắt buộc INBOUND / SIBLING phải có match_count >= 1.
        4. Min Score Threshold: Loại bỏ 100% ứng viên có điểm < min_score_threshold (mặc định 0.85).
        """
        if not seed_chunk_ids:
            return []

        # -------------------------------------------------------------
        # TRỤ CỘT 1: HIERARCHICAL ARTICLE BUNDLING (GOM KHOẢN CÙNG ĐIỀU)
        # -------------------------------------------------------------
        expanded_seed_ids = list(seed_chunk_ids)
        if all_chunks_by_id:
            for sid in seed_chunk_ids[:4]:
                sc = all_chunks_by_id.get(sid)
                if not sc:
                    continue
                doc_id = sc.get("document_id")
                art_num = sc.get("article")
                if not doc_id or not art_num:
                    continue
                # Tìm các Khoản anh em trong cùng một Điều
                for cid, c in all_chunks_by_id.items():
                    if cid not in expanded_seed_ids and c.get("document_id") == doc_id and c.get("article") == art_num:
                        expanded_seed_ids.append(cid)

        candidates_map: Dict[str, Dict[str, Any]] = {}
        missing_keywords = []
        if missing_elements:
            for me in missing_elements:
                if me == "consequence":
                    missing_keywords.extend(["bồi thường", "phạt", "chịu trách nhiệm", "xử phạt", "chế tài", "nghĩa vụ", "vô hiệu", "đơn phương chấm dứt", "chấm dứt hợp đồng"])
                elif me == "action_condition":
                    missing_keywords.extend(["điều kiện", "thủ tục", "thời hạn", "trường hợp", "quy trình", "hành vi"])
                elif me == "subject":
                    missing_keywords.extend(["người lao động", "người sử dụng lao động", "doanh nghiệp", "bên bán", "bên mua"])
                else:
                    missing_keywords.append(me.lower())

        if query_text:
            # Lọc các từ khóa có nghĩa từ câu hỏi (loại bỏ hư từ)
            stop_words = {"nhưng", "trong", "ngoài", "được", "người", "những", "cho", "với", "theo", "này", "của", "và"}
            words = [w for w in query_text.lower().split() if len(w) >= 3 and w not in stop_words]
            missing_keywords.extend(words[:12])

        # Loại bỏ trùng lặp từ khóa
        missing_keywords = list(dict.fromkeys(missing_keywords))

        # -------------------------------------------------------------
        # 1. TRUY VẤN QUA NEO4J NẾU KHẢ DỤNG
        # -------------------------------------------------------------
        if self.is_available():
            scored_cypher = """
            UNWIND $seed_ids AS seed_id
            MATCH (seed:Chunk {chunk_id: seed_id})
            
            // 1. OUTBOUND (seed -> target) W=1.0
            OPTIONAL MATCH (seed)-[:REFERENCES]->(t_out)
            OPTIONAL MATCH (t_out)-[:CONTAINS*0..1]->(chk_out:Chunk)
            WITH seed, COALESCE(chk_out, CASE WHEN t_out:Chunk THEN t_out ELSE NULL END) AS out_node
            
            // 2. INBOUND (target -> seed) W=0.7
            OPTIONAL MATCH (seed)<-[:REFERENCES]-(t_in)
            OPTIONAL MATCH (t_in)-[:CONTAINS*0..1]->(chk_in:Chunk)
            WITH seed, out_node, COALESCE(chk_in, CASE WHEN t_in:Chunk THEN t_in ELSE NULL END) AS in_node
            
            RETURN
                out_node.chunk_id AS out_id, out_node.content AS out_content, out_node.status AS out_status,
                in_node.chunk_id AS in_id, in_node.content AS in_content, in_node.status AS in_status
            LIMIT 40
            """
            try:
                with self.driver.session(database=self.database) as session:
                    res = session.run(scored_cypher, seed_ids=expanded_seed_ids)
                    for rec in res:
                        if rec["out_id"] and rec["out_id"] not in seed_chunk_ids:
                            candidates_map[rec["out_id"]] = {
                                "chunk_id": rec["out_id"],
                                "content": rec["out_content"] or "",
                                "rel_type": "OUTBOUND",
                                "base_weight": 1.0
                            }
                        if rec["in_id"] and rec["in_id"] not in seed_chunk_ids and rec["in_id"] not in candidates_map:
                            candidates_map[rec["in_id"]] = {
                                "chunk_id": rec["in_id"],
                                "content": rec["in_content"] or "",
                                "rel_type": "INBOUND",
                                "base_weight": 0.7
                            }
            except Exception as e:
                print(f"[GraphRAG] Lỗi truy vấn đa chiều Neo4j: {e}. Tự động fallback sang In-Memory.")

        # -------------------------------------------------------------
        # 2. IN-MEMORY FALLBACK VÀ BỔ SUNG SIBLING BUNDLING
        # -------------------------------------------------------------
        if all_chunks_by_id:
            seed_set = set(seed_chunk_ids)

            # Thêm các Khoản anh em (Siblings) vào candidates_map nếu có liên quan
            for sid in seed_chunk_ids[:4]:
                sc = all_chunks_by_id.get(sid)
                if not sc:
                    continue
                doc_id = sc.get("document_id")
                art_num = sc.get("article")
                if not doc_id or not art_num:
                    continue
                for cid, c in all_chunks_by_id.items():
                    if cid not in seed_set and cid not in candidates_map and c.get("document_id") == doc_id and c.get("article") == art_num:
                        candidates_map[cid] = {
                            "chunk_id": cid,
                            "content": c.get("content", ""),
                            "rel_type": "SIBLING",
                            "base_weight": 0.7
                        }

            # A. OUTBOUND
            for sid in expanded_seed_ids:
                sc = all_chunks_by_id.get(sid)
                if not sc:
                    continue
                for ref_id in sc.get("references", []):
                    if ref_id not in seed_set and ref_id in all_chunks_by_id and ref_id not in candidates_map:
                        candidates_map[ref_id] = {
                            "chunk_id": ref_id,
                            "content": all_chunks_by_id[ref_id].get("content", ""),
                            "rel_type": "OUTBOUND",
                            "base_weight": 1.0
                        }

            # B. INBOUND (quét các chunk khác dẫn chiếu tới bất kỳ seed nào)
            for cid, c in all_chunks_by_id.items():
                if cid in seed_set or cid in candidates_map:
                    continue
                refs = c.get("references", [])
                if any(sid in refs for sid in expanded_seed_ids):
                    candidates_map[cid] = {
                        "chunk_id": cid,
                        "content": c.get("content", ""),
                        "rel_type": "INBOUND",
                        "base_weight": 0.7
                    }

        # -------------------------------------------------------------
        # 3. GRAPH RELEVANCE SCORING, THRESHOLD FILTER & RANKING
        # -------------------------------------------------------------
        scored_results = []
        for cid, cand in candidates_map.items():
            content_lower = cand["content"].lower()
            base_w = cand["base_weight"]

            # Đếm số từ khóa câu hỏi / missing elements khớp trong nội dung
            match_count = sum(1 for kw in missing_keywords if kw in content_lower)

            # LỌC NGHIÊM NGẶT: Nếu là quan hệ INBOUND hoặc SIBLING mà không khớp bất kỳ từ khóa nào -> BỎ QUA NGAY
            if cand["rel_type"] in ("INBOUND", "SIBLING") and match_count == 0:
                continue

            semantic_multiplier = 1.0 + (min(match_count, 5) * 0.25)
            final_score = round(base_w * semantic_multiplier, 4)

            # LỌC THEO NGƯỠNG SÀN (MIN SCORE THRESHOLD)
            if final_score < min_score_threshold:
                continue

            # Lấy thông tin chunk hoàn chỉnh từ all_chunks_by_id nếu có
            chunk_data = all_chunks_by_id.get(cid, cand).copy() if all_chunks_by_id else cand.copy()
            chunk_data["graph_rel_type"] = cand["rel_type"]
            chunk_data["graph_score"] = final_score
            chunk_data["graph_match_count"] = match_count

            scored_results.append(chunk_data)

        # Sắp xếp theo Graph Relevance Score giảm dần
        scored_results.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
        return scored_results[:top_n]

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


_GLOBAL_GRAPH_RETRIEVER: Optional[LegalGraphRetriever] = None
_CACHED_ALL_CHUNKS_BY_ID: Optional[Dict[str, dict]] = None


def get_default_graph_retriever() -> LegalGraphRetriever:
    """Trả về instance singleton của LegalGraphRetriever."""
    global _GLOBAL_GRAPH_RETRIEVER
    if _GLOBAL_GRAPH_RETRIEVER is None:
        _GLOBAL_GRAPH_RETRIEVER = LegalGraphRetriever()
    return _GLOBAL_GRAPH_RETRIEVER


def get_all_chunks_by_id() -> Dict[str, dict]:
    """Lazy load toàn bộ chunks theo chunk_id phục vụ fallback duyệt đồ thị."""
    global _CACHED_ALL_CHUNKS_BY_ID
    if _CACHED_ALL_CHUNKS_BY_ID is None:
        chunks_file = Path(__file__).resolve().parent.parent / "data" / "processed" / "legal_chunks.json"
        if chunks_file.exists():
            try:
                data = json.loads(chunks_file.read_text(encoding="utf-8"))
                _CACHED_ALL_CHUNKS_BY_ID = {c["chunk_id"]: c for c in data if "chunk_id" in c}
            except Exception as e:
                print(f"Lỗi khi đọc file chunks: {e}")
                _CACHED_ALL_CHUNKS_BY_ID = {}
        else:
            _CACHED_ALL_CHUNKS_BY_ID = {}
    return _CACHED_ALL_CHUNKS_BY_ID

