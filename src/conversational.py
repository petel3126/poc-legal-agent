"""
Module Quản lý Hội thoại & Xử lý Truy vấn Đa lượt (Conversational Legal RAG).
Cung cấp:
1. ConversationTurn & SessionManager: Quản lý lịch sử hội thoại stateful (lưu user_query, rewritten_query, retrieved_chunks, answer).
2. Follow-up Detector (Hybrid Rule + Fast LLM): Phân loại câu hỏi thành Type A (Explanation), Type B (Contextual Follow-up), Type C (Topic Shift / Independent).
3. Legal Context-Aware Rewriter: Tái cấu trúc truy vấn đảm bảo nguyên tắc 'Resolve context, but do not introduce new legal facts/assumptions'.
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.llm_generation import get_gemini_client

try:
    from google.genai import types
except ImportError:
    types = None


@dataclass
class ConversationTurn:
    """Lưu trữ đầy đủ trạng thái của một lượt hội thoại."""
    user_query: str
    rewritten_query: str = ""
    query_type: str = "INDEPENDENT"  # INDEPENDENT | TYPE_A_EXPLANATION | TYPE_B_FOLLOWUP | TYPE_C_TOPIC_SHIFT
    intent: str = "LEGAL_RAG"        # LEGAL_RAG | HR_DATABASE | CONTRACT_RISK | GREETING | HYBRID
    retrieved_chunks: List[dict] = field(default_factory=list)
    expanded_chunks: List[dict] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    answer: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "user_query": self.user_query,
            "rewritten_query": self.rewritten_query,
            "query_type": self.query_type,
            "intent": self.intent,
            "citations": self.citations,
            "retrieved_chunks_count": len(self.retrieved_chunks),
            "expanded_chunks_count": len(self.expanded_chunks),
            "answer": self.answer[:150] + "..." if len(self.answer) > 150 else self.answer,
            "timestamp": self.timestamp
        }


def extract_cited_articles(text: str) -> List[str]:
    """Trích xuất danh sách các Điều luật được dẫn chiếu trong câu trả lời."""
    matches = re.findall(r"(?:Khoản\s+\d+\s+)?Điều\s+\d+(?:\s+(?:Bộ\s+luật|Luật)\s+[\w\s\d]+)?", text, re.IGNORECASE)
    seen = set()
    unique_matches = []
    for m in matches:
        clean_m = m.strip()
        if clean_m not in seen and len(clean_m) >= 5:
            seen.add(clean_m)
            unique_matches.append(clean_m)
    return unique_matches


class SessionManager:
    """Quản lý phiên hội thoại trong bộ nhớ RAM (Thread-safe)."""

    def __init__(self, max_history_turns: int = 10):
        self._sessions: Dict[str, List[ConversationTurn]] = {}
        self.max_history_turns = max_history_turns

    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[ConversationTurn]:
        """Lấy danh sách các lượt chat gần nhất của session."""
        if not session_id or session_id not in self._sessions:
            return []
        turns = self._sessions[session_id]
        limit = max_turns or self.max_history_turns
        return turns[-limit:]

    def get_last_turn(self, session_id: str) -> Optional[ConversationTurn]:
        """Lấy lượt hội thoại gần nhất."""
        history = self.get_history(session_id)
        return history[-1] if history else None

    def add_turn(self, session_id: str, turn: ConversationTurn):
        """Thêm một lượt hội thoại mới vào session."""
        if not session_id:
            session_id = "default"
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(turn)
        # Giới hạn kích thước lịch sử
        if len(self._sessions[session_id]) > self.max_history_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history_turns:]

    def clear_session(self, session_id: str):
        """Xóa lịch sử của một phiên."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Khởi tạo singleton SessionManager toàn cục
GLOBAL_SESSION_MANAGER = SessionManager()


# =====================================================================
# 1. FOLLOW-UP DETECTOR (HYBRID RULE + FAST LLM)
# =====================================================================

TYPE_A_PATTERNS = [
    r"^(tại sao|vì sao|sao lại|tại sao lại|sao thế)(\s+.*)?[\s\?\.]*$",
    r"^(giải thích|nói rõ|làm rõ|chi tiết hơn|phân tích rõ|trình bày rõ)(\s+.*)?[\s\?\.]*$",
    r"^(căn cứ|căn cứ vào đâu|dựa vào đâu|luật nào quy định|cơ sở pháp lý|căn cứ pháp lý)(\s+.*)?[\s\?\.]*$",
    r"^ý bạn là sao[\s\?\.]*$",
    r"^(cho tôi biết|cho xin)\s+(lý do|nguyên nhân|căn cứ)[\s\?\.]*$"
]

TYPE_B_PATTERNS = [
    r"^(vậy|thế|nếu|còn)\s+(trường hợp của tôi|tôi|ông ấy|bà ấy|anh ấy|chị ấy|A|B|C|D|họ|người này|người đó|bên A|bên B)\s+(thì sao|có được không|như thế nào)",
    r"^(vậy còn|thế còn|nếu như vậy thì|trong trường hợp này thì|nếu thế thì)\b",
    r"^(vậy|thế)\s+.*\s+(có được không|thì sao|như thế nào|bao lâu|phải làm sao|tính thế nào)\??$",
    r"^(nếu\s+.*\s+thì\s+.*\s*(sao|không|\?))$",
    r"\b(như vậy|trong trường hợp đó|khi đó)\s+.*\s+(có được|phải làm gì|xử lý thế nào)"
]


def classify_followup_intent(
    query: str,
    history: List[ConversationTurn],
    gemini_client=None
) -> Tuple[str, bool]:
    """
    Phân loại câu hỏi đa lượt:
    - 'INDEPENDENT': Câu hỏi độc lập hoàn chỉnh, không phụ thuộc lịch sử.
    - 'TYPE_A_EXPLANATION': Câu hỏi yêu cầu giải thích sâu / làm rõ kết luận trước (dùng lại context cũ).
    - 'TYPE_B_FOLLOWUP': Câu hỏi nối tiếp phát sinh tình tiết / khía cạnh mới (cần rewrite và retrieve mới).
    - 'TYPE_C_TOPIC_SHIFT': Người dùng đổi chủ đề hoàn toàn (RAG độc lập).

    Trả về: (query_type, is_followup)
    """
    if not history:
        return "INDEPENDENT", False

    q_clean = query.strip()
    q_lower = q_clean.lower()

    # 1. Rule Check cho Type A (Explanation / Deep-dive)
    # Áp dụng cho các câu hỏi ngắn hoặc câu hỏi làm rõ có đại từ chỉ định
    for p in TYPE_A_PATTERNS:
        if re.search(p, q_lower, re.IGNORECASE):
            # Nếu câu ngắn (<= 10 từ) hoặc chứa đại từ quy chiếu kết luận trước
            if len(q_clean.split()) <= 10 or any(kw in q_lower for kw in ["kết luận", "trên", "như vậy", "thế này", "nói trên", "ở trên"]):
                return "TYPE_A_EXPLANATION", True

    # 2. Rule Check cho Type B (Contextual Follow-up mở rộng đối tượng/tình tiết)
    for p in TYPE_B_PATTERNS:
        if re.search(p, q_lower, re.IGNORECASE):
            return "TYPE_B_FOLLOWUP", True

    # 3. Kiểm tra đại từ chỉ định phụ thuộc context
    pronouns = [r"\bthế này\b", r"\bnhư vậy\b", r"\btrường hợp này\b", r"\bngười này\b", r"\bông ta\b", r"\bbà ta\b"]
    has_pronoun = any(re.search(pn, q_lower) for pn in pronouns)
    if has_pronoun and len(q_clean.split()) <= 15:
        return "TYPE_B_FOLLOWUP", True

    # 4. Nếu câu ngắn (< 7 từ) trong khi đang có hội thoại
    if len(q_clean.split()) <= 6:
        if any(w in q_lower for w in ["tại sao", "vì sao", "sao thế", "rõ hơn", "căn cứ", "lý do"]):
            return "TYPE_A_EXPLANATION", True
        if any(w in q_lower for w in ["thế còn", "vậy còn", "nếu vậy", "thì sao", "ai được"]):
            return "TYPE_B_FOLLOWUP", True

    # 5. Fast LLM Fallback (khi quy tắc chưa chắc chắn)
    client = gemini_client or get_gemini_client()
    if not client or types is None:
        return "INDEPENDENT", False

    last_turn = history[-1]
    prompt = f"""Bạn là bộ phân loại ý định hội thoại (Conversational Intent Classifier) cho hệ thống AI Pháp luật.
Hãy xác định câu hỏi hiện tại của người dùng có phụ thuộc vào câu trả lời trước đó hay không.

LỊCH SỬ GẦN NHẤT:
- Người dùng hỏi trước: {last_turn.user_query}
- AI trả lời trước (tóm tắt): {last_turn.answer[:250]}...

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG: "{q_clean}"

Phân loại thành 1 trong 3 nhóm duy nhất:
1. TYPE_A: Yêu cầu giải thích, làm rõ nguyên nhân hoặc căn cứ của câu trả lời trước (ví dụ: "Tại sao?", "Căn cứ vào đâu?", "Giải thích rõ hơn").
2. TYPE_B: Hỏi tiếp về tình huống trước nhưng mở rộng đối tượng/tình tiết mới (ví dụ: "Vậy còn B thì sao?", "Nếu A mất sau thì thế nào?").
3. INDEPENDENT: Câu hỏi độc lập hoặc đổi hẳn sang chủ đề/luật khác hoàn toàn không liên quan đến vụ việc trước.

Trả về DUY NHẤT một mã: TYPE_A, TYPE_B hoặc INDEPENDENT."""

    try:
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=20
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        label = resp.text.strip().upper()
        if "TYPE_A" in label:
            return "TYPE_A_EXPLANATION", True
        elif "TYPE_B" in label:
            return "TYPE_B_FOLLOWUP", True
        else:
            return "INDEPENDENT", False
    except Exception:
        return "INDEPENDENT", False


# =====================================================================
# 2. LEGAL CONTEXT-AWARE QUERY REWRITER
# =====================================================================

REWRITER_SYSTEM_PROMPT = """Bạn là chuyên gia tái cấu trúc truy vấn pháp lý (Legal Conversational Query Rewriter).
Nhiệm vụ: Viết lại câu hỏi phụ thuộc ngữ cảnh của người dùng thành một câu hỏi ĐỘC LẬP, HOÀN CHỈNH VỀ NỘI DUNG VÀ THUẬT NGỮ PHÁP LÝ để phục vụ tra cứu RAG.

NGUYÊN TẮC BẤT DI BẤT DỊCH:
1. CHỈ khôi phục các đại từ thay thế (như "tôi", "ông ấy", "A", "thế này", "trường hợp đó") dựa trên sự thật đã nêu trong lịch sử.
2. TUYỆT ĐỐI KHÔNG tự ý suy diễn, không thêm thắt các tình tiết mới, không đưa ra phán đoán hoặc kết luận pháp lý thay cho người dùng (Ví dụ: KHÔNG được tự thêm từ "chỉ được hưởng", "do vi phạm hợp đồng", "bị xử phạt").
3. Giữ nguyên mục đích hỏi ban đầu của người dùng nhưng diễn đạt rõ ràng, đầy đủ các chủ thể và quan hệ pháp lý.
4. Trả về DUY NHẤT câu truy vấn đã được viết lại, không giải thích hay mở ngoặc."""


def sanitize_rewritten_query(raw_rewritten: str, original_query: str) -> str:
    """Loại bỏ tiền tố thừa và kiểm tra tính toàn vẹn của câu hỏi được viết lại."""
    cleaned = raw_rewritten.strip().strip('"\'`')
    # Loại bỏ tiền tố phụ nếu có
    cleaned = re.sub(r"^(câu\s+hỏi\s+(được\s+)?viết\s+lại(\s+là)?|truy\s+vấn\s+mới|standalone\s+query)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    # Nếu câu quá ngắn hoặc bị lỗi, fallback về câu gốc
    if len(cleaned) < 5 or len(cleaned.split()) < 3:
        return original_query
    return cleaned


def rewrite_conversational_query(
    query: str,
    history: List[ConversationTurn],
    gemini_client=None
) -> str:
    """Tái cấu trúc câu hỏi nối tiếp thành câu hỏi độc lập đầy đủ ngữ cảnh và định hướng pháp lý chuẩn xác."""
    if not history:
        return query

    client = gemini_client or get_gemini_client()
    if not client or types is None:
        return query

    # Chuẩn bị lịch sử kèm theo các điều luật đã trích dẫn ở các lượt trước
    recent_history_text = ""
    for idx, turn in enumerate(history[-3:], 1):
        cited_laws = []
        for c in turn.retrieved_chunks[:3]:
            doc = c.get("doc_name", "")
            art = c.get("article", "")
            title = c.get("article_title", "")
            if doc and art:
                cited_laws.append(f"{doc} - Điều {art} ({title})")

        laws_str = f" [Căn cứ pháp lý đã áp dụng: {', '.join(cited_laws)}]" if cited_laws else ""
        recent_history_text += f"[Lượt {idx}]{laws_str}\n- Người dùng hỏi: {turn.user_query}\n- AI kết luận: {turn.answer[:220]}...\n\n"

    prompt = f"""Dưới đây là lịch sử hội thoại pháp lý gần nhất:
{recent_history_text}
CÂU HỎI PHỤ THUỘC CỦA NGƯỜI DÙNG:
"{query}"

Hãy viết lại câu hỏi trên thành một câu hỏi ĐỘC LẬP, ĐẦY ĐỦ CHỦ THỂ VÀ QUAN HỆ PHÁP LÝ (Không thêm bớt tình tiết hay kết luận suy đoán):"""

    candidate_models = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    for model_name in candidate_models:
        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=REWRITER_SYSTEM_PROMPT
            )
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            raw_text = resp.text.strip()
            rewritten = sanitize_rewritten_query(raw_text, query)
            if rewritten and len(rewritten) >= 5:
                return rewritten
        except Exception:
            continue

    return query
