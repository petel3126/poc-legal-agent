"""
Hierarchical Legal Chunking — theo thiết kế ở TDD mục 4 (module: chunker,
tương ứng bước tiền xử lý trước khi đưa vào embedding) và schema chunk
ở SRS mục 6.

Cấu trúc nhận diện: Chương -> Mục (tuỳ chọn) -> Điều -> Khoản -> Điểm.
Đơn vị chunk cơ sở: Điều (nếu chỉ có 1 khoản hoặc không có khoản) hoặc
từng Khoản (nếu Điều có từ 2 khoản trở lên) — đúng theo SRS mục 6.

Input:  data/raw/blld_45_2019_qh14_partial.txt (plain text, đã làm sạch)
Output: data/processed/blld_45_2019_qh14_chunks.json
"""

import re
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "blld_45_2019_qh14_partial.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "blld_45_2019_qh14_chunks.json"

DOCUMENT_META = {
    "document_id": "45-2019-QH14",
    "document_type": "Bộ luật",
    "document_number": "45/2019/QH14",
    "title": "Bộ luật Lao động",
    "issuing_authority": "Quốc hội",
    "issue_date": "2019-11-20",
    "effective_date": "2021-01-01",
    "expiration_date": None,
    "status": "ACTIVE",  # "Còn hiệu lực" theo thuvienphapluat.vn tại thời điểm crawl
    "source_url": "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-Luat-lao-dong-2019-333670.aspx",
    "amended_by": None,
    "replaces": "10/2012/QH13",
    "replaced_by": None,
}

CHAPTER_RE = re.compile(r"^Chương\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
MUC_RE = re.compile(r"^Mục\s+(\d+)\.\s*(.+)$")
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\.\s*(.+)$")
CLAUSE_RE = re.compile(r"^(\d+)\.\s+(.*)$")
POINT_RE = re.compile(r"^([a-zđ])\)\s+(.*)$")

# Bắt các dẫn chiếu dạng "Điều 35", "khoản 2 Điều 36", "điểm b khoản 1 Điều 36"
REF_CLAUSE_ARTICLE_RE = re.compile(r"[Kk]hoản\s+(\d+)\s+Điều\s+(\d+)")
REF_ARTICLE_RE = re.compile(r"Điều\s+(\d+)")


def extract_references(text: str, self_article: str, doc_id: str):
    """Tìm các dẫn chiếu tới Điều/Khoản khác trong nội dung, loại bỏ tự tham chiếu."""
    refs = set()
    for m in REF_CLAUSE_ARTICLE_RE.finditer(text):
        clause, article = m.group(1), m.group(2)
        if article != self_article:
            refs.add(f"{doc_id}-đ{article}-k{clause}")
    for m in REF_ARTICLE_RE.finditer(text):
        article = m.group(1)
        if article != self_article:
            # chỉ thêm reference cấp điều nếu chưa có reference cấp khoản cụ thể hơn cho điều đó
            if not any(r.startswith(f"{doc_id}-đ{article}-k") for r in refs):
                refs.add(f"{doc_id}-đ{article}")
    return sorted(refs)


def parse(raw_text: str):
    lines = [l.rstrip() for l in raw_text.split("\n")]
    chunks = []

    cur_chapter = None
    cur_muc = None
    cur_article_num = None
    cur_article_title = None
    cur_article_lines = []  # raw lines belonging to current article (before split into khoản)

    def flush_article():
        """Khi kết thúc 1 Điều: quyết định chunk theo Điều hay theo Khoản."""
        nonlocal cur_article_num, cur_article_title, cur_article_lines
        if cur_article_num is None:
            return
        body_lines = cur_article_lines
        # Tìm các khoản trong body
        clause_spans = []  # (clause_num, start_idx)
        for i, l in enumerate(body_lines):
            m = CLAUSE_RE.match(l)
            if m:
                clause_spans.append((m.group(1), i))

        if len(clause_spans) >= 2:
            # Chunk theo từng khoản
            for idx, (clause_num, start_i) in enumerate(clause_spans):
                end_i = clause_spans[idx + 1][1] if idx + 1 < len(clause_spans) else len(body_lines)
                clause_lines = body_lines[start_i:end_i]
                # dòng đầu là "N. nội dung..." -> bỏ số thứ tự khoản khỏi content
                first = CLAUSE_RE.match(clause_lines[0])
                content_lines = [first.group(2)] + clause_lines[1:]
                content = "\n".join(l for l in content_lines if l.strip())
                chunk_id = f"{DOCUMENT_META['document_id']}-đ{cur_article_num}-k{clause_num}"
                chunks.append(make_chunk(chunk_id, content, clause_num, None))
        else:
            # Không tách khoản -> cả Điều là 1 chunk
            content = "\n".join(l for l in body_lines if l.strip())
            chunk_id = f"{DOCUMENT_META['document_id']}-đ{cur_article_num}"
            chunks.append(make_chunk(chunk_id, content, None, None))

        cur_article_num = None
        cur_article_title = None
        cur_article_lines = []

    def make_chunk(chunk_id, content, clause, point):
        full_content = f"Điều {cur_article_num}. {cur_article_title}\n{content}".strip()
        refs = extract_references(content, cur_article_num, DOCUMENT_META["document_id"])
        chunk = {
            "chunk_id": chunk_id,
            **DOCUMENT_META,
            "chapter": cur_chapter,
            "mục": cur_muc,
            "article": cur_article_num,
            "article_title": cur_article_title,
            "clause": clause,
            "point": point,
            "content": full_content,
            "references": refs,
        }
        return chunk

    for line in lines:
        if not line.strip():
            continue

        m_chap = CHAPTER_RE.match(line.strip())
        if m_chap:
            flush_article()
            cur_chapter = f"Chương {m_chap.group(1)}"
            cur_muc = None
            continue

        m_muc = MUC_RE.match(line.strip())
        if m_muc:
            flush_article()
            cur_muc = f"Mục {m_muc.group(1)}. {m_muc.group(2)}"
            continue

        m_art = ARTICLE_RE.match(line.strip())
        if m_art:
            flush_article()
            cur_article_num = m_art.group(1)
            cur_article_title = m_art.group(2)
            cur_article_lines = []
            continue

        # Bỏ qua dòng tiêu đề Chương (dòng in hoa ngay sau "Chương X") — không phải nội dung Điều
        if cur_article_num is None:
            continue

        cur_article_lines.append(line)

    flush_article()
    return chunks


def run_chunking(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH, force_rechunk: bool = False):
    if out_path.exists() and not force_rechunk:
        print(f"[INFO] File chunks đã tồn tại tại {out_path}. Sử dụng dữ liệu đã lưu (Bỏ qua bước tách chunk).")
        chunks = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Tổng số chunk đã tải: {len(chunks)}")
        return chunks

    print(f"Reading raw legal text từ {raw_path}...")
    raw_text = raw_path.read_text(encoding="utf-8")
    chunks = parse(raw_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


    print(f"Tổng số chunk: {len(chunks)}")
    lengths = [len(c["content"]) for c in chunks]
    print(f"Độ dài content (ký tự) — min: {min(lengths)}, max: {max(lengths)}, "
          f"trung bình: {sum(lengths)//len(lengths)}")
    n_with_refs = sum(1 for c in chunks if c["references"])
    print(f"Số chunk có reference tới Điều/Khoản khác: {n_with_refs}")
    print(f"\nVí dụ 2 chunk đầu tiên:")
    for c in chunks[:2]:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        print("---")
    return chunks

