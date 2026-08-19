"""
Hierarchical Legal Chunking cho Hệ Thống Đa Luật (Multi-Document Legal RAG)
Tự động quét tất cả các file văn bản luật trong data/raw/, thực hiện chunking
theo chuẩn: Chương -> Mục -> Điều -> Khoản -> Điểm.

Đầu ra DUY NHẤT:
  - data/processed/legal_chunks.json (Tập hợp toàn bộ chunks của 9 bộ luật)
"""

import re
import json
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_CHUNKS_FILE = PROCESSED_DIR / "legal_chunks.json"

# Cấu hình Metadata cho 9 bộ luật trọng điểm
DOCUMENTS_CONFIG = {
    "blds_91_2015_qh13.txt": {
        "document_id": "91-2015-QH13",
        "document_type": "Bộ luật",
        "document_number": "91/2015/QH13",
        "title": "Bộ luật Dân sự 2015",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "blld_45_2019_qh14_partial.txt": {
        "document_id": "45-2019-QH14",
        "document_type": "Bộ luật",
        "document_number": "45/2019/QH14",
        "title": "Bộ luật Lao động 2019",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "bvqlntd_19_2023_qh15.txt": {
        "document_id": "19-2023-QH15",
        "document_type": "Luật",
        "document_number": "19/2023/QH15",
        "title": "Luật Bảo vệ quyền lợi người tiêu dùng 2023",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "lanm_24_2018_qh14.txt": {
        "document_id": "24-2018-QH14",
        "document_type": "Luật",
        "document_number": "24/2018/QH14",
        "title": "Luật An ninh mạng 2018",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "ldn_59_2020_qh14.txt": {
        "document_id": "59-2020-QH14",
        "document_type": "Luật",
        "document_number": "59/2020/QH14",
        "title": "Luật Doanh nghiệp 2020",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "ldt_61_2020_qh14.txt": {
        "document_id": "61-2020-QH14",
        "document_type": "Luật",
        "document_number": "61/2020/QH14",
        "title": "Luật Đầu tư 2020",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "lqlt_38_2019_qh14.txt": {
        "document_id": "38-2019-QH14",
        "document_type": "Luật",
        "document_number": "38/2019/QH14",
        "title": "Luật Quản lý thuế 2019",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "lshtt_50_2005_qh11.txt": {
        "document_id": "50-2005-QH11",
        "document_type": "Luật",
        "document_number": "50/2005/QH11",
        "title": "Luật Sở hữu trí tuệ 2005",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    },
    "ltm_36_2005_qh11.txt": {
        "document_id": "36-2005-QH11",
        "document_type": "Luật",
        "document_number": "36/2005/QH11",
        "title": "Luật Thương mại 2005",
        "issuing_authority": "Quốc hội",
        "status": "ACTIVE"
    }
}

CHAPTER_RE = re.compile(r"^Chương\s+([IVXLCDM\d]+)", re.IGNORECASE)
MUC_RE = re.compile(r"^Mục\s+(\d+)[\.:]\s*(.+)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)[\.:]\s*(.+)$")
CLAUSE_RE = re.compile(r"^(\d+)[\.:]\s+(.*)$")

REF_CLAUSE_ARTICLE_RE = re.compile(r"[Kk]hoản\s+(\d+)\s+Điều\s+(\d+)")
REF_ARTICLE_RE = re.compile(r"Điều\s+(\d+)")


def extract_references(text: str, self_article: str, doc_id: str):
    """Tìm các dẫn chiếu tới Điều/Khoản khác trong nội dung."""
    refs = set()
    for m in REF_CLAUSE_ARTICLE_RE.finditer(text):
        clause, article = m.group(1), m.group(2)
        if article != self_article:
            refs.add(f"{doc_id}-đ{article}-k{clause}")
    for m in REF_ARTICLE_RE.finditer(text):
        article = m.group(1)
        if article != self_article:
            if not any(r.startswith(f"{doc_id}-đ{article}-k") for r in refs):
                refs.add(f"{doc_id}-đ{article}")
    return sorted(refs)


def parse_legal_text(raw_text: str, doc_meta: dict):
    lines = [l.rstrip() for l in raw_text.split("\n")]
    chunks = []

    cur_chapter = None
    cur_muc = None
    cur_article_num = None
    cur_article_title = None
    cur_article_lines = []

    def flush_article():
        nonlocal cur_article_num, cur_article_title, cur_article_lines
        if cur_article_num is None:
            return

        body_lines = cur_article_lines
        clause_spans = []
        for i, l in enumerate(body_lines):
            m = CLAUSE_RE.match(l)
            if m:
                clause_spans.append((m.group(1), i))

        if len(clause_spans) >= 2:
            # Chunk theo từng Khoản nếu Điều có từ 2 Khoản trở lên
            for idx, (clause_num, start_i) in enumerate(clause_spans):
                end_i = clause_spans[idx + 1][1] if idx + 1 < len(clause_spans) else len(body_lines)
                clause_lines = body_lines[start_i:end_i]
                first = CLAUSE_RE.match(clause_lines[0])
                content_lines = [first.group(2)] + clause_lines[1:]
                content = "\n".join(l for l in content_lines if l.strip())
                chunk_id = f"{doc_meta['document_id']}-đ{cur_article_num}-k{clause_num}"
                chunks.append(make_chunk(chunk_id, content, clause_num))
        else:
            # Nếu chỉ có 1 Khoản hoặc không chia Khoản -> Chunk theo Điều
            content = "\n".join(l for l in body_lines if l.strip())
            chunk_id = f"{doc_meta['document_id']}-đ{cur_article_num}"
            chunks.append(make_chunk(chunk_id, content, None))

        cur_article_num = None
        cur_article_title = None
        cur_article_lines = []

    def make_chunk(chunk_id, content, clause):
        title_str = cur_article_title if cur_article_title else ""
        full_content = f"Điều {cur_article_num}. {title_str}\n{content}".strip()
        refs = extract_references(content, cur_article_num, doc_meta["document_id"])
        return {
            "chunk_id": chunk_id,
            "document_id": doc_meta["document_id"],
            "document_type": doc_meta["document_type"],
            "document_number": doc_meta["document_number"],
            "title": doc_meta["title"],
            "issuing_authority": doc_meta.get("issuing_authority", "Quốc hội"),
            "status": doc_meta.get("status", "ACTIVE"),
            "chapter": cur_chapter,
            "mục": cur_muc,
            "article": cur_article_num,
            "article_title": cur_article_title,
            "clause": clause,
            "point": None,
            "content": full_content,
            "references": refs
        }

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        ch_match = CHAPTER_RE.match(line_str)
        if ch_match:
            cur_chapter = line_str
            cur_muc = None
            continue

        muc_match = MUC_RE.match(line_str)
        if muc_match:
            cur_muc = line_str
            continue

        art_match = ARTICLE_RE.match(line_str)
        if art_match:
            flush_article()
            cur_article_num = art_match.group(1)
            cur_article_title = art_match.group(2)
            cur_article_lines = []
            continue

        if cur_article_num is not None:
            cur_article_lines.append(line_str)

    flush_article()
    return chunks


def main():
    print("=== BẮT ĐẦU CHUNKING TOÀN BỘ BỘ LUẬT VÀO 1 FILE DUY NHẤT ===")
    all_chunks = []

    for filename, meta in DOCUMENTS_CONFIG.items():
        file_path = RAW_DIR / filename
        if not file_path.exists():
            print(f"⚠️  Bỏ qua {filename} do không tìm thấy file.")
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        chunks = parse_legal_text(raw_text, meta)
        all_chunks.extend(chunks)
        print(f"  ✓ {meta['title']} ({meta['document_id']}): {len(chunks):,} chunks")

    # Dọn dẹp tất cả các file JSON cũ rác trong data/processed/
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith(".json"):
            (PROCESSED_DIR / f).unlink()

    # Ghi toàn bộ chunks vào file duy nhất: data/processed/legal_chunks.json
    OUT_CHUNKS_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"HOÀN THÀNH! Đã lưu duy nhất 1 file chunk tổng hợp chứa {len(all_chunks):,} chunks:")
    print(f"  -> {OUT_CHUNKS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
