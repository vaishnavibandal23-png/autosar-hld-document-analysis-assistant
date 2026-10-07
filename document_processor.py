from pathlib import Path
import re
import fitz

from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 140:
        return False
    if re.match(r"^(\d+(\.\d+)*[\)\.]?|[A-Z][\.\)])\s+\S+", line):
        return True
    if line.isupper() and 3 <= len(line.split()) <= 12:
        return True
    return False


def split_into_chunks(text: str, page: int, document: str):
    text = clean_text(text)
    if not text:
        return []

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    section = ""
    chunks = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        body = " ".join(current).strip()
        if body:
            chunks.append({
                "text": body,
                "page": page,
                "section": section,
                "document": document
            })
        current = []

    for line in lines:
        if detect_heading(line):
            flush()
            section = line
            current.append(line)
        else:
            current.append(line)

    flush()

    # Further split very large chunks while retaining page/section metadata.
    final = []
    for item in chunks:
        text_value = item["text"]
        if len(text_value) <= CHUNK_SIZE:
            final.append(item)
            continue

        start = 0
        while start < len(text_value):
            end = min(start + CHUNK_SIZE, len(text_value))
            if end < len(text_value):
                cut = text_value.rfind(" ", start, end)
                if cut > start + CHUNK_SIZE // 2:
                    end = cut
            piece = text_value[start:end].strip()
            if piece:
                copy = dict(item)
                copy["text"] = piece
                final.append(copy)
            if end >= len(text_value):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)

    return final


def extract_pdf(pdf_path: Path):
    chunks = []
    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        page_chunks = split_into_chunks(
            text,
            page_number,
            pdf_path.name
        )
        chunks.extend(page_chunks)
    doc.close()
    return chunks


def load_all_documents(documents_dir: Path):
    all_chunks = []
    pdfs = sorted(documents_dir.glob("*.pdf"))
    for pdf in pdfs:
        all_chunks.extend(extract_pdf(pdf))
    return all_chunks, pdfs
