"""Read digital PDF text or render a scanned page as an image."""

from __future__ import annotations

from pathlib import Path

MIN_DIGITAL_CHARS = 120


def pdf_text(path: Path, max_pages: int = 8) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages[:max_pages]:
        chunk = page.extract_text() or ""
        if chunk.strip():
            chunks.append(chunk)
    return "\n".join(chunks).strip()


def enough_digital_text(text: str) -> bool:
    return sum(ch.isalnum() for ch in text) >= MIN_DIGITAL_CHARS


def prepare_pdf(orig: Path) -> tuple[str, Path | None, str]:
    """Digital PDFs return text only. Scanned PDFs get a page render and the photo crop."""
    text = pdf_text(orig)
    if enough_digital_text(text):
        return text, None, "digital"
    rendered = render_pdf_page(orig)
    if not rendered:
        return text, None, "failed (pdf render)"
    page_path = orig.with_name(orig.stem + "_page.jpg")
    page_path.write_bytes(rendered)
    from crop import straighten_slip_hybrid

    jpeg, status = straighten_slip_hybrid(page_path)
    stem = orig.stem
    if stem.endswith("_orig"):
        dest = orig.with_name(stem[: -len("_orig")] + "_scan.jpg")
    else:
        dest = orig.with_name(stem + "_scan.jpg")
    if jpeg:
        dest.write_bytes(jpeg)
        if page_path != dest and page_path.exists():
            page_path.unlink()
        return text, dest, status
    return text, page_path, status if status.startswith("failed") else "rendered"


def render_pdf_page(path: Path, page_index: int = 0, scale: float = 2.0) -> bytes | None:
    import pypdfium2 as pdfium
    from io import BytesIO

    doc = pdfium.PdfDocument(str(path))
    try:
        if page_index < 0 or page_index >= len(doc):
            return None
        page = doc[page_index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    finally:
        doc.close()
