import io
from pypdf import PdfReader

def read_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF byte string."""
    with io.BytesIO(pdf_bytes) as f:
        reader = PdfReader(f)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()