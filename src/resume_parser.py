"""Extract plain text from resume PDFs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(source: "str | Path | bytes | BytesIO") -> str:
    """Return the concatenated text of all pages in a resume PDF.

    ``source`` may be a filesystem path or raw PDF bytes (e.g. an uploaded
    file from the Streamlit UI).
    """
    if isinstance(source, (bytes, bytearray)):
        file_obj: "BytesIO | Path" = BytesIO(source)
    else:
        file_obj = Path(source)

    reader = PdfReader(file_obj)
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No text could be extracted from the PDF. Scanned/image-only "
            "resumes are not supported - please upload a text-based PDF."
        )
    return text


if __name__ == "__main__":
    import sys

    print(extract_text_from_pdf(sys.argv[1]))
