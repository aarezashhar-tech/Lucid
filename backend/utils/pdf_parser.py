import os
from typing import Tuple

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, bool, str]:
    """
    Extracts plain text from a PDF file using PyPDF2 or pypdf.
    Returns:
        (extracted_text, success, message)
    """
    if not os.path.exists(pdf_path):
        return "", False, f"File not found: {pdf_path}"

    extracted_pages = []

    # Try pypdf / PyPDF2
    try:
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
        except ImportError:
            import PyPDF2
            reader = PyPDF2.PdfReader(pdf_path)

        num_pages = len(reader.pages)
        if num_pages == 0:
            return "", False, "The PDF document contains 0 pages."

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages).strip()
        if not full_text:
            return "", True, "Document loaded, but no extractable text layer was found (it may be a scanned PDF)."

        return full_text, True, f"Successfully parsed {num_pages} page(s)."

    except Exception as e:
        return "", False, f"PDF extraction error: {str(e)}"
