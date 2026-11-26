import os
import io
import re
import logging
import concurrent.futures
from typing import Tuple, Optional
from PIL import Image

from utils.image_ocr import get_gemini_client, GENAI_AVAILABLE, extract_text_from_image

logger = logging.getLogger("lucid.pdf")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [PDF] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def extract_text_via_gemini_pdf(
    pdf_path: str,
    max_pages: int = 5,
    max_bytes: int = 5 * 1024 * 1024,
    timeout_seconds: float = 8.0
) -> Tuple[str, bool]:
    """
    Sends PDF bytes (capped at <= 5MB and <= 5 pages) to gemini-3.5-flash-lite
    to extract visible text natively from scanned/image PDFs with an 8s timeout.
    Returns (extracted_text, success).
    """
    client = get_gemini_client()
    if client is None:
        return "", False

    try:
        try:
            import pypdf
        except ImportError:
            import PyPDF2 as pypdf

        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if total_pages == 0:
            return "", False

        file_size = os.path.getsize(pdf_path)
        # If the file is already <= max_pages and <= max_bytes, pass directly
        if total_pages <= max_pages and file_size <= max_bytes:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        else:
            # Crop to first max_pages (up to 5 pages)
            writer = pypdf.PdfWriter()
            pages_to_add = min(total_pages, max_pages)
            for i in range(pages_to_add):
                writer.add_page(reader.pages[i])

            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()

            if len(pdf_bytes) > max_bytes:
                logger.warning(
                    f"[PDF] Cropped PDF bytes ({len(pdf_bytes)}) exceed {max_bytes} bytes limit. Skipping native Gemini PDF call."
                )
                return "", False

        from google.genai import types
        part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        prompt = (
            "Extract all readable text visible in this PDF document verbatim. "
            "Preserve the document layout, headings, and paragraph structure. "
            "Do not include any commentary, explanations, introductory text, markdown code fences, or notes. "
            "If the document contains no readable text, reply with exactly: NONE"
        )

        def _generate():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[prompt, part],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            response = future.result(timeout=timeout_seconds)

        raw_text = (response.text or "").strip()
        if not raw_text or raw_text.upper() == "NONE":
            return "", False

        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text).strip()

        return raw_text, True

    except Exception as e:
        logger.warning(f"[PDF] Native Gemini PDF extraction failed or timed out ({timeout_seconds}s limit): {e}")
        return "", False


def extract_text_via_embedded_images(pdf_path: str, max_pages: int = 5) -> Tuple[str, bool]:
    """
    Extracts embedded raster images from the first max_pages of a PDF
    and runs the image OCR function on each image.
    Returns (extracted_text, success).
    """
    try:
        try:
            import pypdf
        except ImportError:
            import PyPDF2 as pypdf

        reader = pypdf.PdfReader(pdf_path)
        extracted_snippets = []

        pages_to_check = min(len(reader.pages), max_pages)
        for page_idx in range(pages_to_check):
            page = reader.pages[page_idx]
            if hasattr(page, "images"):
                for img_idx, img_file in enumerate(page.images):
                    try:
                        pil_img = Image.open(io.BytesIO(img_file.data))
                        ocr_txt, ocr_ok, _ = extract_text_from_image(pil_img)
                        if ocr_ok and ocr_txt.strip():
                            extracted_snippets.append(ocr_txt.strip())
                    except Exception as img_err:
                        logger.warning(f"[PDF] Failed to OCR embedded image {img_idx} on page {page_idx}: {img_err}")

        if extracted_snippets:
            combined = "\n\n".join(extracted_snippets).strip()
            return combined, bool(combined)

        return "", False

    except Exception as e:
        logger.warning(f"[PDF] Embedded image extraction error: {e}")
        return "", False


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, bool, str]:
    """
    Extracts plain text from a PDF file.
    
    Pipeline:
    1. Standard text extraction (pdfplumber / PyPDF2 / pypdf) across all pages (up to first 10 pages).
    2. If extracted text is empty or < 20 characters (scanned/image PDF):
       a. Tries gemini-3.5-flash-lite native multimodal PDF extraction (capped at <= 5MB / first 5 pages, 8s timeout).
       b. Tries extracting embedded images from PDF pages and running OCR.
    3. If extraction still yields no readable text, returns ("", False, "Could not extract readable text from this PDF.").
    
    Returns:
        (extracted_text, success, message)
    """
    if not os.path.exists(pdf_path):
        return "", False, f"File not found: {pdf_path}"

    extracted_pages = []
    num_pages = 0

    # 1. First try standard text extraction using pdfplumber if installed
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages == 0:
                return "", False, "The PDF document contains 0 pages."
            for page in pdf.pages[:10]:
                t = page.extract_text()
                if t and t.strip():
                    extracted_pages.append(t.strip())
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[PDF] pdfplumber standard extraction failed: {e}")

    # 2. If pdfplumber was not available or found nothing, try pypdf / PyPDF2
    if not extracted_pages:
        try:
            try:
                import pypdf
                reader = pypdf.PdfReader(pdf_path)
            except ImportError:
                import PyPDF2 as pypdf
                reader = pypdf.PdfReader(pdf_path)

            num_pages = len(reader.pages)
            if num_pages == 0:
                return "", False, "The PDF document contains 0 pages."

            for page in reader.pages[:10]:
                t = page.extract_text()
                if t and t.strip():
                    extracted_pages.append(t.strip())
        except Exception as e:
            logger.warning(f"[PDF] pypdf / PyPDF2 standard extraction failed: {e}")

    full_text = "\n\n".join(extracted_pages).strip()

    # If standard text layer has adequate content (>= 20 characters), return immediately
    if len(full_text) >= 20:
        return full_text, True, f"Successfully parsed {num_pages} page(s)."

    logger.info(
        f"[PDF] Standard extraction yielded only {len(full_text)} chars (< 20). Triggering scanned/image PDF fallback."
    )

    # 3. Scanned / Image PDF Fallback: Native Gemini 3.5 Flash Lite PDF extraction
    gemini_text, gemini_ok = extract_text_via_gemini_pdf(
        pdf_path, max_pages=5, max_bytes=5 * 1024 * 1024, timeout_seconds=8.0
    )
    if gemini_ok and len(gemini_text.strip()) >= 20:
        logger.info(f"[PDF] Successfully extracted {len(gemini_text)} chars via Gemini native PDF vision.")
        return gemini_text.strip(), True, "Successfully extracted text via AI PDF vision."

    # 4. Embedded Images Fallback: Extract raster images and run OCR
    img_text, img_ok = extract_text_via_embedded_images(pdf_path, max_pages=5)
    if img_ok and len(img_text.strip()) >= 20:
        logger.info(f"[PDF] Successfully extracted {len(img_text)} chars from embedded images.")
        return img_text.strip(), True, "Successfully extracted text from embedded PDF images."

    # If Gemini or embedded OCR yielded some non-empty text, combine with standard text
    best_candidate = gemini_text.strip() or img_text.strip() or full_text
    if len(best_candidate) >= 20:
        return best_candidate, True, "Successfully extracted text from PDF."

    # 5. Extraction STILL yields no readable text -> Return explicit failure
    logger.warning("[PDF] Extraction yielded no readable text across standard layer, Gemini vision, and image OCR.")
    return "", False, "Could not extract readable text from this PDF."
