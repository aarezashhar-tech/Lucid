"""
Unit tests for Image OCR speed/downscaling and PDF content extraction fixes.

Covers:
1. Large image downscaling preserving aspect ratio and converting to RGB.
2. Image OCR using gemini-3.5-flash-lite as primary fast extractor with downscaled Tesseract fallback.
3. Clean JSON response for /api/ocr endpoint.
4. Standard text extraction from PDF documents.
5. Scanned/image PDF fallback extraction via Gemini native vision and embedded image OCR.
6. HTTP 422 returned for truly empty PDFs (never verifying the placeholder string).
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as app_module
from app import app
from utils.image_ocr import (
    downscale_and_preprocess_image,
    extract_text_from_image,
    extract_text_via_gemini_vision,
    extract_text_via_tesseract,
)
from utils.pdf_parser import (
    extract_text_from_pdf,
    extract_text_via_gemini_pdf,
    extract_text_via_embedded_images,
)


# ===========================================================================
# 1. Image Downscaling & Preprocessing Tests
# ===========================================================================

def test_large_image_downscaling_aspect_ratio_and_mode():
    """Verify images > 1280px are downscaled preserving aspect ratio and converted to RGB."""
    # 2560 x 1440 (16:9 ratio, RGBA mode)
    img = Image.new("RGBA", (2560, 1440), color=(255, 0, 0, 128))
    processed = downscale_and_preprocess_image(img, max_dim=1280)

    assert processed.mode == "RGB", "Mode should be converted to RGB"
    assert processed.width <= 1280, f"Width {processed.width} exceeds max 1280"
    assert processed.height <= 1280, f"Height {processed.height} exceeds max 1280"

    # Check 16:9 aspect ratio preserved (1280 x 720)
    assert processed.width == 1280
    assert processed.height == 720


def test_portrait_image_downscaling():
    """Verify portrait images have height constrained to 1280 while preserving aspect ratio."""
    # 1000 x 3000 (1:3 ratio)
    img = Image.new("RGB", (1000, 3000), color="blue")
    processed = downscale_and_preprocess_image(img, max_dim=1280)

    assert processed.height == 1280
    assert processed.width == round(1000 * (1280 / 3000))  # 427


def test_small_image_not_upscaled():
    """Verify images <= 1280px are not upscaled."""
    img = Image.new("RGB", (640, 480), color="green")
    processed = downscale_and_preprocess_image(img, max_dim=1280)

    assert (processed.width, processed.height) == (640, 480)


def test_extract_text_from_image_resizes_on_disk(tmp_path):
    """Verify extract_text_from_image overwrites large images on disk with downscaled versions."""
    large_path = str(tmp_path / "huge_test_image.png")
    img = Image.new("RGB", (2000, 2000), color="white")
    img.save(large_path, format="PNG")

    with patch("utils.image_ocr.extract_text_via_gemini_vision", return_value=("Sample text", True)):
        text, success, msg = extract_text_from_image(large_path)

    assert success is True
    assert text == "Sample text"

    # Verify on-disk file was downscaled to <= 1280px
    reloaded = Image.open(large_path)
    assert reloaded.width <= 1280
    assert reloaded.height <= 1280


# ===========================================================================
# 2. Image OCR: Gemini Primary + Tesseract Fallback Tests
# ===========================================================================

def test_image_ocr_gemini_primary_success(tmp_path):
    """Verify gemini-3.5-flash-lite is used as primary fast text extractor when configured."""
    img_path = str(tmp_path / "test.jpg")
    Image.new("RGB", (100, 100), color="white").save(img_path)

    expected_text = "Breaking News: Major Discovery Announced Today."

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"}), \
         patch("utils.image_ocr.GENAI_AVAILABLE", True), \
         patch("utils.image_ocr.extract_text_via_gemini_vision", return_value=(expected_text, True)) as mock_gemini, \
         patch("utils.image_ocr.extract_text_via_tesseract") as mock_tess:

        text, success, msg, details = extract_text_from_image(img_path, return_details=True)

        assert success is True
        assert text == expected_text
        assert details["extractor"] == "gemini-3.5-flash-lite"
        assert details["confidence"] == 95.0
        assert mock_gemini.called
        assert not mock_tess.called, "Tesseract should NOT be called when Gemini succeeds"


def test_image_ocr_gemini_failure_falls_back_to_tesseract(tmp_path):
    """Verify automatic fallback to downscaled Tesseract (--oem 1 --psm 6) when Gemini fails."""
    img_path = str(tmp_path / "test.jpg")
    Image.new("RGB", (100, 100), color="white").save(img_path)

    tess_text = "Fallback Text Extracted by Tesseract"

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"}), \
         patch("utils.image_ocr.GENAI_AVAILABLE", True), \
         patch("utils.image_ocr.extract_text_via_gemini_vision", return_value=("", False)), \
         patch("utils.image_ocr.extract_text_via_tesseract", return_value=(tess_text, 88.0, 5, True, "")) as mock_tess:

        text, success, msg, details = extract_text_from_image(img_path, return_details=True)

        assert success is True
        assert text == tess_text
        assert details["extractor"] == "tesseract"
        assert details["confidence"] == 88.0
        assert mock_tess.called


def test_image_ocr_empty_image_returns_clean_message(tmp_path):
    """Verify empty image returns success=True with clear empty notification."""
    img_path = str(tmp_path / "blank.png")
    Image.new("RGB", (100, 100), color="white").save(img_path)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"}), \
         patch("utils.image_ocr.GENAI_AVAILABLE", True), \
         patch("utils.image_ocr.extract_text_via_gemini_vision", return_value=("", True)):

        text, success, msg, details = extract_text_from_image(img_path, return_details=True)

        assert success is True
        assert text == ""
        assert details["isLowConfidence"] is True
        assert "no discernible text" in msg.lower()


# ===========================================================================
# 3. /api/ocr Endpoint Tests
# ===========================================================================

def test_api_ocr_returns_clean_json_fast():
    """Verify /api/ocr returns clean JSON structure with 200 OK."""
    client = app.test_client()

    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with patch("app.extract_text_from_image", return_value=(
        "Extracted headline", True, "Text extracted successfully.", {
            "previewText": "Extracted headline",
            "confidence": 95.0,
            "isLowConfidence": False,
            "warning": "",
            "qualityWarning": "",
            "wordCount": 2,
        }
    )), patch.object(app_module.rate_limiter, "is_allowed", return_value=(True, 0)):

        res = client.post(
            "/api/ocr",
            data={"file": (buf, "sample.png")},
            content_type="multipart/form-data"
        )

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["extractedText"] == "Extracted headline"
    assert data["previewText"] == "Extracted headline"
    assert data["confidence"] == 95.0


def test_api_ocr_rejects_non_image_file():
    """Verify /api/ocr rejects non-image formats (e.g. PDF or TXT) with 400."""
    client = app.test_client()

    txt_buf = io.BytesIO(b"Just some plain text content.")
    with patch.object(app_module.rate_limiter, "is_allowed", return_value=(True, 0)):
        res = client.post(
            "/api/ocr",
            data={"file": (txt_buf, "document.txt")},
            content_type="multipart/form-data"
        )

    assert res.status_code == 400
    assert "not supported" in res.get_json()["error"].lower()


# ===========================================================================
# 4. PDF Content Extraction Tests
# ===========================================================================

def test_pdf_standard_text_extraction(tmp_path):
    """Verify standard text extraction succeeds when PDF text layer contains >= 20 chars."""
    pdf_path = str(tmp_path / "standard.pdf")

    try:
        import pypdf
        writer = pypdf.PdfWriter()
        # Create a page with standard extractable text
        writer.add_blank_page(width=400, height=400)
    except ImportError:
        pass

    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a legitimate standard PDF text layer containing plenty of characters."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        # Create dummy file
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        text, success, msg = extract_text_from_pdf(pdf_path)

        assert success is True
        assert "This is a legitimate standard PDF" in text
        assert "Successfully parsed" in msg


def test_pdf_scanned_image_gemini_fallback(tmp_path):
    """Verify scanned PDF (< 20 chars in text layer) uses gemini-3.5-flash-lite fallback."""
    pdf_path = str(tmp_path / "scanned.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 dummy scanned content")

    gemini_scanned_text = "Article Title: Major Scientific Advancement Verified By Researchers."

    with patch("pypdf.PdfReader") as mock_reader_cls, \
         patch("utils.pdf_parser.extract_text_via_gemini_pdf", return_value=(gemini_scanned_text, True)) as mock_gemini_pdf:

        # Standard reader returns empty text (scanned document)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        text, success, msg = extract_text_from_pdf(pdf_path)

        assert success is True
        assert text == gemini_scanned_text
        assert mock_gemini_pdf.called


def test_pdf_embedded_images_ocr_fallback(tmp_path):
    """Verify PDF with empty text layer and failed Gemini falls back to embedded images OCR."""
    pdf_path = str(tmp_path / "embedded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 dummy content")

    ocr_embedded_text = "Headline from scanned screenshot inside PDF document."

    with patch("pypdf.PdfReader") as mock_reader_cls, \
         patch("utils.pdf_parser.extract_text_via_gemini_pdf", return_value=("", False)), \
         patch("utils.pdf_parser.extract_text_via_embedded_images", return_value=(ocr_embedded_text, True)) as mock_img_ocr:

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "   "  # whitespace / < 20 chars
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        text, success, msg = extract_text_from_pdf(pdf_path)

        assert success is True
        assert text == ocr_embedded_text
        assert mock_img_ocr.called


def test_pdf_truly_empty_returns_explicit_failure(tmp_path):
    """Verify extract_text_from_pdf returns explicit failure if no text could be extracted."""
    pdf_path = str(tmp_path / "empty.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 empty")

    with patch("pypdf.PdfReader") as mock_reader_cls, \
         patch("utils.pdf_parser.extract_text_via_gemini_pdf", return_value=("", False)), \
         patch("utils.pdf_parser.extract_text_via_embedded_images", return_value=("", False)):

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        text, success, msg = extract_text_from_pdf(pdf_path)

        assert success is False
        assert text == ""
        assert "Could not extract readable text from this PDF." in msg


# ===========================================================================
# 5. /api/analyze-file PDF 422 & Placeholder Protection Tests
# ===========================================================================

def test_analyze_file_empty_pdf_returns_422_and_never_verifies_placeholder():
    """
    CRITICAL TEST:
    Verify that an empty or unreadable PDF returns HTTP 422
    and NEVER passes 'PDF document ... contained no discernible text.' into engine.analyze.
    """
    client = app.test_client()

    # Valid PDF magic bytes header
    empty_pdf_bytes = b"%PDF-1.4\n%empty\n"

    mock_engine = MagicMock()

    with patch("app.extract_text_from_pdf", return_value=("", False, "Could not extract readable text from this PDF.")), \
         patch("app.get_verification_engine", return_value=mock_engine), \
         patch.object(app_module.rate_limiter, "is_allowed", return_value=(True, 0)):

        res = client.post(
            "/api/analyze-file",
            data={"file": (io.BytesIO(empty_pdf_bytes), "test_blank.pdf")},
            content_type="multipart/form-data"
        )

    # 1. Must return HTTP 422 Unprocessable Entity
    assert res.status_code == 422, f"Expected 422 but got {res.status_code}: {res.get_data(as_text=True)}"
    data = res.get_json()
    assert "Could not extract readable text from this PDF." in data["error"]

    # 2. engine.analyze must NEVER have been called with the placeholder error string
    assert not mock_engine.analyze.called, "Verification engine should NOT be called for an unreadable PDF"
    for call_args in mock_engine.analyze.call_args_list:
        passed_text = str(call_args)
        assert "contained no discernible text" not in passed_text, (
            "Placeholder error string must never be passed to verification engine"
        )
