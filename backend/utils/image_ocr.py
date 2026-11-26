import os
import sys
import io
import re
import logging
import traceback
import shutil
import concurrent.futures
import threading
from typing import Tuple, Optional, Dict, Any, Union
from PIL import Image

from utils.image_publisher_detector import detect_publisher_from_image

logger = logging.getLogger("lucid.ocr")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [OCR] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Google GenAI SDK Setup (gemini-3.5-flash-lite)
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False
    logger.warning("google-genai package not found. Native Gemini OCR vision disabled.")

_gemini_client = None
_gemini_client_lock = threading.Lock()

def get_gemini_client() -> Optional[Any]:
    """Returns singleton Gemini client configured with 10s deadline."""
    global _gemini_client
    if not GENAI_AVAILABLE:
        return None
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    with _gemini_client_lock:
        if _gemini_client is None:
            try:
                _gemini_client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(
                        timeout=10000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    )
                )
                logger.info("Initialized Gemini Client for native multimodal image OCR.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for OCR: {e}")
                return None
        return _gemini_client

# ---------------------------------------------------------------------------
# Tesseract OCR Setup (Fallback)
# ---------------------------------------------------------------------------
def resolve_tesseract_cmd() -> Optional[str]:
    # 1. Explicit environment variable override
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and (os.path.exists(env_cmd) or shutil.which(env_cmd)):
        return env_cmd

    # 2. Windows standard installations (ONLY on Windows)
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    # 3. Linux/Docker/Render: auto-detect via shutil.which("tesseract")
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    for c in ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]:
        if os.path.exists(c):
            return c

    return None

TESSERACT_CMD = resolve_tesseract_cmd()

try:
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        logger.info(f"Explicitly configured pytesseract.pytesseract.tesseract_cmd = r'{TESSERACT_CMD}'")
    else:
        logger.warning("Tesseract binary not found on host. OCR will fail gracefully if invoked.")
except ImportError as imp_err:
    pytesseract = None
    logger.error(f"Failed to import pytesseract: {imp_err}")


# ---------------------------------------------------------------------------
# Image Preprocessing & Downscaling
# ---------------------------------------------------------------------------
def downscale_and_preprocess_image(image: Image.Image, max_dim: int = 1280) -> Image.Image:
    """
    Downscales any image whose width or height exceeds max_dim (preserving aspect ratio)
    and converts to RGB so it never spikes RAM or times out on Render's 512MB free tier.
    """
    # 1. Convert to RGB if in RGBA, P, LA, CMYK, or other mode
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    elif image.mode == 'L':
        image = image.convert('RGB')

    # 2. Downscale preserving aspect ratio if larger than max_dim
    w, h = image.size
    if w > max_dim or h > max_dim:
        logger.info(f"[OCR] Downscaling image from {w}x{h} to max dimension {max_dim}px (preserving aspect ratio)")
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        logger.info(f"[OCR] New dimensions after downscaling: {image.width}x{image.height}")

    return image


# ---------------------------------------------------------------------------
# Primary Extractor: Gemini 3.5 Flash Lite Vision
# ---------------------------------------------------------------------------
def extract_text_via_gemini_vision(image: Image.Image, timeout_seconds: float = 8.0) -> Tuple[str, bool]:
    """
    Uses gemini-3.5-flash-lite native multimodal vision as the primary fast text extractor.
    Enforces a strict 8s timeout via ThreadPoolExecutor.
    Returns (extracted_text, success).
    """
    client = get_gemini_client()
    if client is None:
        return "", False

    try:
        buf = io.BytesIO()
        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        rgb_image.save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()

        part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        prompt = (
            "Extract all readable text visible in this image verbatim. "
            "Maintain the original layout, headings, and paragraph structure where possible. "
            "Do not include any commentary, analysis, introductory text, markdown backticks, or notes. "
            "If the image contains no readable text, reply with exactly: NONE"
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
            return "", True

        # Strip accidental code fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text).strip()

        return raw_text, True

    except Exception as e:
        logger.warning(f"[OCR] Gemini vision extraction failed or timed out ({timeout_seconds}s limit): {e}")
        return "", False


# ---------------------------------------------------------------------------
# Fallback Extractor: Downscaled Tesseract (--oem 1 --psm 6)
# ---------------------------------------------------------------------------
def extract_text_via_tesseract(image: Image.Image) -> Tuple[str, float, int, bool, str]:
    """
    Fallback extractor using downscaled Tesseract with '--oem 1 --psm 6'.
    Returns (extracted_text, avg_confidence, word_count, success, error_message).
    """
    if pytesseract is None:
        return "", 0.0, 0, False, "pytesseract library is not installed in the active environment."

    tess_cmd = resolve_tesseract_cmd() or TESSERACT_CMD
    if not tess_cmd or (not os.path.exists(tess_cmd) and not shutil.which(tess_cmd)):
        return "", 0.0, 0, False, "Tesseract OCR binary is not installed or available on the host system."

    pytesseract.pytesseract.tesseract_cmd = tess_cmd

    try:
        tess_config = "--oem 1 --psm 6"
        extracted_text = pytesseract.image_to_string(image, config=tess_config)
        clean_result = (extracted_text or "").strip()

        ocr_confidences: list[int] = []
        try:
            data = pytesseract.image_to_data(image, config=tess_config, output_type=pytesseract.Output.DICT)
            if data and 'conf' in data:
                ocr_confidences = [int(c) for c in data['conf'] if int(c) >= 0]
        except Exception as conf_err:
            logger.warning(f"[OCR] Could not compute word-level confidence: {conf_err}")

        avg_confidence = round(sum(ocr_confidences) / len(ocr_confidences), 1) if ocr_confidences else 0.0
        word_count = len(ocr_confidences)

        return clean_result, avg_confidence, word_count, True, ""

    except Exception as tess_err:
        logger.error(f"[OCR] Tesseract extraction error: {tess_err}")
        return "", 0.0, 0, False, str(tess_err)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def extract_text_from_image(image_input: Union[str, Image.Image], return_details: bool = False):
    """
    Extracts text from an image file path or PIL Image.
    
    Pipeline:
    1. Downscales image if width or height > 1280px (preserving aspect ratio) and converts to RGB.
    2. Uses gemini-3.5-flash-lite (multimodal vision, 8s timeout) as primary extractor when GEMINI_API_KEY is set.
    3. Automatically falls back to downscaled Tesseract (--oem 1 --psm 6).
    4. Computes confidence and publisher detection metrics.
    
    Returns:
        If return_details is False:
            (extracted_text, success, message)
        If return_details is True:
            (extracted_text, success, message, details_dict)
    """
    file_name = "image_object"
    image_path = None

    if isinstance(image_input, str):
        image_path = image_input
        file_name = os.path.basename(image_path)
        ext = file_name.rsplit('.', 1)[1].lower() if '.' in file_name else "unknown"
        logger.info(f"Received file name: {file_name} (ext: {ext})")

        if not os.path.exists(image_path):
            err_msg = f"File not found on disk: {image_path}"
            logger.error(f"OCR failed: {err_msg}")
            details = {
                "confidence": 0.0, "isLowConfidence": False, "warning": "",
                "qualityWarning": "", "previewText": "", "wordCount": 0
            }
            if return_details:
                return "", False, err_msg, details
            return "", False, err_msg

        try:
            image = Image.open(image_path)
        except Exception as open_err:
            err_msg = f"Failed to open image: {open_err}"
            logger.error(f"OCR failed: {err_msg}")
            details = {
                "confidence": 0.0, "isLowConfidence": False, "warning": "",
                "qualityWarning": "", "previewText": "", "wordCount": 0
            }
            if return_details:
                return "", False, err_msg, details
            return "", False, err_msg
    else:
        image = image_input

    # 1. Downscale & convert to RGB (never exceeds 1280px in either dimension)
    image = downscale_and_preprocess_image(image, max_dim=1280)

    # Save downscaled version back to disk if path is provided so subsequent tasks don't reload giant images
    if image_path:
        try:
            save_format = "PNG" if image_path.lower().endswith(".png") else "JPEG"
            image.save(image_path, format=save_format, quality=85)
        except Exception as save_err:
            logger.warning(f"[OCR] Could not re-save downscaled image to disk: {save_err}")

    details = {
        "confidence": 0.0,
        "isLowConfidence": False,
        "warning": "",
        "qualityWarning": "",
        "previewText": "",
        "wordCount": 0,
        "extractor": "none"
    }

    clean_result = ""
    used_gemini = False
    ocr_success = False
    err_message = ""
    avg_confidence = 0.0
    word_count = 0

    # 2. Try Gemini 3.5 Flash Lite Vision (Primary)
    if os.getenv("GEMINI_API_KEY", "").strip() and GENAI_AVAILABLE:
        logger.info(f"[OCR] Running gemini-3.5-flash-lite vision extraction for {file_name}")
        gemini_text, gemini_ok = extract_text_via_gemini_vision(image, timeout_seconds=8.0)
        if gemini_ok:
            clean_result = gemini_text.strip()
            used_gemini = True
            ocr_success = True
            word_count = len(clean_result.split()) if clean_result else 0
            avg_confidence = 95.0 if clean_result else 0.0
            details["extractor"] = "gemini-3.5-flash-lite"
            logger.info(f"[OCR] Gemini vision succeeded for {file_name}: {len(clean_result)} chars")
        else:
            logger.warning(f"[OCR] Gemini vision failed or timed out for {file_name}. Falling back to Tesseract.")

    # 3. Fallback: Downscaled Tesseract (--oem 1 --psm 6)
    if not used_gemini:
        logger.info(f"[OCR] Running Tesseract OCR (--oem 1 --psm 6) for {file_name}")
        tess_text, tess_conf, tess_words, tess_ok, tess_err = extract_text_via_tesseract(image)
        if tess_ok:
            clean_result = tess_text
            avg_confidence = tess_conf
            word_count = tess_words
            ocr_success = True
            details["extractor"] = "tesseract"
            logger.info(f"[OCR] Tesseract succeeded for {file_name}: {len(clean_result)} chars, conf {avg_confidence}%")
        else:
            ocr_success = False
            err_message = tess_err or "OCR extraction failed across both Gemini and Tesseract."
            logger.error(f"[OCR] Tesseract also failed for {file_name}: {err_message}")

    # If neither extractor succeeded
    if not ocr_success:
        if return_details:
            return "", False, err_message, details
        return "", False, err_message

    # 4. Confidence & Warnings
    is_low_conf = False
    warning = ""
    quality_warning = ""

    if not clean_result:
        is_low_conf = True
        quality_warning = "Text extraction quality may be low."
        warning = "Some text may have been extracted inaccurately from the image."
    elif not used_gemini and word_count > 0 and (avg_confidence < 65.0 or (word_count < 4 and avg_confidence < 75.0)):
        is_low_conf = True
        quality_warning = "Text extraction quality may be low."
        warning = "Some text may have been extracted inaccurately from the image."
        logger.warning(f"[OCR] Low confidence detected: {avg_confidence}% (words: {word_count})")

    publisher_info = {}
    if image_path and os.path.exists(image_path):
        publisher_info = detect_publisher_from_image(image_path, full_ocr_text=clean_result)

    details.update({
        "confidence": avg_confidence,
        "isLowConfidence": is_low_conf,
        "warning": warning,
        "qualityWarning": quality_warning,
        "previewText": clean_result[:500],
        "wordCount": word_count,
        "publisherInfo": publisher_info,
        "publisher": publisher_info.get("publisher", "Unknown"),
        "domain": publisher_info.get("domain", "N/A")
    })

    first_300 = clean_result[:300]
    logger.info(f"OCR completed for {file_name} (Length: {len(clean_result)})")
    if first_300:
        logger.info(f"First 300 characters of extracted text:\n{first_300}")

    if not clean_result:
        msg = "Image analyzed, but no discernible text was found."
        logger.info(f"OCR warning: {msg}")
        if return_details:
            return "", True, msg, details
        return "", True, msg

    if return_details:
        return clean_result, True, "Text extracted successfully.", details
    return clean_result, True, "Text extracted successfully."
