import os
import sys
import logging
import traceback
from PIL import Image
from typing import Tuple

from utils.image_publisher_detector import detect_publisher_from_image

logger = logging.getLogger("lucid.ocr")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [OCR] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configure tesseract_cmd dynamically: support Windows, Linux (Docker/Render), and custom env var
import shutil
from typing import Optional

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


def extract_text_from_image(image_path: str, return_details: bool = False):
    """
    Extracts text from an image file (png, jpg, jpeg) using Tesseract OCR.
    Explicitly uses 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'.
    
    Logs:
    - Received file name
    - File extension
    - OCR started
    - OCR completed
    - Extracted text length
    - First 300 characters of extracted text
    - Exact OCR error if failed

    Returns:
        If return_details is False:
            (extracted_text, success, message)
        If return_details is True:
            (extracted_text, success, message, details_dict)
    """
    file_name = os.path.basename(image_path)
    ext = file_name.rsplit('.', 1)[1].lower() if '.' in file_name else "unknown"

    logger.info(f"Received file name: {file_name}")
    logger.info(f"File extension: {ext}")
    print(f"[OCR] Received file name: {file_name}")
    print(f"[OCR] File extension: {ext}")

    details = {
        "confidence": 0.0,
        "isLowConfidence": False,
        "warning": "",
        "qualityWarning": "",
        "previewText": "",
        "wordCount": 0
    }

    if not os.path.exists(image_path):
        err_msg = f"File not found on disk: {image_path}"
        logger.error(f"OCR failed: {err_msg}")
        print(f"[OCR ERROR] {err_msg}")
        if return_details:
            return "", False, err_msg, details
        return "", False, err_msg

    if pytesseract is None:
        err_msg = "pytesseract library is not installed in the active Python environment."
        logger.error(f"OCR failed: {err_msg}")
        print(f"[OCR ERROR] {err_msg}")
        if return_details:
            return "", False, err_msg, details
        return "", False, err_msg

    tess_cmd = resolve_tesseract_cmd() or TESSERACT_CMD
    if not tess_cmd or (not os.path.exists(tess_cmd) and not shutil.which(tess_cmd)):
        err_msg = "Tesseract OCR binary is not installed or available on the host system."
        logger.warning(f"OCR skipped: {err_msg}")
        if return_details:
            return "", False, err_msg, details
        return "", False, err_msg

    # Ensure explicit path is set before execution
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

    try:
        logger.info(f"OCR started on {file_name}")
        print(f"[OCR] OCR started on {file_name}")

        image = Image.open(image_path)
        # Convert RGBA / P mode images to RGB for tesseract compatibility
        if image.mode in ('RGBA', 'P', 'LA'):
            image = image.convert('RGB')

        # 1. Extract text
        extracted_text = pytesseract.image_to_string(image)
        clean_result = extracted_text.strip()
        text_length = len(clean_result)
        first_300 = clean_result[:300]
        preview_500 = clean_result[:500]

        # 2. Extract confidence scores using image_to_data
        ocr_confidences: list[int] = []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            if data and 'conf' in data:
                ocr_confidences = [int(c) for c in data['conf'] if int(c) >= 0]
        except Exception as conf_err:
            logger.warning(f"Could not compute word-level confidence: {conf_err}")

        avg_confidence = round(sum(ocr_confidences) / len(ocr_confidences), 1) if ocr_confidences else 0.0
        word_count = len(ocr_confidences)

        # Confidence is poor if avg < 65% or very few confident words found
        is_low_conf = False
        warning = ""
        quality_warning = ""

        if not clean_result:
            is_low_conf = True
            quality_warning = "Text extraction quality may be low."
            warning = "Some text may have been extracted inaccurately from the image."
        elif word_count > 0 and (avg_confidence < 65.0 or (word_count < 4 and avg_confidence < 75.0)):
            is_low_conf = True
            quality_warning = "Text extraction quality may be low."
            warning = "Some text may have been extracted inaccurately from the image."
            logger.warning(f"[OCR] Low confidence detected: {avg_confidence}% (words: {word_count})")
            print(f"[OCR WARNING] Low confidence ({avg_confidence}%): {quality_warning}")

        publisher_info = detect_publisher_from_image(image_path, full_ocr_text=clean_result)

        details = {
            "confidence": avg_confidence,
            "isLowConfidence": is_low_conf,
            "warning": warning,
            "qualityWarning": quality_warning,
            "previewText": preview_500,
            "wordCount": word_count,
            "publisherInfo": publisher_info,
            "publisher": publisher_info.get("publisher", "Unknown"),
            "domain": publisher_info.get("domain", "N/A")
        }

        logger.info(f"OCR completed for {file_name}")
        logger.info(f"Extracted text length: {text_length}")
        logger.info(f"First 300 characters of extracted text:\n{first_300}")
        print(f"[OCR] OCR completed for {file_name}")
        print(f"[OCR] Extracted text length: {text_length}")
        print(f"[OCR] First 300 characters of extracted text:\n{first_300}")

        if not clean_result:
            msg = "Image analyzed, but no discernible text was found."
            logger.info(f"OCR warning: {msg}")
            print(f"[OCR] {msg}")
            if return_details:
                return "", True, msg, details
            return "", True, msg

        if return_details:
            return clean_result, True, "Text extracted successfully.", details
        return clean_result, True, "Text extracted successfully."

    except Exception as ocr_err:
        exact_err = f"{type(ocr_err).__name__}: {str(ocr_err)}"
        tb = traceback.format_exc()
        logger.error(f"OCR failed for {file_name}. Exact OCR error: {exact_err}")
        logger.error(f"Traceback:\n{tb}")
        print(f"[OCR ERROR] OCR failed for {file_name}. Exact OCR error: {exact_err}")
        print(f"[OCR ERROR] Traceback:\n{tb}")
        if return_details:
            return "", False, exact_err, details
        return "", False, exact_err
