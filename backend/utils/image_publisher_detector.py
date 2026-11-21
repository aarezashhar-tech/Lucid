"""
Watermark, Logo, and Publisher Detection inside uploaded images.
Detects broadcast bugs, corner watermarks, publisher logos (BBC, Reuters, Al Jazeera, etc.),
and header/footer credits.
"""
import os
import re
import logging
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("lucid.publisher_detector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [PublisherDetector] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Canonical publisher watermark patterns
PUBLISHER_WATERMARK_PATTERNS: Dict[str, list] = {
    "BBC": [
        r'\bBBC(?:\s+NEWS|\s+WORLD|\s+SPORT|\s+BREAKING|\s+MONITORING)?\b',
        r'\bBBC\.CO(?:\.UK)?\b',
        r'\bBBC\.COM\b',
    ],
    "Reuters": [
        r'\bREUTERS\b',
        r'\bTHOMSON\s+REUTERS\b',
        r'\(REUTERS\)',
        r'\bREUTERS\.COM\b',
    ],
    "Al Jazeera": [
        r'\bAL\s*JAZEERA\b',
        r'\bALJAZEERA\b',
        r'\bALJAZEERA\.COM\b',
        r'\bALJAZEERA\.NET\b',
    ],
    "Associated Press": [
        r'\bASSOCIATED\s+PRESS\b',
        r'\bAP\s+NEWS\b',
        r'\bAPNEWS\.COM\b',
    ],
    "CNN": [
        r'\bCNN(?:\s+NEWS|\s+WORLD|\s+ORIGINAL)?\b',
        r'\bCNN\.COM\b',
    ],
    "Bloomberg": [
        r'\bBLOOMBERG(?:\s+NEWS)?\b',
        r'\bBLOOMBERG\.COM\b',
    ],
    "The Guardian": [
        r'\bTHE\s+GUARDIAN\b',
        r'\bGUARDIAN\.CO(?:\.UK)?\b',
        r'\bTHEGUARDIAN\.COM\b',
    ],
    "The New York Times": [
        r'\bNEW\s+YORK\s+TIMES\b',
        r'\bNYTIMES(?:\.COM)?\b',
    ],
    "The Washington Post": [
        r'\bWASHINGTON\s+POST\b',
        r'\bWAPO(?:\.COM)?\b',
    ]
}

PUBLISHER_DOMAINS: Dict[str, str] = {
    "BBC": "bbc.com",
    "Reuters": "reuters.com",
    "Al Jazeera": "aljazeera.com",
    "Associated Press": "apnews.com",
    "CNN": "cnn.com",
    "Bloomberg": "bloomberg.com",
    "The Guardian": "theguardian.com",
    "The New York Times": "nytimes.com",
    "The Washington Post": "washingtonpost.com"
}


def match_publisher_text(text: str) -> Optional[str]:
    """Matches text against known news publisher watermark/logo patterns."""
    if not text:
        return None
    for pub, patterns in PUBLISHER_WATERMARK_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return pub
    return None


def detect_publisher_from_image(image_path: str, full_ocr_text: str = "") -> Dict[str, Any]:
    """
    Detects publisher name, watermark, or logo inside an image.
    1. Inspects full OCR text for publisher watermarks.
    2. Inspects corner crops (Top-Left, Top-Right, Bottom-Left, Bottom-Right) and banners
       where TV bugs and news logos are located.
    3. Applies contrast enhancement, upscaling, and binary thresholding for low-contrast watermarks.
    
    Returns:
        {
            "detected": bool,
            "publisher": str ("BBC", "Reuters", "Al Jazeera", or "Unknown"),
            "domain": str ("bbc.com", "reuters.com", "aljazeera.com", or "N/A"),
            "confidence": int,
            "confidenceLevel": str ("High", "Medium", "Low"),
            "method": str
        }
    """
    # 1. Check full OCR text first
    if full_ocr_text:
        pub = match_publisher_text(full_ocr_text)
        if pub:
            logger.info(f"Publisher detected from full OCR text: {pub}")
            return {
                "detected": True,
                "publisher": pub,
                "domain": PUBLISHER_DOMAINS.get(pub, "N/A"),
                "confidence": 95,
                "confidenceLevel": "High",
                "method": "Full Image OCR Text Match"
            }

    if not os.path.exists(image_path):
        return {
            "detected": False,
            "publisher": "Unknown",
            "domain": "N/A",
            "confidence": 0,
            "confidenceLevel": "Low",
            "method": "None"
        }

    try:
        import pytesseract
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size

        # Corner and banner regions where channel bugs, watermarks, and logos appear
        crops = [
            ("Top-Left", img.crop((0, 0, int(w * 0.35), int(h * 0.25)))),
            ("Top-Right", img.crop((int(w * 0.65), 0, w, int(h * 0.25)))),
            ("Bottom-Left", img.crop((0, int(h * 0.75), int(w * 0.35), h))),
            ("Bottom-Right", img.crop((int(w * 0.65), int(h * 0.75), w, h))),
            ("Top-Banner", img.crop((0, 0, w, int(h * 0.15)))),
            ("Bottom-Banner", img.crop((0, int(h * 0.85), w, h)))
        ]

        for region_name, crop_img in crops:
            cw, ch = crop_img.size
            if cw < 10 or ch < 10:
                continue

            # Upscale 2x for small logo legibility
            scaled = crop_img.resize((cw * 2, ch * 2), Image.Resampling.LANCZOS)

            # Normal crop OCR
            try:
                crop_text = pytesseract.image_to_string(scaled, config="--psm 6")
                pub = match_publisher_text(crop_text)
                if pub:
                    logger.info(f"Publisher detected via corner watermark OCR ({region_name}): {pub}")
                    return {
                        "detected": True,
                        "publisher": pub,
                        "domain": PUBLISHER_DOMAINS.get(pub, "N/A"),
                        "confidence": 96,
                        "confidenceLevel": "High",
                        "method": f"Corner Watermark OCR ({region_name})"
                    }
            except Exception as ocr_e:
                logger.debug(f"Corner OCR error for {region_name}: {ocr_e}")

            # Contrast / Inverted threshold OCR (for white-on-dark or translucent logos)
            try:
                gray = np.array(scaled.convert('L'))
                _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                thresh_text = pytesseract.image_to_string(thresh, config="--psm 6")
                pub = match_publisher_text(thresh_text)
                if pub:
                    logger.info(f"Publisher detected via watermark contrast filter ({region_name}): {pub}")
                    return {
                        "detected": True,
                        "publisher": pub,
                        "domain": PUBLISHER_DOMAINS.get(pub, "N/A"),
                        "confidence": 96,
                        "confidenceLevel": "High",
                        "method": f"Watermark Contrast Filter ({region_name})"
                    }
            except Exception as thresh_e:
                logger.debug(f"Threshold OCR error for {region_name}: {thresh_e}")

    except Exception as e:
        logger.error(f"Error during image publisher detection: {e}")

    return {
        "detected": False,
        "publisher": "Unknown",
        "domain": "N/A",
        "confidence": 0,
        "confidenceLevel": "Low",
        "method": "None"
    }
