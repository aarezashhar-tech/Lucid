"""
Language Detection and Multilingual Processing Service.
Detects language across 50+ languages, provides English translation bridge
for news wire verification when needed, and preserves original text.
"""
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("lucid.language_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [LanguageService] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Mapping of ISO-639-1 language codes to descriptive English names
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "fa": "Persian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "el": "Greek",
    "he": "Hebrew",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "th": "Thai",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "sw": "Swahili"
}

RTL_LANGUAGES = {"ar", "he", "fa", "ur"}


def detect_language_heuristics(text: str) -> Dict[str, Any]:
    """Fallback heuristic script and token detector when langdetect is uncertain or short."""
    if not text:
        return {"code": "en", "name": "English", "confidence": 0.5, "direction": "ltr"}

    # Script regex checks
    if re.search(r'[\u0600-\u06FF]', text):
        return {"code": "ar", "name": "Arabic", "confidence": 0.95, "direction": "rtl"}
    if re.search(r'[\u0900-\u097F]', text):
        return {"code": "hi", "name": "Hindi", "confidence": 0.95, "direction": "ltr"}
    if re.search(r'[\u0980-\u09FF]', text):
        return {"code": "bn", "name": "Bengali", "confidence": 0.95, "direction": "ltr"}
    if re.search(r'[\u0400-\u04FF]', text):
        return {"code": "ru", "name": "Russian", "confidence": 0.95, "direction": "ltr"}
    if re.search(r'[\u3040-\u30FF]', text):
        return {"code": "ja", "name": "Japanese", "confidence": 0.95, "direction": "ltr"}
    if re.search(r'[\uAC00-\uD7AF]', text):
        return {"code": "ko", "name": "Korean", "confidence": 0.95, "direction": "ltr"}
    if re.search(r'[\u4E00-\u9FFF]', text):
        return {"code": "zh", "name": "Chinese", "confidence": 0.95, "direction": "ltr"}

    # Common European stopword heuristics
    lower = text.lower()
    words = set(re.findall(r'\b[a-záéíóúüñäößàèùâêîôûç]+\b', lower))

    spanish_markers = {'el', 'la', 'los', 'las', 'un', 'una', 'en', 'para', 'por', 'con', 'del', 'al', 'presidente', 'gobierno'}
    if len(words.intersection(spanish_markers)) >= 2:
        return {"code": "es", "name": "Spanish", "confidence": 0.90, "direction": "ltr"}

    french_markers = {'le', 'la', 'les', 'des', 'du', 'dans', 'pour', 'avec', 'sur', 'est', 'sont', 'président', 'gouvernement'}
    if len(words.intersection(french_markers)) >= 2:
        return {"code": "fr", "name": "French", "confidence": 0.90, "direction": "ltr"}

    german_markers = {'der', 'die', 'das', 'und', 'mit', 'für', 'auf', 'ist', 'sind', 'nicht', 'bundesregierung', 'präsident'}
    if len(words.intersection(german_markers)) >= 2:
        return {"code": "de", "name": "German", "confidence": 0.90, "direction": "ltr"}

    return {"code": "en", "name": "English", "confidence": 0.85, "direction": "ltr"}


class LanguageService:
    """Manages language identification and transparent multi-language processing."""

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detects the natural language of the input text safely."""
        if not text or not text.strip():
            return {
                "languageCode": "en",
                "detectedLanguage": "English",
                "confidence": 1.0,
                "isEnglish": True,
                "direction": "ltr"
            }

        clean = re.sub(r'https?://\S+', '', text).strip()
        if len(clean) < 4:
            clean = text.strip()

        code = "en"
        confidence = 0.85

        try:
            from langdetect import detect_langs
            predictions = detect_langs(clean)
            if predictions:
                top = predictions[0]
                code = str(top.lang).lower()
                confidence = round(float(top.prob), 2)
        except Exception as e:
            logger.debug(f"langdetect fallback triggered: {e}")
            heur = detect_language_heuristics(clean)
            code = heur["code"]
            confidence = heur["confidence"]

        lang_name = LANGUAGE_NAMES.get(code, code.upper())
        is_english = code in ("en", "eng")
        direction = "rtl" if code in RTL_LANGUAGES else "ltr"

        return {
            "code": code,
            "languageCode": code,
            "language": lang_name,
            "detectedLanguage": lang_name,
            "confidence": confidence,
            "isEnglish": is_english,
            "isForeignLanguage": not is_english,
            "direction": direction
        }

    def translate_to_english(self, text: str, source_lang: str = "auto") -> Optional[str]:
        """
        Attempts machine translation to English for verification pipeline.
        Gracefully returns None on rate limit, offline status, or error without crashing.
        """
        if not text or not text.strip():
            return None

        # Truncate to reasonable length for translation API
        to_translate = text[:3000].strip()

        try:
            from deep_translator import GoogleTranslator
            src = source_lang if source_lang in LANGUAGE_NAMES and source_lang != "en" else "auto"
            translated = GoogleTranslator(source=src, target="en").translate(to_translate)
            if translated and translated.strip():
                logger.info(f"Successfully translated {len(to_translate)} chars from {src} to English.")
                return translated.strip()
        except Exception as trans_err:
            logger.warning(f"Translation service paused or rate-limited ({type(trans_err).__name__}). Falling back to native multilingual analysis.")

        return None

    def process_content(self, text: str) -> Dict[str, Any]:
        """
        Processes content for multi-language handling:
        Detects language, optionally translates for backend verification,
        and returns consolidated metadata.
        """
        detection = self.detect_language(text)
        is_english = detection["isEnglish"]

        translated_text = None
        if not is_english and len(text.strip()) > 5:
            translated_text = self.translate_to_english(text, source_lang=detection["languageCode"])

        return {
            "code": detection["languageCode"],
            "languageCode": detection["languageCode"],
            "language": detection["detectedLanguage"],
            "detectedLanguage": detection["detectedLanguage"],
            "confidence": detection["confidence"],
            "isEnglish": is_english,
            "isForeignLanguage": not is_english,
            "direction": detection["direction"],
            "originalText": text,
            "translatedText": translated_text if translated_text else text,
            "analysisText": translated_text if translated_text else text,
            "wasTranslated": bool(translated_text and translated_text != text)
        }


language_service = LanguageService()

def get_language_service() -> LanguageService:
    return language_service
