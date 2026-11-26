import os
import re
import json
import time
import uuid
import logging
import threading
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    from firebase_admin import credentials as firebase_credentials
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    firebase_admin = None
    firebase_auth = None
    firebase_credentials = None
    FIREBASE_ADMIN_AVAILABLE = False

# Load environment variables from backend/.env or root .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

from utils.image_ocr import extract_text_from_image
from utils.pdf_parser import extract_text_from_pdf
from utils.url_extractor import fetch_and_extract_article, validate_url_security
from services.ai_service import get_verification_engine
from services.ai_reasoning import TRANSPARENCY_WARNING

FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

app = Flask(
    __name__,
    static_folder=FRONTEND_DIST if os.path.exists(FRONTEND_DIST) else None,
    static_url_path=""
)

# Trust exactly one proxy hop (e.g. Render's load balancer) so request.remote_addr
# reflects the real client IP without trusting raw client-supplied X-Forwarded-For.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

# Production / Environment CORS Configuration
is_production = os.getenv("FLASK_ENV") == "production" or os.getenv("ENV") == "production"

cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    allowed_origins = [
        origin.strip().rstrip("/")
        for origin in cors_origins_env.split(",")
        if origin.strip()
    ]
else:
    # Strict Vercel production origin + local development (never wildcard *)
    allowed_origins = [
        "https://weblucid.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
            "expose_headers": ["Content-Type", "Authorization"]
        }
    },
    supports_credentials=True
)

@app.route('/api/<path:subpath>', methods=['OPTIONS'])
def options_preflight(subpath):
    response = jsonify({"status": "preflight-ok"})
    return response, 200

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        origin_clean = origin.rstrip("/")
        if origin_clean in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    # Never set Access-Control-Allow-Origin to "*"
    return response

# ==========================================
# SECURITY: RATE LIMITING & TOKEN VERIFICATION
# ==========================================

class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter per user/IP."""
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = defaultdict(list)

    def is_allowed(self, identifier: str, endpoint_category: str, max_requests: int, window_seconds: int = 60) -> Tuple[bool, int]:
        now = time.time()
        key = f"{identifier}:{endpoint_category}"
        with self.lock:
            cutoff = now - window_seconds
            recent = [t for t in self.requests.get(key, []) if t > cutoff]
            if recent:
                self.requests[key] = recent
            else:
                # Delete empty keys instead of leaving stale buckets behind.
                self.requests.pop(key, None)
            if max_requests <= 0:
                return False, max(window_seconds, 1)
            if len(recent) >= max_requests:
                retry_after = int(recent[0] + window_seconds - now) + 1
                return False, max(retry_after, 1)
            recent.append(now)
            self.requests[key] = recent
            return True, 0

rate_limiter = SlidingWindowRateLimiter()

# In-memory TTL cache for verified Firebase ID tokens: token -> (user_info, expiry_ts)
TOKEN_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
TOKEN_CACHE_LOCK = threading.Lock()
TOKEN_CACHE_MAX_ENTRIES = 500

FIREBASE_SERVICE_ACCOUNT_ENV = "FIREBASE_SERVICE_ACCOUNT_JSON"
_firebase_app = None
_firebase_project_id: Optional[str] = None
_firebase_init_lock = threading.Lock()
_firebase_init_attempted = False


def _get_firebase_app():
    """Initializes and returns the Firebase Admin app from the service-account JSON env var."""
    global _firebase_app, _firebase_project_id, _firebase_init_attempted

    if not FIREBASE_ADMIN_AVAILABLE:
        return None

    with _firebase_init_lock:
        if _firebase_init_attempted:
            return _firebase_app
        _firebase_init_attempted = True

        raw_service_account = os.getenv(FIREBASE_SERVICE_ACCOUNT_ENV, "").strip()
        if not raw_service_account:
            app.logger.warning(
                f"{FIREBASE_SERVICE_ACCOUNT_ENV} is not configured; Firebase tokens will be rejected."
            )
            return None

        try:
            service_account_info = json.loads(raw_service_account)
            project_id = str(service_account_info.get("project_id", "")).strip()
            if not project_id:
                raise ValueError("service account JSON is missing project_id")
            credential = firebase_credentials.Certificate(service_account_info)
            _firebase_app = firebase_admin.initialize_app(credential, {"projectId": project_id})
            _firebase_project_id = project_id
            app.logger.info(f"Firebase Admin initialized for project '{project_id}'.")
        except Exception as err:
            _firebase_app = None
            _firebase_project_id = None
            app.logger.error(f"Failed to initialize Firebase Admin from {FIREBASE_SERVICE_ACCOUNT_ENV}: {err}")

        return _firebase_app


def _purge_expired_tokens(now: float) -> None:
    """Removes expired token cache entries and bounds cache size to max 500 entries."""
    with TOKEN_CACHE_LOCK:
        expired = [tok for tok, (_, expiry) in TOKEN_CACHE.items() if now >= expiry]
        for tok in expired:
            TOKEN_CACHE.pop(tok, None)
        if len(TOKEN_CACHE) > TOKEN_CACHE_MAX_ENTRIES:
            sorted_by_expiry = sorted(TOKEN_CACHE.items(), key=lambda it: it[1][1])
            excess = len(TOKEN_CACHE) - TOKEN_CACHE_MAX_ENTRIES
            for tok, _ in sorted_by_expiry[:excess]:
                TOKEN_CACHE.pop(tok, None)


def _guest_auth_context(ip: str) -> Dict[str, Any]:
    return {
        "is_authenticated": False,
        "user_id": None,
        "email": None,
        "is_guest": True,
        "identifier": ip
    }


def get_auth_context() -> Dict[str, Any]:
    """
    Extracts and verifies a Firebase ID token from Authorization: Bearer <token>
    using firebase-admin verify_id_token(). Tokens issued for a different Firebase
    project are rejected. Guests are identified by request.remote_addr only.
    """
    ip = request.remote_addr or "127.0.0.1"
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return _guest_auth_context(ip)

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return _guest_auth_context(ip)

    now = time.time()
    _purge_expired_tokens(now)
    with TOKEN_CACHE_LOCK:
        cached = TOKEN_CACHE.get(token)
        if cached:
            info, expiry = cached
            if now < expiry:
                return info
            TOKEN_CACHE.pop(token, None)

    firebase_app = _get_firebase_app()
    if firebase_app is None or firebase_auth is None:
        app.logger.warning("Firebase Admin is unavailable; treating request as unauthenticated.")
        return _guest_auth_context(ip)

    try:
        decoded = firebase_auth.verify_id_token(token, app=firebase_app)
        audience = str(decoded.get("aud", "")).strip()
        if not _firebase_project_id or audience != _firebase_project_id:
            app.logger.warning(
                f"Rejected Firebase token for unexpected project audience '{audience}' "
                f"(expected '{_firebase_project_id}')."
            )
            return _guest_auth_context(ip)

        user_id = decoded.get("uid") or decoded.get("sub") or decoded.get("user_id")
        email = decoded.get("email", "")
        exp = float(decoded.get("exp", now + 300))
        info = {
            "is_authenticated": True,
            "user_id": user_id,
            "email": email,
            "is_guest": False,
            "identifier": user_id or email or ip
        }
        with TOKEN_CACHE_LOCK:
            if len(TOKEN_CACHE) >= TOKEN_CACHE_MAX_ENTRIES:
                oldest_tok = min(TOKEN_CACHE.keys(), key=lambda k: TOKEN_CACHE[k][1], default=None)
                if oldest_tok:
                    TOKEN_CACHE.pop(oldest_tok, None)
            TOKEN_CACHE[token] = (info, min(exp, now + 300))
        return info
    except Exception as err:
        app.logger.warning(f"Failed to verify Firebase token: {err}")

    # Fall back to unauthenticated / guest with client IP identifier
    return _guest_auth_context(ip)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf', 'txt'}
PROHIBITED_EXTENSIONS = {
    'exe', 'bat', 'cmd', 'sh', 'bash', 'bin', 'dll', 'msi', 'com', 'ps1',
    'vbs', 'vbe', 'js', 'jse', 'wsf', 'wsh', 'scr', 'pif', 'py', 'pl', 'php'
}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB max

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_uploaded_file(file_storage) -> Tuple[bool, str]:
    """
    Validates uploaded file: verifies extension, magic bytes header,
    blocks executable types, and checks non-empty size.
    """
    if not file_storage or not file_storage.filename:
        return False, "No file provided in request."

    filename = secure_filename(file_storage.filename)
    if not filename or '.' not in filename:
        return False, "Invalid filename or missing extension."

    ext = filename.rsplit('.', 1)[1].lower()
    if ext in PROHIBITED_EXTENSIONS:
        return False, "Executable and script files are strictly prohibited."

    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File format .{ext} is not allowed. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Inspect magic bytes header
    header = file_storage.read(2048)
    file_storage.seek(0)

    if len(header) == 0:
        return False, "The uploaded file is empty."

    if ext in ('jpg', 'jpeg'):
        if not header.startswith(b'\xff\xd8\xff'):
            return False, "File is not a valid JPEG image."
    elif ext == 'png':
        if not header.startswith(b'\x89PNG\r\n\x1a\n'):
            return False, "File is not a valid PNG image."
    elif ext == 'pdf':
        if not header.startswith(b'%PDF-'):
            return False, "File is not a valid PDF document."
    elif ext == 'txt':
        if b'\x00' in header:
            return False, "Text files must not contain binary null bytes."
        try:
            header.decode('utf-8')
        except UnicodeDecodeError:
            return False, "Text files must be valid UTF-8/ASCII."

    return True, ""

@app.route('/api/health', methods=['GET'])
@app.route('/api/health/', methods=['GET'])
def health():
    engine = get_verification_engine()
    return jsonify({
        "status": "healthy",
        "service": "Lucid Evidence-Based Fact Verification Platform",
        "architecture": {
            "layer1": "Source Reputation Engine",
            "layer2": "Claim Extraction Engine",
            "layer3": "Evidence Verification",
            "layer4": "Multi-Factor Scoring (70% Evidence, 10% Source, 10% Consistency, 5% ML, 5% Language)",
            "layer5": "Modern 6-Tier Verdict Generation",
            "layer6": "Investigation Dashboard Support (Overview, Claims, Evidence, Sources, Reasoning)",
            "layer7": "Transparency Warning Disclaimers",
            "layer8": "Multi-Perspective AI Reasoning"
        },
        "modelLoaded": engine.model is not None,
        "vectorizerLoaded": engine.vectorizer is not None,
        "supportedUploads": list(ALLOWED_EXTENSIONS),
        "disclaimer": TRANSPARENCY_WARNING,
        "futureReady": {
            "geminiConfigured": bool(os.getenv("GEMINI_API_KEY")),
            "openAiConfigured": bool(os.getenv("OPENAI_API_KEY")),
            "factCheckConfigured": bool(os.getenv("FACTCHECK_API_KEY")),
            "gnewsConfigured": bool(os.getenv("GNEWS_API_KEY")),
            "wikipediaConfigured": True
        }
    })

def run_shared_verification_pipeline(
    text: str,
    source_hint: str = "",
    url: str = "",
    title: str = "",
    input_type: str = "text",
    is_image: bool = False,
    file_type: str = "",
    image_publisher = None
):
    """
    Shared Unified Pipeline for both pasted text and URL inputs (Task B).
    Ensures identical claim extraction, evidence retrieval, and scoring.
    Logs extracted text length and each score component to the server logs.
    """
    engine = get_verification_engine()
    app.logger.info(
        f"[SharedPipeline] Processing {input_type.upper()} input | Extracted Text Length: {len(text)} chars | Title: '{title[:60]}'"
    )
    print(
        f"[SharedPipeline] Processing {input_type.upper()} input | Extracted Text Length: {len(text)} chars | Title: '{title[:60]}'"
    )

    result = engine.analyze(
        text=text,
        source_hint=source_hint,
        url_or_domain=url,
        article_title=title,
        is_image=is_image,
        file_type=file_type,
        image_publisher=image_publisher
    )

    factors = result.get("scoringFactors", {})
    ml_contrib = factors.get("mlPrediction", {}).get("contribution", 0)
    src_contrib = factors.get("sourceReputation", {}).get("contribution", 0)
    evi_contrib = factors.get("evidenceStrength", {}).get("contribution", 0)
    con_contrib = factors.get("claimConsistency", {}).get("contribution", 0)
    lan_contrib = factors.get("languageQuality", {}).get("contribution", 0)

    ts_display = result.get('trustScore')
    ts_str = f"{ts_display}/100" if isinstance(ts_display, (int, float)) else str(ts_display)
    log_line = (
        f"[SharedPipeline Scoring] {input_type.upper()} -> Text Length: {len(text)} chars | Trust Score: {ts_str} | Verdict: {result.get('verdict')} | "
        f"Components: [Evidence(70%): {evi_contrib}, Source(10%): {src_contrib}, Consistency(10%): {con_contrib}, ML(5%): {ml_contrib}, Language(5%): {lan_contrib}]"
    )
    app.logger.info(log_line)
    print(log_line)

    return result

@app.route('/api/analyze', methods=['POST'])
@app.route('/api/analyze/', methods=['POST'])
def analyze():
    """
    Multi-Layer Evidence-Based Analysis Endpoint:
    Accepts text, optional source_hint or url.
    Returns 5-factor scoring, atomic claims, evidence corroboration,
    source reputation, and AI reasoning.
    Protected by Firebase token auth verification and sliding-window rate limiting.
    """
    try:
        # 1. Auth context & Rate limiting (stricter limits for unauthenticated guests)
        auth_ctx = get_auth_context()
        limit = 30 if auth_ctx["is_authenticated"] else 8
        allowed, retry_after = rate_limiter.is_allowed(auth_ctx["identifier"], "verify", limit, 60)
        if not allowed:
            return jsonify({
                "error": f"Rate limit reached ({limit} requests/min). Please wait {retry_after} seconds before submitting another verification.",
                "retryAfter": retry_after
            }), 429

        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()
        source_hint = data.get("source", "") or data.get("sourceHint", "")
        url_input = data.get("url", "")

        if not text:
            return jsonify({
                "error": "No text provided for analysis",
                "prediction": "UNKNOWN",
                "verdict": "UNVERIFIED",
                "confidence": 0,
                "credibility": 0,
                "trustScore": 0,
                "riskLevel": "UNKNOWN",
                "disclaimer": TRANSPARENCY_WARNING,
                "explanation": ["Empty text input received."]
            }), 400

        # Input length validation
        if len(text) > 50000:
            return jsonify({"error": "Input text exceeds maximum allowed length of 50,000 characters."}), 400

        # If input text is actually a web URL, fetch and extract article content
        is_url_input = bool(re.match(r'^https?://', text, re.I)) or ('.' in text and ' ' not in text and not text.endswith('.') and len(text.split('\n')) == 1)
        if is_url_input:
            extraction = fetch_and_extract_article(text)
            if not extraction.get("success"):
                return jsonify({
                    "error": extraction.get("error", "Could not read this page. Paste the article text instead.")
                }), 400
            result = run_shared_verification_pipeline(
                text=extraction["text"],
                source_hint=extraction["domain"],
                url=extraction["url"],
                title=extraction["title"],
                input_type="url"
            )
            result["extractedText"] = extraction["text"]
            result["sourceDomain"] = extraction["domain"]
            result["articleTitle"] = extraction["title"] or result.get("articleTitle", "")
            return jsonify(result), 200

        result = run_shared_verification_pipeline(
            text=text,
            source_hint=source_hint,
            url=url_input,
            input_type="text"
        )
        return jsonify(result), 200

    except Exception as e:
        app.logger.exception(f"[analyze] Internal verification error: {e}")
        return jsonify({
            "error": "An internal error occurred while processing the verification request. Please try again.",
            "prediction": "UNKNOWN",
            "verdict": "UNVERIFIED",
            "confidence": 0,
            "credibility": 0,
            "trustScore": 0,
            "riskLevel": "UNKNOWN",
            "disclaimer": TRANSPARENCY_WARNING,
            "explanation": ["The system encountered an unexpected error during analysis."]
        }), 500

@app.route('/api/analyze-file', methods=['POST'])
@app.route('/api/analyze-file/', methods=['POST'])
def analyze_file():
    """
    Handles file upload (images with OCR, PDFs with PyPDF2, and TXT files),
    extracts content, and runs multi-factor verification pipeline.
    Enforces token verification, guest rate limits, magic byte validation,
    and blocks prohibited executable formats.
    """
    # 1. Rate Limiting Check
    auth_ctx = get_auth_context()
    limit = 15 if auth_ctx["is_authenticated"] else 4
    allowed, retry_after = rate_limiter.is_allowed(auth_ctx["identifier"], "upload", limit, 60)
    if not allowed:
        return jsonify({
            "error": f"Upload rate limit reached ({limit} files/min). Please wait {retry_after} seconds.",
            "retryAfter": retry_after
        }), 429

    # Robust file parameter extraction (accepts 'file', 'image', 'document', or first multipart entry)
    file = request.files.get('file') or request.files.get('image') or request.files.get('document')
    if file is None and request.files:
        file = next(iter(request.files.values()), None)

    # 2. Strict Magic Byte & File Type Validation
    is_valid_f, val_err = validate_uploaded_file(file)
    if not is_valid_f:
        return jsonify({"error": val_err}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ""
    stored_filename = f"{uuid.uuid4().hex}.{ext}" if ext else uuid.uuid4().hex
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    file.save(save_path)

    app.logger.info(f"[analyze-file] Received file: {filename} (Ext: {ext}) stored as {stored_filename}")

    extracted_text = ""
    extraction_note = ""
    ocr_preview = None
    text_override = request.form.get('textOverride') or request.form.get('text')

    try:
        if ext in ('jpg', 'jpeg', 'png'):
            extracted_text, success, msg, ocr_details = extract_text_from_image(save_path, return_details=True)

            if not success:
                app.logger.error(f"[analyze-file] OCR failed for {filename}: {msg}")
                return jsonify({
                    "error": f"OCR failed: {msg}",
                    "exactError": msg,
                    "fileName": filename,
                    "fileExtension": ext
                }), 422

            extraction_note = f"OCR ({ext.upper()}): {msg}"
            
            ocr_preview = {
                "previewText": ocr_details.get("previewText", extracted_text[:500]),
                "extractedText": extracted_text,
                "confidence": ocr_details.get("confidence", 0.0),
                "isLowConfidence": ocr_details.get("isLowConfidence", False),
                "warning": ocr_details.get("warning", ""),
                "qualityWarning": ocr_details.get("qualityWarning", "")
            }

            # If user reviewed and submitted edited text in the preview screen, prioritize it
            if text_override and text_override.strip():
                extracted_text = text_override.strip()
                ocr_preview["extractedText"] = extracted_text
                ocr_preview["previewText"] = extracted_text[:500]
                extraction_note = f"OCR ({ext.upper()}): Reviewed by user."

            if not extracted_text:
                extracted_text = f"Image {filename} was processed via OCR, but no clear text was detected."

        elif ext == 'pdf':
            extracted_text, success, msg = extract_text_from_pdf(save_path)
            if not success or not extracted_text or not extracted_text.strip():
                app.logger.error(f"[analyze-file] PDF extraction failed for {filename}: {msg}")
                return jsonify({
                    "error": "Could not extract readable text from this PDF.",
                    "fileName": filename,
                    "fileExtension": ext
                }), 422
            extraction_note = f"PDF Parser: {msg}"

        elif ext == 'txt':
            with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read().strip()
            extraction_note = "Text file loaded directly."

        # Analyze extracted content
        if not extracted_text.strip():
            if ext == 'pdf':
                return jsonify({
                    "error": "Could not extract readable text from this PDF.",
                    "fileName": filename,
                    "fileExtension": ext
                }), 422
            extracted_text = f"File {filename} contained no discernible text."

        is_image_file = ext in ('jpg', 'jpeg', 'png')
        img_publisher_info = ocr_details.get("publisherInfo") if (ocr_preview and "publisherInfo" in ocr_details) else None
        engine = get_verification_engine()
        result = engine.analyze(
            extracted_text,
            source_hint=filename,
            is_image=is_image_file,
            file_type="image" if is_image_file else ext,
            image_publisher=img_publisher_info
        )
        result["extractedText"] = extracted_text
        result["fileName"] = filename
        result["extractionNote"] = extraction_note

        # Use the OCR-derived title as the official article title for image uploads
        if ext in ('jpg', 'jpeg', 'png') and extracted_text.strip():
            derived_title = result.get("articleSummary", {}).get("title")
            if not derived_title or derived_title in ("News Article", "Extracted Content", "Extracted Article Content", "Article Analysis") or derived_title.endswith(('.png', '.jpg', '.jpeg')):
                derived_title = engine.summarizer._derive_title(extracted_text)
            result["articleTitle"] = derived_title
            if "articleSummary" in result and isinstance(result["articleSummary"], dict):
                result["articleSummary"]["title"] = derived_title
        if ocr_preview:
            result["ocrPreview"] = ocr_preview
            result["ocrConfidence"] = ocr_preview.get("confidence")
            result["isLowConfidence"] = ocr_preview.get("isLowConfidence", False)
            result["ocrWarning"] = ocr_preview.get("warning", "")
            result["ocrQualityWarning"] = ocr_preview.get("qualityWarning", "")

        return jsonify(result), 200

    except Exception as e:
        app.logger.exception(f"[analyze-file] File processing failed: {e}")
        return jsonify({"error": "An internal error occurred while processing the uploaded file. Please try again."}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass

@app.route('/api/ocr', methods=['POST', 'OPTIONS'])
@app.route('/api/ocr/', methods=['POST', 'OPTIONS'])
def perform_ocr():
    """
    Dedicated OCR endpoint for image preview workflow.
    Extracts text from uploaded image, computes confidence metrics,
    and returns first 300-500 characters preview with low-confidence alerts.
    """
    if request.method == 'OPTIONS':
        return ('', 200)

    # Rate Limiting Check
    auth_ctx = get_auth_context()
    limit = 15 if auth_ctx["is_authenticated"] else 4
    allowed, retry_after = rate_limiter.is_allowed(auth_ctx["identifier"], "upload", limit, 60)
    if not allowed:
        return jsonify({
            "error": f"OCR rate limit reached ({limit} requests/min). Please wait {retry_after} seconds.",
            "retryAfter": retry_after
        }), 429

    file = request.files.get('file') or request.files.get('image') or request.files.get('document')
    if file is None and request.files:
        file = next(iter(request.files.values()), None)

    # Validate file type and magic bytes
    is_valid_f, val_err = validate_uploaded_file(file)
    if not is_valid_f:
        return jsonify({"error": val_err}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ""
    if ext not in ('jpg', 'jpeg', 'png'):
        return jsonify({"error": f"File format .{ext} is not supported for image OCR. Supported formats: jpg, jpeg, png."}), 400

    stored_filename = f"{uuid.uuid4().hex}.{ext}" if ext else uuid.uuid4().hex
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    file.save(save_path)

    app.logger.info(f"[api/ocr] Running OCR preview for: {filename} (stored as {stored_filename})")
    try:
        extracted_text, success, msg, details = extract_text_from_image(save_path, return_details=True)

        if not success:
            return jsonify({
                "error": f"OCR failed: {msg}",
                "exactError": msg,
                "fileName": filename
            }), 422

        return jsonify({
            "success": True,
            "fileName": filename,
            "fileExtension": filename.rsplit('.', 1)[1].lower(),
            "extractedText": extracted_text,
            "previewText": details.get("previewText", extracted_text[:500]),
            "confidence": details.get("confidence", 0.0),
            "isLowConfidence": details.get("isLowConfidence", False),
            "warning": details.get("warning", ""),
            "qualityWarning": details.get("qualityWarning", ""),
            "wordCount": details.get("wordCount", 0),
            "message": msg
        }), 200
    except Exception as e:
        app.logger.exception(f"[api/ocr] Error during OCR: {e}")
        return jsonify({"error": "An internal error occurred during text extraction."}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass

@app.route('/api/analyze-url', methods=['POST'])
@app.route('/api/analyze-url/', methods=['POST'])
def analyze_url():
    """
    Scrapes article text from a given news URL, extracts title and main text,
    and runs the unified verification pipeline (Task A & Task B).
    If the page cannot be fetched or has insufficient content, returns
    HTTP 400 with: 'Could not read this page. Paste the article text instead.'
    Strictly enforces 10s timeout, browser User-Agent, and SSRF prevention.
    """
    auth_ctx = get_auth_context()
    limit = 30 if auth_ctx["is_authenticated"] else 8
    allowed, retry_after = rate_limiter.is_allowed(auth_ctx["identifier"], "verify", limit, 60)
    if not allowed:
        return jsonify({
            "error": f"Rate limit reached ({limit} requests/min). Please wait {retry_after} seconds before submitting another verification.",
            "retryAfter": retry_after
        }), 429

    data = request.get_json(force=True, silent=True) or {}
    raw_input = data.get("url", "").strip()

    if not raw_input:
        return jsonify({"error": "Missing URL parameter"}), 400

    if len(raw_input) > 2048:
        return jsonify({"error": "URL exceeds maximum allowed length of 2,048 characters."}), 400

    try:
        # Auto-detect whether input is actually a URL or raw text/claim
        is_url = bool(re.match(r'^https?://', raw_input, re.I)) or ('.' in raw_input and ' ' not in raw_input and len(raw_input.split('\n')) == 1)

        if not is_url:
            result = run_shared_verification_pipeline(
                text=raw_input,
                input_type="text"
            )
            result["extractedText"] = raw_input
            result["articleTitle"] = result.get("articleTitle") or raw_input[:100]
            return jsonify(result), 200

        extraction = fetch_and_extract_article(raw_input)
        if not extraction.get("success"):
            return jsonify({
                "error": extraction.get("error", "Could not read this page. Paste the article text instead.")
            }), 400

        result = run_shared_verification_pipeline(
            text=extraction["text"],
            source_hint=extraction["domain"],
            url=extraction["url"],
            title=extraction["title"],
            input_type="url"
        )
        result["extractedText"] = extraction["text"]
        result["sourceDomain"] = extraction["domain"]
        result["articleTitle"] = extraction["title"] or result.get("articleTitle", "")

        return jsonify(result), 200
    except Exception as e:
        app.logger.exception(f"[analyze-url] Error verifying URL: {e}")
        return jsonify({"error": "Could not read this page. Paste the article text instead."}), 500

@app.errorhandler(429)
def handle_429(e):
    return jsonify({"error": "Too many requests. Please slow down and try again later."}), 429

@app.errorhandler(500)
def handle_500(e):
    app.logger.exception(f"Internal 500 error: {e}")
    return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

@app.errorhandler(Exception)
def handle_general_exception(e):
    # Let Flask/Werkzeug preserve HTTP semantics (404, 413, 405, etc.).
    if isinstance(e, HTTPException):
        return e
    app.logger.exception(f"Unhandled server exception: {e}")
    return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """
    In production, serves the built React frontend from frontend/dist.
    Enables single-container or unified cloud deployment.
    """
    if path.startswith('api/') or path == 'api':
        return jsonify({"error": f"API endpoint /{path} not found"}), 404

    if app.static_folder and os.path.exists(app.static_folder):
        target_path = os.path.join(app.static_folder, path)
        if path and os.path.exists(target_path) and os.path.isfile(target_path):
            return send_from_directory(app.static_folder, path)
        index_file = os.path.join(app.static_folder, 'index.html')
        if os.path.exists(index_file):
            return send_from_directory(app.static_folder, 'index.html')

    return jsonify({
        "status": "online",
        "service": "Lucid Fact Verification API",
        "message": "Frontend build not detected in frontend/dist. Run 'npm run build' inside frontend.",
        "endpoints": ["/api/health", "/api/analyze", "/api/analyze-file", "/api/analyze-url"]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Lucid Evidence-Based Fact Verification Backend on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
