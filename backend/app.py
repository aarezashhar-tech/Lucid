import os
import re
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from backend/.env or root .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

from utils.image_ocr import extract_text_from_image
from utils.pdf_parser import extract_text_from_pdf
from services.ai_service import get_verification_engine
from services.ai_reasoning import TRANSPARENCY_WARNING

FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

app = Flask(
    __name__,
    static_folder=FRONTEND_DIST if os.path.exists(FRONTEND_DIST) else None,
    static_url_path=""
)

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
    # Default fallback origins for local development
    allowed_origins = [
        "https://www.yourdomain.com",
        "https://yourdomain.com",
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
        if origin_clean in allowed_origins or (not is_production and ("localhost" in origin_clean or "127.0.0.1" in origin_clean)):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    # Never set Access-Control-Allow-Origin to "*" in production
    return response

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf', 'txt'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB max

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            "layer4": "Multi-Factor Scoring (30% ML, 25% Source, 25% Evidence, 10% Consistency, 10% Language)",
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
            "gnewsConfigured": bool(os.getenv("GNEWS_API_KEY"))
        }
    })

@app.route('/api/analyze', methods=['POST'])
@app.route('/api/analyze/', methods=['POST'])
def analyze():
    """
    Multi-Layer Evidence-Based Analysis Endpoint:
    Accepts text, optional source_hint or url.
    Returns 5-factor scoring, atomic claims, evidence corroboration,
    source reputation, and AI reasoning.
    """
    try:
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

        engine = get_verification_engine()
        result = engine.analyze(text=text, source_hint=source_hint, url_or_domain=url_input)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "prediction": "UNKNOWN",
            "verdict": "UNVERIFIED",
            "confidence": 0,
            "credibility": 0,
            "trustScore": 0,
            "riskLevel": "UNKNOWN",
            "disclaimer": TRANSPARENCY_WARNING,
            "explanation": [f"Analysis error: {str(e)}"]
        }), 500

@app.route('/api/analyze-file', methods=['POST'])
@app.route('/api/analyze-file/', methods=['POST'])
def analyze_file():
    """
    Handles file upload (images with OCR, PDFs with PyPDF2, and TXT files),
    extracts content, and runs multi-factor verification pipeline.
    """
    # Robust file parameter extraction (accepts 'file', 'image', 'document', or first multipart entry)
    file = request.files.get('file') or request.files.get('image') or request.files.get('document')
    if file is None and request.files:
        file = next(iter(request.files.values()), None)

    if file is None or not file.filename:
        return jsonify({
            "error": "No file part in request. Expected form-data key 'file'.",
            "receivedKeys": list(request.files.keys())
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ""
    app.logger.info(f"[analyze-file] Received file name: {filename}")
    app.logger.info(f"[analyze-file] File extension: {ext}")
    print(f"[analyze-file] Received file name: {filename}")
    print(f"[analyze-file] File extension: {ext}")

    extracted_text = ""
    extraction_note = ""
    ocr_preview = None
    text_override = request.form.get('textOverride') or request.form.get('text')

    try:
        if ext in ('jpg', 'jpeg', 'png'):
            app.logger.info(f"[analyze-file] OCR started for image: {filename}")
            print(f"[analyze-file] OCR started for image: {filename}")
            extracted_text, success, msg, ocr_details = extract_text_from_image(save_path, return_details=True)

            if not success:
                app.logger.error(f"[analyze-file] OCR failed for {filename}. Exact error: {msg}")
                print(f"[analyze-file ERROR] OCR failed for {filename}. Exact error: {msg}")
                return jsonify({
                    "error": f"OCR failed: {msg}",
                    "exactError": msg,
                    "fileName": filename,
                    "fileExtension": ext
                }), 422

            app.logger.info(f"[analyze-file] OCR completed for image: {filename}")
            app.logger.info(f"[analyze-file] Extracted text length: {len(extracted_text)}")
            app.logger.info(f"[analyze-file] First 300 characters of extracted text:\n{extracted_text[:300]}")
            print(f"[analyze-file] OCR completed for image: {filename}")
            print(f"[analyze-file] Extracted text length: {len(extracted_text)}")
            print(f"[analyze-file] First 300 characters of extracted text:\n{extracted_text[:300]}")
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
            app.logger.info(f"[analyze-file] PDF extraction started for: {filename}")
            extracted_text, success, msg = extract_text_from_pdf(save_path)
            if not success and not extracted_text:
                app.logger.error(f"[analyze-file] PDF extraction failed for {filename}: {msg}")
                return jsonify({
                    "error": f"PDF extraction failed: {msg}",
                    "exactError": msg,
                    "fileName": filename,
                    "fileExtension": ext
                }), 422
            extraction_note = f"PDF Parser: {msg}"
            if not extracted_text:
                extracted_text = f"PDF document {filename} contained no discernible text."

        elif ext == 'txt':
            with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read().strip()
            extraction_note = "Text file loaded directly."

        # Analyze extracted content
        if not extracted_text.strip():
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
        app.logger.error(f"[analyze-file] File processing failed with exception: {str(e)}")
        print(f"[analyze-file ERROR] File processing failed: {str(e)}")
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500

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

    file = request.files.get('file') or request.files.get('image') or request.files.get('document')
    if file is None and request.files:
        file = next(iter(request.files.values()), None)

    if file is None or not file.filename:
        return jsonify({
            "error": "No image file part provided. Expected 'file' form data."
        }), 400

    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ""
    if ext not in ('jpg', 'jpeg', 'png'):
        return jsonify({
            "error": f"OCR requires an image file (JPG, JPEG, PNG). Received .{ext}"
        }), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    app.logger.info(f"[api/ocr] Running OCR preview for: {filename}")
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
        "fileExtension": ext,
        "extractedText": extracted_text,
        "previewText": details.get("previewText", extracted_text[:500]),
        "confidence": details.get("confidence", 0.0),
        "isLowConfidence": details.get("isLowConfidence", False),
        "warning": details.get("warning", ""),
        "qualityWarning": details.get("qualityWarning", ""),
        "wordCount": details.get("wordCount", 0),
        "message": msg
    }), 200

@app.route('/api/analyze-url', methods=['POST'])
@app.route('/api/analyze-url/', methods=['POST'])
def analyze_url():
    """
    Scrapes article text from a given news URL, extracts domain and source reputation,
    and runs multi-factor verification pipeline. Gracefully falls back to direct text
    analysis if a query or statement is provided instead of a valid web URL.
    """
    data = request.get_json(force=True, silent=True) or {}
    raw_input = data.get("url", "").strip()

    if not raw_input:
        return jsonify({"error": "Missing URL or text parameter"}), 400

    # Auto-detect whether input is actually a URL or raw text/claim
    is_url = bool(re.match(r'^(https?://)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(/\S*)?$', raw_input)) and ' ' not in raw_input
    if not is_url:
        engine = get_verification_engine()
        result = engine.analyze(raw_input)
        result["extractedText"] = raw_input
        result["articleTitle"] = result.get("articleTitle") or raw_input[:100]
        return jsonify(result), 200

    url = raw_input
    if not re.match(r'^https?://', url):
        url = 'https://' + url

    try:
        domain = urlparse(url).netloc
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 LucidVerifier/2.0'
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Extract title (og:title, twitter:title, h1, or title tag)
        title = ""
        og_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'og:title'}) or soup.find('meta', attrs={'name': 'twitter:title'})
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove non-content elements before extracting body (navigation, ads, captions, bylines)
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'time']):
            tag.decompose()

        # Remove known UI widgets, byline elements, share buttons, topic tags, and ad slots
        for selector in [
            '.byline', '.author', '[data-component="byline-block"]',
            '.share-tools', '.social-share', '.article-share',
            '.tags', '.topics', '[data-testid="topics-list"]',
            '.related-topics', '.related-content', '.related-articles',
            '.advertisement', '.ad-slot', '.sponsored-content'
        ]:
            for el in soup.select(selector):
                el.decompose()

        # Extract article body
        article_el = soup.find('article') or soup.find('main') or soup.body
        paragraphs = []
        if article_el:
            for p in article_el.find_all('p'):
                text = p.get_text(strip=True)
                text_lower = text.lower()
                if len(text) > 25 and not any(bp in text_lower for bp in [
                    'cookie', 'subscribe', 'sign up', 'all rights reserved', 'terms of service',
                    'add as preferred', 'image caption', 'photo caption', 'share this story'
                ]):
                    paragraphs.append(text)
        
        full_content = (title + "\n\n" + "\n".join(paragraphs[:25])).strip()
        if not full_content:
            full_content = title or f"Article content from {domain}"

        engine = get_verification_engine()
        result = engine.analyze(full_content, source_hint=domain, url_or_domain=url, article_title=title)
        result["extractedText"] = full_content
        result["sourceDomain"] = domain
        result["articleTitle"] = title or result.get("articleTitle", "")

        return jsonify(result), 200

    except Exception as e:
        # Graceful fallback: Analyze the input as text so verification never crashes
        engine = get_verification_engine()
        result = engine.analyze(raw_input)
        result["extractedText"] = raw_input
        result["articleTitle"] = result.get("articleTitle") or raw_input[:100]
        return jsonify(result), 200

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
