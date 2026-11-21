# Lucid — AI Fake News Detection & Verification Platform

Lucid is a multi-screen, AI-powered misinformation detection and verification platform built with a React + TypeScript frontend and a Python Flask + Scikit-Learn backend.

## 🌟 Platform Highlights

- **Multi-Screen Workflow**: Distinct investigative spaces modeled after ChatGPT, Perplexity, and Notion (no scrolling landing pages).
  - `Authentication Screen`: Continue with Google, Microsoft, or Guest; stores session locally in `localStorage`.
  - `Verification Selection Screen`: Specialized cards for News Articles, URLs, Images/Screenshots, PDFs, and Social Media posts.
  - `Upload & Investigation Workspace`: Drag-and-drop file upload (`jpg`, `jpeg`, `png`, `pdf`, `txt`), direct URL ingestion, and rich text pasting.
  - `Investigation Screen`: Animated 6-stage forensics pipeline with dynamic progress indicators.
  - `Results Dashboard`: Circular Trust Meter, Confidence Progress Ring, Evidence Cards, Analysis Timeline, and detailed Claim-by-Claim verification.
  - `Explanation Panel`: Detailed signals for emotional manipulation, clickbait indicators, source credibility, and model certainty.
  - `Report Export Screen`: Print/download PDF reports, save to local history, or generate shareable links.
- **Machine Learning & NLP Pipeline**:
  - Preprocessing: lowercasing, HTML/URL stripping, tokenization, stopword removal via NLTK, and linguistic sensationalism detection.
  - TF-IDF Vectorization (`ngram_range=(1, 2)`, sublinear TF scaling).
  - Logistic Regression Classifier trained on verified Fake and Real News datasets.
  - Persistent serialized models in `model.pkl` and `vectorizer.pkl`.
- **Multimodal Support**:
  - Image OCR with `pytesseract` and `Pillow`.
  - PDF document parsing with `PyPDF2` / `pypdf`.
  - Web article extraction with `requests` and `BeautifulSoup4`.
- **Future-Ready Architecture**: Pluggable provider interface ready for Gemini API, OpenAI API, and Google Fact Check Tools API (`services/ai_service.py`).

---

## 📁 Project Structure

```
Lucid/
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Multi-screen application UI & state
│   │   ├── index.css        # Lucid dark-mode design system & animations
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts       # Vite proxy & configuration
│   └── tsconfig.json
│
├── backend/
│   ├── app.py               # Flask REST API with CORS enabled
│   ├── train_model.py       # Model training & serialization script
│   ├── test_api.py          # Automated verification test suite
│   ├── model.pkl            # Trained Logistic Regression model
│   ├── vectorizer.pkl       # Fitted TfidfVectorizer
│   ├── requirements.txt
│   ├── services/
│   │   └── ai_service.py    # Verification engine & future API bridge
│   ├── utils/
│   │   ├── text_preprocessing.py # Text cleaning & linguistic signals
│   │   ├── image_ocr.py          # Pytesseract OCR extraction
│   │   └── pdf_parser.py         # PyPDF2 document parser
│   └── uploads/             # Safe storage for uploaded evidence
```

---

## 🚀 Getting Started

### 1. Backend Setup & Run

Make sure Python is installed.

```bash
cd backend

# Install dependencies (already pre-installed in workspace)
pip install -r requirements.txt

# (Optional) Re-train the model
python train_model.py

# Run automated tests
python test_api.py

# Start the Flask API server (runs on port 5000)
python app.py
```

### 2. Frontend Setup & Run

```bash
cd frontend

# Install packages
npm install

# Build check
npm run build

# Start the dev server
npm run dev
```

Visit the app in your browser at `http://localhost:5173`.

---

## 🔌 API Reference

### `POST /api/analyze`
Analyzes raw text for misinformation.

**Request:**
```json
{
  "text": "The International Energy Agency reported that global renewable energy capacity grew by 50% in 2023."
}
```

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 88,
  "credibility": 88,
  "riskLevel": "LOW",
  "verdict": "VERIFIED",
  "trustScore": 92,
  "explanation": [
    "Trusted language patterns detected.",
    "No major misinformation indicators found.",
    "Neutral, attributable tone characteristic of verified reporting."
  ],
  "claims": [...],
  "sources": [...],
  "timeline": [...]
}
```

### `POST /api/analyze-file`
Accepts `multipart/form-data` with `file` (`.jpg`, `.jpeg`, `.png`, `.pdf`, `.txt`), extracts content using OCR / PDF parser, and runs the ML verification pipeline.

### `POST /api/analyze-url`
Accepts `{"url": "https://..."}`, scrapes the news story, and evaluates the claim and domain credibility.

### `GET /api/health`
Returns service status, loaded model indicators, and external API readiness.
