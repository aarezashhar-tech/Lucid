# ==========================================
# Python Production Runtime with OCR Support
# ==========================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies including Tesseract OCR and runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Verify critical headless CV and ML dependencies inside container before runtime
RUN python -c "import cv2; import numpy; import PIL; print('OpenCV successfully verified:', cv2.__version__)"

# Download required NLTK tokenizers & stopwords
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy backend files and pre-trained models
COPY backend/ ./backend/

WORKDIR /app/backend

EXPOSE 5000

# Run with production Gunicorn WSGI server, dynamically respecting Render's $PORT
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 app:app"]
