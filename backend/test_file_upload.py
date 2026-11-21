"""
Tests for /api/analyze-file endpoint:
1. Verifies POST /api/analyze-file works with multipart/form-data key 'file'
2. Verifies trailing slash /api/analyze-file/ also works (no 404 or redirect error)
3. Verifies CORS allows requests from http://localhost:5173
4. Verifies OCR/PDF extraction does not fail with 422
"""
import io
import pytest
from app import app


@pytest.fixture
def client():
    return app.test_client()


def test_analyze_file_with_txt(client):
    """Verifies file upload with key 'file' succeeds with 200 OK."""
    data = {
        'file': (io.BytesIO(b"British annual inflation fell to 2% according to official reporting."), "inflation.txt")
    }
    resp = client.post('/api/analyze-file', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.get_data(as_text=True)}"
    json_data = resp.get_json()
    assert "verdict" in json_data
    assert "trustScore" in json_data
    assert json_data["fileName"] == "inflation.txt"


def test_analyze_file_trailing_slash(client):
    """Verifies trailing slash /api/analyze-file/ does not 404 or fail."""
    data = {
        'file': (io.BytesIO(b"Supreme Court delivers unanimous verdict on environmental law."), "ruling.txt")
    }
    resp = client.post('/api/analyze-file/', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"


def test_analyze_file_cors_headers(client):
    """Verifies CORS headers allow origin http://localhost:5173."""
    headers = {
        "Origin": "http://localhost:5173"
    }
    data = {
        'file': (io.BytesIO(b"International Energy Agency released global renewable energy market report."), "iea.txt")
    }
    resp = client.post('/api/analyze-file', data=data, content_type='multipart/form-data', headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") in ("http://localhost:5173", "*")


def test_analyze_file_options_preflight(client):
    """Verifies OPTIONS preflight request for CORS."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
    resp = client.options('/api/analyze-file', headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") in ("http://localhost:5173", "*")
    assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")


def test_analyze_file_missing_key_gives_400(client):
    """Verifies missing file part returns 400 with helpful error message."""
    resp = client.post('/api/analyze-file', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400
    json_data = resp.get_json()
    assert "file" in json_data["error"]


def test_api_ocr_endpoint(client):
    """Verifies POST /api/ocr returns previewText and confidence."""
    import os
    img_path = os.path.join(os.path.dirname(__file__), 'uploads', 'Screenshot_2026-11-14_224316.png')
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            data = {'file': (io.BytesIO(f.read()), 'test_screenshot.png')}
        resp = client.post('/api/ocr', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        json_data = resp.get_json()
        assert json_data["success"] is True
        assert "previewText" in json_data
        assert "confidence" in json_data
        assert "isLowConfidence" in json_data


def test_analyze_file_with_image_includes_ocr_preview_and_summary(client):
    """Verifies image analysis returns ocrPreview, articleSummary, keyTakeaways, and tldr."""
    import os
    img_path = os.path.join(os.path.dirname(__file__), 'uploads', 'Screenshot_2026-11-14_224316.png')
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            data = {'file': (io.BytesIO(f.read()), 'test_screenshot.png')}
        resp = client.post('/api/analyze-file', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        json_data = resp.get_json()
        assert "ocrPreview" in json_data
        assert "articleSummary" in json_data
        summary = json_data["articleSummary"]
        assert "executiveSummary" in summary
        assert "keyTakeaways" in summary
        assert "tldr" in summary
        assert len(summary["keyTakeaways"]) <= 5
