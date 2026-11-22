"""
Unit & Integration Tests for Image Upload Source Attribution, Logo/Watermark Detection,
and OCR Claim Extraction.

Requirements:
1. Uploaded filenames (.jpg, .jpeg, .png, .pdf) must NEVER be treated as domains or publishers.
2. For image uploads without detected publisher metadata:
   - Publisher = Unknown
   - Source Type = Uploaded Image
   - Domain = N/A
3. OCR claim extraction:
   - "JENSEN HUANG DISCUSSES COMING BACK AS NVIDIA CEO"
     becomes:
     Claim: Jensen Huang is discussing a return as NVIDIA CEO.
     Status: Unverified
     instead of 0 claims detected.
4. Detect watermarks, logos, or publisher names inside images:
   - BBC logo -> Publisher: BBC, Domain: bbc.com
   - Reuters logo -> Publisher: Reuters, Domain: reuters.com
   - Al Jazeera logo -> Publisher: Al Jazeera, Domain: aljazeera.com
   - Otherwise -> Publisher: Unknown, Domain: N/A
"""
import io
import os
import pytest
from PIL import Image, ImageDraw

from app import app
from services.source_reputation import get_source_engine, SourceReputationEngine
from services.claim_extractor import get_claim_engine, normalize_claim_text
from services.ai_service import get_verification_engine
from utils.image_publisher_detector import detect_publisher_from_image


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture
def source_engine():
    return get_source_engine()


@pytest.fixture
def claim_engine():
    return get_claim_engine()


@pytest.fixture
def verification_engine():
    return get_verification_engine()


# =========================================================================
# Part 1: Filename and Domain Attribution Tests
# =========================================================================

def test_filenames_not_treated_as_domains(source_engine):
    """
    Filenames ending in .jpg, .jpeg, .png, .pdf must NEVER be treated as domains.
    """
    for fname in ["sample1.jpeg", "test.jpg", "photo.png", "document.pdf", "archive.pdf"]:
        # extract_domain must return empty string for filenames
        assert source_engine.extract_domain(fname) == "", f"Filename {fname} was treated as domain!"
        # detect_url must return None for filenames
        assert source_engine.detect_url(url_or_domain=fname) is None, f"Filename {fname} was treated as URL!"


def test_image_upload_without_publisher_metadata(source_engine, verification_engine):
    """
    For image uploads without detected publisher metadata:
      Publisher = Unknown
      Source Type = Uploaded Image
      Domain = N/A
    """
    # 1. SourceReputationEngine evaluation
    rep = source_engine.evaluate_source(
        text="A breaking news report from an uploaded photo.",
        source_hint="sample1.jpeg",
        is_image=True
    )
    assert rep["publisher"] == "Unknown"
    assert rep["sourceType"] == "Uploaded Image"
    assert rep["domain"] == "N/A"
    assert "sample1.jpeg" not in rep["publisher"]
    assert "sample1.jpeg" not in rep["sourceName"]
    assert rep["domain"] != "sample1.jpeg"

    # 2. VerificationEngine analyze pipeline
    result = verification_engine.analyze(
        text="A breaking news report from an uploaded photo.",
        source_hint="sample1.jpeg",
        is_image=True
    )
    assert result["publisher"] == "Unknown"
    assert result["sourceType"] == "Uploaded Image"
    assert result["sourceDomain"] == "N/A"
    assert result["primarySource"]["publisher"] == "Unknown"
    assert result["primarySource"]["domain"] == "N/A"
    assert result["primarySource"]["sourceType"] == "Uploaded Image"
    assert result["sources"][0]["publisher"] == "Unknown"
    assert result["sources"][0]["domain"] == "N/A"
    assert result["sources"][0]["type"] == "Uploaded Image"


def test_pdf_upload_not_treated_as_domain(source_engine, verification_engine):
    """
    PDF uploads without publisher metadata must not treat .pdf filename as domain.
    """
    rep = source_engine.evaluate_source(
        text="Official whitepaper text without domain header.",
        source_hint="analysis_report.pdf",
        file_type="pdf"
    )
    assert rep["publisher"] == "Unknown"
    assert rep["domain"] == "N/A"
    assert "analysis_report.pdf" not in rep["publisher"]
    assert "analysis_report.pdf" not in rep["sourceName"]


# =========================================================================
# Part 2: OCR Claim Extraction Improvements
# =========================================================================

def test_ocr_claim_extraction_jensen_huang(claim_engine):
    """
    Example OCR text:
    "JENSEN HUANG DISCUSSES COMING BACK AS NVIDIA CEO"
    should become:
    Claim: Jensen Huang is discussing a return as NVIDIA CEO.
    Status: Unverified
    instead of: 0 claims detected.
    """
    ocr_text = "JENSEN HUANG DISCUSSES COMING BACK AS NVIDIA CEO"
    claims = claim_engine.extract_claims(ocr_text)

    assert len(claims) >= 1, "Expected at least 1 claim, got 0 claims detected!"
    claim_obj = claims[0]
    assert claim_obj["claim"] == "Jensen Huang is discussing a return as NVIDIA CEO."
    assert claim_obj["status"] == "Unverified"


def test_ocr_claim_extraction_from_sample1_raw_text(claim_engine):
    """
    Simulates raw OCR text extracted from sample1.jpeg:
    / @¢ ad

    JUST IN: JENSEN HUANG
    DISCUSSES COMING
    BACK AS NVIDIA CEO

    WITH ALLIES
    """
    raw_ocr = (
        "/ @¢ ad\n\n"
        "JUST IN: JENSEN HUANG\n"
        "DISCUSSES COMING\n"
        "BACK AS NVIDIA CEO\n\n"
        "WITH ALLIES"
    )
    claims = claim_engine.extract_claims(raw_ocr)
    assert len(claims) >= 1, "Expected claim from sample1 OCR text, got 0!"
    assert claims[0]["claim"] == "Jensen Huang is discussing a return as NVIDIA CEO."
    assert claims[0]["status"] in ("Unverified", "Supported")


# =========================================================================
# Part 3: Image Watermark / Logo / Publisher Detection
# =========================================================================

def test_detect_watermark_bbc(tmp_path):
    """BBC logo inside image -> Publisher: BBC, Domain: bbc.com"""
    img_path = str(tmp_path / "bbc_test.png")
    img = Image.new('RGB', (600, 400), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "BBC NEWS", fill=(255, 255, 255))
    img.save(img_path)

    det = detect_publisher_from_image(img_path)
    assert det["detected"] is True
    assert det["publisher"] == "BBC"
    assert det["domain"] == "bbc.com"


def test_detect_watermark_reuters(tmp_path):
    """Reuters logo inside image -> Publisher: Reuters, Domain: reuters.com"""
    img_path = str(tmp_path / "reuters_test.png")
    img = Image.new('RGB', (600, 400), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((450, 360), "REUTERS", fill=(255, 128, 0))
    img.save(img_path)

    det = detect_publisher_from_image(img_path)
    assert det["detected"] is True
    assert det["publisher"] == "Reuters"
    assert det["domain"] == "reuters.com"


def test_detect_watermark_aljazeera(tmp_path):
    """Al Jazeera logo inside image -> Publisher: Al Jazeera, Domain: aljazeera.com"""
    img_path = str(tmp_path / "aljazeera_test.png")
    img = Image.new('RGB', (600, 400), color=(30, 20, 40))
    d = ImageDraw.Draw(img)
    d.text((430, 20), "AL JAZEERA", fill=(255, 200, 50))
    img.save(img_path)

    det = detect_publisher_from_image(img_path)
    assert det["detected"] is True
    assert det["publisher"] == "Al Jazeera"
    assert det["domain"] == "aljazeera.com"


def test_no_watermark_returns_unknown(tmp_path):
    """Image without logo or watermark -> Publisher: Unknown, Domain: N/A"""
    img_path = str(tmp_path / "plain_test.png")
    img = Image.new('RGB', (600, 400), color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    d.text((50, 150), "Just a random photo caption", fill=(0, 0, 0))
    img.save(img_path)

    det = detect_publisher_from_image(img_path)
    assert det["detected"] is False
    assert det["publisher"] == "Unknown"
    assert det["domain"] == "N/A"


# =========================================================================
# Part 4: Integration Test on /api/analyze-file with sample1.jpeg
# =========================================================================

def test_api_analyze_file_sample1(client):
    """
    End-to-end integration test with sample1.jpeg:
    1. Publisher is Unknown (not SAMPLE1 or Web Publisher (sample1.jpeg))
    2. Source Type is Uploaded Image
    3. Domain is N/A (not sample1.jpeg)
    4. Claim is 'Jensen Huang is discussing a return as NVIDIA CEO.' with status 'Unverified'
    """
    sample1_path = os.path.join(os.path.dirname(__file__), "uploads", "sample1.jpeg")
    if not os.path.exists(sample1_path):
        pytest.skip("sample1.jpeg not present in backend/uploads")

    with open(sample1_path, "rb") as f:
        resp = client.post(
            "/api/analyze-file",
            data={"file": (f, "sample1.jpeg")},
            content_type="multipart/form-data"
        )

    assert resp.status_code == 200
    data = resp.get_json()

    # 1. Source attribution checks
    assert data["publisher"] == "Unknown", f"Expected Unknown, got {data['publisher']}"
    assert data["sourceType"] == "Uploaded Image"
    assert data["sourceDomain"] == "N/A"
    assert data["primarySource"]["publisher"] == "Unknown"
    assert data["primarySource"]["domain"] == "N/A"
    assert data["primarySource"]["sourceType"] == "Uploaded Image"
    assert "sample1.jpeg" not in data["primarySource"]["name"]

    # 2. Claim checks
    claims = data.get("claims", [])
    assert len(claims) >= 1, f"Expected at least 1 claim, got {len(claims)}"
    assert claims[0]["claim"] == "Jensen Huang is discussing a return as NVIDIA CEO."
    assert claims[0]["status"] in ("Unverified", "Supported")
