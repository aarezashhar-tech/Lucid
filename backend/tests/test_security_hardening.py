"""
Offline tests for the security hardening changes.

Covers:
1. Firebase Admin verify_id_token() + project audience rejection.
2. ProxyFix / remote_addr rate limiting and empty-key pruning.
3. SSRF-safe manual redirects and validated-IP pinning.
4. defusedxml RSS/XML parsing.
5. UUID upload filenames + finally cleanup.
6. Gemini key header + unified GEMINI_MODEL.
7. HTTPException passthrough (404/413 keep status).
8. evidenceUnavailable flag for failed Wikipedia/GNews/FactCheck lookups.
"""

import io
import json
import os
import re
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from werkzeug.exceptions import NotFound

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as app_module
from app import app
from services import factcheck_service, gnews_service, rag_retriever, wikipedia_service
from services.ai_reasoning import AIReasoningEngine
from services.evidence_availability import evidence_tracker
from services.gemini_stance_classifier import GeminiStanceClassifier
from services.summarizer import ArticleSummarizer
from utils import url_extractor as url_extractor_module


def _guest_ctx(identifier="ip"):
    return {
        "is_authenticated": False,
        "user_id": None,
        "email": None,
        "is_guest": True,
        "identifier": identifier,
    }


# ---------------------------------------------------------------------------
# 1. Firebase Admin token verification
# ---------------------------------------------------------------------------

def test_firebase_token_for_expected_project_authenticates():
    if app_module.firebase_auth is None:
        pytest.skip("firebase-admin is not installed")
    app_module.TOKEN_CACHE.clear()
    decoded = {"uid": "user-1", "email": "u@example.com", "aud": "lucid-prod", "exp": time.time() + 300}
    with patch.object(app_module, "_get_firebase_app", return_value=object()), \
         patch.object(app_module, "_firebase_project_id", "lucid-prod"), \
         patch.object(app_module.firebase_auth, "verify_id_token", return_value=decoded):
        with app.test_request_context("/", headers={"Authorization": "Bearer valid-token"}):
            ctx = app_module.get_auth_context()
    assert ctx["is_authenticated"] is True
    assert ctx["user_id"] == "user-1"
    assert ctx["identifier"] == "user-1"


def test_firebase_token_for_other_project_is_rejected():
    if app_module.firebase_auth is None:
        pytest.skip("firebase-admin is not installed")
    app_module.TOKEN_CACHE.clear()
    decoded = {"uid": "user-1", "email": "u@example.com", "aud": "attacker-project", "exp": time.time() + 300}
    with patch.object(app_module, "_get_firebase_app", return_value=object()), \
         patch.object(app_module, "_firebase_project_id", "lucid-prod"), \
         patch.object(app_module.firebase_auth, "verify_id_token", return_value=decoded):
        with app.test_request_context("/", headers={"Authorization": "Bearer other-token"}):
            ctx = app_module.get_auth_context()
    assert ctx["is_authenticated"] is False
    assert ctx["is_guest"] is True


# ---------------------------------------------------------------------------
# 2. ProxyFix, remote_addr, limiter/cache key cleanup
# ---------------------------------------------------------------------------

def test_proxyfix_is_installed():
    from werkzeug.middleware.proxy_fix import ProxyFix
    assert isinstance(app.wsgi_app, ProxyFix)


def test_remote_addr_is_used_not_raw_x_forwarded_for():
    with app.test_request_context(
        "/",
        headers={"X-Forwarded-For": "1.2.3.4"},
        environ_overrides={"REMOTE_ADDR": "9.9.9.9"},
    ):
        ctx = app_module.get_auth_context()
    assert ctx["identifier"] == "9.9.9.9"


def test_rate_limiter_deletes_empty_keys():
    limiter = app_module.SlidingWindowRateLimiter()
    assert limiter.is_allowed("user", "cat", 1, 60)[0] is True
    key = "user:cat"
    limiter.requests[key] = [time.time() - 120]
    allowed, _ = limiter.is_allowed("user", "cat", 0, 60)
    assert allowed is False
    assert key not in limiter.requests


def test_token_cache_purges_expired_keys():
    app_module.TOKEN_CACHE.clear()
    app_module.TOKEN_CACHE["expired"] = ({"is_authenticated": True}, time.time() - 1)
    app_module._purge_expired_tokens(time.time())
    assert "expired" not in app_module.TOKEN_CACHE


# ---------------------------------------------------------------------------
# 3. SSRF protection
# ---------------------------------------------------------------------------

def test_private_ip_is_blocked():
    ok, _err, _url, _ip, _host, _port = url_extractor_module.resolve_and_validate_url("http://127.0.0.1/admin")
    assert ok is False


def test_safe_get_follows_at_most_five_redirects():
    calls = []

    def fake_fetch(url, headers, timeout, max_bytes):
        calls.append(url)
        return {
            "status_code": 302,
            "headers": {"Location": "/next"},
            "content": b"",
            "encoding": None,
            "apparent_encoding": None,
        }

    with patch.object(url_extractor_module, "_fetch_once", side_effect=fake_fetch):
        with pytest.raises(url_extractor_module.TooManyRedirectsError):
            url_extractor_module.safe_get("https://example.com/start", {}, timeout=1, max_redirects=5)
    assert len(calls) == 6


def test_fetch_once_validates_before_connecting():
    with patch.object(
        url_extractor_module,
        "resolve_and_validate_url",
        return_value=(False, "blocked", "", "", "", 443),
    ) as validator:
        with pytest.raises(url_extractor_module.SSRFBlockedError):
            url_extractor_module._fetch_once("http://evil.example", {}, timeout=1, max_bytes=1024)
    validator.assert_called_once()


def test_pinned_adapter_connects_to_validated_ip_and_preserves_host():
    adapter = url_extractor_module._PinnedHTTPAdapter("203.0.113.10", "example.com", "https")
    prepared = requests.Request("GET", "https://example.com/path?q=1").prepare()
    with patch("requests.adapters.HTTPAdapter.send", return_value=MagicMock()) as send_mock:
        adapter.send(prepared)
    sent_request = send_mock.call_args[0][0]
    assert sent_request.url.startswith("https://203.0.113.10/path")
    assert sent_request.headers["Host"] == "example.com"


# ---------------------------------------------------------------------------
# 4. defusedxml
# ---------------------------------------------------------------------------

def test_all_rss_xml_parsing_uses_defusedxml():
    from defusedxml import ElementTree as DefusedElementTree
    assert gnews_service.ET is DefusedElementTree
    assert rag_retriever.ET is DefusedElementTree


# ---------------------------------------------------------------------------
# 5. Upload filenames + cleanup
# ---------------------------------------------------------------------------

def test_upload_uses_uuid_filename_and_deletes_file(tmp_path):
    client = app.test_client()
    old_upload_folder = app.config["UPLOAD_FOLDER"]
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    seen = {}

    class FakeSummarizer:
        def _derive_title(self, text):
            return "Test title"

    class FakeEngine:
        summarizer = FakeSummarizer()

        def analyze(self, text, **kwargs):
            seen["files"] = os.listdir(tmp_path)
            return {"verdict": "UNVERIFIED", "trustScore": 0, "articleSummary": {}}

    try:
        with patch.object(app_module, "get_verification_engine", return_value=FakeEngine()), \
             patch.object(app_module, "get_auth_context", return_value=_guest_ctx()), \
             patch.object(app_module.rate_limiter, "is_allowed", return_value=(True, 0)):
            response = client.post(
                "/api/analyze-file",
                data={"file": (io.BytesIO(b"Hello world, this is a text upload."), "note.txt")},
                content_type="multipart/form-data",
            )
    finally:
        app.config["UPLOAD_FOLDER"] = old_upload_folder

    assert response.status_code == 200
    assert seen["files"], "expected the upload to be saved before analysis"
    assert re.fullmatch(r"[0-9a-f]{32}\.txt", seen["files"][0])
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# 6. Gemini key header + GEMINI_MODEL
# ---------------------------------------------------------------------------

def _gemini_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "candidates": [{
            "content": {"parts": [{"text": json.dumps({
                "executiveSummary": "Summary sentence one. Summary sentence two. Summary sentence three.",
                "keyTakeaways": ["One", "Two", "Three"],
                "tldr": "One sentence."
            })}]}
        }]
    }
    return response


def test_summarizer_sends_gemini_key_in_header_and_uses_model_env():
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _gemini_response()

    with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key", "GEMINI_MODEL": "gemini-test-model"}):
        summarizer = ArticleSummarizer()
        with patch("services.summarizer.requests.post", side_effect=fake_post):
            result = summarizer._query_gemini("Some article text", "Title")

    assert result is not None
    assert "key=" not in captured["url"]
    assert "gemini-test-model" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret-key"


def test_ai_reasoning_sends_gemini_key_in_header_and_uses_model_env():
    captured = {}

    def fake_post(url, headers=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "synthesis": "s",
                "whyAppearsTrue": [],
                "whyAppearsFalse": [],
                "evidenceFound": [],
                "evidenceMissing": []
            })}]}}]
        }
        return response

    with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key", "GEMINI_MODEL": "gemini-test-model"}):
        reasoner = AIReasoningEngine()
        with patch("services.ai_reasoning.requests.post", side_effect=fake_post):
            result = reasoner._query_gemini("text", [], {}, {})

    assert result is not None
    assert "key=" not in captured["url"]
    assert "gemini-test-model" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret-key"


def test_stance_classifier_uses_unified_gemini_model_env():
    with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-test-model"}):
        classifier = GeminiStanceClassifier()
    assert classifier.preferred_model == "gemini-test-model"


# ---------------------------------------------------------------------------
# 7. HTTPException passthrough
# ---------------------------------------------------------------------------

def test_http_exception_is_returned_unchanged():
    exc = NotFound()
    assert app_module.handle_general_exception(exc) is exc


def test_unknown_api_route_keeps_404_status():
    response = app.test_client().get("/api/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 8. evidenceUnavailable flags
# ---------------------------------------------------------------------------

def test_factcheck_failure_marks_evidence_unavailable():
    evidence_tracker.reset()
    with patch.dict(os.environ, {"FACTCHECK_API_KEY": "test-key"}):
        with patch("services.factcheck_service.requests.get", side_effect=requests.RequestException("boom")):
            results = factcheck_service.query_google_factcheck("some claim")
    assert results == []
    assert "FactCheck" in evidence_tracker.snapshot()


def test_gnews_failure_marks_evidence_unavailable():
    evidence_tracker.reset()
    with patch.dict(os.environ, {"GNEWS_API_KEY": "test-key"}):
        with patch("services.gnews_service.requests.get", side_effect=requests.RequestException("boom")):
            results = gnews_service.search_gnews("some claim")
    assert results == []
    assert "GNews" in evidence_tracker.snapshot()


def test_wikipedia_failure_marks_evidence_unavailable():
    evidence_tracker.reset()
    service = wikipedia_service.WikipediaService()
    with patch.object(service.session, "get", side_effect=requests.RequestException("boom")):
        results = service.search_wikipedia("some query")
    assert results == []
    assert "Wikipedia" in evidence_tracker.snapshot()


def test_pipeline_surfaces_evidence_unavailable_flag():
    from services.ai_service import get_verification_engine

    engine = get_verification_engine()
    evidence_tracker.reset()
    minimal_evidence = {
        "claims": [],
        "supportingEvidenceCount": 0,
        "contradictingEvidenceCount": 0,
        "supportingEvidence": [],
        "contradictingEvidence": [],
        "factCheckResults": [],
        "gnewsArticles": [],
        "evidenceScore": 50,
        "evidenceCalculation": {},
    }

    def failing_gnews(_claims):
        evidence_tracker.mark_failure("GNews", "simulated outage")
        return {}

    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "OPENAI_API_KEY": ""}), \
         patch.object(engine.summarizer, "gemini_key", None), \
         patch.object(engine.summarizer, "openai_key", None), \
         patch.object(engine.ai_reasoner, "gemini_key", None), \
         patch.object(engine.ai_reasoner, "openai_key", None), \
         patch.object(engine, "query_external_factcheck", return_value=[]), \
         patch("services.ai_service.gnews_service.search_gnews_for_claims", side_effect=failing_gnews), \
         patch.object(engine.rag_retriever, "retrieve_evidence_for_claim", return_value=[]), \
         patch.object(engine.evidence_verifier, "verify_all_claims", return_value=minimal_evidence):
        result = engine.analyze("The moon is made of cheese and this is a test claim.")

    assert result["evidenceUnavailable"] is True
    assert "GNews" in result["evidenceUnavailableSources"]
    assert result["evidence"]["evidenceUnavailable"] is True
    assert result["evidence"]["supportingEvidenceStatus"] == "EVIDENCE UNAVAILABLE"
    assert "not evidence of absence" in result["emptyEvidenceAudit"]["reason"]
