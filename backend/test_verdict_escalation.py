"""
Unit tests for verdict escalation logic:
1. True article with no citations -> UNVERIFIED
2. False fact-check result -> KNOWN FALSE
3. Mixed evidence -> NEEDS REVIEW
4. Strong corroboration -> LIKELY TRUE or VERIFIED

Safeguards verified:
- Missing evidence must NEVER be treated as evidence of falsity.
- KNOWN FALSE only if Google Fact Check returns a false/debunked rating
  OR contradicting evidence exceeds supporting evidence by defined threshold.
- LIKELY FALSE when some contradicting evidence exists but not enough for definitive false.
- UNVERIFIED when no reliable supporting or contradicting citations attached,
  or evidence score below threshold due to missing citations.
"""
import os
import sys
from unittest.mock import patch, MagicMock
import pytest

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app
from services.multi_factor_scorer import get_scoring_engine, MultiFactorScoringEngine
from services.ai_service import get_verification_engine


@pytest.fixture
def scoring_engine():
    return get_scoring_engine()


@pytest.fixture
def verification_engine():
    return get_verification_engine()


@pytest.fixture
def client():
    return app.test_client()


# =========================================================================
# TEST 1: True article with no citations -> UNVERIFIED
# =========================================================================
def test_true_article_with_no_citations_is_unverified(scoring_engine, verification_engine, client):
    """
    Test 1: True article with no citations -> UNVERIFIED.
    Missing evidence must NEVER be treated as evidence of falsity.
    Even with high ML confidence or standard objective prose, absent attached
    verifiable citations the verdict must remain UNVERIFIED.
    """
    # 1. Scoring Engine level check
    result = scoring_engine.compute_verdict_and_score(
        ml_score=85,
        source_score=50,
        evidence_score=50,  # Baseline evidence score when citations are missing
        claims=[{"claim": "Researchers noted 14 fish species in the local river basin.", "status": "Unverified"}],
        ling_signals={"sensational_score": 5, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=0,
        contradicting_evidence_count=0,
        fact_check_results=[]
    )

    assert result["verdict"] == "UNVERIFIED", f"Expected UNVERIFIED, got {result['verdict']}"
    assert result["riskLevel"] == "MEDIUM"
    assert "no reliable supporting or contradicting citations attached" in result["summaryReason"].lower()

    # 2. Even with low ML score (suspicious classifier), missing citations MUST NOT yield KNOWN FALSE or LIKELY FALSE
    low_ml_res = scoring_engine.compute_verdict_and_score(
        ml_score=15,
        source_score=30,
        evidence_score=40,
        claims=[{"claim": "Local community council passed zoning amendment.", "status": "Unverified"}],
        ling_signals={"sensational_score": 10, "clickbait_score": 5, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=0,
        contradicting_evidence_count=0,
        fact_check_results=[]
    )
    assert low_ml_res["verdict"] == "UNVERIFIED", (
        f"Missing evidence must NEVER be treated as falsity! Got {low_ml_res['verdict']}"
    )
    assert low_ml_res["verdict"] not in ("KNOWN FALSE", "LIKELY FALSE")

    # 3. Pipeline / API level check
    true_article_no_citations = (
        "Researchers at the Regional Environmental Institute documented fourteen native fish species "
        "across three distinct river basins during an ecological survey conducted last spring."
    )
    with patch("services.factcheck_service.query_google_factcheck", return_value=[]):
        with patch("services.gnews_service.search_gnews_for_claims", return_value={}):
            api_res = verification_engine.analyze(true_article_no_citations)
            assert api_res["verdict"] == "UNVERIFIED", f"Expected UNVERIFIED via pipeline, got {api_res['verdict']}"
            assert api_res["evidence"]["supportingCount"] == 0
            assert api_res["evidence"]["contradictingCount"] == 0
            assert api_res["verdict"] not in ("KNOWN FALSE", "LIKELY FALSE", "FALSE")


# =========================================================================
# TEST 2: False fact-check result -> KNOWN FALSE
# =========================================================================
def test_false_factcheck_result_is_known_false(scoring_engine, verification_engine):
    """
    Test 2: False fact-check result -> KNOWN FALSE.
    When Google Fact Check returns a false or debunked rating, the verdict must
    escalate to KNOWN FALSE with CRITICAL risk level.
    """
    # 1. Scoring Engine level check
    mock_fc_results = [
        {
            "claim": "Secret 5G mind control microchips in drinking water",
            "claimant": "Viral Posts",
            "reviewer": "PolitiFact",
            "rating": "Pants on Fire",
            "url": "https://www.politifact.com/factchecks/5g-water-hoax/",
            "source": "Google Fact Check API"
        }
    ]
    result = scoring_engine.compute_verdict_and_score(
        ml_score=20,
        source_score=50,
        evidence_score=20,
        claims=[{"claim": "Secret 5G microchips in drinking water", "status": "Contradicted"}],
        ling_signals={"sensational_score": 40, "clickbait_score": 35, "exclamation_count": 2, "caps_ratio": 0.15},
        supporting_evidence_count=0,
        contradicting_evidence_count=1,
        fact_check_results=mock_fc_results
    )
    assert result["verdict"] == "KNOWN FALSE", f"Expected KNOWN FALSE, got {result['verdict']}"
    assert result["riskLevel"] == "CRITICAL"
    assert "google fact check" in result["summaryReason"].lower()

    # 2. Check various debunk keywords ("Debunked", "False", "Incorrect")
    for debunk_rating in ["Debunked", "False", "Incorrect", "Mostly False", "Fake"]:
        debunk_fc = [{"reviewer": "FactCheck.org", "rating": debunk_rating, "url": "https://factcheck.org"}]
        res = scoring_engine.compute_verdict_and_score(
            ml_score=50,
            source_score=50,
            evidence_score=30,
            claims=[],
            ling_signals={},
            supporting_evidence_count=0,
            contradicting_evidence_count=1,
            fact_check_results=debunk_fc
        )
        assert res["verdict"] == "KNOWN FALSE", f"Rating '{debunk_rating}' should trigger KNOWN FALSE, got {res['verdict']}"

    # 3. Pipeline level check with mocked Google Fact Check API
    mock_payload = {
        "claims": [
            {
                "text": "Drinking seawater cures all chronic viral infections",
                "claimant": "Social Media",
                "claimReview": [
                    {
                        "publisher": {"name": "Snopes", "site": "snopes.com"},
                        "textualRating": "False",
                        "url": "https://www.snopes.com/fact-check/seawater-cure-hoax/",
                        "reviewDate": "2024-02-01T00:00:00Z"
                    }
                ]
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.dict(os.environ, {"FACTCHECK_API_KEY": "test-key"}):
        with patch("requests.get", return_value=mock_resp):
            pipe_res = verification_engine.analyze("Drinking seawater cures all chronic viral infections.")
            assert pipe_res["verdict"] in ("KNOWN FALSE", "FALSE"), f"Expected KNOWN FALSE or FALSE, got {pipe_res['verdict']}"
            assert pipe_res["riskLevel"] == "CRITICAL"
            assert len(pipe_res["factCheckResults"]) > 0


# =========================================================================
# TEST 3: Mixed evidence -> NEEDS REVIEW
# =========================================================================
def test_mixed_evidence_is_needs_review(scoring_engine):
    """
    Test 3: Mixed evidence -> NEEDS REVIEW.
    When an article has both supporting and contradicting citations, and the
    contradictions do not exceed support by the defined threshold, the verdict
    must be NEEDS REVIEW with MEDIUM risk.
    """
    # Case A: Balanced mixed evidence (2 supporting, 2 contradicting)
    res_balanced = scoring_engine.compute_verdict_and_score(
        ml_score=60,
        source_score=75,
        evidence_score=50,
        claims=[
            {"claim": "Claim A", "status": "Supported"},
            {"claim": "Claim B", "status": "Contradicted"}
        ],
        ling_signals={"sensational_score": 10, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=2,
        contradicting_evidence_count=2,
        fact_check_results=[]
    )
    assert res_balanced["verdict"] == "NEEDS REVIEW", f"Expected NEEDS REVIEW, got {res_balanced['verdict']}"
    assert res_balanced["riskLevel"] == "MEDIUM"

    # Case B: 1 supporting vs 1 contradicting
    res_equal = scoring_engine.compute_verdict_and_score(
        ml_score=55,
        source_score=70,
        evidence_score=50,
        claims=[{"claim": "Claim X", "status": "Unverified"}],
        ling_signals={},
        supporting_evidence_count=1,
        contradicting_evidence_count=1,
        fact_check_results=[]
    )
    assert res_equal["verdict"] == "NEEDS REVIEW", f"Expected NEEDS REVIEW, got {res_equal['verdict']}"

    # Case C: 2 supporting vs 1 contradicting (contradictions do not exceed support)
    res_lean_support = scoring_engine.compute_verdict_and_score(
        ml_score=70,
        source_score=80,
        evidence_score=65,
        claims=[{"claim": "Claim Y", "status": "Supported"}],
        ling_signals={},
        supporting_evidence_count=2,
        contradicting_evidence_count=1,
        fact_check_results=[]
    )
    assert res_lean_support["verdict"] == "NEEDS REVIEW", f"Expected NEEDS REVIEW, got {res_lean_support['verdict']}"


# =========================================================================
# TEST 4: Strong corroboration -> LIKELY TRUE or VERIFIED
# =========================================================================
def test_strong_corroboration_is_likely_true_or_verified(scoring_engine, verification_engine):
    """
    Test 4: Strong corroboration -> LIKELY TRUE or VERIFIED.
    When multiple supporting citations corroborate claims with 0 contradicting flags:
    - Extremely high authority/score yields VERIFIED.
    - Strong corroboration yields LIKELY TRUE.
    """
    # 1. Strong corroboration yielding LIKELY TRUE
    res_likely_true = scoring_engine.compute_verdict_and_score(
        ml_score=75,
        source_score=75,
        evidence_score=85,
        claims=[{"claim": "Official treaty signed in Geneva", "status": "Supported"}],
        ling_signals={"sensational_score": 0, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=3,
        contradicting_evidence_count=0,
        fact_check_results=[]
    )
    assert res_likely_true["verdict"] in ("LIKELY TRUE", "VERIFIED"), (
        f"Expected LIKELY TRUE or VERIFIED, got {res_likely_true['verdict']}"
    )
    assert res_likely_true["riskLevel"] == "LOW"

    # 2. Exceptional corroboration from accredited source yielding VERIFIED
    res_verified = scoring_engine.compute_verdict_and_score(
        ml_score=92,
        source_score=95,
        evidence_score=96,
        claims=[{"claim": "Major economic index released", "status": "Supported"}],
        ling_signals={"sensational_score": 0, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=4,
        contradicting_evidence_count=0,
        fact_check_results=[]
    )
    assert res_verified["verdict"] == "VERIFIED", f"Expected VERIFIED, got {res_verified['verdict']}"
    assert res_verified["riskLevel"] == "LOW"


# =========================================================================
# ADDITIONAL VERIFICATION: Contradiction threshold & Likely False
# =========================================================================
def test_contradiction_threshold_and_likely_false(scoring_engine):
    """
    Verifies defined threshold logic:
    - Contradicting evidence exceeds supporting by threshold (default 2) -> KNOWN FALSE.
    - Some contradicting evidence exists (< threshold, e.g. 1) -> LIKELY FALSE.
    """
    # Contradicting exceeds supporting by 2 (threshold): KNOWN FALSE
    res_exceeds = scoring_engine.compute_verdict_and_score(
        ml_score=40,
        source_score=50,
        evidence_score=20,
        claims=[{"claim": "Contradicted claim", "status": "Contradicted"}],
        ling_signals={},
        supporting_evidence_count=0,
        contradicting_evidence_count=2,
        fact_check_results=[]
    )
    assert res_exceeds["verdict"] == "KNOWN FALSE"
    assert res_exceeds["riskLevel"] == "CRITICAL"

    # 3 contradicting vs 1 supporting: difference is 2 -> KNOWN FALSE
    res_diff_exceeds = scoring_engine.compute_verdict_and_score(
        ml_score=40,
        source_score=50,
        evidence_score=25,
        claims=[],
        ling_signals={},
        supporting_evidence_count=1,
        contradicting_evidence_count=3,
        fact_check_results=[]
    )
    assert res_diff_exceeds["verdict"] == "KNOWN FALSE"

    # Single contradicting citation (1), 0 supporting -> LIKELY FALSE
    res_likely_false = scoring_engine.compute_verdict_and_score(
        ml_score=45,
        source_score=50,
        evidence_score=30,
        claims=[],
        ling_signals={},
        supporting_evidence_count=0,
        contradicting_evidence_count=1,
        fact_check_results=[]
    )
    assert res_likely_false["verdict"] == "LIKELY FALSE"
    assert res_likely_false["riskLevel"] == "HIGH"
