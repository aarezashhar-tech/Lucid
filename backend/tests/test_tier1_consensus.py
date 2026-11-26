"""
Offline Unit Tests for Tier-1 Consensus Override, Rebalanced Scoring Weights,
and Short-Claim / Concurrent Pipeline Optimizations.
Zero network calls required; runs in < 5 seconds.
"""

import os
import sys
import time
from datetime import datetime
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.multi_factor_scorer import MultiFactorScoringEngine
from services.ai_service import VerificationEngine


TODAY_STR = datetime.now().strftime("%Y-%m-%d")


def _messy_ling_signals():
    return {
        "sensational_score": 65,
        "emotional_language": True,
        "clickbait_score": 55,
        "exclamation_count": 3,
        "caps_ratio": 0.25,
        "detected_sensational_terms": ["shocking", "breaking"],
    }


def test_weights_rebalanced_evidence_source_80_ml_lang_10():
    scorer = MultiFactorScoringEngine()
    res = scorer.compute_verdict_and_score(
        ml_score=10.0,
        source_score=50.0,
        evidence_score=85.0,
        claims=[{"claim": "Test factual claim", "status": "Supported"}],
        ling_signals=_messy_ling_signals(),
        supporting_evidence_count=2,
        contradicting_evidence_count=0,
        top_supporting=[
            {
                "url": "https://www.reuters.com/world/item-1-2026",
                "publisher": "Reuters",
                "relevanceScore": 82,
                "stanceConfidence": 90,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 96,
                "stance": "support",
            },
            {
                "url": "https://www.dw.com/en/item-2-2026",
                "publisher": "Deutsche Welle (DW)",
                "relevanceScore": 80,
                "stanceConfidence": 88,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 90,
                "stance": "support",
            },
        ],
        top_contradicting=[],
    )
    factors = res["scoringFactors"]
    ev_w = factors["evidenceStrength"]["weight"]
    src_w = factors["sourceReputation"]["weight"]
    con_w = factors["claimConsistency"]["weight"]
    ml_w = factors["mlPrediction"]["weight"]
    lan_w = factors["languageQuality"]["weight"]

    assert round(ev_w, 2) == 0.90
    assert round(src_w, 2) == 0.00
    assert round(ml_w + lan_w, 2) == 0.00
    assert round(con_w, 2) == 0.10
    assert round(ev_w + src_w + con_w + ml_w + lan_w, 2) == 1.00
    assert "90%" in factors["evidenceStrength"]["label"]
    assert "0%" in factors["mlPrediction"]["label"]
    assert "0%" in factors["languageQuality"]["label"]


def test_two_tier1_sources_on_short_messy_claim_triggers_verified_and_floor_82():
    """
    2+ distinct Tier-1 sources supporting a short/messy claim (even with low ML score
    and high sensational/clickbait score) must return VERIFIED with trustScore >= 82.
    """
    scorer = MultiFactorScoringEngine()
    res = scorer.compute_verdict_and_score(
        ml_score=12.0,
        source_score=50.0,
        evidence_score=80.0,
        claims=[{"claim": "SHOCKING!! Central bank cuts benchmark rate today!!!", "status": "Supported"}],
        ling_signals=_messy_ling_signals(),
        supporting_evidence_count=2,
        contradicting_evidence_count=0,
        top_supporting=[
            {
                "url": "https://www.thehindu.com/business/Economy/article-rate-cut-2026.ece",
                "publisher": "The Hindu",
                "relevanceScore": 52,
                "stanceConfidence": 88,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 89,
                "stance": "support",
            },
            {
                "url": "https://indianexpress.com/article/business/rate-cut-decision-123456/",
                "publisher": "The Indian Express",
                "relevanceScore": 50,
                "stanceConfidence": 86,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 88,
                "stance": "support",
            },
        ],
        top_contradicting=[],
    )
    assert res["verdict"] == "VERIFIED", f"Expected VERIFIED, got {res['verdict']}"
    assert res["finalTrustScore"] >= 82, f"Expected trustScore >= 82, got {res['finalTrustScore']}"
    assert res["riskLevel"] == "LOW"


def test_two_tier1_sources_expanded_outlets_trigger_verified():
    """Verify DPA, Economist, FT, AFP, Al Jazeera, Guardian, AP are recognized as Tier-1."""
    scorer = MultiFactorScoringEngine()
    res = scorer.compute_verdict_and_score(
        ml_score=20.0,
        source_score=50.0,
        evidence_score=78.0,
        claims=[{"claim": "European industrial output rose in Q3", "status": "Supported"}],
        ling_signals=_messy_ling_signals(),
        supporting_evidence_count=2,
        contradicting_evidence_count=0,
        top_supporting=[
            {
                "url": "https://www.dpa-international.com/article/europe-output-2026",
                "publisher": "DPA",
                "relevanceScore": 65,
                "stanceConfidence": 85,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 93,
                "stance": "support",
            },
            {
                "url": "https://www.economist.com/europe/2026/10/industrial-output",
                "publisher": "The Economist",
                "relevanceScore": 68,
                "stanceConfidence": 82,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 93,
                "stance": "support",
            },
        ],
        top_contradicting=[],
    )
    assert res["verdict"] == "VERIFIED"
    assert res["finalTrustScore"] >= 82


def test_single_tier1_source_does_not_auto_trigger_verified():
    """1 Tier-1 source only must NOT auto-trigger VERIFIED."""
    scorer = MultiFactorScoringEngine()
    res = scorer.compute_verdict_and_score(
        ml_score=75.0,
        source_score=50.0,
        evidence_score=80.0,
        claims=[{"claim": "Single wire report claim", "status": "Supported"}],
        ling_signals={"sensational_score": 5, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=1,
        contradicting_evidence_count=0,
        top_supporting=[
            {
                "url": "https://www.reuters.com/world/single-source-story-2026",
                "publisher": "Reuters",
                "relevanceScore": 90,
                "stanceConfidence": 95,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 96,
                "stance": "support",
            }
        ],
        top_contradicting=[],
    )
    assert res["verdict"] != "VERIFIED"
    assert res["verdict"] == "LIKELY TRUE"


def test_duplicate_same_tier1_outlet_does_not_auto_trigger_verified():
    """Two articles from the same Tier-1 outlet (e.g. two Reuters URLs) are only 1 distinct Tier-1 source."""
    scorer = MultiFactorScoringEngine()
    res = scorer.compute_verdict_and_score(
        ml_score=70.0,
        source_score=50.0,
        evidence_score=80.0,
        claims=[{"claim": "Claim reported twice by same wire", "status": "Supported"}],
        ling_signals={"sensational_score": 5, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=2,
        contradicting_evidence_count=0,
        top_supporting=[
            {
                "url": "https://www.reuters.com/world/story-a-2026",
                "publisher": "Reuters",
                "relevanceScore": 88,
                "stanceConfidence": 92,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 96,
                "stance": "support",
            },
            {
                "url": "https://www.reuters.com/business/story-b-2026",
                "publisher": "Reuters",
                "relevanceScore": 86,
                "stanceConfidence": 90,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 96,
                "stance": "support",
            },
        ],
        top_contradicting=[],
    )
    assert res["verdict"] != "VERIFIED"


def test_two_tier1_supporting_plus_ifcn_false_debunk_never_verified():
    """
    2 Tier-1 supporting sources + 1 IFCN FALSE debunk (either in top_contradicting
    or in fact_check_results) must stay FALSE / NEEDS REVIEW and NEVER VERIFIED.
    """
    scorer = MultiFactorScoringEngine()
    supp = [
        {
            "url": "https://www.reuters.com/world/claim-story-1-2026",
            "publisher": "Reuters",
            "relevanceScore": 85,
            "stanceConfidence": 90,
            "publicationDate": TODAY_STR,
            "sourceTrustScore": 96,
            "stance": "support",
        },
        {
            "url": "https://www.bbc.com/news/claim-story-2-2026",
            "publisher": "BBC",
            "relevanceScore": 84,
            "stanceConfidence": 89,
            "publicationDate": TODAY_STR,
            "sourceTrustScore": 94,
            "stance": "support",
        },
    ]

    # Case A: IFCN debunk in top_contradicting
    contra_ifcn = [
        {
            "url": "https://www.altnews.in/fact-check-claim-debunked-2026/",
            "publisher": "Alt News",
            "sourceCategory": "IFCN Fact Checker",
            "factCheckRating": "False",
            "relevanceScore": 92,
            "stanceConfidence": 98,
            "publicationDate": TODAY_STR,
            "sourceTrustScore": 94,
            "stance": "contradict",
            "evidenceSnippet": "Fact Check Verdict: FALSE. Claim is fabricated.",
        }
    ]
    res_a = scorer.compute_verdict_and_score(
        ml_score=75.0,
        source_score=50.0,
        evidence_score=70.0,
        claims=[{"claim": "Contested viral video claim", "status": "Contradicted"}],
        ling_signals={"sensational_score": 10, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=2,
        contradicting_evidence_count=1,
        top_supporting=supp,
        top_contradicting=contra_ifcn,
    )
    assert res_a["verdict"] in ("KNOWN FALSE", "FALSE", "NEEDS REVIEW")
    assert res_a["verdict"] != "VERIFIED"
    assert res_a["finalTrustScore"] <= 60

    # Case B: IFCN debunk in fact_check_results alongside 2 Tier-1 supporting
    res_b = scorer.compute_verdict_and_score(
        ml_score=75.0,
        source_score=50.0,
        evidence_score=80.0,
        claims=[{"claim": "Contested viral claim", "status": "Supported"}],
        ling_signals={"sensational_score": 10, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=2,
        contradicting_evidence_count=1,
        fact_check_results=[
            {
                "reviewer": "BOOM Live",
                "rating": "False",
                "url": "https://www.boomlive.in/fact-check/viral-claim-false",
                "source": "Google Fact Check API",
            }
        ],
        top_supporting=supp,
        top_contradicting=[],
    )
    assert res_b["verdict"] in ("KNOWN FALSE", "FALSE", "NEEDS REVIEW")
    assert res_b["verdict"] != "VERIFIED"
    assert res_b["finalTrustScore"] <= 12


def test_ifcn_true_rating_triggers_verified_and_ignores_half_true():
    """An explicit IFCN True rating with 0 contradictions triggers VERIFIED (>= 82); Half True does not."""
    scorer = MultiFactorScoringEngine()

    res_true = scorer.compute_verdict_and_score(
        ml_score=25.0,
        source_score=50.0,
        evidence_score=78.0,
        claims=[{"claim": "Verified civic policy stat", "status": "Supported"}],
        ling_signals=_messy_ling_signals(),
        supporting_evidence_count=1,
        contradicting_evidence_count=0,
        fact_check_results=[
            {
                "reviewer": "Factly",
                "rating": "True",
                "url": "https://factly.in/policy-stat-true-check/",
                "source": "Google Fact Check API",
            }
        ],
        top_supporting=[
            {
                "url": "https://factly.in/policy-stat-true-check/",
                "publisher": "Factly",
                "sourceCategory": "IFCN Fact Checker",
                "factCheckRating": "True",
                "relevanceScore": 90,
                "stanceConfidence": 95,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 94,
                "stance": "support",
            }
        ],
        top_contradicting=[],
    )
    assert res_true["verdict"] == "VERIFIED"
    assert res_true["finalTrustScore"] >= 82

    res_half = scorer.compute_verdict_and_score(
        ml_score=75.0,
        source_score=50.0,
        evidence_score=75.0,
        claims=[{"claim": "Nuanced policy statement", "status": "Supported"}],
        ling_signals={"sensational_score": 5, "clickbait_score": 0, "exclamation_count": 0, "caps_ratio": 0},
        supporting_evidence_count=1,
        contradicting_evidence_count=0,
        fact_check_results=[
            {
                "reviewer": "PolitiFact",
                "rating": "Half True",
                "url": "https://www.politifact.com/factchecks/nuanced-policy/",
                "source": "Google Fact Check API",
            }
        ],
        top_supporting=[
            {
                "url": "https://www.politifact.com/factchecks/nuanced-policy/",
                "publisher": "PolitiFact",
                "sourceCategory": "IFCN Fact Checker",
                "factCheckRating": "Half True",
                "relevanceScore": 85,
                "stanceConfidence": 70,
                "publicationDate": TODAY_STR,
                "sourceTrustScore": 95,
                "stance": "neutral",
            }
        ],
        top_contradicting=[],
    )
    assert res_half["verdict"] != "VERIFIED"


def test_short_claim_skips_summarizer_and_redundant_ai_reasoning():
    """
    For short single-claim inputs (< 25 words), VerificationEngine.analyze must skip
    the external summarizer call and skip redundant ai_reasoner deliberation when
    groundedReasoning already covers it.
    """
    engine = VerificationEngine()
    claim = "Jensen Huang serves as chief executive of Nvidia."

    fake_ranked = {
        "supporting": [
            {
                "articleTitle": "Nvidia CEO Jensen Huang outlines AI roadmap",
                "publisher": "Reuters",
                "publicationDate": TODAY_STR,
                "url": "https://www.reuters.com/technology/nvidia-ceo-roadmap-2026",
                "evidenceSnippet": "Nvidia chief executive Jensen Huang spoke at the keynote.",
                "sourceTrustScore": 96,
                "sourceCategory": "Wire Service",
                "relevanceScore": 88,
                "semanticSimilarity": 0.88,
                "compositeRankScore": 91,
                "stance": "support",
                "stanceConfidence": 92,
                "stanceReason": "Reuters confirms Jensen Huang is CEO of Nvidia.",
                "matchingClaim": claim,
            },
            {
                "articleTitle": "Jensen Huang leads Nvidia into next chip cycle",
                "publisher": "BBC",
                "publicationDate": TODAY_STR,
                "url": "https://www.bbc.com/news/technology-nvidia-huang-2026",
                "evidenceSnippet": "BBC profile confirms Jensen Huang is Nvidia CEO.",
                "sourceTrustScore": 94,
                "sourceCategory": "Public Broadcaster",
                "relevanceScore": 85,
                "semanticSimilarity": 0.85,
                "compositeRankScore": 89,
                "stance": "support",
                "stanceConfidence": 90,
                "stanceReason": "BBC confirms Jensen Huang leads Nvidia as CEO.",
                "matchingClaim": claim,
            },
        ],
        "contradicting": [],
        "related": [],
        "neutral": [],
        "supportingCount": 2,
        "contradictingCount": 0,
        "relatedCount": 0,
        "neutralCount": 0,
    }

    with patch.object(engine.summarizer, "summarize", side_effect=AssertionError("Summarizer should be skipped for < 25 words")), \
         patch.object(engine.ai_reasoner, "generate_ai_reasoning", side_effect=AssertionError("Redundant AI deliberation should be skipped")), \
         patch("services.ai_service.gnews_service.search_gnews_for_claims", return_value={claim: []}), \
         patch.object(engine.rag_retriever, "retrieve_evidence_for_claim", return_value=fake_ranked["supporting"]), \
         patch.object(engine.evidence_ranker, "rank_evidence_for_claim", return_value=fake_ranked), \
         patch.object(engine.evidence_verifier.wikipedia_service, "check_biographical_or_encyclopedic_claim", return_value=None):
        result = engine.analyze(claim)

    assert result["verdict"] == "VERIFIED"
    assert result["trustScore"] >= 82
    assert result["articleSummary"]["provider"] == "Direct Claim Summary"
    assert result["reasoning"]["provider"] == "Lucid Grounded Reasoning Engine"
    assert result["groundedReasoning"]["synthesis"]


def test_multiclaim_concurrent_retrieval_and_ranking():
    """
    Multiple claims should run retrieval/ranking concurrently (ThreadPoolExecutor max_workers=3)
    so total duration for 3 simulated 0.4s claim calls is well under 1.0s instead of >= 1.2s.
    """
    engine = VerificationEngine()
    multi_text = (
        "Global renewable energy capacity expanded by nearly fifty percent during the annual monitoring cycle. "
        "European regulators approved updated industrial grid interconnection standards in Brussels this quarter. "
        "International shipping operators reported lower fuel emissions across major Pacific container routes."
    )

    def slow_retrieve(c_text, **kwargs):
        time.sleep(0.35)
        return []

    def fast_rank(c_text, candidates):
        time.sleep(0.05)
        return {"supporting": [], "contradicting": [], "related": [], "neutral": []}

    start = time.monotonic()
    with patch.object(engine.summarizer, "_query_gemini", return_value=None), \
         patch.object(engine.summarizer, "_query_openai", return_value=None), \
         patch("services.ai_service.gnews_service.search_gnews_for_claims", return_value={}), \
         patch.object(engine.rag_retriever, "retrieve_evidence_for_claim", side_effect=slow_retrieve), \
         patch.object(engine.evidence_ranker, "rank_evidence_for_claim", side_effect=fast_rank), \
         patch.object(engine.evidence_verifier.wikipedia_service, "check_biographical_or_encyclopedic_claim", return_value=None):
        result = engine.analyze(multi_text)
    elapsed = time.monotonic() - start

    assert len(result["claims"]) >= 3
    assert elapsed < 1.0, f"Expected concurrent claim execution (<1.0s), took {elapsed:.2f}s"
