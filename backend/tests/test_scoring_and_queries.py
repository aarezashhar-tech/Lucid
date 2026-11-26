import pytest
from services.multi_factor_scorer import MultiFactorScoringEngine, WEIGHT_EVIDENCE_DIRECT, WEIGHT_CONSISTENCY_DIRECT
from services.rag_retriever import RAGRetrievalEngine
from services.evidence_ranker import EvidenceRanker
from services.wikipedia_service import get_wikipedia_service

def test_ml_and_language_weights_zero():
    scorer = MultiFactorScoringEngine()
    result = scorer.compute_verdict_and_score(
        ml_score=95.0,
        source_score=80.0,
        evidence_score=85.0,
        claims=[{"claim": "Test claim", "claimType": "factual"}],
        ling_signals={"clickbaitScore": 10},
        supporting_evidence_count=3,
        contradicting_evidence_count=0,
        is_direct_submission=False,
        source_domain="reuters.com"
    )
    factors = result["scoringFactors"]
    assert factors["mlPrediction"]["weight"] == 0.0
    assert factors["mlPrediction"]["contribution"] == 0.0
    assert factors["languageQuality"]["weight"] == 0.0
    assert factors["languageQuality"]["contribution"] == 0.0
    assert factors["evidenceStrength"]["weight"] == 0.80
    assert factors["sourceReputation"]["weight"] == 0.10
    assert factors["claimConsistency"]["weight"] == 0.10

def test_direct_text_submission_weights():
    scorer = MultiFactorScoringEngine()
    result = scorer.compute_verdict_and_score(
        ml_score=50.0,
        source_score=50.0,
        evidence_score=80.0,
        claims=[{"claim": "Direct claim", "claimType": "factual"}],
        ling_signals={},
        supporting_evidence_count=2,
        contradicting_evidence_count=0,
        is_direct_submission=True
    )
    factors = result["scoringFactors"]
    assert factors["evidenceStrength"]["weight"] == 0.90
    assert factors["sourceReputation"]["weight"] == 0.00
    assert factors["claimConsistency"]["weight"] == 0.10

def test_query_cleaner_and_conversational_retrieval():
    engine = RAGRetrievalEngine()
    claim = "after jensen huang return's as the ceo , the nvidia stock goes up again."
    cleaned = engine._clean_claim_query(claim)
    assert "Jensen" in cleaned or "Huang" in cleaned or "CEO" in cleaned or "Nvidia" in cleaned
    assert "return's" not in cleaned
    assert "after" not in cleaned.lower().split()

def test_superlative_contradiction_milky_way():
    wiki = get_wikipedia_service()
    claim = "milkyway is the largest galaxy in the entire world"
    check = wiki.check_biographical_or_encyclopedic_claim(claim)
    assert check is not None
    assert check["isContradicted"] is True
    assert check["verdict"] == "FALSE"
    assert check["evidenceItem"]["stance"] == "contradict"

def test_milky_way_verdict_likely_false():
    scorer = MultiFactorScoringEngine()
    wiki = get_wikipedia_service()
    claim = "milkyway is the largest galaxy in the entire world"
    check = wiki.check_biographical_or_encyclopedic_claim(claim)
    contra_items = [check["evidenceItem"]]
    
    result = scorer.compute_verdict_and_score(
        ml_score=50.0,
        source_score=50.0,
        evidence_score=10.0,
        claims=[{"claim": claim, "claimType": "factual"}],
        ling_signals={},
        supporting_evidence_count=0,
        contradicting_evidence_count=1,
        top_supporting=[],
        top_contradicting=contra_items,
        is_direct_submission=True
    )
    assert result["verdict"] in ("FALSE", "LIKELY FALSE", "KNOWN FALSE")
    assert result["finalTrustScore"] <= 30
