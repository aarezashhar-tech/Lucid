import pytest
from services.language_service import get_language_service
from services.article_classifier import get_article_classifier
from services.evidence_verifier import get_evidence_verifier
from services.multi_factor_scorer import get_scoring_engine
from services.ai_service import get_verification_engine


def test_language_detection():
    lang_service = get_language_service()

    # Spanish
    es_res = lang_service.detect_language("El presidente del gobierno anunció nuevas medidas económicas en Madrid.")
    assert es_res["languageCode"] == "es"
    assert es_res["code"] == "es"
    assert es_res["isForeignLanguage"] is True

    # French
    fr_res = lang_service.detect_language("Le ministre des affaires étrangères a déclaré une nouvelle initiative à Paris.")
    assert fr_res["languageCode"] == "fr"
    assert fr_res["code"] == "fr"
    assert fr_res["isForeignLanguage"] is True

    # Arabic script heuristic
    ar_res = lang_service.detect_language("أعلنت وزارة الصحة عن افتتاح مستشفى جديد في العاصمة")
    assert ar_res["languageCode"] == "ar"
    assert ar_res["code"] == "ar"
    assert ar_res["isForeignLanguage"] is True

    # English
    en_res = lang_service.detect_language("The Federal Reserve announced an unexpected interest rate cut in Washington today.")
    assert en_res["languageCode"] == "en"
    assert en_res["code"] == "en"
    assert en_res["isForeignLanguage"] is False


def test_safe_translation_fallback():
    lang_service = get_language_service()
    # Process text safely (never throws unhandled exceptions even if rate-limited or offline)
    res = lang_service.process_content("El banco central subió los tipos de interés.")
    assert "language" in res or "detectedLanguage" in res
    assert "analysisText" in res
    assert len(res["analysisText"]) > 0


def test_article_type_classification():
    classifier = get_article_classifier()

    # News Report
    news_text = "WASHINGTON (Reuters) - The US Treasury Department announced new sanctions on Friday targeting offshore entities."
    res_news = classifier.classify_article_type(news_text)
    assert res_news["articleType"] == "News Report"

    # Opinion Piece
    op_text = "Opinion: In my view, the current economic trajectory is deeply flawed. We must reconsider our foundational approach to fiscal spending."
    res_op = classifier.classify_article_type(op_text)
    assert res_op["articleType"] == "Opinion Piece"

    # Historical Analysis
    hist_text = "Historical Analysis: In 1914, the diplomatic breakdown of the Archduke's assassination illuminated centuries of imperial alliances."
    res_hist = classifier.classify_article_type(hist_text)
    assert res_hist["articleType"] == "Historical Analysis"

    # Feature Article
    feat_text = "Feature: Inside the hidden lives of deep-sea oceanographers exploring the Mariana Trench."
    res_feat = classifier.classify_article_type(feat_text)
    assert res_feat["articleType"] == "Feature Article"


def test_source_quality_assessment_for_historical_analysis():
    classifier = get_article_classifier()
    text = "In the archives of the Library of Congress, historical manuscripts reveal the nuanced treaty negotiations of 1783."
    source_context = {
        "sourceName": "National Archives Historical Press",
        "domain": "archives.gov",
        "trustScore": 92,
        "tier": "Tier 1: Wire Services & Public Archives"
    }
    assessment = classifier.generate_source_quality_assessment(text, source_context, article_type="Historical Analysis")
    assert assessment["overallRating"] in ("Authoritative Historical / Academic Standard", "High Scholarly / Archival Authority", "Reputable Historical Review")
    assert assessment["sourceQualityScore"] >= 80
    assert "scholarlyStandards" in assessment
    assert "methodologicalRigor" in assessment
    assert "archivalIntegrity" in assessment
    assert "sourceAuthority" in assessment


def test_evidence_retrieval_transparency_when_citations_zero():
    verifier = get_evidence_verifier()
    # Verify a novel claim where no citations exist
    claim = {
        "id": "claim-1",
        "claim": "Astronauts discovered a hidden subterranean lake beneath the lunar south pole yesterday.",
        "claimNumber": 1,
        "status": "Unverified",
        "supportingSources": [],
        "contradictingSources": [],
        "supportingEvidenceCount": 0,
        "contradictingEvidenceCount": 0
    }
    res = verifier.verify_all_claims(
        claims=[claim],
        source_context={"sourceName": "Unknown Blog", "trustScore": 50, "tier": "Tier 3: Low Trust"},
        article_text="Astronauts discovered a hidden subterranean lake beneath the lunar south pole yesterday.",
        external_factchecks_map={},
        gnews_map={}
    )

    audit = res.get("emptyEvidenceAudit")
    assert audit is not None
    assert audit["resultsFound"] == 0
    assert audit["reason"] == "No sufficiently relevant corroborating sources were found."
    assert audit["sourcesSearched"] == ["Reuters", "AP", "BBC", "Google Fact Check", "GNews"]


def test_scoring_genre_differentiation():
    scorer = get_scoring_engine()
    classifier = get_article_classifier()

    # 1. Historical Analysis should not be penalized by zero breaking-news wire citations
    hist_source_data = {"sourceName": "Oxford University Press", "trustScore": 94, "tier": "Tier 1: Reputable Academic Press"}
    hist_assessment = classifier.generate_source_quality_assessment(
        "A historical study of the Roman Senate's fiscal reforms in the late Republic.",
        hist_source_data,
        article_type="Historical Analysis"
    )
    hist_verdict = scorer.compute_verdict_and_score(
        ml_score=80,
        source_score=94,
        evidence_score=0,
        claims=[],
        ling_signals={"sensational_score": 0, "emotional_language": False},
        supporting_evidence_count=0,
        contradicting_evidence_count=0,
        article_type="Historical Analysis",
        source_quality_assessment=hist_assessment
    )
    assert "HISTORICAL ANALYSIS" in hist_verdict["verdict"]
    assert hist_verdict["finalTrustScore"] >= 80

    # 2. Opinion Piece should yield Opinion - not fact-checked / OPINION verdict and "Not scored"
    op_verdict = scorer.compute_verdict_and_score(
        ml_score=75,
        source_score=85,
        evidence_score=0,
        claims=[],
        ling_signals={"sensational_score": 5, "emotional_language": False},
        supporting_evidence_count=0,
        contradicting_evidence_count=0,
        article_type="Opinion Piece"
    )
    assert op_verdict["verdict"] in ("Opinion - not fact-checked", "OPINION / COMMENTARY")
    assert op_verdict["finalTrustScore"] == "Not scored"


def test_end_to_end_multilingual_and_genres():
    engine = get_verification_engine()

    # Spanish text
    spanish_article = "El Ministerio de Economía de España informó sobre el crecimiento del producto interno bruto en el último trimestre."
    result = engine.analyze(spanish_article, source_hint="elpais.com")

    assert result["detectedLanguage"] == "Spanish" or result["language"]["languageCode"] == "es"
    assert "articleType" in result
    assert "emptyEvidenceAudit" in result or result["evidence"]["supportingCount"] >= 0
    assert result["trustScore"] > 0
