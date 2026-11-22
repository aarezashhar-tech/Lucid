"""
Unit and Integration Tests for scoring-consistency review fixes:
1. Publisher attribution from Google News RSS (<source> and title suffix, neutral 50 for unknown, no UN keyword hallucination).
2. Specific claim matching (geographic/agency/metric alignment, stance 'related' vs 'support').
3. Authentic Wikipedia extract & Wikidata death checks with [Derived check] label.
4. Zero evidence score cap (<= 50) and low confidence flag.
5. Short text language detection (< 20 chars defaults to 'en', no template prose praise).
6. Number / Name altered article score drop validation (Article 1, Article 2, Article 3).
"""
import pytest
from app import run_shared_verification_pipeline
from services.rag_retriever import get_rag_retriever
from services.evidence_ranker import get_evidence_ranker
from services.wikipedia_service import get_wikipedia_service
from services.language_service import get_language_service


def test_publisher_attribution_unknown_neutral_50():
    """
    Publishers like renewable-energy-industry.com, Carbon Brief, Fair Observer
    must get their real publisher names and neutral 50 trust score,
    never 'United Nations (UN)'.
    """
    rag = get_rag_retriever()
    test_cases = [
        ("renewable-energy-industry.com", "https://news.google.com/rss/articles/CBMiRGh0dHBzOi8vd3d3LnJlbmV3YWJsZS1lbmVyZ3ktaW5kdXN0cnkuY29tL25ld3MvcmVuZXdhYmxlLWVuZXJneS1pZWEtcmVwb3J00gEA"),
        ("Carbon Brief", "https://news.google.com/rss/articles/CBMiP2h0dHBzOi8vd3d3LmNhcmJvbmJyaWVmLm9yZy9kZXRhaWxlZC1xLWEtaWVhLWZhc3Rlc3QtcmVuZXdhYmxlLXJlcG9ydC_SAQA"),
        ("Fair Observer", "https://news.google.com/rss/articles/CBMiSWh0dHBzOi8vd3d3LmZhaXJvYnNlcnZlci5jb20vd29ybGQtbmV3cy9pZWEtcmVuZXdhYmxlcy1nbG9iYWwtY2FwYWNpdHktMjAyMy_SAQA")
    ]
    for pub_name, url in test_cases:
        meta = rag._match_publisher_metadata(pub_name, url)
        assert meta["publisher"] == pub_name, f"Expected publisher {pub_name}, got {meta['publisher']}"
        assert meta["score"] == 50, f"Expected neutral 50 score, got {meta['score']}"
        assert meta["publisher"] != "United Nations (UN)"


def test_evidence_matching_specific_claim_vs_related():
    """
    A headline with topic similarity but different geographic/institutional scope
    must be classified as 'related', never 'support'.
    """
    ranker = get_evidence_ranker()
    claim_text = "The World Health Organization (WHO) noted health benefits from reduced air pollution, while Reuters reported similar findings across European utilities."
    unrelated_headline = "Government approves action plan to reduce air pollution in Latin American cities - Reuters"
    
    ranked_res = ranker.rank_evidence_for_claim(
        claim_text=claim_text,
        candidate_items=[{
            "articleTitle": unrelated_headline,
            "publisher": "Reuters",
            "url": "https://reuters.com/latin-america-air-pollution",
            "evidenceSnippet": "Ministers in Latin America approved a regional pact targeting urban emissions.",
            "sourceTrustScore": 92,
            "sourceCategory": "News Agency / Wire"
        }]
    )
    assert ranked_res["supportingCount"] == 0, f"Specific claim mismatch must NOT be marked as 'support'! Got supportingCount={ranked_res['supportingCount']}"
    assert ranked_res["relatedCount"] == 1, f"Expected 1 related item, got {ranked_res['relatedCount']}"
    assert ranked_res["related"][0]["stance"] == "related"


def test_wikipedia_authentic_snippet_and_derived_label():
    """
    Wikipedia checks must show real API extracts / Wikidata checks and be prefixed with [Derived check].
    """
    wiki = get_wikipedia_service()
    res = wiki.check_biographical_or_encyclopedic_claim("PM Modi died")
    assert res.get("isContradicted") is True
    assert "[Derived check]" in res.get("reason", "")
    ev_item = res.get("evidenceItem", {})
    snippet = ev_item.get("evidenceSnippet", "")
    assert "[Derived check]" in snippet
    assert "no date of death recorded" in snippet


def test_zero_evidence_capped_at_50():
    """
    When no corroborating evidence is found, trust score must be capped at <= 50 with low confidence.
    """
    res = run_shared_verification_pipeline(
        text="A local resident observed three stray cats crossing the alley behind Elm Street on Tuesday evening.",
        title="Cats crossing Elm Street alley",
        input_type="text"
    )
    assert res["trustScore"] <= 50
    assert "Low confidence" in res["summaryReason"]
    assert res["confidence"] <= 40


def test_short_claim_language_detection_and_no_prose_template():
    """
    Texts under 20 chars must default to English and not trigger prose praise template.
    """
    lang = get_language_service()
    short_claim = "PM Modi died"
    detected = lang.detect_language(short_claim)
    assert detected["languageCode"] == "en"

    res = run_shared_verification_pipeline(
        text=short_claim,
        title=short_claim,
        input_type="text"
    )
    has_prose_template = any("Objective, attributable journalistic prose detected" in exp for exp in res["explanation"])
    assert not has_prose_template


def test_score_drop_on_3_altered_articles():
    """
    Requirement 7: Changing one number or one name in real articles must produce
    a clear score drop without special tuning to the examples.
    """
    # Article 1: IEA Renewable Energy (50% -> 2%)
    art1_orig = (
        "Global renewable energy capacity additions increased by nearly 50% in 2023, "
        "marking the fastest growth rate in two decades according to the International Energy Agency."
    )
    art1_alt = (
        "Global renewable energy capacity additions increased by nearly 2% in 2023, "
        "marking the fastest growth rate in two decades according to the International Energy Agency."
    )
    res1_orig = run_shared_verification_pipeline(text=art1_orig, title="IEA Renewables 50%", input_type="text")
    res1_alt = run_shared_verification_pipeline(text=art1_alt, title="IEA Renewables 2%", input_type="text")
    assert res1_orig["trustScore"] > res1_alt["trustScore"], f"Article 1 drop expected: {res1_orig['trustScore']} vs {res1_alt['trustScore']}"

    # Article 2: NVIDIA CEO (Jensen Huang -> Sundar Pichai)
    art2_orig = (
        "Jensen Huang continues as NVIDIA founder and Chief Executive Officer, leading the company "
        "in artificial intelligence chip development and data center computing infrastructure."
    )
    art2_alt = (
        "Sundar Pichai continues as NVIDIA founder and Chief Executive Officer, leading the company "
        "in artificial intelligence chip development and data center computing infrastructure."
    )
    res2_orig = run_shared_verification_pipeline(text=art2_orig, title="Jensen Huang NVIDIA", input_type="text")
    res2_alt = run_shared_verification_pipeline(text=art2_alt, title="Sundar Pichai NVIDIA", input_type="text")
    assert res2_orig["trustScore"] > res2_alt["trustScore"], f"Article 2 drop expected: {res2_orig['trustScore']} vs {res2_alt['trustScore']}"

    # Article 3: NDTV / RBI Policy (6.5% -> 1.2%)
    art3_orig = (
        "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent following the conclusion "
        "of its Monetary Policy Committee meeting."
    )
    art3_alt = (
        "The Reserve Bank of India kept the repo rate unchanged at 1.2 percent following the conclusion "
        "of its Monetary Policy Committee meeting."
    )
    res3_orig = run_shared_verification_pipeline(text=art3_orig, source_hint="ndtv.com", title="RBI repo 6.5%", input_type="text")
    res3_alt = run_shared_verification_pipeline(text=art3_alt, source_hint="ndtv.com", title="RBI repo 1.2%", input_type="text")
    assert res3_orig["trustScore"] > res3_alt["trustScore"], f"Article 3 drop expected: {res3_orig['trustScore']} vs {res3_alt['trustScore']}"
