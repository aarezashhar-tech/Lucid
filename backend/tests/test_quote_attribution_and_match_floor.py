"""
Unit tests for Quote Speaker Attribution and Evidence Match Floor.

Covers:
1. Attributed quote-card text ("My family is in Al-Mawasi... - Ehab Abu Jazar") preserves
   speaker attribution in extracted claims.
2. First-person pronoun rewriting ('I' -> 'he', 'My' -> 'his', 'We' -> 'they').
3. Skipping pure philosophical/generic truisms containing no verifiable person, place, or event.
4. Weak 7% topical match with no entity overlap is NOT classified as SUPPORT.
5. Minimum semantic similarity threshold (>= 0.15 / 15%) and entity match floor for SUPPORT.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.claim_extractor import (
    claim_engine,
    extract_speaker_attribution,
    rewrite_first_person_quote,
    is_generic_philosophical_truism,
)
from services.evidence_ranker import EvidenceRanker


# ===========================================================================
# 1. Speaker Attribution & Pronoun Rewriting Tests
# ===========================================================================

def test_extract_speaker_attribution_dash():
    """Verify speaker attribution is extracted from trailing dash format."""
    text1 = "My family is in Al-Mawasi... - Ehab Abu Jazar"
    assert extract_speaker_attribution(text1) == "Ehab Abu Jazar"

    text2 = "My family is living in tents in Al-Mawasi - Ehab Abu Jazar, Palestine football coach"
    assert extract_speaker_attribution(text2) == "Ehab Abu Jazar"

    text3 = "We have nowhere else to go. — Ehab Abu Jazar"
    assert extract_speaker_attribution(text3) == "Ehab Abu Jazar"


def test_extract_speaker_attribution_multiline():
    """Verify speaker attribution is extracted from multi-line quote cards."""
    text = (
        '"My family is living in tents in Al-Mawasi"\n'
        '- Ehab Abu Jazar, Palestine football coach'
    )
    assert extract_speaker_attribution(text) == "Ehab Abu Jazar"


def test_rewrite_first_person_quote_pronouns():
    """Verify first-person pronouns ('My', 'I', 'We') are rewritten and attributed."""
    speaker = "Ehab Abu Jazar"

    s1 = "My family is in Al-Mawasi"
    r1 = rewrite_first_person_quote(s1, speaker)
    assert r1 == "Ehab Abu Jazar stated his family is in Al-Mawasi."

    s2 = "We have no food or water"
    r2 = rewrite_first_person_quote(s2, speaker)
    assert r2 == "Ehab Abu Jazar stated they have no food or water."

    s3 = "I was forced to leave our camp"
    r3 = rewrite_first_person_quote(s3, speaker)
    assert r3 == "Ehab Abu Jazar stated he was forced to leave their camp."


def test_claim_extractor_attributes_quote_card():
    """Verify full claim extractor preserves speaker attribution for quote-card text."""
    text = "My family is in Al-Mawasi... - Ehab Abu Jazar"
    claims = claim_engine.extract_claims(text)

    assert len(claims) >= 1
    extracted_text = claims[0]["claim"]
    assert "Ehab Abu Jazar" in extracted_text
    assert "stated his family is in Al-Mawasi" in extracted_text


def test_claim_extractor_multi_sentence_quote_card():
    """Verify multiple sentences in an attributed quote card all receive attribution."""
    text = (
        "My family is living in tents in Al-Mawasi. "
        "We have nowhere else to go. "
        "- Ehab Abu Jazar, Palestine football coach"
    )
    claims = claim_engine.extract_claims(text)

    assert len(claims) >= 1
    assert "Ehab Abu Jazar" in claims[0]["claim"]
    assert "his family is living in tents in Al-Mawasi" in claims[0]["claim"]
    if len(claims) > 1:
        assert "Ehab Abu Jazar" in claims[1]["claim"]
        assert "they have nowhere else to go" in claims[1]["claim"]


# ===========================================================================
# 2. Philosophical / Generic Truism Skipping Tests
# ===========================================================================

def test_is_generic_philosophical_truism():
    """Verify pure philosophical truisms without verifiable entities/places are identified."""
    assert is_generic_philosophical_truism("Football is more than just a game.") is True
    assert is_generic_philosophical_truism("Hope is what keeps humanity alive.") is True
    assert is_generic_philosophical_truism("Life is a journey of self discovery.") is True
    assert is_generic_philosophical_truism("Peace begins with a smile.") is True

    # Verifiable claims must NOT be flagged as truisms
    assert is_generic_philosophical_truism("My family is living in tents in Al-Mawasi.") is False
    assert is_generic_philosophical_truism("Ehab Abu Jazar stated his family is in Al-Mawasi.") is False
    assert is_generic_philosophical_truism("The Palestine football team qualified for the 2024 tournament.") is False
    assert is_generic_philosophical_truism("Earth is flat.") is False


def test_claim_extractor_skips_truisms_in_quote_card():
    """Verify claim extractor skips generic truisms and only extracts verifiable claims."""
    text = (
        "Football is more than just a game, it is hope. "
        "My family is living in tents in Al-Mawasi. "
        "- Ehab Abu Jazar"
    )
    claims = claim_engine.extract_claims(text)

    extracted_texts = [c["claim"] for c in claims]
    # Truism should be skipped
    assert not any("more than just a game" in t.lower() for t in extracted_texts)
    # Verifiable attributed claim must be present
    assert any("Ehab Abu Jazar" in t and "tents in Al-Mawasi" in t for t in extracted_texts)


# ===========================================================================
# 3. Minimum Semantic & Entity Match Floor Tests
# ===========================================================================

def test_weak_topical_match_not_classified_as_support():
    """
    CRITICAL TEST:
    Verify that a weak 7% topical match with no entity overlap is NOT classified as SUPPORT
    (downgraded to neutral/related).
    """
    ranker = EvidenceRanker()
    claim = "Ehab Abu Jazar stated his family is living in tents in Al-Mawasi."

    # Generic article with ~7% similarity and 0 entity overlap
    weak_candidate = {
        "articleTitle": "World Financial News & Market Summary",
        "evidenceSnippet": "Stock prices surged globally following positive retail growth statistics.",
        "url": "https://www.reuters.com/markets/stocks-summary-2024",
        "sourceTrustScore": 90,
    }

    # Force compute_similarity to return weak 7% (0.07)
    with patch.object(ranker.semantic_engine, "compute_similarity", return_value=0.07):
        result = ranker.rank_evidence_for_claim(claim, [weak_candidate], allow_gemini=False)

    # Must NOT be in supporting
    assert len(result["supporting"]) == 0, (
        f"Weak 7% match was incorrectly placed into supporting: {result['supporting']}"
    )

    # Must be categorized as neutral or related, never support
    all_evaluated = result["supporting"] + result["related"] + result["neutral"]
    assert len(all_evaluated) == 1
    assert all_evaluated[0]["stance"] in ("neutral", "related")
    assert all_evaluated[0]["stance"] != "support"


def test_gemini_support_downgraded_if_below_match_floor():
    """
    Verify that even if Gemini classifies an article as SUPPORT,
    if semantic similarity is < 0.15 or lacks entity overlap, it is downgraded.
    """
    ranker = EvidenceRanker()
    claim = "Ehab Abu Jazar stated his family is living in tents in Al-Mawasi."

    weak_candidate = {
        "articleTitle": "Middle East Regional Overview",
        "evidenceSnippet": "General commentary on broader Middle Eastern diplomacy and international summits.",
        "url": "https://www.bbc.com/news/world-middle-east-overview",
        "sourceTrustScore": 88,
    }

    # Mock Gemini returning 'support' on this candidate
    mock_gemini_res = {
        0: {"stance": "support", "confidence": 85, "reason": "General Middle East topic overlap."}
    }

    with patch.object(ranker.semantic_engine, "compute_similarity", return_value=0.08), \
         patch.object(ranker.gemini_classifier, "classify_candidates_for_claim", return_value=mock_gemini_res):

        result = ranker.rank_evidence_for_claim(claim, [weak_candidate], allow_gemini=True)

    # Match floor must override and reject SUPPORT
    assert len(result["supporting"]) == 0
    all_evaluated = result["supporting"] + result["related"] + result["neutral"]
    assert len(all_evaluated) == 1
    assert all_evaluated[0]["stance"] in ("neutral", "related")


def test_high_similarity_and_entity_match_allows_support():
    """Verify that an article with >= 15% similarity AND entity overlap CAN be classified as SUPPORT."""
    ranker = EvidenceRanker()
    claim = "Ehab Abu Jazar stated his family is living in tents in Al-Mawasi."

    matching_candidate = {
        "articleTitle": "Palestine Olympic football coach Ehab Abu Jazar speaks on family in Gaza",
        "evidenceSnippet": "Ehab Abu Jazar confirmed his family is living in tents in Al-Mawasi after displacement.",
        "url": "https://www.reuters.com/world/middle-east/palestine-coach-ehab-abu-jazar-family-2024",
        "sourceTrustScore": 95,
        "publicationDate": "2024-07-20",
    }

    with patch.object(ranker.semantic_engine, "compute_similarity", return_value=0.68):
        result = ranker.rank_evidence_for_claim(claim, [matching_candidate], allow_gemini=False)

    assert len(result["supporting"]) == 1
    assert result["supporting"][0]["stance"] == "support"
