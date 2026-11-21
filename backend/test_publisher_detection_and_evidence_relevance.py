"""
Unit Tests for:
1. Publisher metadata detection from copied article text submissions:
   - BBC patterns (BBC, BBC News, bbc.com, bbc.co.uk, BBC topic links)
   - Al Jazeera patterns (Al Jazeera, aljazeera.com)
   - Reuters patterns (Reuters, (Reuters))
   - Confidence > 90% -> Possible Publisher: <Publisher>, Confidence: High
2. Evidence relevance filtering:
   - Military clashes, wars, drone attacks, killings, local incident reports
   - Valid evidence: BBC, Reuters, AP, Al Jazeera, UN
   - Invalid evidence: WHO, Nature Journal, The Lancet must NOT be attached
   - Assign NO SUPPORTING EVIDENCE FOUND when no corroboration exists.
"""
import pytest
from services.source_reputation import get_source_engine
from services.evidence_verifier import (
    get_evidence_verifier,
    is_conflict_or_military_event,
    is_health_or_academic_source,
    is_evidence_topic_relevant
)
from services.ai_service import get_verification_engine


@pytest.fixture
def source_engine():
    return get_source_engine()


@pytest.fixture
def evidence_verifier():
    return get_evidence_verifier()


@pytest.fixture
def verification_engine():
    return get_verification_engine()


# =========================================================================
# Part 1: Publisher Detection from Copied Article Text Submissions
# =========================================================================

def test_detect_publisher_bbc_news(source_engine):
    """
    BBC article pattern: BBC News
    Expected: Possible Publisher: BBC, Confidence: High
    """
    text = (
        "BBC News - West Bank\n\n"
        "Israeli forces have shot dead a 14-year-old Palestinian boy in the occupied West Bank, "
        "Palestinian health officials reported on Sunday."
    )
    rep = source_engine.evaluate_source(text=text)
    assert rep["publisher"] == "BBC"
    assert "Possible Publisher: BBC" in rep["sourceName"]
    assert rep["publisherConfidence"] == "High"
    assert rep["confidence"] > 90
    assert rep["domain"] == "bbc.com"


def test_detect_publisher_bbc_standalone(source_engine):
    """
    BBC article pattern: BBC standalone header / marker
    Expected: Possible Publisher: BBC, Confidence: High
    """
    text = (
        "BBC - Middle East\n\n"
        "Israeli forces have shot dead a Palestinian boy during clashes in the northern West Bank."
    )
    rep = source_engine.evaluate_source(text=text)
    assert rep["publisher"] == "BBC"
    assert "Possible Publisher: BBC" in rep["sourceName"]
    assert rep["publisherConfidence"] == "High"


def test_detect_publisher_bbc_domain_and_topic_links(source_engine):
    """
    BBC article patterns: bbc.com, bbc.co.uk, BBC topic links
    Expected: Possible Publisher: BBC, Confidence: High
    """
    # bbc.com link
    text_com = (
        "Israeli forces have shot dead a 14-year-old Palestinian boy.\n"
        "Read the full report at bbc.com/news/world-middle-east-68000000"
    )
    rep_com = source_engine.evaluate_source(text=text_com)
    assert rep_com["publisher"] == "BBC"
    assert "Possible Publisher: BBC" in rep_com["sourceName"]
    assert rep_com["publisherConfidence"] == "High"

    # bbc.co.uk topic link
    text_co_uk = (
        "Renewable energy investments expanded significantly across Europe.\n"
        "Source link: https://www.bbc.co.uk/news/science-environment-12345"
    )
    rep_co_uk = source_engine.evaluate_source(text=text_co_uk)
    assert rep_co_uk["publisher"] == "BBC"
    assert "Possible Publisher: BBC" in rep_co_uk["sourceName"]
    assert rep_co_uk["publisherConfidence"] == "High"


def test_detect_publisher_aljazeera(source_engine):
    """
    Al Jazeera article patterns: Al Jazeera, aljazeera.com
    Expected: Possible Publisher: Al Jazeera, Confidence: High
    """
    text_aj = (
        "Al Jazeera - Gaza conflict updates.\n\n"
        "Israeli military operations continued across central Gaza on Saturday morning."
    )
    rep = source_engine.evaluate_source(text=text_aj)
    assert rep["publisher"] == "Al Jazeera"
    assert "Possible Publisher: Al Jazeera" in rep["sourceName"]
    assert rep["publisherConfidence"] == "High"
    assert rep["confidence"] > 90

    text_link = (
        "Diplomatic delegations arrived in Doha for truce negotiations.\n"
        "Full dispatch: https://www.aljazeera.com/news/2026/11/15/doha-talks"
    )
    rep_link = source_engine.evaluate_source(text=text_link)
    assert rep_link["publisher"] == "Al Jazeera"
    assert "Possible Publisher: Al Jazeera" in rep_link["sourceName"]
    assert rep_link["publisherConfidence"] == "High"


def test_detect_publisher_reuters_dateline(source_engine):
    """
    Reuters article patterns: (Reuters), Reuters
    Expected: Possible Publisher: Reuters, Confidence: High
    """
    # (Reuters) dateline
    text_dateline = (
        "(Reuters) - Saudi Arabia intercepted three ballistic missiles near the southern border."
    )
    rep = source_engine.evaluate_source(text=text_dateline)
    assert rep["publisher"] == "Reuters"
    assert "Possible Publisher: Reuters" in rep["sourceName"]
    assert rep["publisherConfidence"] == "High"
    assert rep["confidence"] > 90

    # Reuters header
    text_header = (
        "Reuters - Saudi air defenses destroyed incoming drone targets over the weekend."
    )
    rep_hdr = source_engine.evaluate_source(text=text_header)
    assert rep_hdr["publisher"] == "Reuters"
    assert "Possible Publisher: Reuters" in rep_hdr["sourceName"]
    assert rep_hdr["publisherConfidence"] == "High"

    # reuters.com link
    text_link = (
        "Global financial ministers concluded annual summit in Basel.\n"
        "Read on reuters.com/markets/europe/basel-meeting-2026"
    )
    rep_link = source_engine.evaluate_source(text=text_link)
    assert rep_link["publisher"] == "Reuters"
    assert "Possible Publisher: Reuters" in rep_link["sourceName"]
    assert rep_link["publisherConfidence"] == "High"


def test_body_citation_remains_unspecified(source_engine):
    """
    Passing body-text citations ('According to Reuters') without copied article metadata
    must remain Unspecified / Self-Published Content.
    """
    body_text = (
        "Clean energy investments reached new highs last quarter. "
        "According to Reuters, solar installations doubled in several regions. "
        "The report was welcomed by energy ministers."
    )
    rep = source_engine.evaluate_source(text=body_text)
    assert rep["publisher"] == "Unknown"
    assert "Unspecified / Self-Published Content" in rep["sourceName"]
    assert rep["tier"] == "Unverified"


# =========================================================================
# Part 2: Evidence Relevance Filtering
# =========================================================================

def test_conflict_and_military_event_detection():
    """
    Tests identification of military clashes, wars, drone attacks, killings, local incident reports.
    """
    assert is_conflict_or_military_event("Israeli forces killed a Palestinian boy") is True
    assert is_conflict_or_military_event("Military clashes broke out near the northern border") is True
    assert is_conflict_or_military_event("A drone attack struck an oil depot yesterday") is True
    assert is_conflict_or_military_event("Soldiers launched an offensive near the frontline") is True
    assert is_conflict_or_military_event("Police reported three people fatally shot in local incident") is True

    # Non-conflict topics
    assert is_conflict_or_military_event("Renewable solar energy capacity grew 50% globally") is False
    assert is_conflict_or_military_event("Central bank decided to lower interest rates by 25 basis points") is False


def test_health_and_academic_sources_identified():
    """
    Tests identification of WHO, Nature, Lancet.
    """
    assert is_health_or_academic_source("World Health Organization (WHO)") is True
    assert is_health_or_academic_source("WHO") is True
    assert is_health_or_academic_source("The Lancet") is True
    assert is_health_or_academic_source("Nature Journal") is True
    assert is_health_or_academic_source("Nature") is True

    # General news and wire publishers are not medical journals
    assert is_health_or_academic_source("BBC") is False
    assert is_health_or_academic_source("Reuters") is False
    assert is_health_or_academic_source("Associated Press (AP)") is False
    assert is_health_or_academic_source("Al Jazeera") is False
    assert is_health_or_academic_source("UN") is False


def test_topic_relevance_rejects_who_nature_lancet_for_conflict():
    """
    Claim: 'Israeli forces killed a Palestinian boy'
    Valid Evidence: BBC, Reuters, AP, Al Jazeera, UN
    Invalid Evidence: WHO, Nature Journal, The Lancet
    """
    claim = "Israeli forces killed a Palestinian boy"

    # Invalid sources: MUST be rejected
    assert is_evidence_topic_relevant(claim, "World Health Organization (WHO)") is False
    assert is_evidence_topic_relevant(claim, "WHO") is False
    assert is_evidence_topic_relevant(claim, "Nature Journal") is False
    assert is_evidence_topic_relevant(claim, "Nature") is False
    assert is_evidence_topic_relevant(claim, "The Lancet") is False

    # Valid sources: MUST be accepted
    assert is_evidence_topic_relevant(claim, "BBC") is True
    assert is_evidence_topic_relevant(claim, "Reuters") is True
    assert is_evidence_topic_relevant(claim, "Associated Press (AP)") is True
    assert is_evidence_topic_relevant(claim, "Al Jazeera") is True
    assert is_evidence_topic_relevant(claim, "UN") is True


def test_evidence_verification_claim_israeli_forces_killed_palestinian_boy(evidence_verifier):
    """
    Verifies that for 'Israeli forces killed a Palestinian boy',
    WHO, Nature Journal, and The Lancet are NEVER attached as supporting citations.
    Only valid conflict wire/human rights reporting is attached.
    """
    claim = {
        "id": "claim-1",
        "claim": "Israeli forces killed a Palestinian boy",
        "supportingSources": [],
        "contradictingSources": [],
        "supportingEvidenceCount": 0,
        "contradictingEvidenceCount": 0
    }
    source_context = {
        "sourceName": "BBC",
        "publisher": "BBC",
        "trustScore": 92,
        "isTrusted": True,
        "domain": "bbc.com"
    }

    result = evidence_verifier.verify_single_claim(
        claim=claim,
        source_context=source_context,
        full_article_text="Israeli forces have shot dead a 14-year-old Palestinian boy in the West Bank."
    )

    # 1. Invalid sources MUST NOT be present
    for src in result["supportingSources"]:
        src_lower = src.lower()
        assert "who" not in src_lower and "health" not in src_lower
        assert "nature" not in src_lower
        assert "lancet" not in src_lower

    # 2. Valid sources are allowed (BBC, Reuters, AP, Al Jazeera, UN)
    for src in result["supportingSources"]:
        assert any(
            valid in src
            for valid in ["BBC", "Reuters", "Associated Press", "AP", "Al Jazeera", "UN"]
        )


def test_no_supporting_evidence_found_assigned_when_no_corroboration(evidence_verifier):
    """
    When evidence does not meet topic relevance or semantic matching,
    system must assign: NO SUPPORTING EVIDENCE FOUND.
    """
    claim = {
        "id": "claim-1",
        "claim": "An unverified obscure local incident occurred in an undisclosed town.",
        "supportingSources": [],
        "contradictingSources": [],
        "supportingEvidenceCount": 0,
        "contradictingEvidenceCount": 0
    }
    source_context = {
        "sourceName": "Unspecified / Self-Published Content",
        "publisher": "Unknown",
        "trustScore": 50,
        "isTrusted": False,
        "domain": "direct-submission"
    }

    res = evidence_verifier.verify_single_claim(
        claim=claim,
        source_context=source_context,
        full_article_text="An unverified obscure local incident occurred in an undisclosed town."
    )

    assert len(res["supportingSources"]) == 0
    assert "NO SUPPORTING EVIDENCE FOUND" in res["analysisNotes"]

    all_res = evidence_verifier.verify_all_claims(
        claims=[res],
        source_context=source_context,
        article_text="An unverified obscure local incident occurred in an undisclosed town."
    )
    assert all_res["supportingEvidenceCount"] == 0
    assert all_res["supportingEvidenceStatus"] == "NO SUPPORTING EVIDENCE FOUND"
    assert "NO SUPPORTING EVIDENCE FOUND" in all_res["evidenceCalculation"]["explanation"]
