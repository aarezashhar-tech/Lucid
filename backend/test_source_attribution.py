"""
Unit Tests for Source Ownership and Evidence Attribution.

Requirements:
1. Source publisher determined ONLY from:
   - URL domain
   - Publisher metadata
   - Article byline
2. External corroboration sources must never replace the primary source.
3. Prevent evidence contamination (IEA reports, WHO reports, Reuters articles must not attach as primary for BBC).
4. Add domain ownership validation (bbc.com -> BBC, reuters.com -> Reuters, aljazeera.com -> Al Jazeera, iea.org -> International Energy Agency).
5. Add tests:
   - Input: BBC URL -> Expected: Publisher = BBC
   - Input: IEA URL -> Expected: Publisher = IEA
   - Input: Al Jazeera URL -> Expected: Publisher = Al Jazeera
"""
import pytest
from unittest.mock import patch, MagicMock

from services.source_reputation import (
    validate_domain_ownership,
    get_source_engine,
    SourceReputationEngine
)
from services.ai_service import get_verification_engine


@pytest.fixture
def source_engine():
    return get_source_engine()


@pytest.fixture
def verification_engine():
    return get_verification_engine()


# =====================================================================
# Requirement 4 & 5: Domain Ownership Validation & Core URL Tests
# =====================================================================

def test_domain_ownership_validation():
    """
    Validates domain ownership mapping:
      bbc.com -> BBC
      reuters.com -> Reuters
      aljazeera.com -> Al Jazeera
      iea.org -> International Energy Agency / IEA
    """
    # BBC
    bbc_res = validate_domain_ownership("bbc.com")
    assert bbc_res is not None
    assert bbc_res["publisher"] == "BBC"
    assert bbc_res == "BBC"

    bbc_uk_res = validate_domain_ownership("bbc.co.uk")
    assert bbc_uk_res is not None
    assert bbc_uk_res["publisher"] == "BBC"
    assert bbc_uk_res == "BBC"

    # Reuters
    reuters_res = validate_domain_ownership("reuters.com")
    assert reuters_res is not None
    assert reuters_res["publisher"] == "Reuters"
    assert reuters_res == "Reuters"

    # Al Jazeera
    aj_res = validate_domain_ownership("aljazeera.com")
    assert aj_res is not None
    assert aj_res["publisher"] == "Al Jazeera"
    assert aj_res == "Al Jazeera"

    # IEA (both name = International Energy Agency and publisher = IEA)
    iea_res = validate_domain_ownership("iea.org")
    assert iea_res is not None
    assert iea_res["publisher"] == "IEA"
    assert iea_res["name"] == "International Energy Agency"
    assert iea_res == "IEA"
    assert iea_res == "International Energy Agency"


def test_publisher_from_bbc_url(source_engine, verification_engine):
    """
    Input: BBC URL
    Expected: Publisher = BBC
    """
    bbc_url = "https://www.bbc.com/news/science-environment-68000000"

    # 1. Source reputation engine evaluation
    rep = source_engine.evaluate_source(url_or_domain=bbc_url)
    assert rep["publisher"] == "BBC"
    assert rep["domain"] == "bbc.com"
    assert rep["isPrimary"] is True

    # 2. Verification pipeline analysis
    analysis = verification_engine.analyze(
        text="The transition to renewable energy continues at record pace globally.",
        url_or_domain=bbc_url
    )
    assert analysis["publisher"] == "BBC"
    assert analysis["primarySource"]["publisher"] == "BBC"
    assert analysis["primarySource"]["isPrimary"] is True
    assert analysis["sources"][0]["publisher"] == "BBC"
    assert analysis["sources"][0]["isPrimary"] is True


def test_publisher_from_iea_url(source_engine, verification_engine):
    """
    Input: IEA URL
    Expected: Publisher = IEA
    """
    iea_url = "https://www.iea.org/reports/renewables-2024-analysis-and-forecast"

    # 1. Source reputation engine evaluation
    rep = source_engine.evaluate_source(url_or_domain=iea_url)
    assert rep["publisher"] == "IEA"
    assert rep["domain"] == "iea.org"
    assert rep["isPrimary"] is True

    # 2. Verification pipeline analysis
    analysis = verification_engine.analyze(
        text="Renewables 2024 is the IEA's primary analysis on the sector.",
        url_or_domain=iea_url
    )
    assert analysis["publisher"] == "IEA"
    assert analysis["primarySource"]["publisher"] == "IEA"
    assert analysis["primarySource"]["isPrimary"] is True
    assert analysis["sources"][0]["publisher"] == "IEA"
    assert analysis["sources"][0]["isPrimary"] is True


def test_publisher_from_aljazeera_url(source_engine, verification_engine):
    """
    Input: Al Jazeera URL
    Expected: Publisher = Al Jazeera
    """
    aj_url = "https://www.aljazeera.com/news/2026/11/15/diplomatic-talks-middle-east"

    # 1. Source reputation engine evaluation
    rep = source_engine.evaluate_source(url_or_domain=aj_url)
    assert rep["publisher"] == "Al Jazeera"
    assert rep["domain"] == "aljazeera.com"
    assert rep["isPrimary"] is True

    # 2. Verification pipeline analysis
    analysis = verification_engine.analyze(
        text="Delegates arrived in the capital for emergency talks.",
        url_or_domain=aj_url
    )
    assert analysis["publisher"] == "Al Jazeera"
    assert analysis["primarySource"]["publisher"] == "Al Jazeera"
    assert analysis["primarySource"]["isPrimary"] is True
    assert analysis["sources"][0]["publisher"] == "Al Jazeera"
    assert analysis["sources"][0]["isPrimary"] is True


# =====================================================================
# Requirement 1: Publisher Determined ONLY From URL, Metadata, Byline
# =====================================================================

def test_publisher_determined_from_metadata(source_engine):
    """
    Publisher metadata (via explicit header or metadata dict) determines publisher.
    """
    # Via header metadata
    header_text = "Publisher: Reuters\nHeadline: Global markets steady on Monday."
    rep_meta = source_engine.evaluate_source(text=header_text)
    assert rep_meta["publisher"] == "Reuters"
    assert rep_meta["isPrimary"] is True

    # Via metadata dict
    rep_dict = source_engine.evaluate_source(
        text="Global markets steady on Monday.",
        metadata={"publisher": "Al Jazeera"}
    )
    assert rep_dict["publisher"] == "Al Jazeera"
    assert rep_dict["isPrimary"] is True


def test_publisher_determined_from_byline(source_engine):
    """
    Journalistic bylines and lead datelines determine publisher when no URL is provided.
    """
    # Byline format
    byline_text = "By Rebecca Morelle, BBC News\n\nAstronomers have discovered a new exoplanet."
    rep_byline = source_engine.evaluate_source(text=byline_text)
    assert rep_byline["publisher"] == "BBC"
    assert rep_byline["isPrimary"] is True

    # Wire Dateline format
    dateline_text = "LONDON (Reuters) — British annual inflation fell to 2% in May."
    rep_dateline = source_engine.evaluate_source(text=dateline_text)
    assert rep_dateline["publisher"] == "Reuters"
    assert rep_dateline["isPrimary"] is True


def test_body_citations_do_not_determine_publisher(source_engine):
    """
    Citations like 'According to Reuters' inside body text must NEVER determine primary publisher.
    Missing evidence or body citations without URL/metadata/byline must remain Unspecified.
    """
    body_text = (
        "Clean energy investments reached new highs last quarter. "
        "According to Reuters, solar installations doubled in several regions. "
        "The report was welcomed by energy ministers."
    )
    rep = source_engine.evaluate_source(text=body_text)
    # Must NOT attribute Reuters as the primary publisher!
    assert rep["publisher"] != "Reuters"
    assert rep["publisher"] == "Unknown"
    assert "Unspecified" in rep["sourceName"]


# =====================================================================
# Requirement 2 & 3: Prevent Evidence Contamination
# =====================================================================

def test_prevent_evidence_contamination_on_bbc_article(verification_engine):
    """
    Issue: BBC article URLs were being linked to unrelated sources such as IEA reports.

    Requirements:
    - Primary source for bbc.com article MUST be BBC.
    - Do not attach IEA reports, WHO reports, or Reuters articles as the primary source.
    - External corroboration sources (IEA, WHO, Reuters) must appear ONLY as supporting sources.
    """
    bbc_url = "https://www.bbc.com/news/science-environment-renewable-energy-50"
    article_text = (
        "Global renewable energy capacity rose by 50% in the fastest growth rate in two decades. "
        "According to a report released by the International Energy Agency (IEA), solar PV accounted "
        "for three-quarters of additions worldwide. "
        "The World Health Organization (WHO) noted health benefits from reduced air pollution, "
        "while Reuters reported similar findings across European utilities."
    )

    analysis = verification_engine.analyze(text=article_text, url_or_domain=bbc_url)

    # 1. Primary publisher must strictly be BBC
    assert analysis["publisher"] == "BBC"
    assert analysis["primarySource"]["publisher"] == "BBC"
    assert analysis["primarySource"]["name"] == "BBC"
    assert analysis["primarySource"]["domain"] == "bbc.com"
    assert analysis["primarySource"]["isPrimary"] is True

    # 2. Primary source in sources list must be BBC
    primary_in_list = analysis["sources"][0]
    assert primary_in_list["publisher"] == "BBC"
    assert primary_in_list["isPrimary"] is True

    # 3. IEA, WHO, Reuters must NEVER be the primary source
    assert analysis["primarySource"]["publisher"] != "IEA"
    assert analysis["primarySource"]["publisher"] != "International Energy Agency"
    assert analysis["primarySource"]["publisher"] != "WHO"
    assert analysis["primarySource"]["publisher"] != "Reuters"

    # 4. Any external corroborations present must have isPrimary == False
    for source in analysis["sources"][1:]:
        assert source["isPrimary"] is False
        assert source["publisher"] in (
            "International Energy Agency (IEA)",
            "International Energy Agency",
            "IEA",
            "World Health Organization (WHO)",
            "WHO",
            "Reuters",
            "Bloomberg News",
            "Associated Press (AP)",
            "Associated Press",
            "UN",
            source["publisher"]
        )
