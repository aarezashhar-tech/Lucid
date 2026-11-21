"""
Unit tests for Claim Extraction Quality and Content-Cleaning Stage
Verifies that navigation elements, author names, image captions, and UI text
are strictly ignored, and only verifiable factual claims are extracted.
"""
import pytest
from services.claim_extractor import get_claim_engine


@pytest.fixture
def claim_engine():
    return get_claim_engine()


def test_valid_factual_claim_accepted(claim_engine):
    """Verifies that clear, verifiable factual statements are accepted."""
    valid_claim = "Israeli forces have shot dead a 14-year-old Palestinian boy."
    assert claim_engine.is_valid_factual_claim(valid_claim) is True

    claims = claim_engine.extract_claims(valid_claim)
    assert len(claims) == 1
    assert "shot dead" in claims[0]["claim"].lower()


def test_invalid_author_names_ignored(claim_engine):
    """Author and reporter names must never be treated as claims."""
    assert claim_engine.is_valid_factual_claim("David Gritten") is False
    assert claim_engine.is_valid_factual_claim("By David Gritten") is False
    assert claim_engine.is_valid_factual_claim("David Gritten, BBC News") is False
    assert claim_engine.is_valid_factual_claim("Reported by Sarah Jones") is False


def test_invalid_ui_elements_ignored(claim_engine):
    """UI widgets, newsroom prompts, and action buttons must be ignored."""
    assert claim_engine.is_valid_factual_claim("Add as preferred on Google") is False
    assert claim_engine.is_valid_factual_claim("Add as preferred") is False
    assert claim_engine.is_valid_factual_claim("Share this story on WhatsApp") is False
    assert claim_engine.is_valid_factual_claim("Save article") is False
    assert claim_engine.is_valid_factual_claim("Follow us on Google News") is False
    assert claim_engine.is_valid_factual_claim("Subscribe to our newsletter") is False


def test_invalid_navigation_and_related_links_ignored(claim_engine):
    """Navigation text and related article links must be ignored."""
    assert claim_engine.is_valid_factual_claim("Related") is False
    assert claim_engine.is_valid_factual_claim("Related Topics") is False
    assert claim_engine.is_valid_factual_claim("Related Stories") is False
    assert claim_engine.is_valid_factual_claim("More on this story") is False
    assert claim_engine.is_valid_factual_claim("Read more") is False
    assert claim_engine.is_valid_factual_claim("Top Stories") is False


def test_invalid_category_and_location_labels_ignored(claim_engine):
    """Isolated category, topic, or location labels must be ignored."""
    assert claim_engine.is_valid_factual_claim("Israel") is False
    assert claim_engine.is_valid_factual_claim("West Bank") is False
    assert claim_engine.is_valid_factual_claim("Gaza") is False
    assert claim_engine.is_valid_factual_claim("Middle East") is False
    assert claim_engine.is_valid_factual_claim("World") is False
    assert claim_engine.is_valid_factual_claim("Politics") is False


def test_invalid_image_captions_and_ads_ignored(claim_engine):
    """Image captions, photo credits, and advertisements must be ignored."""
    assert claim_engine.is_valid_factual_claim("Image caption: An ambulance arriving at the hospital yesterday.") is False
    assert claim_engine.is_valid_factual_claim("Photo credit: Getty Images / AFP") is False
    assert claim_engine.is_valid_factual_claim("Advertisement: Sponsored financial content") is False


def test_content_cleaning_stage_with_noisy_article(claim_engine):
    """
    Tests an article with embedded author names, UI widgets, topic tags,
    and image captions. Only legitimate factual claims must be extracted.
    """
    raw_article = """
    David Gritten
    BBC News
    Add as preferred on Google

    Israeli forces have shot dead a 14-year-old Palestinian boy during a raid in the occupied West Bank.

    Related Topics
    Israel
    West Bank
    Share this story

    Medics said the teenager died after suffering a bullet wound to the chest.

    Image caption: An ambulance arriving at the hospital.
    Photo by Reuters.

    The military stated that soldiers had opened fire after being targeted by explosive devices.

    Advertisement
    All rights reserved.
    """

    cleaned = claim_engine.clean_article_content(raw_article)

    # Content-cleaning assertions
    assert "David Gritten" not in cleaned
    assert "Add as preferred on Google" not in cleaned
    assert "Related Topics" not in cleaned
    assert "Advertisement" not in cleaned

    # Claim extraction assertions
    claims = claim_engine.extract_claims(raw_article)
    extracted_sentences = [c["claim"] for c in claims]

    # Verify invalid items are NOT extracted
    for invalid in ["David Gritten", "Add as preferred on Google", "Related", "Israel", "West Bank", "Advertisement"]:
        assert not any(invalid.lower() == s.lower().strip('.') for s in extracted_sentences), f"Found invalid claim: {invalid}"

    # Verify legitimate factual statements ARE extracted
    assert any("shot dead a 14-year-old Palestinian boy" in s for s in extracted_sentences)
    assert any("died after suffering a bullet wound" in s for s in extracted_sentences)
    assert any("soldiers had opened fire" in s for s in extracted_sentences)


def test_integration_with_ai_service():
    """Tests that the full verification engine ignores UI noise when extracting claims."""
    from services.ai_service import get_verification_engine

    engine = get_verification_engine()
    noisy_input = (
        "David Gritten\n"
        "Add as preferred on Google\n"
        "Israeli forces have shot dead a 14-year-old Palestinian boy in the West Bank.\n"
        "Related\n"
        "Israel\n"
        "West Bank\n"
        "Local officials confirmed that thirty-three people were safely evacuated from the surrounding area.\n"
    )

    result = engine.analyze(noisy_input, source_hint="bbc.com", url_or_domain="https://www.bbc.com/news/article-123")
    extracted_claims = [c["claim"] for c in result.get("claims", [])]

    # Ensure no noisy UI elements or author names are treated as claims
    for noise in ["David Gritten", "Add as preferred on Google", "Related", "Israel", "West Bank"]:
        assert not any(noise.lower() == c.lower().strip('.') for c in extracted_claims), f"Found noise in claims: {noise}"

    # Ensure real factual claims are extracted
    assert len(extracted_claims) >= 1
    assert any("shot dead" in c.lower() for c in extracted_claims)
