"""
Unit and Integration Tests for Wikidata / Wikipedia Living-Person Relative & Associate Check
Validates:
1. "CM Yogi of UP mother died" does NOT trigger a Wikidata living-person contradiction for Yogi Adityanath.
2. "PM Modi died" still correctly triggers the Wikidata living-person contradiction for Narendra Modi.
3. Relative and associate keywords (mother, father, wife, son, brother, aide, friend, etc.)
   and possessive forms ('s mother, of ... mother) are correctly identified.
4. RAG retrieval layer does not inject a synthetic living-person contradiction when a relative is mentioned.
"""
import pytest
from services.wikipedia_service import (
    get_wikipedia_service,
    claim_refers_to_relative_or_associate,
    RELATIVE_ASSOCIATE_REGEX
)
from services.rag_retriever import get_rag_retriever


def test_claim_refers_to_relative_or_associate_detection():
    """
    Tests detection of relative and associate references across various phrasing styles.
    """
    # Relative terms and possessive forms
    assert claim_refers_to_relative_or_associate("CM Yogi of UP mother died") is True
    assert claim_refers_to_relative_or_associate("cm yogi of up mother died?") is True
    assert claim_refers_to_relative_or_associate("Yogi Adityanath's mother passed away") is True
    assert claim_refers_to_relative_or_associate("Mother of Chief Minister Yogi died") is True
    assert claim_refers_to_relative_or_associate("PM Modi's mother passed away") is True
    assert claim_refers_to_relative_or_associate("Minister's father hospitalized") is True
    assert claim_refers_to_relative_or_associate("President's wife attended gala") is True
    assert claim_refers_to_relative_or_associate("Governor's husband died") is True
    assert claim_refers_to_relative_or_associate("Joe Biden's son Beau Biden died") is True
    assert claim_refers_to_relative_or_associate("Ambassador's daughter missing") is True
    assert claim_refers_to_relative_or_associate("Prime Minister's brother visited temple") is True
    assert claim_refers_to_relative_or_associate("Senator's sister passed away") is True
    assert claim_refers_to_relative_or_associate("King's uncle passed away") is True
    assert claim_refers_to_relative_or_associate("Chancellor's aunt died") is True
    assert claim_refers_to_relative_or_associate("Chief Minister's close friend died in accident") is True
    assert claim_refers_to_relative_or_associate("Senior political aide of CM passed away") is True
    assert claim_refers_to_relative_or_associate("Relative of UP CM passed away") is True

    # Non-relative claims should return False
    assert claim_refers_to_relative_or_associate("PM Modi died") is False
    assert claim_refers_to_relative_or_associate("Yogi Adityanath died") is False
    assert claim_refers_to_relative_or_associate("King Charles III passed away") is False
    assert claim_refers_to_relative_or_associate("Sundar Pichai continues as CEO") is False

    # Intrinsic name edge cases (e.g. Mother Teresa)
    assert claim_refers_to_relative_or_associate("Mother Teresa died", entity_name="Mother Teresa") is False
    assert claim_refers_to_relative_or_associate("Mother Teresa's aide died", entity_name="Mother Teresa") is True


def test_yogi_mother_does_not_trigger_wikidata_contradiction():
    """
    Requirement 1 & 2:
    "CM Yogi of UP mother died" must NOT trigger a Wikidata living-person contradiction
    for Yogi Adityanath.
    """
    wiki = get_wikipedia_service()

    # Test "CM Yogi of UP mother died"
    res1 = wiki.check_biographical_or_encyclopedic_claim("CM Yogi of UP mother died")
    assert res1 is None or not res1.get("isContradicted"), (
        f"Expected no contradiction for 'CM Yogi of UP mother died', got: {res1}"
    )

    # Test "cm yogi of up mother died?" (with question mark and lowercase)
    res2 = wiki.check_biographical_or_encyclopedic_claim("cm yogi of up mother died?")
    assert res2 is None or not res2.get("isContradicted"), (
        f"Expected no contradiction for 'cm yogi of up mother died?', got: {res2}"
    )

    # Test Wikipedia evidence retrieval items do not include a living-person contradiction
    ev_items = wiki.get_wikipedia_evidence_for_claim("CM Yogi of UP mother died")
    for item in ev_items:
        snippet = str(item.get("evidenceSnippet", "")).lower()
        stance = item.get("stance")
        assert stance != "contradict", (
            f"Wikipedia evidence item should not have stance 'contradict', got: {item}"
        )
        assert "no date of death recorded (living person)" not in snippet, (
            f"Synthetic living-person contradiction found in snippet: {snippet}"
        )


def test_pm_modi_died_still_triggers_wikidata_contradiction():
    """
    Requirement 2:
    "PM Modi died" still correctly triggers the living-person contradiction for Narendra Modi.
    """
    wiki = get_wikipedia_service()
    res = wiki.check_biographical_or_encyclopedic_claim("PM Modi died")

    assert res is not None, "Expected contradiction result for 'PM Modi died', got None"
    assert res.get("isContradicted") is True
    assert res.get("verdict") == "FALSE"
    assert "no date of death recorded" in res.get("reason", "").lower()
    assert res.get("entity") in ("Narendra Modi", "PM Modi")

    ev_item = res.get("evidenceItem", {})
    assert ev_item.get("stance") == "contradict"
    snippet = ev_item.get("evidenceSnippet", "")
    assert "[Derived check]" in snippet
    assert "no date of death recorded (living person)" in snippet


def test_rag_retrieval_relative_death_check_filtering():
    """
    RAG retriever must not return a living person contradiction for Yogi's mother,
    while correctly returning it for PM Modi.
    """
    retriever = get_rag_retriever()

    # Yogi mother claim: no living person contradiction in RAG results
    yogi_results = retriever.retrieve_evidence_for_claim("CM Yogi of UP mother died")
    yogi_contra = [
        r for r in yogi_results
        if r.get("stance") == "contradict" or "living person" in str(r.get("evidenceSnippet", "")).lower()
    ]
    assert len(yogi_contra) == 0, f"Expected 0 contradicting living-person items, got: {yogi_contra}"

    # PM Modi death claim: living person contradiction MUST be present
    modi_results = retriever.retrieve_evidence_for_claim("PM Modi died")
    modi_contra = [
        r for r in modi_results
        if r.get("stance") == "contradict" and "no date of death recorded" in str(r.get("evidenceSnippet", "")).lower()
    ]
    assert len(modi_contra) >= 1, f"Expected >=1 contradicting living-person item for PM Modi, got: {modi_contra}"
