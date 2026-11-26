"""
Unit Tests for Temporal Stance Understanding and Event/Timeframe Disambiguation.

Covers:
1. Reversal/Pullback snippet on a subsequent day confirming the claimed event occurred:
   Claim: "after Jensen Huang returns as CEO, Nvidia stock goes up"
   Snippet: "Nvidia's stock fell 5% on Tuesday, reversing from a strong 31% rally on Monday, after the board reinstated Jensen Huang as CEO"
   -> Must classify as SUPPORT (not CONTRADICT).
2. Direct contradiction without confirmation of the claimed event:
   Claim: "after Jensen Huang returns as CEO, Nvidia stock goes up"
   Snippet: "Nvidia stock plunged immediately after reinstatement with no rally"
   -> Must classify as CONTRADICT.
3. Explicit Event & Timeframe Rule in Gemini stance classifier prompt.
"""

import os
import sys
import pytest

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.evidence_ranker import EvidenceRanker
from services.gemini_stance_classifier import GeminiStanceClassifier


def test_temporal_reversal_classified_as_support():
    """
    When an evidence snippet confirms that the claimed event occurred (a strong rally on Monday
    after Jensen Huang was reinstated as CEO) but also reports a subsequent reversal/pullback on
    a later day (fell 5% on Tuesday), it must be classified as SUPPORT, not CONTRADICT.
    """
    ranker = EvidenceRanker()
    claim = "after Jensen Huang returns as CEO, Nvidia stock goes up"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia's stock fell 5% on Tuesday, reversing from a strong 31% rally on Monday, after the board reinstated Jensen Huang as CEO",
        "url": "https://finance.yahoo.com/news/nvidia-stock-rally-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    # 1. Direct heuristic stance classification
    stance, conf = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support', got '{stance}'"
    assert conf >= 70, f"Expected confidence >= 70, got {conf}"

    # 2. Evidence ranker output (heuristic fallback mode)
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=False)
    assert ranked["supportingCount"] == 1, f"Expected 1 supporting source, got {ranked['supportingCount']}"
    assert ranked["contradictingCount"] == 0, f"Expected 0 contradicting sources, got {ranked['contradictingCount']}"
    assert ranked["supporting"][0]["stance"] == "support"


def test_direct_contradiction_plunged_no_rally_classified_as_contradict():
    """
    When evidence directly disputes that the claimed event happened at all
    (e.g., stock plunged immediately with no rally), it must be classified as CONTRADICT.
    """
    ranker = EvidenceRanker()
    claim = "after Jensen Huang returns as CEO, Nvidia stock goes up"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock plunged immediately after reinstatement with no rally",
        "url": "https://finance.yahoo.com/news/nvidia-stock-plunge-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    # 1. Direct heuristic stance classification
    stance, conf = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "contradict", f"Expected 'contradict', got '{stance}'"
    assert conf >= 80, f"Expected confidence >= 80, got {conf}"

    # 2. Evidence ranker output (heuristic fallback mode)
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=False)
    assert ranked["contradictingCount"] == 1, f"Expected 1 contradicting source, got {ranked['contradictingCount']}"
    assert ranked["supportingCount"] == 0, f"Expected 0 supporting sources, got {ranked['supportingCount']}"
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_gemini_prompt_contains_event_and_timeframe_rule():
    """
    Verifies that the prompt in GeminiStanceClassifier contains the explicit
    Event & Timeframe Rule to prevent false CONTRADICT classifications.
    """
    classifier = GeminiStanceClassifier()
    # Check that rule 11 is defined in the source/prompt
    import inspect
    source = inspect.getsource(classifier.classify_candidates_for_claim)
    assert "EVENT & TIMEFRAME RULE" in source
    assert "reversing from a strong 31% rally on Monday after he returned" in source
    assert "SUPPORT" in source
    assert "Only classify as CONTRADICT if the evidence directly disputes that the claimed event happened at all" in source


def test_pre_event_rally_with_post_event_plunge_classified_as_contradict():
    """
    When an evidence snippet states the stock rallied BEFORE the return, but plunged immediately
    after reinstatement with no rally, the pre-event rally must not suppress the direct contradiction.
    """
    ranker = EvidenceRanker()
    claim = "after Jensen Huang returns as CEO, Nvidia stock goes up"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock rallied before Jensen Huang returned as CEO, but plunged immediately after reinstatement with no rally",
        "url": "https://finance.yahoo.com/news/nvidia-pre-rally-post-plunge-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    stance, conf = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "contradict", f"Expected 'contradict', got '{stance}'"
    assert conf >= 80, f"Expected confidence >= 80, got {conf}"


def test_speculative_movement_wording_returns_neutral():
    """
    Speculative wording ('may rally') must return neutral and not support for a past/occurred event claim.
    """
    ranker = EvidenceRanker()
    claim = "Nvidia stock rallied after Jensen Huang returned as CEO"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock may rally after Jensen Huang returns as CEO, analysts predict",
        "url": "https://finance.yahoo.com/news/nvidia-may-rally-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "neutral", f"Expected 'neutral', got '{stance}'"


def test_officeholder_subject_validation_preserved_with_descriptive_market_term():
    """
    Claims asserting leadership with descriptive market terms ('Alice Smith is CEO of Acme market research')
    must preserve officeholder subject validation and not be bypassed to accept a different officeholder.
    """
    ranker = EvidenceRanker()
    claim = "Alice Smith is CEO of Acme market research"
    evidence_item = {
        "articleTitle": "Business Wire",
        "evidenceSnippet": "Bob Jones is CEO of Acme market research, following the annual board meeting",
        "url": "https://businesswire.com/news/acme-leadership-2024.html",
        "publisher": "Business Wire",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Did not expect 'support' when subject officeholder does not match, got '{stance}'"


def test_conflicting_percentage_preserves_contradiction():
    """
    When evidence reports a conflicting percentage metric (2% vs 10%), it must not be overridden to support.
    """
    ranker = EvidenceRanker()
    claim = "Nvidia stock rose 10% after Jensen Huang returned as CEO"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock rose 2% after Jensen Huang returned as CEO",
        "url": "https://finance.yahoo.com/news/nvidia-gain-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Did not expect 'support' with conflicting percentage, got '{stance}'"


def test_different_returnee_in_movement_claim_does_not_support():
    """
    When claim attributes the return to Jensen Huang, evidence naming a different person
    ('Bob Jones returned as CEO') must not be classified as support just because company matches.
    """
    ranker = EvidenceRanker()
    claim = "Jensen Huang returned as CEO and Nvidia stock rose"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock rose after Bob Jones returned as CEO following board shakeup",
        "url": "https://finance.yahoo.com/news/nvidia-leadership-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Did not expect 'support' when returnee is a different person, got '{stance}'"


def test_different_returnee_in_company_first_claim_does_not_support():
    """
    For claim 'Acme stock rose after Alice Smith returned as CEO', evidence 'Acme stock rose after Bob Jones returned as CEO'
    must not be treated as support just because company 'Acme' is in the evidence.
    """
    ranker = EvidenceRanker()
    claim = "Acme stock rose after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme stock rose after Bob Jones returned as CEO",
        "url": "https://marketwatch.com/acme-bob-jones.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Expected non-support for different returnee in company-first claim, got '{stance}'"


def test_causal_confirmation_guard_before_movement_support():
    """
    Causal movement claims ('causes blood pressure to rise') require explicit causal confirmation;
    unconfirmed correlation must return neutral.
    """
    ranker = EvidenceRanker()
    claim = "Acme treatment causes blood pressure to rise"
    evidence_item = {
        "articleTitle": "Medical News",
        "evidenceSnippet": "Acme reports blood pressure rose during treatment in preliminary trials",
        "url": "https://mednews.com/acme-trials-2024.html",
        "publisher": "Medical News",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "neutral", f"Expected 'neutral' for unconfirmed causal claim, got '{stance}'"


def test_gemini_override_requires_percentage_corroboration(monkeypatch):
    """
    When a claim asserts a specific percentage ('rose 10%'), but the evidence has no percentage,
    Gemini's contradiction must NOT be overridden to support.
    """
    ranker = EvidenceRanker()
    claim = "Nvidia stock rose 10% after Jensen Huang returned as CEO"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock rallied after Jensen Huang returned as CEO, but pulled back later in the week",
        "url": "https://finance.yahoo.com/news/nvidia-stock-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    # Mock Gemini returning contradict
    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 90, "reason": "Conflicting percentage"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_gemini_override_rejects_pre_event_rally(monkeypatch):
    """
    When the rally occurred BEFORE the event and was followed by a post-reinstatement plunge,
    Gemini's contradiction must NOT be overridden to support.
    """
    ranker = EvidenceRanker()
    claim = "after Jensen Huang returns as CEO, Nvidia stock goes up"
    evidence_item = {
        "articleTitle": "Yahoo Finance",
        "evidenceSnippet": "Nvidia stock rallied before Jensen Huang returned as CEO, then later fell after reinstatement with no rally",
        "url": "https://finance.yahoo.com/news/nvidia-pre-rally-2024.html",
        "publisher": "Yahoo Finance",
        "sourceTrustScore": 85
    }

    # Mock Gemini returning contradict
    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 92, "reason": "Direct contradiction"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_gemini_override_preserves_contradiction_for_different_year(monkeypatch):
    """
    When claim asserts an event occurred in 2024, but evidence reports it occurred in 2023,
    Gemini's contradiction must NOT be overridden to support despite subsequent reversal wording.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO in 2024"
    evidence_item = {
        "articleTitle": "Acme News",
        "evidenceSnippet": "Acme shares rose 10% after Alice Smith returned as CEO in 2023, then later fell",
        "url": "https://news.example.com/acme-2023.html",
        "publisher": "Acme News",
        "sourceTrustScore": 85
    }

    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 90, "reason": "Conflicting event year"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_timeframe_reversal_upward_classified_as_support():
    """
    When claim is about Monday ('Acme shares rose 10% on Monday'), and evidence confirms Monday's 10% rise
    then mentions a Tuesday plunge, it should be classified as support, not contradict.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% on Monday"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares rose 10% on Monday, then plunged on Tuesday",
        "url": "https://marketwatch.com/acme-monday.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support' for Monday gain despite Tuesday reversal, got '{stance}'"


def test_timeframe_reversal_downward_classified_as_support():
    """
    When claim is about Monday ('Acme shares fell 10% on Monday'), and evidence confirms Monday's 10% decline
    then mentions a Tuesday rally, it should be classified as support, not contradict.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares fell 10% on Monday"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares fell 10% on Monday, then jumped on Tuesday",
        "url": "https://marketwatch.com/acme-monday-down.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support' for Monday decline despite Tuesday jump, got '{stance}'"


def test_leadership_context_required_for_omitted_subject_reinstatement():
    """
    When claim is 'Alice Smith returned as Acme CEO and Acme stock rose', evidence saying
    'Acme stock rose after the company reinstated its dividend' must not match because
    dividend reinstatement is not a leadership event.
    """
    ranker = EvidenceRanker()
    claim = "Alice Smith returned as Acme CEO and Acme stock rose"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme stock rose after the company reinstated its dividend",
        "url": "https://marketwatch.com/acme-dividend.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Expected non-support for corporate dividend reinstatement without leadership subject, got '{stance}'"


def test_gemini_override_rejects_standalone_reversal_without_return_link(monkeypatch):
    """
    When evidence has 'reversing from a 10% rally last week' but explicitly contradicts the return event
    ('Alice Smith has never left the CEO role'), Gemini's contradiction must NOT be overridden to support.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares plunged, reversing from a 10% rally last week. Alice Smith has never left the CEO role.",
        "url": "https://marketwatch.com/acme-plunge.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 92, "reason": "Return contradicted"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_company_mismatch_in_override_preserves_contradiction(monkeypatch):
    """
    When claim is about 'Acme shares rose 10%', but evidence reports 'Beta shares rose 10%',
    Gemini's contradiction must NOT be overridden to support.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Beta shares rose 10% after Alice Smith returned as CEO, then later fell",
        "url": "https://marketwatch.com/beta-shares.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 90, "reason": "Company mismatch"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_percentage_mismatch_attached_to_rise_preserves_contradiction(monkeypatch):
    """
    When evidence reports a 2% rise after CEO return and a later 10% decline, for a 10% rise claim,
    the 10% attached to the decline must not satisfy pct_corroborated.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares rose 2% after Alice Smith returned as CEO, then later fell 10%",
        "url": "https://marketwatch.com/acme-rise-2.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 92, "reason": "Percentage mismatch"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_claimed_day_contradiction_prevents_override(monkeypatch):
    """
    When claim is about a Tuesday rise ('Acme shares rose on Tuesday after Alice Smith returned as CEO'),
    evidence reporting a Tuesday fall (even after Monday post-return rally) must preserve contradiction.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose on Tuesday after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares fell on Tuesday, reversing from a strong rally on Monday after the board reinstated Alice Smith as CEO",
        "url": "https://marketwatch.com/acme-tuesday-fall.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 90, "reason": "Fell on Tuesday"}}
    )

    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_plain_opposite_movement_classified_as_contradict():
    """
    When claim asserts 'Acme shares rose 10% on Monday', evidence stating 'Acme shares fell 10% on Monday'
    must be classified as contradict (not support).
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% on Monday"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares fell 10% on Monday",
        "url": "https://marketwatch.com/acme-monday-fell.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "contradict", f"Expected 'contradict' for plain opposite movement on claimed day, got '{stance}'"


def test_pre_return_rally_separated_by_day_classified_as_contradict():
    """
    When evidence says 'Acme shares rallied 10% on Monday before Alice Smith returned as CEO, then later fell after reinstatement with no rally',
    it must be classified as contradict.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares rallied 10% on Monday before Alice Smith returned as CEO, then later fell after reinstatement with no rally",
        "url": "https://marketwatch.com/acme-pre-rally.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "contradict", f"Expected 'contradict' for pre-return rally with post-return plunge, got '{stance}'"


def test_shared_year_with_different_day_is_not_a_conflict():
    """
    When claim is 'Acme shares rose 10% on Monday in 2024', and evidence says
    'Acme shares rose 10% on Monday in 2024, then plunged on Tuesday in 2024',
    the shared year 2024 must not cause Tuesday plunge to be treated as a same-day conflict with Monday.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% on Monday in 2024"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares rose 10% on Monday in 2024, then plunged on Tuesday in 2024",
        "url": "https://marketwatch.com/acme-2024-days.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support' for confirmed Monday rise despite Tuesday plunge in same year, got '{stance}'"


def test_day_first_with_movement_verb_classified_as_support():
    """
    When evidence is 'On Monday, Acme shares rose 10%; then plunged on Tuesday',
    the day-first format with movement verb 'rose' must be recognized as confirming the Monday movement.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% on Monday"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "On Monday, Acme shares rose 10%; then plunged on Tuesday",
        "url": "https://marketwatch.com/acme-monday-day-first.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support' for day-first movement confirmation, got '{stance}'"


def test_override_requires_same_movement_for_company_and_percentage(monkeypatch):
    """
    When evidence mentions a rise for Acme with 2% and a separate rise for Beta with 10%,
    the 10% from Beta must not satisfy Acme's claimed 10% rise override.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market News",
        "evidenceSnippet": "Acme shares rose 2% after Alice Smith returned as CEO, then later fell. Beta shares rose 10% after Bob Jones returned as CEO.",
        "url": "https://marketnews.com/acme-beta.html",
        "publisher": "Market News",
        "sourceTrustScore": 85
    }

    monkeypatch.setattr(
        ranker.gemini_classifier,
        "classify_candidates_for_claim",
        lambda c, items, **kwargs: {0: {"stance": "contradict", "confidence": 85, "reason": "Acme rose only 2%, not 10%"}}
    )
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_pm_title_case_insensitively_matched_in_stopwords():
    """
    Claim 'PM Alice Smith returned to office' with evidence 'Prime minister Alice Smith returned to office'
    must recognize uppercase 'PM' as role acronym and not reject evidence.
    """
    ranker = EvidenceRanker()
    claim = "PM Alice Smith returned to office"
    raw_combined = "Prime minister Alice Smith returned to office"
    is_match, reason = ranker._check_specific_claim_match(claim, raw_combined)
    assert is_match, f"Expected specific match for PM role, got reason: {reason}"


def test_tuesday_plunge_not_assigned_to_monday():
    """
    Evidence 'Acme shares rose 10% on Monday then plunged on Tuesday' must not attribute
    the Tuesday plunge to Monday, and must classify as support for Monday rise.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% on Monday"
    evidence_item = {
        "articleTitle": "Finance Daily",
        "evidenceSnippet": "Acme shares rose 10% on Monday then plunged on Tuesday",
        "url": "https://financedaily.com/acme-mon-tue.html",
        "publisher": "Finance Daily",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support', got '{stance}'"


def test_claimed_month_constraint_validated():
    """
    Claim 'Acme shares rose on Monday in January 2024' with evidence
    'Acme shares rose on Monday in February 2024' must not return support due to month mismatch.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose on Monday in January 2024"
    evidence_item = {
        "articleTitle": "Market Watch",
        "evidenceSnippet": "Acme shares rose on Monday in February 2024",
        "url": "https://marketwatch.com/acme-feb.html",
        "publisher": "Market Watch",
        "sourceTrustScore": 85
    }

    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Expected stance != 'support' due to conflicting month, got '{stance}'"


def test_role_phrase_its_ceo_not_treated_as_competing_person():
    """
    Claim 'Acme stock rose after Alice Smith returned as CEO' with evidence
    'Acme stock rose after the board reinstated its CEO' must not treat 'its CEO' as a competing person.
    """
    ranker = EvidenceRanker()
    claim = "Acme stock rose after Alice Smith returned as CEO"
    raw_combined = "Acme stock rose after the board reinstated its CEO"
    is_match, reason = ranker._check_specific_claim_match(claim, raw_combined)
    assert "different person" not in reason.lower(), f"Unexpected different person error: {reason}"


def test_conflicting_given_name_rejected_as_competing_person():
    """
    Claim 'Acme stock rose after Alice Smith returned as CEO' with evidence
    'Acme stock rose after the board reinstated Bob Smith as CEO' must reject
    due to conflicting given name (Bob vs Alice) despite matching surname.
    """
    ranker = EvidenceRanker()
    claim = "Acme stock rose after Alice Smith returned as CEO"
    raw_combined = "Acme stock rose after the board reinstated Bob Smith as CEO"
    is_match, reason = ranker._check_specific_claim_match(claim, raw_combined)
    assert not is_match, "Expected mismatch due to conflicting given name"
    assert "different person" in reason.lower() or "not mention" in reason.lower()


def test_surname_only_without_conflict_matches_person():
    """
    Claim 'Acme stock rose after Alice Smith returned as CEO' with evidence
    'Acme stock rose after the board reinstated Smith as CEO' allows surname-only
    reference when no conflicting given name is present.
    """
    ranker = EvidenceRanker()
    claim = "Acme stock rose after Alice Smith returned as CEO"
    raw_combined = "Acme stock rose after the board reinstated Smith as CEO"
    is_match, reason = ranker._check_specific_claim_match(claim, raw_combined)
    assert is_match, f"Expected surname-only match to succeed, got reason: {reason}"


def test_override_rejects_corporate_asset_reinstatement():
    """
    Evidence citing dividend reinstatement instead of leadership return must not
    satisfy the Gemini contradict override.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Acme Corp Update",
        "evidenceSnippet": "Acme shares rose 10% after the company reinstated its dividend, but later fell.",
        "url": "https://example.com/acme-dividend.html",
        "publisher": "Example News",
        "sourceTrustScore": 85
    }
    ranker.gemini_classifier.classify_candidates_for_claim = lambda c, items, **kwargs: {
        0: {"stance": "contradict", "confidence": 90, "reason": "Mentions dividend, not CEO return"}
    }
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_speculative_movement_rejected_by_gemini_override():
    """
    Speculative wording ('may rally 10%') followed by later reversal must not satisfy
    the Gemini override to support.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Forecast Daily",
        "evidenceSnippet": "Acme shares may rally 10% after Alice Smith returns as CEO. Acme shares later fell.",
        "url": "https://example.com/acme-spec.html",
        "publisher": "Forecast Daily",
        "sourceTrustScore": 85
    }
    ranker.gemini_classifier.classify_candidates_for_claim = lambda c, items, **kwargs: {
        0: {"stance": "contradict", "confidence": 90, "reason": "Speculative rally, not confirmed"}
    }
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_override_percentage_bound_to_company_movement():
    """
    In multi-company text, the percentage must be bound to the claimed company's movement.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Market Round",
        "evidenceSnippet": "Acme shares rose 2% after Alice Smith returned as CEO, and Beta shares rose 10% after Alice Smith returned as CEO, then later fell.",
        "url": "https://example.com/market-round.html",
        "publisher": "Market Round",
        "sourceTrustScore": 85
    }
    ranker.gemini_classifier.classify_candidates_for_claim = lambda c, items, **kwargs: {
        0: {"stance": "contradict", "confidence": 90, "reason": "Acme only rose 2%"}
    }
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_negated_inflected_movement_verb_rejected_by_override():
    """
    'Acme shares never rose 10%' must be caught as negated rise and not satisfy the override.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Correction Report",
        "evidenceSnippet": "Acme shares never rose 10% after Alice Smith returned as CEO, then later fell.",
        "url": "https://example.com/acme-never-rose.html",
        "publisher": "Correction Report",
        "sourceTrustScore": 85
    }
    ranker.gemini_classifier.classify_candidates_for_claim = lambda c, items, **kwargs: {
        0: {"stance": "contradict", "confidence": 90, "reason": "Never rose 10%"}
    }
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["contradictingCount"] == 1
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_month_case_normalization_in_gemini_override():
    """
    Capitalized month in evidence must normalize and match claimed month.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose 10% in january after Alice Smith returned as CEO"
    evidence_item = {
        "articleTitle": "Monthly Review",
        "evidenceSnippet": "Acme shares rose 10% in January after Alice Smith returned as CEO, but subsequently fell in February.",
        "url": "https://example.com/acme-jan.html",
        "publisher": "Monthly Review",
        "sourceTrustScore": 85
    }
    ranker.gemini_classifier.classify_candidates_for_claim = lambda c, items, **kwargs: {
        0: {"stance": "contradict", "confidence": 90, "reason": "Fell in February"}
    }
    ranked = ranker.rank_evidence_for_claim(claim, [evidence_item], allow_gemini=True)
    assert ranked["supportingCount"] == 1
    assert ranked["supporting"][0]["stance"] == "support"


def test_movement_speculation_guard_independent_of_other_completed_movements():
    """
    Claim 'Acme shares rose on Monday' with evidence 'Acme shares had fallen on Tuesday; Acme shares may rally on Monday'
    must not count 'may rally on Monday' as support.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose on Monday"
    evidence_item = {
        "articleTitle": "Weekly Outlook",
        "evidenceSnippet": "Acme shares had fallen on Tuesday; Acme shares may rally on Monday",
        "url": "https://example.com/weekly-outlook.html",
        "publisher": "Weekly Outlook",
        "sourceTrustScore": 85
    }
    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance != "support", f"Expected stance != 'support', got '{stance}'"


def test_date_constraints_bound_to_claimed_movement_timeframe():
    """
    Claim 'Acme shares rose on Monday in January 2024' with evidence
    'Acme shares rose on Monday in February 2024. January 2024 review.' must return related, not support.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares rose on Monday in January 2024"
    evidence_item = {
        "articleTitle": "Review Summary",
        "evidenceSnippet": "Acme shares rose on Monday in February 2024. January 2024 review.",
        "url": "https://example.com/review-sum.html",
        "publisher": "Review Summary",
        "sourceTrustScore": 85
    }
    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "related", f"Expected 'related', got '{stance}'"


def test_increased_movement_verb_recognized_consistently():
    """
    'increased' movement verb must be recognized across affirmative and timeframe matching.
    """
    ranker = EvidenceRanker()
    claim = "Acme shares increased 10% on Monday"
    evidence_item = {
        "articleTitle": "Finance Alert",
        "evidenceSnippet": "Acme shares increased 10% on Monday",
        "url": "https://example.com/acme-increased.html",
        "publisher": "Finance Alert",
        "sourceTrustScore": 85
    }
    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "support", f"Expected 'support', got '{stance}'"


def test_conflicting_capital_returns_contradict():
    """
    Claim 'Sydney is the capital of Australia' with evidence 'Canberra is the capital of Australia'
    must return contradict.
    """
    ranker = EvidenceRanker()
    claim = "Sydney is the capital of Australia"
    evidence_item = {
        "articleTitle": "Australia Facts",
        "evidenceSnippet": "Canberra is the capital of Australia and the seat of government.",
        "url": "https://example.com/australia-facts.html",
        "publisher": "Australia Facts",
        "sourceTrustScore": 85
    }
    stance, _ = ranker.classify_stance(claim, evidence_item, 0.75)
    assert stance == "contradict", f"Expected 'contradict', got '{stance}'"

