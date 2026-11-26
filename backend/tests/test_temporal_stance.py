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
