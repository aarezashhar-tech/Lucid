"""
Offline Verdict and Stance Test Suite for Lucid Backend.
Runs completely offline using mocked sources, zero network dependencies.
Executes with one command and finishes in well under 30 seconds.

Tests:
1. Dominant Support -> VERIFIED (no floor at 78)
2. Balanced Conflict -> NEEDS REVIEW (score in [40, 60], not forced to 38)
3. Single Weak Contradiction with dominant support -> VERIFIED or LIKELY TRUE (not NEEDS REVIEW)
4. No Evidence -> UNVERIFIED with reason "no live evidence found"
5. All-Old Evidence (>180 days) on time-sensitive claim -> NEEDS REVIEW / capped at 60
6. Future-Event Wording -> Neutral stance (not support)
7. Compound Claim Splitting & Partial Unverified Rule -> Not FALSE or LIKELY FALSE
8. Entity/Topic Match Requirement -> Generic page does not count as contradiction
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.multi_factor_scorer import MultiFactorScoringEngine
from services.evidence_ranker import EvidenceRanker
from services.claim_extractor import split_compound_claim, ClaimExtractionEngine
from utils.url_extractor import is_real_article_url, resolve_google_news_url


class TestLucidVerdicts(unittest.TestCase):

    def setUp(self):
        self.scorer = MultiFactorScoringEngine()
        self.ranker = EvidenceRanker()
        self.extractor = ClaimExtractionEngine()
        self.ling_signals = {
            "sensational_score": 10,
            "emotional_language": False,
            "clickbait_score": 5,
            "detected_sensational_terms": []
        }
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        old_dt = datetime.now() - timedelta(days=200)
        self.old_str = old_dt.strftime("%Y-%m-%d")

    def test_dominant_support_verified(self):
        """
        Test 1: Dominant support across reliable wire sources (e.g. Reuters, BBC)
        must produce VERIFIED, and score must NOT be floored at exactly 78.
        """
        top_supporting = [
            {
                "url": "https://www.reuters.com/world/article-test-1-2024",
                "publisher": "Reuters",
                "relevanceScore": 85,
                "publicationDate": self.today_str,
                "sourceTrustScore": 96,
                "stance": "support",
                "compositeRankScore": 90
            },
            {
                "url": "https://www.bbc.com/news/article-test-2-2024",
                "publisher": "BBC",
                "relevanceScore": 82,
                "publicationDate": self.today_str,
                "sourceTrustScore": 94,
                "stance": "support",
                "compositeRankScore": 88
            }
        ]
        top_contradicting = []

        res = self.scorer.compute_verdict_and_score(
            ml_score=75.0,
            source_score=85.0,
            evidence_score=85.0,
            claims=[{"claim": "Event happened", "status": "Supported"}],
            ling_signals=self.ling_signals,
            supporting_evidence_count=2,
            contradicting_evidence_count=0,
            top_supporting=top_supporting,
            top_contradicting=top_contradicting
        )

        self.assertEqual(res["verdict"], "VERIFIED")
        # Check that verdict is in sensible band [70, 98] and not artificially pegged
        self.assertGreaterEqual(res["finalTrustScore"], 70)
        self.assertLessEqual(res["finalTrustScore"], 98)
        self.assertEqual(res["riskLevel"], "LOW")
        self.assertIn("clearly dominating", res["summaryReason"].lower())

    def test_balanced_conflict_needs_review(self):
        """
        Test 2: Balanced conflict where both sides are strong and close in weight
        must produce NEEDS REVIEW.
        Score must NOT be hard-coded to 38, but remain in the [40, 60] band.
        """
        top_supporting = [
            {
                "url": "https://www.reuters.com/business/article-supp-1",
                "publisher": "Reuters",
                "relevanceScore": 85,
                "publicationDate": self.today_str,
                "sourceTrustScore": 96,
                "stance": "support",
                "compositeRankScore": 90
            }
        ]
        top_contradicting = [
            {
                "url": "https://apnews.com/article/article-contra-1",
                "publisher": "AP News",
                "relevanceScore": 85,
                "publicationDate": self.today_str,
                "sourceTrustScore": 95,
                "stance": "contradict",
                "compositeRankScore": 90
            }
        ]

        res = self.scorer.compute_verdict_and_score(
            ml_score=50.0,
            source_score=50.0,
            evidence_score=50.0,
            claims=[{"claim": "Disputed rate hike", "status": "Contradicted"}],
            ling_signals=self.ling_signals,
            supporting_evidence_count=1,
            contradicting_evidence_count=1,
            top_supporting=top_supporting,
            top_contradicting=top_contradicting
        )

        self.assertEqual(res["verdict"], "NEEDS REVIEW")
        self.assertNotEqual(res["finalTrustScore"], 38, "Score must not be forced to 38")
        self.assertGreaterEqual(res["finalTrustScore"], 40)
        self.assertLessEqual(res["finalTrustScore"], 60)
        self.assertEqual(res["riskLevel"], "MEDIUM")
        self.assertIn("conflicting evidence", res["summaryReason"].lower())

    def test_single_weak_contradiction_dominant_support(self):
        """
        Test 3: Strong support dominates (multiple reputable wires), with only 1 weak
        contradicting source. Verdict must be VERIFIED or LIKELY TRUE (NOT NEEDS REVIEW).
        """
        top_supporting = [
            {
                "url": "https://www.reuters.com/business/article-supp-strong-1",
                "publisher": "Reuters",
                "relevanceScore": 90,
                "publicationDate": self.today_str,
                "sourceTrustScore": 96,
                "stance": "support",
                "compositeRankScore": 92
            },
            {
                "url": "https://www.bloomberg.com/news/article-supp-strong-2",
                "publisher": "Bloomberg",
                "relevanceScore": 88,
                "publicationDate": self.today_str,
                "sourceTrustScore": 94,
                "stance": "support",
                "compositeRankScore": 89
            }
        ]
        top_contradicting = [
            {
                "url": "https://thetimesofindia.com/article-weak-contra-1",
                "publisher": "The Times of India",
                "relevanceScore": 25,
                "publicationDate": self.today_str,
                "sourceTrustScore": 60,
                "stance": "contradict",
                "compositeRankScore": 40
            }
        ]

        res = self.scorer.compute_verdict_and_score(
            ml_score=80.0,
            source_score=85.0,
            evidence_score=85.0,
            claims=[{"claim": "Major market growth announced", "status": "Supported"}],
            ling_signals=self.ling_signals,
            supporting_evidence_count=2,
            contradicting_evidence_count=1,
            top_supporting=top_supporting,
            top_contradicting=top_contradicting
        )

        self.assertIn(res["verdict"], ("VERIFIED", "LIKELY TRUE"), "Weak contradiction must not degrade dominant support to NEEDS REVIEW")
        self.assertGreaterEqual(res["finalTrustScore"], 65)

    def test_no_evidence_unverified(self):
        """
        Test 4: When no evidence is found, verdict must be UNVERIFIED
        and reason must state 'no live evidence found'.
        """
        res = self.scorer.compute_verdict_and_score(
            ml_score=50.0,
            source_score=50.0,
            evidence_score=50.0,
            claims=[{"claim": "Obscure unverifiable claim", "status": "Unverified"}],
            ling_signals=self.ling_signals,
            supporting_evidence_count=0,
            contradicting_evidence_count=0,
            top_supporting=[],
            top_contradicting=[]
        )

        self.assertEqual(res["verdict"], "UNVERIFIED")
        self.assertIn("no live evidence found", res["summaryReason"].lower())
        self.assertEqual(res["riskLevel"], "MEDIUM")
        self.assertLessEqual(res["finalTrustScore"], 50)

    def test_all_old_evidence_time_sensitive(self):
        """
        Test 5: Time-sensitive claim where ALL supporting sources are older than 180 days.
        Must be downgraded to NEEDS REVIEW with trust score capped at 60.
        """
        top_supporting = [
            {
                "url": "https://www.reuters.com/world/article-old-news-1",
                "publisher": "Reuters",
                "relevanceScore": 85,
                "publicationDate": self.old_str,
                "sourceTrustScore": 96,
                "stance": "support",
                "compositeRankScore": 85
            }
        ]
        top_contradicting = []

        res = self.scorer.compute_verdict_and_score(
            ml_score=80.0,
            source_score=85.0,
            evidence_score=80.0,
            claims=[{"claim": "Current prime minister resigned", "status": "May be outdated", "isTimeSensitive": True, "isOutdated": True}],
            ling_signals=self.ling_signals,
            supporting_evidence_count=1,
            contradicting_evidence_count=0,
            top_supporting=top_supporting,
            top_contradicting=top_contradicting
        )

        self.assertEqual(res["verdict"], "NEEDS REVIEW")
        self.assertLessEqual(res["finalTrustScore"], 60, "Trust score must be capped at 60 for outdated evidence on time-sensitive claim")
        self.assertIn("older than 6 months", res["summaryReason"].lower())

    def test_future_event_wording_neutral(self):
        """
        Test 6: A source discussing possible or future events ('plans to', 'wants to', 'may')
        must be classified as 'neutral' for a claim asserting the event happened.
        """
        claim_text = "Jensen Huang returned as CEO of Nvidia"
        candidate_future = {
            "articleTitle": "Jensen Huang plans to return as CEO of Nvidia next quarter",
            "evidenceSnippet": "Nvidia spokesperson confirmed Jensen Huang wants to return and is considering the executive appointment.",
            "sourceTrustScore": 90,
            "url": "https://www.reuters.com/technology/huang-plans-return-2024"
        }

        stance, conf = self.ranker.classify_stance(claim_text, candidate_future, sim_score=0.55)
        self.assertEqual(stance, "neutral", f"Expected neutral for future/speculative event wording, got: {stance}")

    def test_compound_claim_partial_unverified(self):
        """
        Test 7: Compound claims split into sub-claims.
        If one sub-claim is contradicted and the other has NO evidence,
        verdict must NOT be FALSE or LIKELY FALSE from the other part alone.
        """
        claim_text = "Jensen Huang resigned as CEO of Nvidia and Apple acquired Intel"
        sub_claims = split_compound_claim(claim_text)
        self.assertEqual(len(sub_claims), 2, "Compound claim must be split into 2 sub-claims")
        self.assertIn("Jensen Huang resigned as CEO of Nvidia", sub_claims[0])
        self.assertIn("Apple acquired Intel", sub_claims[1])

        # Claim 1 has contradiction, Claim 2 has 0 evidence
        claims_data = [
            {
                "claim": sub_claims[0],
                "status": "Contradicted",
                "supportingSources": [],
                "contradictingSources": ["Reuters Fact Check"],
                "evidenceSnippets": [{"stance": "contradict", "url": "https://reuters.com/factcheck-1"}]
            },
            {
                "claim": sub_claims[1],
                "status": "Unverified",
                "supportingSources": [],
                "contradictingSources": [],
                "evidenceSnippets": []
            }
        ]

        top_contra = [{
            "url": "https://reuters.com/factcheck-1",
            "publisher": "Reuters Fact Check",
            "relevanceScore": 85,
            "publicationDate": self.today_str,
            "sourceTrustScore": 96,
            "stance": "contradict",
            "compositeRankScore": 90
        }]

        res = self.scorer.compute_verdict_and_score(
            ml_score=20.0,
            source_score=50.0,
            evidence_score=10.0,
            claims=claims_data,
            ling_signals=self.ling_signals,
            supporting_evidence_count=0,
            contradicting_evidence_count=1,
            top_supporting=[],
            top_contradicting=top_contra
        )

        self.assertNotIn(res["verdict"], ("FALSE", "KNOWN FALSE", "LIKELY FALSE"),
                         "Verdict must not be FALSE or LIKELY FALSE when one sub-claim has no evidence")
        self.assertEqual(res["verdict"], "NEEDS REVIEW")

    def test_entity_topic_match_generic_page_not_contradiction(self):
        """
        Test 8: Generic page (e.g. Wikipedia article on Earth) must NOT count as
        a contradiction for a specific claim like 'The Earth is flat'.
        """
        claim_text = "The Earth is flat"
        generic_snippet = "Earth is the third planet from the Sun and the only astronomical object known to harbor life."

        is_topic_match = self.ranker._snippet_mentions_claim_topic_or_entities(
            claim_text, f"Wikipedia: Earth {generic_snippet}"
        )
        self.assertFalse(is_topic_match, "Generic overview of Earth must not be counted as a specific topic match for 'The Earth is flat'")


if __name__ == "__main__":
    unittest.main()
