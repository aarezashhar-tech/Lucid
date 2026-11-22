from __future__ import annotations
"""
Layer 4 & Layer 5: Multi-Factor Scoring & Modern Verdict Engine
Combines 5 independent analytical dimensions:
- 30% ML Prediction
- 25% Source Reputation
- 25% Evidence Strength
- 10% Claim Consistency
- 10% Language Quality

Replaces binary REAL/FAKE with granular verdict types:
- VERIFIED
- LIKELY TRUE
- NEEDS REVIEW
- UNVERIFIED
- LIKELY FALSE
- KNOWN FALSE
"""
from typing import Dict, Any, List, Tuple

TupleScore = Tuple[int, str]

class MultiFactorScoringEngine:
    """
    Computes weighted multi-factor credibility score and assigns
    fact-verification verdict with strict evidence escalation safeguards.
    """

    CONTRADICTION_THRESHOLD = 2
    EVIDENCE_THRESHOLD = 60.0

    FALSE_FACTCHECK_KEYWORDS = [
        "false", "fake", "incorrect", "pants on fire", "misleading", "hoax",
        "debunked", "untrue", "scam", "fiction", "unsubstantiated", "distorted",
        "disproven", "refuted", "fabricated"
    ]

    def calculate_claim_consistency(self, claims: List[Dict[str, Any]]) -> TupleScore:
        """
        Evaluates internal consistency between extracted claims:
        - Checks for contradictory statuses
        - Penalizes mixed contradictory vs supported statements
        """
        if not claims:
            return 70, "No claims available to assess consistency."

        statuses = [c.get("status", "Unverified") for c in claims]
        has_supported = any(s == "Supported" for s in statuses)
        has_contradicted = any(s == "Contradicted" for s in statuses)

        if has_supported and has_contradicted:
            return 35, "Internal conflict detected: article contains both supported and contradicted claims."
        
        if all(s == "Supported" for s in statuses):
            return 94, "High claim consistency: all extracted propositions are mutually corroborating."

        if all(s == "Contradicted" for s in statuses):
            return 20, "Consistent falsehood pattern across all extracted claims."

        if has_supported:
            return 85, "Primary claims are supported with no observed internal contradictions."

        return 65, "Neutral claim consistency with unverified elements."

    def calculate_language_quality(self, ling_signals: Dict[str, Any]) -> TupleScore:
        """
        Calculates language quality score (0-100) based on journalistic objectivity:
        - Absence of sensationalism
        - Absence of clickbait patterns
        - Clean capitalization and punctuation
        """
        sensational = ling_signals.get("sensational_score", 0)
        clickbait = ling_signals.get("clickbait_score", 0)
        excl = ling_signals.get("exclamation_count", 0)
        caps = ling_signals.get("caps_ratio", 0)

        # Baseline 100 minus linguistic penalties
        penalty = (sensational * 0.45) + (clickbait * 0.35) + (excl * 4) + (caps * 30)
        quality = int(round(max(10, min(99, 100 - penalty))))

        if quality >= 85:
            notes = "Objective, standard journalistic phrasing with no manipulative cues."
        elif quality >= 60:
            notes = "Acceptable prose with mild emotional or colloquial tone."
        else:
            notes = "High sensationalism and emotional manipulation indicators detected."

        return quality, notes

    def compute_verdict_and_score(
        self,
        ml_score: float,
        source_score: float,
        evidence_score: float,
        claims: List[Dict[str, Any]],
        ling_signals: Dict[str, Any],
        supporting_evidence_count: int,
        contradicting_evidence_count: int,
        fact_check_results: List[Dict[str, Any]] = None,
        has_false_fact_check: bool = None,
        contradiction_threshold: int = 2,
        evidence_threshold: float = 60.0,
        article_type: str = "News Report",
        source_quality_assessment: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Computes the final 5-factor weighted score and maps to the 6-tier verdict.
        Follows strict evidence escalation rules:
        - KNOWN FALSE:
          Only if Google Fact Check returns a false/debunked rating,
          OR contradicting evidence exceeds supporting evidence by defined threshold.
        - LIKELY FALSE:
          Some contradicting evidence exists but not enough for a definitive false verdict.
        - UNVERIFIED:
          No reliable supporting or contradicting citations attached,
          OR evidence score below threshold due to missing citations.
        - NEEDS REVIEW:
          Mixed evidence (conflicting supporting and contradicting citations).
        - LIKELY TRUE / VERIFIED:
          Strong corroboration with no contradicting evidence.
        - Missing evidence must NEVER be treated as evidence of falsity.
        """
        consistency_score, consistency_notes = self.calculate_claim_consistency(claims)
        language_score, language_notes = self.calculate_language_quality(ling_signals)

        # Ensure scores are within [0, 100]
        ml_s = max(0, min(100, float(ml_score)))
        src_s = max(0, min(100, float(source_score)))
        evi_s = max(0, min(100, float(evidence_score)))
        con_s = max(0, min(100, float(consistency_score)))
        lan_s = max(0, min(100, float(language_score)))

        # -----------------------------------------------------------------
        # 1. OPINION / COMMENTARY PIECES (Problem 3)
        # Opinion must NOT get a fact-based trust score.
        # Return "Not scored" for score and "Opinion - not fact-checked" for verdict.
        # -----------------------------------------------------------------
        if article_type in ("Opinion Piece", "Opinion"):
            return {
                "finalTrustScore": "Not scored",
                "verdict": "Opinion - not fact-checked",
                "riskLevel": "LOW",
                "summaryReason": "Opinion / commentary piece — subjective arguments, perspectives, and commentary are not assigned a fact-based verification score.",
                "scoringFactors": {
                    "evidenceStrength": {
                        "weight": 0.45,
                        "score": "Not scored",
                        "contribution": 0.0,
                        "label": "Evidence Strength (45%) — N/A for Opinion"
                    },
                    "mlPrediction": {
                        "weight": 0.20,
                        "score": "Not scored",
                        "contribution": 0.0,
                        "label": "ML Pattern Classifier (20%) — N/A for Opinion"
                    },
                    "languageQuality": {
                        "weight": 0.15,
                        "score": int(round(lan_s)),
                        "contribution": 0.0,
                        "label": "Language Tone & Rhetoric (15%)",
                        "notes": language_notes
                    },
                    "sourceReputation": {
                        "weight": 0.10,
                        "score": int(round(src_s)),
                        "contribution": 0.0,
                        "label": "Domain & Source Authority (10%)"
                    },
                    "claimConsistency": {
                        "weight": 0.10,
                        "score": int(round(con_s)),
                        "contribution": 0.0,
                        "label": "Viewpoint Consistency (10%)",
                        "notes": consistency_notes
                    }
                }
            }

        if article_type == "Historical Analysis":
            final_trust_score = round(
                (0.40 * src_s) +
                (0.25 * lan_s) +
                (0.20 * ml_s) +
                (0.15 * con_s)
            )
            final_trust_score = max(20, min(98, int(final_trust_score)))

            if src_s >= 80:
                verdict = "HISTORICAL ANALYSIS - VERIFIED AUTHORITY"
                risk_level = "LOW"
            elif src_s >= 60:
                verdict = "HISTORICAL ANALYSIS - REPUTABLE SOURCE"
                risk_level = "LOW"
            else:
                verdict = "HISTORICAL ANALYSIS - UNVERIFIED"
                risk_level = "MEDIUM"

            summary_reason = "Evaluated via Source Quality Assessment on historical authority, archival integrity, and scholarly standards rather than daily breaking news wire feeds."

            return {
                "finalTrustScore": final_trust_score,
                "verdict": verdict,
                "riskLevel": risk_level,
                "summaryReason": summary_reason,
                "scoringFactors": {
                    "evidenceStrength": {
                        "weight": 0.45,
                        "score": int(round(evi_s)),
                        "contribution": round(0.45 * evi_s, 1),
                        "label": "Evidence Strength (45%)"
                    },
                    "mlPrediction": {
                        "weight": 0.20,
                        "score": int(round(ml_s)),
                        "contribution": round(0.20 * ml_s, 1),
                        "label": "ML Text Pattern Classifier (20%)"
                    },
                    "languageQuality": {
                        "weight": 0.15,
                        "score": int(round(lan_s)),
                        "contribution": round(0.15 * lan_s, 1),
                        "label": "Journalistic Language Integrity (15%)",
                        "notes": language_notes
                    },
                    "sourceReputation": {
                        "weight": 0.10,
                        "score": int(round(src_s)),
                        "contribution": round(0.10 * src_s, 1),
                        "label": "Domain & Source Reputation (10%)"
                    },
                    "claimConsistency": {
                        "weight": 0.10,
                        "score": int(round(con_s)),
                        "contribution": round(0.10 * con_s, 1),
                        "label": "Factual Claim Consistency (10%)",
                        "notes": consistency_notes
                    }
                }
            }

        # -----------------------------------------------------------------
        # 2. FACTUAL VERIFICATION MULTI-FACTOR SCORING (Problems 1, 2, 5)
        # Domain reputation is ONE small part (10%, strictly max 10-15%).
        # Weights: Evidence 45%, ML 20%, Language 15%, Source 10%, Consistency 10%.
        # -----------------------------------------------------------------
        computed_score = round(
            (0.45 * evi_s) +
            (0.20 * ml_s) +
            (0.15 * lan_s) +
            (0.10 * src_s) +
            (0.10 * con_s)
        )
        final_trust_score = max(5, min(98, int(computed_score)))

        # Detect Google Fact Check false/debunked rating if not explicitly supplied
        if has_false_fact_check is None:
            has_false_fact_check = False
            if fact_check_results:
                for fc in fact_check_results:
                    rating = (fc.get("rating") or fc.get("textualRating") or "").lower()
                    if any(k in rating for k in self.FALSE_FACTCHECK_KEYWORDS):
                        has_false_fact_check = True
                        break

            if not has_false_fact_check and claims:
                for c in claims:
                    for fc in c.get("factCheckMatches", []):
                        rating = (fc.get("rating") or fc.get("textualRating") or "").lower()
                        if any(k in rating for k in self.FALSE_FACTCHECK_KEYWORDS):
                            has_false_fact_check = True
                            break
                    if has_false_fact_check:
                        break
                    for snip in c.get("evidenceSnippets", []):
                        calc_reason = (snip.get("calculationReason") or "").lower()
                        snip_text = (snip.get("snippet") or "").lower()
                        if ("debunked" in calc_reason or "debunked" in snip_text) and "google fact check" in (calc_reason + snip_text):
                            has_false_fact_check = True
                            break
                    if has_false_fact_check:
                        break

        # Check whether contradicting evidence exists
        contra_exceeds_threshold = (
            contradicting_evidence_count >= contradiction_threshold and
            (contradicting_evidence_count - supporting_evidence_count) >= contradiction_threshold
        )

        # -----------------------------------------------------------------
        # 3. VERDICT ASSIGNMENT WITH STRICT EVIDENCE SAFEGUARDS
        # - A reliable publisher with a false claim inside must score LOW.
        # - An unknown publisher with a true, well-sourced claim must not be punished.
        # - FALSE: Conclusively contradicted by evidence or rated false.
        # - UNVERIFIED: No evidence found (missing evidence is NEVER false!).
        # -----------------------------------------------------------------
        if has_false_fact_check:
            verdict = "KNOWN FALSE"
            final_trust_score = min(final_trust_score, 12)
            risk_level = "CRITICAL"
            summary_reason = "Conclusively disproven. Flagged as false by Google Fact Check API."

        elif contra_exceeds_threshold:
            verdict = "KNOWN FALSE"
            final_trust_score = min(final_trust_score, 12)
            risk_level = "CRITICAL"
            summary_reason = f"Contradicting evidence exceeds supporting by {contradicting_evidence_count - supporting_evidence_count}. Contradicted by authoritative evidence."

        elif supporting_evidence_count > 0 and contradicting_evidence_count > 0:
            verdict = "NEEDS REVIEW"
            final_trust_score = min(final_trust_score, 38)
            risk_level = "MEDIUM"
            summary_reason = f"Mixed evidence detected across wire sources ({supporting_evidence_count} supporting vs {contradicting_evidence_count} contradicting citations). Contains refuted assertions."

        elif contradicting_evidence_count > 0:
            verdict = "LIKELY FALSE"
            final_trust_score = min(final_trust_score, 35)
            risk_level = "HIGH"
            summary_reason = f"Likely inaccurate. Retrieved evidence disputes the claim with {contradicting_evidence_count} contradicting flag(s)."

        elif supporting_evidence_count == 0 and contradicting_evidence_count == 0:
            verdict = "UNVERIFIED"
            final_trust_score = max(45, min(55, final_trust_score))  # Neutral unverified score ~50
            risk_level = "MEDIUM"
            summary_reason = "No reliable supporting or contradicting citations attached. Unverified pending independent evidence."

        elif evidence_score < evidence_threshold and supporting_evidence_count < 2:
            verdict = "UNVERIFIED"
            final_trust_score = max(45, min(55, final_trust_score))
            risk_level = "MEDIUM"
            summary_reason = "Insufficient independent evidence citations. Unverified pending further corroboration."

        elif supporting_evidence_count >= 2 and contradicting_evidence_count == 0:
            verdict = "VERIFIED"
            final_trust_score = max(final_trust_score, 78)
            risk_level = "LOW"
            summary_reason = f"Corroborated across {supporting_evidence_count} independent reliable sources with zero contradicting evidence."

        elif supporting_evidence_count >= 1 and contradicting_evidence_count == 0:
            verdict = "LIKELY TRUE"
            final_trust_score = max(final_trust_score, 68)
            risk_level = "LOW"
            summary_reason = f"Supported by reputable reporting ({supporting_evidence_count} corroborating citation(s)) without contradiction."

        else:
            verdict = "NEEDS REVIEW"
            risk_level = "MEDIUM"
            summary_reason = "Content requires editorial review due to mixed or incomplete evidence signals."

        return {
            "finalTrustScore": final_trust_score,
            "verdict": verdict,
            "riskLevel": risk_level,
            "summaryReason": summary_reason,
            "scoringFactors": {
                "evidenceStrength": {
                    "weight": 0.45,
                    "score": int(round(evi_s)),
                    "contribution": round(0.45 * evi_s, 1),
                    "label": "Evidence Strength (45%)"
                },
                "mlPrediction": {
                    "weight": 0.20,
                    "score": int(round(ml_s)),
                    "contribution": round(0.20 * ml_s, 1),
                    "label": "ML Text Pattern Classifier (20%)"
                },
                "languageQuality": {
                    "weight": 0.15,
                    "score": int(round(lan_s)),
                    "contribution": round(0.15 * lan_s, 1),
                    "label": "Journalistic Language Integrity (15%)",
                    "notes": language_notes
                },
                "sourceReputation": {
                    "weight": 0.10,
                    "score": int(round(src_s)),
                    "contribution": round(0.10 * src_s, 1),
                    "label": "Domain & Source Reputation (10%)"
                },
                "claimConsistency": {
                    "weight": 0.10,
                    "score": int(round(con_s)),
                    "contribution": round(0.10 * con_s, 1),
                    "label": "Factual Claim Consistency (10%)",
                    "notes": consistency_notes
                }
            }
        }

scoring_engine = MultiFactorScoringEngine()

def get_scoring_engine() -> MultiFactorScoringEngine:
    return scoring_engine
