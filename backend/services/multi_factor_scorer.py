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
from typing import Dict, Any, List

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

        # Weighted calculation
        # 30% ML, 25% Source, 25% Evidence, 10% Consistency, 10% Language
        final_trust_score = round(
            (0.30 * ml_s) +
            (0.25 * src_s) +
            (0.25 * evi_s) +
            (0.10 * con_s) +
            (0.10 * lan_s)
        )
        final_trust_score = max(5, min(98, int(final_trust_score)))

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

        # Check whether contradicting evidence exceeds supporting evidence by defined threshold
        contra_exceeds_threshold = (
            contradicting_evidence_count >= contradiction_threshold and
            (contradicting_evidence_count - supporting_evidence_count) >= contradiction_threshold
        )

        # -----------------------------------------------------------------
        # Genre-Specific Handling (Feature, Opinion, Historical Analysis)
        # -----------------------------------------------------------------
        # Feature articles and opinion pieces are not treated as breaking-news claims.
        # Historical / analysis articles receive Source Quality Assessment.
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

        elif article_type == "Opinion Piece":
            final_trust_score = round(
                (0.35 * src_s) +
                (0.35 * lan_s) +
                (0.15 * con_s) +
                (0.15 * ml_s)
            )
            final_trust_score = max(20, min(98, int(final_trust_score)))
            verdict = "OPINION / COMMENTARY"
            risk_level = "LOW" if src_s >= 65 and lan_s >= 65 else "MEDIUM"
            summary_reason = "Evaluated as an Opinion Piece (Subjective Viewpoint / Editorial). Subjective perspectives and rhetorical arguments are not evaluated as factual breaking-news claims."

        elif article_type == "Feature Article":
            final_trust_score = round(
                (0.35 * src_s) +
                (0.30 * lan_s) +
                (0.20 * con_s) +
                (0.15 * ml_s)
            )
            final_trust_score = max(25, min(98, int(final_trust_score)))
            verdict = "FEATURE ARTICLE - CREDIBLE" if (src_s >= 75 and lan_s >= 65) else "FEATURE ARTICLE"
            risk_level = "LOW" if src_s >= 70 else "MEDIUM"
            summary_reason = "Evaluated as a Feature Article (Narrative / Human-Interest Story). Evaluated for thematic depth, narrative consistency, and source credibility rather than daily breaking news wire corroboration."

        else:
            # Standard News Report factual verification
            # 1. KNOWN FALSE:
            #    - Only if Google Fact Check returns a false/debunked rating.
            #    - Or contradicting evidence exceeds supporting evidence by defined threshold.
            if has_false_fact_check or contra_exceeds_threshold:
                verdict = "KNOWN FALSE"
                risk_level = "CRITICAL"

            # 2. LIKELY FALSE:
            #    - Some contradicting evidence exists but not enough for a definitive false verdict.
            elif contradicting_evidence_count > 0 and supporting_evidence_count == 0:
                verdict = "LIKELY FALSE"
                risk_level = "HIGH"

            # 3. Mixed evidence -> NEEDS REVIEW:
            #    - Both supporting and contradicting citations exist, neither meeting KNOWN FALSE threshold.
            elif supporting_evidence_count > 0 and contradicting_evidence_count > 0:
                verdict = "NEEDS REVIEW"
                risk_level = "MEDIUM"

            # 4. UNVERIFIED:
            #    - No reliable supporting or contradicting citations attached.
            #    - Evidence score below threshold due to missing citations.
            #    - Missing evidence must NEVER be treated as evidence of falsity.
            elif supporting_evidence_count == 0 and contradicting_evidence_count == 0:
                verdict = "UNVERIFIED"
                risk_level = "MEDIUM"
            elif evidence_score < evidence_threshold and supporting_evidence_count < 2:
                verdict = "UNVERIFIED"
                risk_level = "MEDIUM"

            # 5. Strong corroboration -> LIKELY TRUE or VERIFIED
            elif final_trust_score >= 88 and supporting_evidence_count >= 2 and contradicting_evidence_count == 0 and src_s >= 80:
                verdict = "VERIFIED"
                risk_level = "LOW"
            elif final_trust_score >= 70 and supporting_evidence_count >= 1:
                verdict = "LIKELY TRUE"
                risk_level = "LOW"
            else:
                verdict = "NEEDS REVIEW"
                risk_level = "MEDIUM"

            # Construct primary deliberation reason
            reason_parts = []
            if has_false_fact_check:
                reason_parts.append("flagged false/debunked by Google Fact Check")

            if src_s >= 85:
                reason_parts.append("trusted source")
            elif src_s <= 35:
                reason_parts.append("low-credibility publisher")
                
            if supporting_evidence_count >= 2:
                reason_parts.append(f"multiple supporting claims ({supporting_evidence_count} corroborations)")
            elif contradicting_evidence_count >= 2:
                reason_parts.append(f"flagged contradicting claims ({contradicting_evidence_count} counter-evidences)")
                
            if lan_s >= 80:
                reason_parts.append("objective journalistic language")
            elif lan_s <= 40:
                reason_parts.append("sensationalist / clickbait styling")

            if supporting_evidence_count == 0 and contradicting_evidence_count == 0:
                reason_parts.append("no reliable supporting or contradicting citations attached")
            elif contradicting_evidence_count == 0:
                reason_parts.append("no significant contradicting evidence detected")
            else:
                reason_parts.append(f"{contradicting_evidence_count} contradiction(s) registered")

            summary_reason = ", ".join(reason_parts).capitalize() + "."

        return {
            "finalTrustScore": final_trust_score,
            "verdict": verdict,
            "riskLevel": risk_level,
            "summaryReason": summary_reason,
            "scoringFactors": {
                "mlPrediction": {
                    "weight": 0.30,
                    "score": int(round(ml_s)),
                    "contribution": round(0.30 * ml_s, 1),
                    "label": "ML Text Pattern Classifier"
                },
                "sourceReputation": {
                    "weight": 0.25,
                    "score": int(round(src_s)),
                    "contribution": round(0.25 * src_s, 1),
                    "label": "Domain & Source Authority"
                },
                "evidenceStrength": {
                    "weight": 0.25,
                    "score": int(round(evi_s)),
                    "contribution": round(0.25 * evi_s, 1),
                    "label": "Corroborated Evidence Network"
                },
                "claimConsistency": {
                    "weight": 0.10,
                    "score": int(round(con_s)),
                    "contribution": round(0.10 * con_s, 1),
                    "label": "Factual Claim Consistency",
                    "notes": consistency_notes
                },
                "languageQuality": {
                    "weight": 0.10,
                    "score": int(round(lan_s)),
                    "contribution": round(0.10 * lan_s, 1),
                    "label": "Journalistic Language Integrity",
                    "notes": language_notes
                }
            }
        }

TupleScore = tuple

scoring_engine = MultiFactorScoringEngine()

def get_scoring_engine() -> MultiFactorScoringEngine:
    return scoring_engine
