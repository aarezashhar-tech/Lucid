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
from datetime import datetime, date
from utils.url_extractor import is_real_article_url

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
        source_quality_assessment: Dict[str, Any] = None,
        source_tier: str = "",
        source_category: str = "",
        top_supporting: List[Dict[str, Any]] = None,
        top_contradicting: List[Dict[str, Any]] = None
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
        # 0. SATIRICAL CONTENT (Requirement 6)
        # When domain or source is known satire, return "Satire" instead of a normal trust score.
        # -----------------------------------------------------------------
        is_satire = (
            source_tier == "Satire"
            or source_category == "Satirical Publication"
            or article_type == "Satire"
            or (source_quality_assessment and (
                source_quality_assessment.get("tier") == "Satire"
                or "Satire" in str(source_quality_assessment.get("sourceAuthority", ""))
                or "Satire" in str(source_quality_assessment.get("category", ""))
            ))
        )
        if is_satire:
            return {
                "finalTrustScore": "Satire",
                "verdict": "Satire",
                "riskLevel": "LOW",
                "summaryReason": "Satirical content — published by a known satire or parody publication and not intended as factual news.",
                "scoringFactors": {
                    "evidenceStrength": {
                        "weight": 0.45,
                        "score": "Satire",
                        "contribution": 0.0,
                        "label": "Evidence Strength (45%) — Satirical publication"
                    },
                    "mlPrediction": {
                        "weight": 0.20,
                        "score": "Satire",
                        "contribution": 0.0,
                        "label": "ML Pattern Classifier (20%) — Satirical publication"
                    },
                    "languageQuality": {
                        "weight": 0.15,
                        "score": int(round(lan_s)),
                        "contribution": 0.0,
                        "label": "Language Tone & Humor (15%)",
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

        # -----------------------------------------------------------------
        # 3. WEIGHTED EVIDENCE SCORING & VERDICT ASSIGNMENT (Requirements 4, 7, 8)
        # Weight = relevance x recency x domain_trust
        # NEEDS REVIEW only when both sides are strong and close.
        # If support clearly dominates, verdict is VERIFIED or LIKELY TRUE even with weak contradiction.
        # Hard caps: score not forced to 38 for NEEDS REVIEW (kept in sensible band 40-60).
        # Outdated: 6-month downgrade ONLY when claim is time-sensitive AND all supporting sources are old.
        # -----------------------------------------------------------------
        def _calc_source_weight(item: Dict[str, Any]) -> float:
            u = item.get("url", "")
            if not is_real_article_url(u):
                return 0.0

            # 1. Relevance (0.1 - 1.0)
            rel_score = item.get("relevanceScore")
            if rel_score is not None:
                rel = max(0.1, min(1.0, float(rel_score) / 100.0))
            else:
                rel = max(0.1, min(1.0, float(item.get("semanticSimilarity", 0.6))))
            stance_conf = item.get("stanceConfidence")
            if stance_conf is not None:
                rel = 0.5 * rel + 0.5 * max(0.1, min(1.0, float(stance_conf) / 100.0))

            # 2. Recency (0.4 - 1.0)
            pub_date = str(item.get("publicationDate") or item.get("publicationTime") or item.get("review_date") or "").strip()[:10]
            recency = 0.75
            if pub_date:
                try:
                    dt = datetime.strptime(pub_date, "%Y-%m-%d").date()
                    delta_days = (date.today() - dt).days
                    if delta_days <= 90:
                        recency = 1.0
                    elif delta_days <= 180:
                        recency = 0.90
                    elif delta_days <= 365:
                        recency = 0.75
                    elif delta_days <= 730:
                        recency = 0.60
                    else:
                        recency = 0.40
                except Exception:
                    recency = 0.75

            # 3. Domain Trust (0.1 - 1.0)
            trust_val = item.get("sourceTrustScore") or item.get("trust") or 50
            domain_trust = max(0.1, min(1.0, float(trust_val) / 100.0))

            return round(rel * recency * domain_trust, 4)

        # Collect verified supporting and contradicting items
        supp_items: List[Dict[str, Any]] = []
        contra_items: List[Dict[str, Any]] = []

        if top_supporting:
            supp_items.extend([it for it in top_supporting if is_real_article_url(it.get("url", ""))])
        if top_contradicting:
            contra_items.extend([it for it in top_contradicting if is_real_article_url(it.get("url", ""))])

        if not supp_items and claims:
            for c in claims:
                for snip in c.get("evidenceSnippets", []):
                    u = snip.get("url", "")
                    if is_real_article_url(u):
                        if snip.get("stance") == "support":
                            supp_items.append(snip)
                        elif snip.get("stance") == "contradict":
                            contra_items.append(snip)

        total_support_weight = sum(_calc_source_weight(s) for s in supp_items)
        total_contradict_weight = sum(_calc_source_weight(c) for c in contra_items)

        # Both sides are strong and close (balanced conflict)
        both_sides_strong = (total_support_weight >= 0.20 and total_contradict_weight >= 0.20)
        weights_close = False
        if both_sides_strong:
            ratio = total_support_weight / max(0.001, total_contradict_weight)
            weights_close = (0.5 <= ratio <= 2.0)

        is_needs_review = both_sides_strong and weights_close
        support_clearly_dominates = (total_support_weight >= 0.45 and (total_contradict_weight == 0 or total_support_weight >= 2.0 * total_contradict_weight))
        contradict_clearly_dominates = (total_contradict_weight >= 0.45 and (total_support_weight == 0 or total_contradict_weight >= 2.0 * total_support_weight))

        if has_false_fact_check:
            verdict = "KNOWN FALSE"
            final_trust_score = min(final_trust_score, 12)
            risk_level = "CRITICAL"
            summary_reason = "Conclusively disproven. Flagged as false by certified fact-check organization."

        elif contradict_clearly_dominates:
            if total_contradict_weight >= 0.8:
                verdict = "KNOWN FALSE"
                final_trust_score = min(final_trust_score, 12)
                risk_level = "CRITICAL"
                summary_reason = f"Conclusively disproven. Contradicting evidence strongly dominates (Support: {round(total_support_weight, 2)} vs Contradict: {round(total_contradict_weight, 2)} weight)."
            else:
                verdict = "LIKELY FALSE"
                final_trust_score = min(final_trust_score, 35)
                risk_level = "HIGH"
                summary_reason = f"Likely inaccurate. Retrieved evidence disputes the claim (Contradict: {round(total_contradict_weight, 2)} vs Support: {round(total_support_weight, 2)} weight)."

        elif is_needs_review:
            verdict = "NEEDS REVIEW"
            risk_level = "MEDIUM"
            # Requirement 7: Do not force score to 38. Keep within sensible band [40, 60]
            final_trust_score = max(40, min(60, int(computed_score)))
            summary_reason = f"Mixed evidence detected across wire sources. Conflicting evidence exists on both sides (Support weight: {round(total_support_weight, 2)} vs Contradict weight: {round(total_contradict_weight, 2)})."

        elif support_clearly_dominates:
            if total_support_weight >= 1.0 and len(supp_items) >= 2:
                verdict = "VERIFIED"
                # Requirement 6: Do not floor at 78. Let weighted score show within sensible band [70, 98]
                final_trust_score = max(70, min(98, int(computed_score)))
                risk_level = "LOW"
                summary_reason = f"Corroborated across reliable wire sources (Support weight: {round(total_support_weight, 2)}) with support clearly dominating."
            else:
                verdict = "LIKELY TRUE"
                # Requirement 6: Do not floor at 68. Let weighted score show within sensible band [60, 85]
                final_trust_score = max(60, min(85, int(computed_score)))
                risk_level = "LOW"
                summary_reason = f"Supported by reputable reporting (Support weight: {round(total_support_weight, 2)}) without substantial contradiction."

        elif total_support_weight < 0.25 and total_contradict_weight < 0.25:
            # Requirement 2: Return UNVERIFIED with "no live evidence found"
            verdict = "UNVERIFIED"
            final_trust_score = min(50, final_trust_score)
            risk_level = "MEDIUM"
            summary_reason = "No live evidence found across news wire feeds or fact-check registries. Low confidence — unverified pending independent evidence."

        elif total_support_weight > total_contradict_weight:
            verdict = "LIKELY TRUE"
            final_trust_score = max(60, min(75, int(computed_score)))
            risk_level = "LOW"
            summary_reason = f"Supported by reporting (Support weight: {round(total_support_weight, 2)} vs Contradict: {round(total_contradict_weight, 2)})."

        else:
            verdict = "UNVERIFIED"
            final_trust_score = min(50, final_trust_score)
            risk_level = "MEDIUM"
            summary_reason = "No live evidence found across news wire feeds or fact-check registries. Low confidence — unverified pending independent evidence."

        # -----------------------------------------------------------------
        # Requirement 4: Compound claims evidence check
        # A source only counts for the sub-claim it actually addresses.
        # If one sub-claim has no evidence, the verdict must not be FALSE or LIKELY FALSE from the other part alone.
        # -----------------------------------------------------------------
        if claims and len(claims) > 1 and verdict in ("KNOWN FALSE", "LIKELY FALSE", "FALSE"):
            has_unverified_subclaim = False
            for c in claims:
                supp_c = len(c.get("supportingSources", [])) + len([s for s in (c.get("evidenceSnippets") or []) if s.get("stance") == "support"])
                contra_c = len(c.get("contradictingSources", [])) + len([s for s in (c.get("evidenceSnippets") or []) if s.get("stance") == "contradict"])
                if supp_c == 0 and contra_c == 0:
                    has_unverified_subclaim = True
                    break

            if has_unverified_subclaim:
                verdict = "NEEDS REVIEW"
                risk_level = "MEDIUM"
                final_trust_score = max(35, min(55, int(computed_score)))
                summary_reason = "Compound claim contains mixed verification status: one part is contradicted, but another sub-claim has no live evidence found and cannot be judged false from the other part alone."

        # -----------------------------------------------------------------
        # Requirement 8: Outdated evidence rule
        # Apply the 6-month downgrade only when the claim is time-sensitive
        # AND all supporting sources are old. Recent supporting sources prevent the downgrade.
        # -----------------------------------------------------------------
        is_time_sensitive_claim = any(
            c.get("isTimeSensitive") or c.get("isOutdated") or c.get("status") == "May be outdated"
            for c in (claims or [])
        )
        if is_time_sensitive_claim and supp_items:
            has_recent_supporting = False
            for s in supp_items:
                p_date = str(s.get("publicationDate") or s.get("publicationTime") or "").strip()[:10]
                if p_date:
                    try:
                        dt = datetime.strptime(p_date, "%Y-%m-%d").date()
                        if (date.today() - dt).days <= 180:
                            has_recent_supporting = True
                            break
                    except Exception:
                        pass

            if not has_recent_supporting:
                final_trust_score = min(final_trust_score, 60)
                if verdict in ("VERIFIED", "LIKELY TRUE"):
                    verdict = "NEEDS REVIEW"
                    risk_level = "MEDIUM"
                summary_reason += " All supporting evidence is older than 6 months for a time-sensitive claim and may be outdated. Trust score capped at 60."

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
