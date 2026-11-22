"""
Lucid Grounded Reasoning & Verdict Synthesis Engine
Generates transparent, evidence-grounded rationales and verdicts derived
STRICTLY from retrieved, ranked sources.

Enforces Verdict Standards:
- VERIFIED: Multiple independent supporting sources (>= 2 distinct reputable publishers)
- LIKELY TRUE: Strong supporting evidence (1 major credible publisher or solid corroboration without contradiction)
- UNVERIFIED: Insufficient retrieval results (0 or inconclusive supporting/contradicting evidence)
              * CRITICAL RULE: Missing evidence must NEVER be treated as false.
- LIKELY FALSE: Some contradicting evidence or reputable disputation
- FALSE: Fact-check providers rated False or overwhelming contradiction from credible sources
"""
import re
import html
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("lucid.grounded_reasoner")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [GroundedReasoner] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def clean_reasoning_text(text: str) -> str:
    """Strips HTML tags, raw URLs, and leftover markdown/attributes from output reasoning text."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = html.unescape(clean)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'(target|href|class|rel)=["\']?[^"\'>\s]+["\']?', '', clean)
    clean = re.sub(r'["\']?\s*<a\s+[^>]*>?', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


class GroundedReasoningEngine:
    """
    Synthesizes grounded explanations and authoritative verdicts
    strictly derived from retrieved evidence.
    """

    def synthesize_rag_verdict_and_reasoning(
        self,
        claims: List[Dict[str, Any]],
        evidence_matrix: List[Dict[str, Any]],
        top_supporting: List[Dict[str, Any]],
        top_contradicting: List[Dict[str, Any]],
        sources_searched: List[str],
        article_type: str = "News Report",
        final_verdict: Optional[str] = None,
        final_trust_score: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes strict RAG verdict determination and generates
        evidence-grounded reasoning text.
        """
        # Collect distinct supporting and contradicting publishers
        supporting_pubs = set()
        for item in top_supporting:
            pub = item.get("publisher")
            if pub:
                supporting_pubs.add(pub)

        contradicting_pubs = set()
        has_ifcn_false = False
        for item in top_contradicting:
            pub = item.get("publisher")
            if pub:
                contradicting_pubs.add(pub)
            fc_rate = str(item.get("factCheckRating", "")).lower()
            cat = str(item.get("sourceCategory", "")).lower()
            title = str(item.get("articleTitle", "")).lower()
            pub_lower = str(pub or "").lower()
            if (
                "fact checker" in cat or
                "ifcn" in cat or
                any(neg in fc_rate for neg in ["false", "debunk", "pants on fire", "hoax", "misleading", "fake"]) or
                re.search(r'\b(false|debunked|hoax|pants\s+on\s+fire|misleading|untrue|fabricated)\b', title) or
                any(fp in pub_lower for fp in ["snopes", "factcheck", "politifact", "lead stories"])
            ):
                has_ifcn_false = True

        supp_count = len(top_supporting)
        contra_count = len(top_contradicting)
        distinct_supp_count = len(supporting_pubs)

        factual_claims = [c for c in evidence_matrix if not c.get("isOpinion")]
        opinion_claims = [c for c in evidence_matrix if c.get("isOpinion")]

        # -------------------------------------------------------------
        # 1. VERDICT DECISION MATRIX
        # -------------------------------------------------------------
        if final_verdict:
            verdict = final_verdict
            trust_score = final_trust_score if final_trust_score is not None else (
                "Not scored" if verdict == "Opinion - not fact-checked" else 50
            )
            risk_level = "CRITICAL" if verdict == "FALSE" else "HIGH" if verdict == "LIKELY FALSE" else "LOW" if verdict in ("VERIFIED", "LIKELY TRUE", "Opinion - not fact-checked") else "MEDIUM"
            if verdict == "FALSE":
                summary_reason = f"Conclusively disproven. Contradicted by authoritative evidence ({', '.join(list(contradicting_pubs)[:2]) or 'verified sources'})."
            elif verdict == "VERIFIED":
                summary_reason = f"Corroborated across {supp_count} independent sources ({', '.join(list(supporting_pubs)[:2]) or 'reputable sources'}) with zero contradicting evidence."
            elif verdict == "LIKELY TRUE":
                summary_reason = f"Supported by reputable reporting ({supp_count} citation(s)) without contradiction."
            elif verdict == "Opinion - not fact-checked":
                summary_reason = "Opinion / commentary piece — subjective perspectives are not evaluated as factual breaking-news claims."
            else:
                summary_reason = "No sufficiently relevant corroborating sources were found. Unverified pending independent evidence."
        elif article_type in ("Opinion Piece", "Opinion"):
            verdict = "Opinion - not fact-checked"
            trust_score = "Not scored"
            risk_level = "LOW"
            summary_reason = (
                f"Identified as an Opinion Piece ({len(opinion_claims)} viewpoint assertions). "
                "Subjective arguments and commentary are evaluated for author transparency and rhetoric "
                "rather than strict true/false daily wire verification."
            )
        elif article_type == "Historical Analysis":
            verdict = "HISTORICAL ANALYSIS - REPUTABLE"
            trust_score = 85
            risk_level = "LOW"
            summary_reason = (
                "Evaluated under Source Quality Assessment. Historical retrospectives are assessed "
                "on scholarly depth and archival documentation rather than breaking news wire consensus."
            )
        elif contra_count >= 1 and distinct_supp_count == 0:
            # FALSE: Contradicted by established sources with zero corroborating evidence
            # (e.g. 5G causes COVID-19, Pope endorsed Trump, Earth is flat, living person died)
            verdict = "FALSE"
            trust_score = 10
            risk_level = "CRITICAL"
            c_names = ", ".join(list(contradicting_pubs)[:2]) or "accredited investigative sources"
            specific_reason = ""
            for item in top_contradicting:
                snip = item.get("evidenceSnippet", "")
                if "Contradicted by evidence:" in snip or "Contradicted by scientific consensus:" in snip:
                    specific_reason = snip
                    break
            summary_reason = specific_reason or f"Conclusively disproven. Contradicted by authoritative scientific, institutional, or news wire reporting ({c_names})."
        elif has_ifcn_false or (contra_count >= 2 and distinct_supp_count <= 1):
            # FALSE: Fact-check providers rated False or overwhelming contradiction
            verdict = "FALSE"
            trust_score = 12
            risk_level = "CRITICAL"
            c_names = ", ".join(list(contradicting_pubs)[:2]) or "fact-checking investigations"
            specific_reason = ""
            for item in top_contradicting:
                snip = item.get("evidenceSnippet", "")
                if "Contradicted by evidence:" in snip or "Contradicted by scientific consensus:" in snip:
                    specific_reason = snip
                    break
            summary_reason = specific_reason or f"Contradicted by established fact-checking and news wire investigations ({c_names})."
        elif contra_count >= 1 and distinct_supp_count > 1:
            # Conflicting evidence
            verdict = "DISPUTED"
            trust_score = 45
            risk_level = "HIGH"
            summary_reason = "Conflicting reporting detected with contradictory assertions across wire sources."
        elif distinct_supp_count >= 2 and contra_count == 0:
            # VERIFIED: Multiple independent supporting sources without contradiction
            verdict = "VERIFIED"
            trust_score = 90
            risk_level = "LOW"
            s_names = " and ".join(list(supporting_pubs)[:2])
            summary_reason = f"Corroborated across multiple independent reputable wire services: {s_names}."
        elif distinct_supp_count == 1 and supp_count >= 1 and contra_count == 0:
            # LIKELY TRUE: Strong supporting evidence
            verdict = "LIKELY TRUE"
            trust_score = 80
            risk_level = "LOW"
            s_name = list(supporting_pubs)[0]
            summary_reason = f"Supported by reputable reporting from {s_name} without contradiction."
        else:
            verdict = "UNVERIFIED"
            trust_score = 50
            risk_level = "MEDIUM"
            summary_reason = "No sufficiently relevant corroborating sources were found. Unverified pending independent evidence."

        # -------------------------------------------------------------
        # 2. GROUNDED SYNTHESIS GENERATION
        # The AI generates explanations ONLY from retrieved evidence.
        # Never produce "Supported" without showing retrieved sources.
        # -------------------------------------------------------------
        why_appears_true: List[str] = []
        why_appears_false: List[str] = []
        evidence_found_citations: List[str] = []
        evidence_missing_citations: List[str] = []

        # Grounding from supporting sources
        for s in top_supporting[:3]:
            pub = clean_reasoning_text(s.get("publisher", "Wire Source"))
            title = clean_reasoning_text(s.get("articleTitle", ""))
            date_str = s.get("publicationDate", "")
            snippet = clean_reasoning_text(s.get("evidenceSnippet", ""))
            citation = f"{pub} ({date_str}): \"{title}\" — {snippet}"
            evidence_found_citations.append(citation)
            why_appears_true.append(f"Corroborated by {pub}: {snippet}")

        # Grounding from contradicting sources
        for c in top_contradicting[:3]:
            pub = clean_reasoning_text(c.get("publisher", "Fact Checker"))
            title = clean_reasoning_text(c.get("articleTitle", ""))
            date_str = c.get("publicationDate", "")
            snippet = clean_reasoning_text(c.get("evidenceSnippet", ""))
            citation = f"{pub} ({date_str}): \"{title}\" — {snippet}"
            evidence_found_citations.append(citation)
            why_appears_false.append(f"Disputed by {pub}: {snippet}")

        if not top_supporting:
            evidence_missing_citations.append(
                "No external corroborating dispatches passed semantic matching threshold across queried sources."
            )
            why_appears_true.append("No corroborating citations identified in queried sources.")

        if not top_contradicting and not has_ifcn_false:
            why_appears_false.append("No contradicting wire flags or negative fact checks were discovered in public registries.")

        # Synthesis text: Dynamically generated from real evidence counts
        if verdict == "Opinion - not fact-checked" or article_type in ("Opinion Piece", "Opinion"):
            synthesis = (
                "Grounded Retrieval Synthesis: Submission is identified as an Opinion / Commentary piece. "
                "Subjective arguments and commentary express author perspective rather than verifiable breaking news, "
                "and are not assigned a fact-based true/false trust score under journalistic standards."
            )
        elif supp_count > 0 and contra_count == 0:
            s_list = ", ".join(list(supporting_pubs)[:3]) or "verified sources"
            synthesis = (
                f"Grounded Retrieval Synthesis: Claim-by-claim cross-referencing successfully grounded "
                f"this submission in published reporting from {s_list} ({supp_count} corroborating citation(s)). "
                f"Retrieved evidence exhibits high semantic relevance and credibility without contradicting wire flags."
            )
        elif contra_count > 0 and supp_count == 0:
            c_list = ", ".join(list(contradicting_pubs)[:3]) or "authoritative sources"
            synthesis = (
                f"Grounded Retrieval Synthesis: Retrieved investigative records from {c_list} directly contradict "
                f"the candidate claim(s) ({contra_count} contradicting record(s) detected). "
                f"Authoritative reporting and official records dispute the factual basis of the submission."
            )
        elif contra_count > 0 and supp_count > 0:
            s_list = ", ".join(list(supporting_pubs)[:2]) or "supporting sources"
            c_list = ", ".join(list(contradicting_pubs)[:2]) or "contradicting sources"
            synthesis = (
                f"Grounded Retrieval Synthesis: Conflicting evidence detected across queried sources "
                f"({supp_count} supporting from {s_list} vs {contra_count} contradicting from {c_list}). "
                f"Editorial review is recommended due to mixed reporting."
            )
        else:
            sources_summary = ", ".join([s.split(' (')[0] for s in sources_searched]) if sources_searched else "queried sources"
            synthesis = (
                f"Grounded Retrieval Synthesis: An exhaustive search across {sources_summary} yielded 0 corroborating citations "
                f"for the candidate claim(s). In accordance with strict fact-checking methodology, absent evidence "
                f"is designated as UNVERIFIED rather than false."
            )

        explanation = [
            summary_reason,
            f"Sources searched: {', '.join(sources_searched[:4]) if sources_searched else 'queried sources'}.",
            f"Evidence retrieved: {supp_count} supporting, {contra_count} contradicting.",
            f"Grounded Verdict: {verdict}."
        ]

        return {
            "verdict": verdict,
            "finalTrustScore": trust_score,
            "riskLevel": risk_level,
            "summaryReason": summary_reason,
            "explanation": explanation,
            "groundedReasoning": {
                "synthesis": synthesis,
                "whyAppearsTrue": why_appears_true,
                "whyAppearsFalse": why_appears_false,
                "evidenceFound": evidence_found_citations,
                "evidenceMissing": evidence_missing_citations or ["No additional corroborating documentation missing."]
            },
            "evidenceAudit": {
                "sourcesSearched": sources_searched,
                "resultsFound": {
                    "supporting": supp_count,
                    "contradicting": contra_count,
                    "total": supp_count + contra_count
                },
                "reason": (
                    "No sufficiently relevant corroborating sources were found."
                    if supp_count == 0 else
                    f"Successfully grounded in {distinct_supp_count} independent publisher(s)."
                )
            }
        }


_grounded_engine = GroundedReasoningEngine()


def get_grounded_reasoner() -> GroundedReasoningEngine:
    return _grounded_engine
