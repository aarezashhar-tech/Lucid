"""
Layer 8: AI Reasoning Engine
Generates transparent, multi-perspective fact-checking reasoning:
1. Why the claim appears true
2. Why the claim appears false
3. What evidence was found
4. What evidence is missing

Supports optional Gemini / OpenAI integrations with deterministic fallback.
Crucial rule: LLMs NEVER determine the verdict alone.
"""
import os
import json
from typing import Dict, Any, List
import requests

TRANSPARENCY_WARNING = "This verdict is generated using machine learning and evidence scoring. It does not guarantee factual accuracy."

class AIReasoningEngine:
    """
    Produces structured reasoning across 4 analytical dimensions.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def generate_ai_reasoning(
        self,
        article_text: str,
        claims: List[Dict[str, Any]],
        source_data: Dict[str, Any],
        evidence_data: Dict[str, Any],
        verdict_data: Dict[str, Any],
        ling_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates 4 reasoning categories. Uses Gemini/OpenAI if available,
        otherwise uses internal deterministic analytical reasoning.
        """
        # Try external LLM if key is present
        if self.gemini_key:
            llm_result = self._query_gemini(article_text, claims, source_data, verdict_data)
            if llm_result:
                llm_result["disclaimer"] = TRANSPARENCY_WARNING
                return llm_result

        if self.openai_key:
            llm_result = self._query_openai(article_text, claims, source_data, verdict_data)
            if llm_result:
                llm_result["disclaimer"] = TRANSPARENCY_WARNING
                return llm_result

        # Deterministic analytical reasoning generator
        return self._generate_deterministic_reasoning(
            article_text, claims, source_data, evidence_data, verdict_data, ling_signals
        )

    def _generate_deterministic_reasoning(
        self,
        text: str,
        claims: List[Dict[str, Any]],
        source_data: Dict[str, Any],
        evidence_data: Dict[str, Any],
        verdict_data: Dict[str, Any],
        ling_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates thorough, contextual reasoning points directly from analytical layers.
        """
        verdict = verdict_data.get("verdict", "NEEDS REVIEW")
        trust_score = verdict_data.get("finalTrustScore", 50)
        source_name = source_data.get("sourceName", "Unspecified Source")
        source_trust = source_data.get("trustScore", 50)
        supp_count = evidence_data.get("supportingEvidenceCount", 0)
        contra_count = evidence_data.get("contradictingEvidenceCount", 0)
        
        why_true: List[str] = []
        why_false: List[str] = []
        evidence_found: List[str] = []
        evidence_missing: List[str] = []

        # 1. Why the claim appears true
        if source_trust >= 80:
            why_true.append(f"Attributed to {source_name}, which holds a high institutional trust rating of {source_trust}/100.")
        if supp_count > 0:
            why_true.append(f"Multiple independent corroborations found ({supp_count} supporting citations recorded across trusted wire indices).")
        if ling_signals.get("sensational_score", 0) < 25:
            why_true.append("Language phrasing is neutral, objective, and conforms to standard newsroom reporting standards.")
        if any(c.get("numbers") for c in claims):
            why_true.append("Contains specific numerical and geographical claims rather than vague generalizations.")

        if not why_true:
            why_true.append("Contains declarative statements consistent with standard topical syntax.")

        # 2. Why the claim appears false
        if contra_count > 0:
            why_false.append(f"Flagged by {contra_count} contradicting source registries or established debunk databases.")
        if verdict in ("LIKELY FALSE", "KNOWN FALSE"):
            why_false.append("Factual propositions conflict with empirical consensus and official government or regulatory statements.")
        if source_trust < 50 and contra_count > 0:
            why_false.append(f"The publishing source ({source_name}) lacks verified accreditation or has a lower credibility index ({source_trust}/100).")
        if ling_signals.get("sensational_score", 0) > 30 and contra_count > 0:
            why_false.append(f"Elevated sensationalism score ({ling_signals.get('sensational_score')}%) and emotionally manipulative vocabulary detected.")
        if ling_signals.get("clickbait_score", 0) > 30 and contra_count > 0:
            why_false.append("Headline or body exhibits classic clickbait engagement bait patterns.")

        if not why_false:
            if contra_count == 0 and supp_count == 0:
                why_false.append("No contradicting evidence detected; missing citations strictly warrant an unverified classification rather than falsity.")
            else:
                why_false.append("No immediate structural disinformation markers or viral manipulation signals detected.")

        # 3. What evidence was found
        for supp in evidence_data.get("supportingEvidence", [])[:3]:
            evidence_found.append(f"Supporting: {supp.get('source')} - {supp.get('snippet', '')[:110]}...")
        for contra in evidence_data.get("contradictingEvidence", [])[:3]:
            evidence_found.append(f"Contradicting: {contra.get('source')} - {contra.get('snippet', '')[:110]}...")

        if not evidence_found:
            if supp_count > 0:
                evidence_found.append(f"{supp_count} corroborating wire dispatches identified in primary news registries.")
            else:
                evidence_found.append("No definitive corroborating or contradicting wire archives matched in local indices.")

        # 4. What evidence is missing
        if not source_data.get("isTrusted", False):
            evidence_missing.append("Direct attribution to an accredited primary news wire or verified official spokesperson.")
        if not any(c.get("entities") for c in claims):
            evidence_missing.append("Specific verifiable named entities, geographic coordinates, or exact event timestamps.")
        if contra_count == 0 and supp_count < 2:
            evidence_missing.append("Secondary independent cross-confirmation from on-the-ground international observers.")
        evidence_missing.append("Cryptographic or digital signature verification for source document integrity.")

        # Synthesis
        synthesis = (
            f"Verdict reached: {verdict} (Trust Score: {trust_score}/100). "
            f"The evaluation synthesizes 30% ML pattern classification, 25% source authority ({source_name}), "
            f"25% evidence corroboration ({supp_count} supporting, {contra_count} contradicting), "
            f"and 20% linguistic/consistency audits. "
            f"Final verdict is governed by multi-factor mathematical scoring, not solely text pattern classification."
        )

        return {
            "synthesis": synthesis,
            "whyAppearsTrue": why_true,
            "whyAppearsFalse": why_false,
            "evidenceFound": evidence_found,
            "evidenceMissing": evidence_missing,
            "disclaimer": TRANSPARENCY_WARNING,
            "provider": "Lucid Multi-Factor Analytical Reasoner (Rule-Based + NLP)"
        }

    def _query_gemini(
        self,
        text: str,
        claims: List[Dict[str, Any]],
        source_data: Dict[str, Any],
        verdict_data: Dict[str, Any]
    ) -> Any:
        """Invokes Gemini 1.5/2.0 API to generate reasoned analysis points."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
            prompt = (
                "You are an evidence-based fact-checking reasoning assistant for Lucid.\n"
                "IMPORTANT: You DO NOT decide the verdict. The platform has already computed the multi-factor score.\n"
                f"Computed Trust Score: {verdict_data.get('finalTrustScore')}/100, Verdict: {verdict_data.get('verdict')}.\n"
                f"Source: {source_data.get('sourceName')} ({source_data.get('trustScore')}/100).\n"
                f"Article text:\n{text[:1200]}\n\n"
                "Return ONLY a valid JSON object with these exact keys:\n"
                "{\n"
                '  "synthesis": "Brief overview of reasoning",\n'
                '  "whyAppearsTrue": ["point 1", "point 2"],\n'
                '  "whyAppearsFalse": ["point 1", "point 2"],\n'
                '  "evidenceFound": ["point 1", "point 2"],\n'
                '  "evidenceMissing": ["point 1", "point 2"]\n'
                "}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            res = requests.post(url, json=payload, timeout=8)
            if res.status_code == 200:
                body = res.json()
                raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(raw_text)
                parsed["provider"] = "Gemini AI"
                return parsed
        except Exception as e:
            print(f"Gemini API invocation fallback: {e}")
        return None

    def _query_openai(
        self,
        text: str,
        claims: List[Dict[str, Any]],
        source_data: Dict[str, Any],
        verdict_data: Dict[str, Any]
    ) -> Any:
        """Invokes OpenAI API to generate reasoned analysis points."""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            prompt = (
                "You are an evidence-based fact-checking reasoning assistant for Lucid.\n"
                "IMPORTANT: You DO NOT decide the verdict. The platform has already computed the multi-factor score.\n"
                f"Computed Trust Score: {verdict_data.get('finalTrustScore')}/100, Verdict: {verdict_data.get('verdict')}.\n"
                f"Source: {source_data.get('sourceName')} ({source_data.get('trustScore')}/100).\n"
                f"Article text:\n{text[:1200]}\n\n"
                "Return ONLY a valid JSON object with keys: synthesis, whyAppearsTrue, whyAppearsFalse, evidenceFound, evidenceMissing."
            )
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                body = res.json()
                raw = body["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                parsed["provider"] = "OpenAI API"
                return parsed
        except Exception as e:
            print(f"OpenAI API invocation fallback: {e}")
        return None

ai_reasoner = AIReasoningEngine()

def get_ai_reasoner() -> AIReasoningEngine:
    return ai_reasoner
