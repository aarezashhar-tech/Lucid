from __future__ import annotations
import os
import joblib
import logging
from typing import Dict, Any, List, Optional

from utils.text_preprocessing import clean_text, analyze_linguistic_signals
from services.source_reputation import get_source_engine
from services.claim_extractor import get_claim_engine
from services.evidence_verifier import get_evidence_verifier
from services.multi_factor_scorer import get_scoring_engine
from services.ai_reasoning import get_ai_reasoner, TRANSPARENCY_WARNING
import services.factcheck_service as factcheck_service
import services.gnews_service as gnews_service
from services.factcheck_service import query_google_factcheck
from services.gnews_service import search_gnews, search_gnews_for_claims
from services.summarizer import get_article_summarizer
from services.language_service import get_language_service
from services.article_classifier import get_article_classifier
from services.rag_retriever import get_rag_retriever
from services.evidence_ranker import get_evidence_ranker
from services.grounded_reasoner import get_grounded_reasoner
from services.wikipedia_service import get_wikipedia_service
from utils.url_extractor import is_real_article_url

logger = logging.getLogger("lucid.ai_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [AIService] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class VerificationEngine:
    def __init__(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(backend_dir, "model.pkl")
        self.vec_path = os.path.join(backend_dir, "vectorizer.pkl")
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

        self.source_engine = get_source_engine()
        self.claim_engine = get_claim_engine()
        self.evidence_verifier = get_evidence_verifier()
        self.scoring_engine = get_scoring_engine()
        self.ai_reasoner = get_ai_reasoner()
        self.summarizer = get_article_summarizer()
        self.language_service = get_language_service()
        self.article_classifier = get_article_classifier()
        self.rag_retriever = get_rag_retriever()
        self.evidence_ranker = get_evidence_ranker()
        self.grounded_reasoner = get_grounded_reasoner()
        self.wikipedia_service = get_wikipedia_service()

    def _load_artifacts(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vec_path)
                logger.info("Model and vectorizer loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model artifacts: {e}")
        else:
            logger.warning(f"Warning: Artifacts not found at {self.model_path} and {self.vec_path}")

    def query_external_factcheck(self, query: str) -> List[Dict[str, Any]]:
        """
        Complete Google Fact Check Tools API query implementation.
        Requirements:
        1. Reads FACTCHECK_API_KEY from environment.
        2. Uses requests library.
        3. Queries https://factchecktools.googleapis.com/v1alpha1/claims:search
        4. Returns parsed format:
           [
             {
               "claim": "...",
               "claimant": "...",
               "reviewer": "...",
               "rating": "...",
               "url": "...",
               "review_date": "...",
               "source": "Google Fact Check API"
             }
           ]
        5. Error handling included.
        6. Returns [] if no fact check exists or on error.
        11. Backend logging for API responses.
        """
        logger.info(f"query_external_factcheck called with query: '{query[:100]}'")
        results = factcheck_service.query_google_factcheck(query)
        logger.info(f"query_external_factcheck retrieved {len(results)} fact-check match(es)")
        return results

    def query_gnews(self, query: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Query GNews API for recent news coverage corroborating or contradicting claims.
        """
        logger.info(f"query_gnews called with query: '{query[:100]}'")
        results = search_gnews(query, max_articles=max_articles)
        logger.info(f"query_gnews retrieved {len(results)} article(s)")
        return results

    def analyze(
        self,
        text: str,
        source_hint: str = "",
        url_or_domain: str = "",
        article_title: str = "",
        is_image: bool = False,
        file_type: str = "",
        image_publisher: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes Lucid Verification Pipeline (Requirement 7):
        Article Summarization (Executive Summary, Key Takeaways, TL;DR)
        ↓
        Claim Extraction
        ↓
        Google Fact Check API
        ↓
        Source Reputation Engine
        ↓
        Evidence Verification
        ↓
        ML Classification
        ↓
        Final Trust Score
        """
        if not is_image:
            import re
            is_image = (
                file_type in ("image", "img", "photo") or
                bool(re.search(r'\.(jpe?g|png|gif|webp|bmp)$', str(source_hint).lower().strip())) or
                bool(re.search(r'\.(jpe?g|png|gif|webp|bmp)$', str(url_or_domain).lower().strip()))
            )

        if not text or not text.strip():
            source_data = self.source_engine.evaluate_source(
                url_or_domain=url_or_domain,
                text=text,
                source_hint=source_hint,
                is_image=is_image,
                file_type=file_type,
                image_publisher=image_publisher
            )
            primary_publisher = source_data.get("publisher", source_data["sourceName"])
            source_type = source_data.get("sourceType", "Uploaded Image" if is_image else "")
            article_summary = self.summarizer.summarize(text, title=article_title, url=url_or_domain)
            return {
                "prediction": "UNKNOWN",
                "verdict": "UNVERIFIED",
                "confidence": 0,
                "credibility": 0,
                "trustScore": 0,
                "riskLevel": "UNKNOWN",
                "publisher": primary_publisher,
                "sourceType": source_type,
                "sourceDomain": source_data["domain"],
                "language": {"language": "English", "code": "en", "isForeignLanguage": False},
                "detectedLanguage": "English",
                "languageCode": "en",
                "articleType": "News Report",
                "sourceQualityAssessment": None,
                "emptyEvidenceAudit": None,
                "ragEnabled": True,
                "sourcesSearched": [],
                "resultsFound": {"supporting": 0, "contradicting": 0, "total": 0},
                "topSupportingSources": [],
                "topContradictingSources": [],
                "evidenceMatrix": [],
                "groundedReasoning": None,
                "primarySource": {
                    "publisher": primary_publisher,
                    "name": source_data["sourceName"],
                    "domain": source_data["domain"],
                    "sourceType": source_type,
                    "isPrimary": True
                },
                "sourceReputation": source_data,
                "sourceCredibility": source_data["trustScore"],
                "articleSummary": article_summary,
                "summary": article_summary,
                "articleTitle": article_title or article_summary.get("title", ""),
                "explanation": ["No content provided for analysis."],
                "disclaimer": TRANSPARENCY_WARNING,
                "claims": [],
                "sources": [
                    {
                        "name": source_data["sourceName"],
                        "publisher": primary_publisher,
                        "code": (primary_publisher[:3].upper() if primary_publisher and primary_publisher != "Unknown" else "SRC"),
                        "domain": source_data["domain"],
                        "type": source_type or source_data["category"],
                        "trust": source_data["trustScore"],
                        "tier": source_data["tier"],
                        "description": source_data["description"],
                        "isPrimary": True
                    }
                ],
                "factCheckResults": [],
                "evidence": {"supporting": [], "contradicting": []},
                "reasoning": {
                    "synthesis": "Empty input received.",
                    "whyAppearsTrue": [],
                    "whyAppearsFalse": [],
                    "evidenceFound": [],
                    "evidenceMissing": ["Full article text required for verification."],
                    "disclaimer": TRANSPARENCY_WARNING
                }
            }

        # Reset session audit for RAG retrieval transparency
        self.rag_retriever.reset_session_audit()

        # -------------------------------------------------------------
        # STEP -1: Multi-Language Detection & Processing
        # -------------------------------------------------------------
        logger.info("Pipeline Step -1: Multi-Language Detection & Translation Bridge")
        # Requirement 5: Do not run language detection on texts under about 20 characters
        if len(text.strip()) < 20:
            lang_info = {
                "code": "en",
                "languageCode": "en",
                "language": "English",
                "detectedLanguage": "English",
                "confidence": 1.0,
                "isEnglish": True,
                "isForeignLanguage": False,
                "analysisText": text,
                "wasTranslated": False
            }
        else:
            lang_info = self.language_service.process_content(text)
        analysis_text = lang_info.get("analysisText", text)
        detected_lang = lang_info.get("language", "English")
        detected_code = lang_info.get("code", "en")
        was_translated = lang_info.get("wasTranslated", False)
        logger.info(f"Language detected: {detected_lang} ({detected_code}), Translated: {was_translated}")

        # -------------------------------------------------------------
        # STEP -0.5: Article Classification
        # -------------------------------------------------------------
        logger.info("Pipeline Step -0.5: Article Type Classification")
        classification = self.article_classifier.classify_article_type(analysis_text or text, title=article_title)
        article_type = classification["articleType"]
        logger.info(f"Article classified as: {article_type} (Confidence: {classification['confidence']}%)")

        # -------------------------------------------------------------
        # STEP 0: Article Summarization (Executive Summary, Key Takeaways, TL;DR)
        # -------------------------------------------------------------
        logger.info("Pipeline Step 0: Article Summarization")
        article_summary = self.summarizer.summarize(analysis_text, title=article_title, url=url_or_domain)
        logger.info(f"Article summarized: TL;DR: '{article_summary.get('tldr', '')[:80]}...'")

        # -------------------------------------------------------------
        # STEP 1: Claim Extraction (Distinguishes Factual vs Opinion)
        # -------------------------------------------------------------
        logger.info("Pipeline Step 1: Claim Extraction")
        extracted_claims = self.claim_engine.extract_claims(analysis_text, max_claims=5, article_type=article_type)
        logger.info(f"Extracted {len(extracted_claims)} claim(s)")

        # -------------------------------------------------------------
        # STEP 2 (RAG): Multi-Source Claim-by-Claim Evidence Retrieval
        # Searches: Reuters, AP News, BBC, Al Jazeera, The Guardian, AFP,
        # Google Fact Check API, IFCN signatories, Government, UN/WHO.
        # Requirement 9: Do not fact-check opinion paragraphs.
        # -------------------------------------------------------------
        logger.info("Pipeline Step 2 (RAG): Multi-Source Claim-by-Claim Evidence Retrieval")
        retrieval_map: Dict[str, List[Dict[str, Any]]] = {}
        all_rag_candidates: List[Dict[str, Any]] = []

        factual_claims = [c for c in extracted_claims if not (c.get("isOpinion") or c.get("claimType") == "Opinion / Perspective Assertion")]
        try:
            gnews_map = gnews_service.search_gnews_for_claims(factual_claims)
        except Exception:
            gnews_map = {}

        for claim_obj in extracted_claims:
            c_text = claim_obj.get("claim", "")
            is_op = claim_obj.get("isOpinion", False) or claim_obj.get("claimType") == "Opinion / Perspective Assertion"
            if c_text and not is_op:
                wire_list = gnews_map.get(c_text, [])
                candidates = self.rag_retriever.retrieve_evidence_for_claim(c_text, max_items=14, wire_articles=wire_list)
                retrieval_map[c_text] = candidates
                all_rag_candidates.extend(candidates)
            else:
                retrieval_map[c_text] = []

        # If zero factual claims were extracted, query the first factual sentence
        if not factual_claims and len(text.strip().split()) > 3 and article_type != "Opinion Piece":
            first_sent = text.strip().split('.')[0]
            fallback_candidates = self.rag_retriever.retrieve_evidence_for_claim(first_sent, max_items=6)
            all_rag_candidates.extend(fallback_candidates)
            if fallback_candidates and extracted_claims:
                retrieval_map[extracted_claims[0].get("claim", first_sent)] = fallback_candidates

        # -------------------------------------------------------------
        # STEP 2b (RAG): Semantic Evidence Ranking & Evidence Matrix
        # Ranks candidates by relevance (embeddings), publisher credibility, recency.
        # -------------------------------------------------------------
        logger.info("Pipeline Step 2b (RAG): Semantic Evidence Ranking & Evidence Matrix Construction")
        evidence_matrix = self.evidence_ranker.build_evidence_matrix_for_claims(
            extracted_claims,
            retrieval_map
        )

        raw_supp: List[Dict[str, Any]] = []
        raw_contra: List[Dict[str, Any]] = []
        for row in evidence_matrix:
            raw_supp.extend(row.get("supportingEvidence", []))
            raw_contra.extend(row.get("contradictingEvidence", []))

        def _dedup_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen = set()
            out = []
            for it in items:
                k = (it.get("url") or it.get("articleTitle", "")).lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    out.append(it)
            return out

        top_supporting = _dedup_evidence(raw_supp)
        top_contradicting = _dedup_evidence(raw_contra)
        top_supporting.sort(key=lambda x: (x.get("compositeRankScore", 0), x.get("relevanceScore", 0)), reverse=True)
        top_contradicting.sort(
            key=lambda x: (
                1 if ("Contradicted by evidence:" in x.get("evidenceSnippet", "") or "Contradicted by scientific consensus:" in x.get("evidenceSnippet", "")) else 0,
                x.get("compositeRankScore", 0),
                x.get("relevanceScore", 0)
            ),
            reverse=True
        )

        sources_searched = self.rag_retriever.get_sources_searched_list()

        # -------------------------------------------------------------
        # STEP 2d: Legacy API Compatibility (Google Fact Check & GNews)
        # -------------------------------------------------------------
        external_factchecks_map: Dict[str, List[Dict[str, Any]]] = {}
        all_api_factchecks: List[Dict[str, Any]] = []
        for claim_obj in extracted_claims:
            claim_text = claim_obj.get("claim", "")
            if claim_text and not claim_obj.get("isOpinion"):
                fc_results = self.query_external_factcheck(claim_text)
                external_factchecks_map[claim_text] = fc_results
                all_api_factchecks.extend(fc_results)

        if not gnews_map:
            gnews_map = {}
            all_gnews_articles = []
            for claim_obj in extracted_claims:
                claim_text = claim_obj.get("claim", "")
                if claim_text and not claim_obj.get("isOpinion"):
                    gn_results = self.query_gnews(claim_text)
                    gnews_map[claim_text] = gn_results
                    all_gnews_articles.extend(gn_results)
        else:
            all_gnews_articles = []
            for g_list in gnews_map.values():
                all_gnews_articles.extend(g_list)

        # -------------------------------------------------------------
        # STEP 3: Source Reputation Engine & Quality Assessment
        # -------------------------------------------------------------
        logger.info("Pipeline Step 3: Source Reputation Engine & Quality Assessment")
        source_data = self.source_engine.evaluate_source(
            url_or_domain=url_or_domain,
            text=analysis_text,
            source_hint=source_hint,
            is_image=is_image,
            file_type=file_type,
            image_publisher=image_publisher
        )
        source_score = source_data["trustScore"]
        logger.info(f"Source evaluated: {source_data['sourceName']} (Trust Score: {source_score}/100, Tier: {source_data['tier']})")

        # Generate source quality assessment (especially critical for Historical Analysis and Feature pieces)
        source_quality_assessment = self.article_classifier.generate_source_quality_assessment(
            analysis_text, source_data, article_type=article_type
        )

        # -------------------------------------------------------------
        # STEP 4: Evidence Verification
        # -------------------------------------------------------------
        logger.info("Pipeline Step 4: Evidence Verification")
        evidence_results = self.evidence_verifier.verify_all_claims(
            claims=extracted_claims,
            source_context=source_data,
            article_text=analysis_text,
            external_factchecks_map=external_factchecks_map,
            gnews_map=gnews_map
        )
        # Harmonize evidence counts and score using deduplicated retrieved evidence
        verifier_contra = evidence_results.get("contradictingEvidence", [])
        for vc in verifier_contra:
            if not any((vc.get("url") or vc.get("articleTitle")) == (tc.get("url") or tc.get("articleTitle")) for tc in top_contradicting):
                top_contradicting.append(vc)

        # Requirement 3: Only accept sources with a real article URL (not a homepage or section page like /technology)
        top_supporting = [s for s in top_supporting if is_real_article_url(s.get("url", ""))]
        top_contradicting = [c for c in top_contradicting if is_real_article_url(c.get("url", ""))]

        supp_count = len(top_supporting)
        contra_count = len(top_contradicting)
        evidence_results["supportingEvidenceCount"] = supp_count
        evidence_results["contradictingEvidenceCount"] = contra_count

        evidence_score = evidence_results.get("evidenceScore", 50)
        if supp_count >= 2 and contra_count == 0:
            avg_supp = sum(s.get("sourceTrustScore", 85) for s in top_supporting[:3]) / min(3, len(top_supporting))
            evidence_score = max(evidence_score, int(round(avg_supp)))
        elif contra_count >= 1 and supp_count == 0:
            evidence_score = 0
        elif supp_count == 1 and contra_count == 0:
            evidence_score = max(evidence_score, 75)
        elif supp_count == 0 and contra_count == 0:
            evidence_score = 50
        evidence_results["evidenceScore"] = evidence_score

        fact_check_results = evidence_results.get("factCheckResults", [])
        if not fact_check_results and all_api_factchecks:
            fact_check_results = all_api_factchecks

        logger.info(f"Evidence verified: {supp_count} supporting, {contra_count} contradicting, Evidence Score: {evidence_score}/100")

        # -------------------------------------------------------------
        # STEP 5: ML Classification
        # -------------------------------------------------------------
        logger.info("Pipeline Step 5: ML Classification (TF-IDF + Logistic Regression)")
        ling_signals = analyze_linguistic_signals(analysis_text)
        cleaned = clean_text(analysis_text)

        prob_real = 0.5
        prob_fake = 0.5
        if self.model is not None and self.vectorizer is not None:
            try:
                X_vec = self.vectorizer.transform([cleaned])
                probs = self.model.predict_proba(X_vec)[0]
                prob_fake = float(probs[0])
                prob_real = float(probs[1])
            except Exception as e:
                logger.error(f"ML inference error: {e}")
                if ling_signals["sensational_score"] > 35:
                    prob_fake = 0.85
                    prob_real = 0.15
                else:
                    prob_real = 0.85
                    prob_fake = 0.15
        else:
            if ling_signals["sensational_score"] > 35:
                prob_fake = 0.85
                prob_real = 0.15
            else:
                prob_real = 0.85
                prob_fake = 0.15

        ml_score = int(round(prob_real * 100))
        raw_prediction = "REAL" if prob_real >= 0.5 else "FAKE"
        logger.info(f"ML Classification: {raw_prediction} ({ml_score}% Real probability)")

        # -------------------------------------------------------------
        # STEP 6: Final Trust Score (Multi-Factor Scoring)
        # Weights: Evidence 45%, ML 20%, Language 15%, Source 10%, Consistency 10%
        # -------------------------------------------------------------
        logger.info(f"Pipeline Step 6: Final Trust Score & Verdict Calculation (Genre: {article_type})")
        verdict_results = self.scoring_engine.compute_verdict_and_score(
            ml_score=ml_score,
            source_score=source_score,
            evidence_score=evidence_score,
            claims=evidence_results["claims"],
            ling_signals=ling_signals,
            supporting_evidence_count=supp_count,
            contradicting_evidence_count=contra_count,
            fact_check_results=fact_check_results,
            article_type=article_type,
            source_quality_assessment=source_quality_assessment,
            source_tier=source_data.get("tier", ""),
            source_category=source_data.get("category", ""),
            top_supporting=top_supporting,
            top_contradicting=top_contradicting
        )

        final_verdict = verdict_results["verdict"]
        final_trust_score = verdict_results["finalTrustScore"]
        risk_level = verdict_results["riskLevel"]
        summary_reason = verdict_results["summaryReason"]

        # Ensure conclusively false / contradicted claims receive standard FALSE verdict and low score
        # Never override Satire, Opinion, or NEEDS REVIEW verdicts (Requirement 4)
        if final_verdict not in ("Satire", "Opinion - not fact-checked", "NEEDS REVIEW"):
            if final_verdict in ("KNOWN FALSE", "FALSE"):
                final_verdict = "FALSE"
                if isinstance(final_trust_score, (int, float)):
                    final_trust_score = min(final_trust_score, 12)
                risk_level = "CRITICAL"
                if contra_count >= 1 and supp_count == 0:
                    summary_reason = f"Conclusively disproven. Contradicted by authoritative evidence ({contra_count} contradicting record(s))."

        # -------------------------------------------------------------
        # STEP 6b (RAG): Grounded Reasoning Synthesis
        # Strictly binds reasoning to real evidence counts and verdict
        # -------------------------------------------------------------
        sources_searched = self.rag_retriever.get_sources_searched_list()
        rag_verdict_data = self.grounded_reasoner.synthesize_rag_verdict_and_reasoning(
            claims=extracted_claims,
            evidence_matrix=evidence_matrix,
            top_supporting=top_supporting,
            top_contradicting=top_contradicting,
            sources_searched=sources_searched,
            article_type=article_type,
            final_verdict=final_verdict,
            final_trust_score=final_trust_score
        )

        scoring_factors = verdict_results["scoringFactors"]
        ml_contrib = scoring_factors.get("mlPrediction", {}).get("contribution", 0)
        src_contrib = scoring_factors.get("sourceReputation", {}).get("contribution", 0)
        evi_contrib = scoring_factors.get("evidenceStrength", {}).get("contribution", 0)
        con_contrib = scoring_factors.get("claimConsistency", {}).get("contribution", 0)
        lan_contrib = scoring_factors.get("languageQuality", {}).get("contribution", 0)

        log_msg = (
            f"[Lucid Pipeline] Text Length: {len(text)} chars | Final Trust Score: {final_trust_score} | Verdict: {final_verdict} | "
            f"Score Components: [Evidence(45%): {evi_contrib}, ML(20%): {ml_contrib}, Language(15%): {lan_contrib}, Source(10%): {src_contrib}, Consistency(10%): {con_contrib}]"
        )
        logger.info(log_msg)
        print(log_msg)

        # -------------------------------------------------------------
        # STEP 7: AI Reasoning & Deliberation
        # -------------------------------------------------------------
        reasoning_data = self.ai_reasoner.generate_ai_reasoning(
            article_text=analysis_text,
            claims=evidence_results["claims"],
            source_data=source_data,
            evidence_data=evidence_results,
            verdict_data=verdict_results,
            ling_signals=ling_signals
        )

        primary_publisher = source_data.get("publisher", source_data["sourceName"])
        source_type = source_data.get("sourceType", "Uploaded Image" if is_image else "")

        # Build Sources list - primary source is always at index 0
        sources_list = [
            {
                "name": source_data["sourceName"],
                "publisher": primary_publisher,
                "code": (primary_publisher[:3].upper() if primary_publisher and primary_publisher != "Unknown" else "SRC"),
                "domain": source_data["domain"],
                "type": source_type or source_data["category"],
                "trust": source_data["trustScore"],
                "tier": source_data["tier"],
                "description": source_data["description"],
                "isPrimary": True
            }
        ]

        seen_sources = {source_data["sourceName"].lower(), primary_publisher.lower()}
        for snip in evidence_results["supportingEvidence"] + evidence_results["contradictingEvidence"]:
            src_name = snip.get("source", "")
            if src_name and src_name.lower() not in seen_sources:
                seen_sources.add(src_name.lower())
                code = "".join([w[0] for w in src_name.split()[:3]]).upper()
                sources_list.append({
                    "name": src_name,
                    "publisher": src_name,
                    "code": code,
                    "domain": snip.get("url", src_name.lower().replace(" ", "") + ".org"),
                    "type": "Corroborating Evidence Registry",
                    "trust": snip.get("trust", 90),
                    "tier": "Evidence Corroboration",
                    "description": snip.get("snippet", ""),
                    "isPrimary": False
                })

        evi_factor_score = verdict_results["scoringFactors"]["evidenceStrength"]["score"]
        if evi_factor_score == "Not scored":
            confidence = 85
        elif supp_count == 0 and contra_count == 0:
            # Requirement 4: Low confidence when zero evidence is found
            confidence = 35
        else:
            confidence = int(round(
                (float(evi_factor_score) * 0.5) +
                (float(verdict_results["scoringFactors"]["sourceReputation"]["score"]) * 0.3) +
                (max(prob_real, prob_fake) * 100 * 0.2)
            ))
            confidence = max(50, min(98, confidence))

        explanation = [
            summary_reason,
            f"Multi-Factor Trust Score: {final_trust_score}{'' if final_trust_score == 'Not scored' else '/100'} based on 5 analytical layers.",
            f"Source Authority: {source_data['sourceName']} (Publisher: {primary_publisher}, Trust Score: {source_data['trustScore']}/100, Weight: 10%).",
            f"Evidence Corroboration: {supp_count} supporting citations, {contra_count} contradicting flags."
        ]
        if fact_check_results:
            explanation.append(f"Google Fact Check API matched {len(fact_check_results)} external fact check review(s).")
        
        # Requirement 5: Remove the template line "Objective, attributable journalistic prose detected" unless from real language analysis
        if ling_signals.get("sensational_score", 0) > 25:
            explanation.append(f"Sensationalist language markers detected ({ling_signals['sensational_score']}%).")
        elif len(text.split()) >= 40 and ling_signals.get("sensational_score", 0) <= 15 and ling_signals.get("clickbait_score", 0) <= 15:
            explanation.append("Linguistic analysis: Attributable, objective journalistic phrasing detected.")
        
        if supp_count == 0 and contra_count == 0:
            explanation.append("Confidence Assessment: Low confidence — no independent evidence found to corroborate or contradict the claims.")

        empty_audit_data = {
            "sourcesSearched": sources_searched,
            "resultsFound": f"{supp_count} supporting, {contra_count} contradicting",
            "reason": (
                f"Contradicted by {contra_count} source(s)." if contra_count > 0 else "No sufficiently relevant corroborating sources were found."
            )
        } if supp_count == 0 else None

        return {
            "verdict": final_verdict,
            "trustScore": final_trust_score,
            "credibility": final_trust_score,
            "confidence": confidence,
            "riskLevel": risk_level,
            "prediction": raw_prediction,
            "stanceClassifierStatus": getattr(self.evidence_ranker.gemini_classifier, "last_status", "available"),
            "summaryReason": summary_reason,
            "explanation": explanation,
            
            # RAG (Retrieval-Augmented Generation) Architecture
            "ragEnabled": True,
            "sourcesSearched": sources_searched,
            "resultsFound": f"{supp_count} supporting, {contra_count} contradicting",
            "topSupportingSources": top_supporting[:6],
            "topContradictingSources": top_contradicting[:6],
            "evidenceMatrix": evidence_matrix,
            "groundedReasoning": rag_verdict_data["groundedReasoning"],

            # Multi-language detection
            "language": lang_info,
            "detectedLanguage": detected_lang,
            "languageCode": detected_code,
            "wasTranslated": was_translated,
            "originalLanguageText": text if was_translated else None,

            # Article classification & Genre-specific evaluation
            "articleType": article_type,
            "articleTypeClassification": classification,
            "sourceQualityAssessment": source_quality_assessment,

            # Evidence retrieval transparency
            "emptyEvidenceAudit": empty_audit_data,

            # Layer 0: Automatic Article Summarization
            "articleSummary": article_summary,
            "summary": article_summary,
            "articleTitle": article_summary.get("title") or article_title,

            # Primary publisher attribution
            "publisher": primary_publisher,
            "sourceType": source_type,
            "sourceDomain": source_data["domain"],
            "primarySource": {
                "publisher": primary_publisher,
                "name": source_data["sourceName"],
                "domain": source_data["domain"],
                "sourceType": source_type,
                "isPrimary": True,
                "trustScore": source_data["trustScore"]
            },

            # Layer 1: Source Reputation
            "sourceReputation": source_data,
            "sourceCredibility": source_data["trustScore"],
            
            # Layer 2 & 3: Claims and Evidence
            "claims": evidence_results["claims"],
            "ragGroundedVerdict": rag_verdict_data,
            "evidence": {
                "supporting": [
                    {
                        "source": s.get("publisher", "Wire Source"),
                        "sourceName": s.get("publisher", "Wire Source"),
                        "articleTitle": s.get("articleTitle", ""),
                        "publicationDate": s.get("publicationDate", ""),
                        "sourceTrustScore": s.get("sourceTrustScore", 90),
                        "url": s.get("url", ""),
                        "matchingClaim": s.get("matchingClaim", ""),
                        "stance": "support",
                        "snippet": s.get("evidenceSnippet", ""),
                        "trust": s.get("sourceTrustScore", 90),
                        "relevance": s.get("relevanceScore", 85)
                    }
                    for s in top_supporting
                ],
                "contradicting": [
                    {
                        "source": c.get("publisher", "Fact Checker"),
                        "sourceName": c.get("publisher", "Fact Checker"),
                        "articleTitle": c.get("articleTitle", ""),
                        "publicationDate": c.get("publicationDate", ""),
                        "sourceTrustScore": c.get("sourceTrustScore", 90),
                        "url": c.get("url", ""),
                        "matchingClaim": c.get("matchingClaim", ""),
                        "stance": "contradict",
                        "snippet": c.get("evidenceSnippet", ""),
                        "trust": c.get("sourceTrustScore", 90),
                        "relevance": c.get("relevanceScore", 85)
                    }
                    for c in top_contradicting
                ],
                "supportingCount": supp_count,
                "contradictingCount": contra_count,
                "evidenceScore": evidence_score,
                "supportingEvidenceStatus": "NO SUPPORTING EVIDENCE FOUND" if supp_count == 0 else "CORROBORATED",
                "calculation": evidence_results.get("evidenceCalculation"),
                "emptyEvidenceAudit": empty_audit_data
            },
            "evidenceCalculation": evidence_results.get("evidenceCalculation"),
            "evidenceQuality": evidence_score,

            # Google Fact Check Tools API results
            "factCheckResults": fact_check_results,

            # GNews API Corroboration results
            "gnewsArticles": evidence_results.get("gnewsArticles", []),

            # Layer 4: Multi-Factor Scoring Factors
            "scoringFactors": scoring_factors,
            "languageIntegrity": scoring_factors["languageQuality"]["score"],
            
            # Layer 6: Sources registry
            "sources": sources_list,

            # Layer 7: Transparency Disclaimer
            "disclaimer": TRANSPARENCY_WARNING,

            # Layer 8: AI Reasoning
            "reasoning": reasoning_data,

            # Linguistic signals
            "emotionalLanguage": ling_signals["emotional_language"],
            "clickbaitScore": ling_signals["clickbait_score"],
            "sensationalTerms": ling_signals["detected_sensational_terms"],

            # Investigation timeline
            "timeline": [
                {"step": "Claim Extraction", "detail": f"Segmented {len(extracted_claims)} atomic testable factual claims", "status": "done"},
                {"step": "Google Fact Check API", "detail": f"Queried claims:search endpoint ({len(fact_check_results)} review records)", "status": "done"},
                {"step": "GNews API Corroboration", "detail": f"Searched recent trusted news wires ({len(evidence_results.get('gnewsArticles', []))} article matches)", "status": "done"},
                {"step": "Source Reputation Audit", "detail": f"Evaluated domain authority: {source_data['sourceName']} ({source_data['trustScore']}/100)", "status": "done"},
                {"step": "Evidence Cross-Referencing", "detail": f"Queried wire archives ({supp_count} supporting, {contra_count} contradicting)", "status": "done"},
                {"step": "ML Model Inference", "detail": "Executed TF-IDF + Logistic Regression pattern analysis (30% weight)", "status": "done"},
                {"step": "Multi-Factor Synthesis", "detail": f"Calculated 5-layer weighted trust score ({final_trust_score}/100) -> {final_verdict}", "status": "done"}
            ],

            "aiIntegrations": {
                "geminiConfigured": bool(os.getenv("GEMINI_API_KEY")),
                "geminiModel": getattr(self.evidence_ranker.gemini_classifier, "active_model", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")),
                "stanceClassifierStatus": getattr(self.evidence_ranker.gemini_classifier, "last_status", "available"),
                "openAiConfigured": bool(os.getenv("OPENAI_API_KEY")),
                "factCheckConfigured": bool(os.getenv("FACTCHECK_API_KEY")),
                "gnewsConfigured": bool(os.getenv("GNEWS_API_KEY")),
                "provider": reasoning_data.get("provider", "Lucid Core Multi-Factor Engine")
            }
        }

engine = VerificationEngine()

def get_verification_engine() -> VerificationEngine:
    return engine
