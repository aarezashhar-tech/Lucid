from __future__ import annotations
"""
Layer 4 & Layer 5: Multi-Factor Scoring & Modern Verdict Engine
Combines 5 independent analytical dimensions:
- 70% Evidence Strength
- 10% Source Reputation
- 10% Claim Consistency
- 5% ML Prediction
- 5% Language Quality

Replaces binary REAL/FAKE with granular verdict types:
- VERIFIED
- LIKELY TRUE
- NEEDS REVIEW
- UNVERIFIED
- LIKELY FALSE
- KNOWN FALSE
"""
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime, date
from urllib.parse import urlparse
from utils.url_extractor import is_real_article_url

TupleScore = Tuple[int, str]

WEIGHT_EVIDENCE_URL = 0.80
WEIGHT_SOURCE_URL = 0.10
WEIGHT_CONSISTENCY_URL = 0.10

WEIGHT_EVIDENCE_DIRECT = 0.90
WEIGHT_SOURCE_DIRECT = 0.00
WEIGHT_CONSISTENCY_DIRECT = 0.10

WEIGHT_EVIDENCE = 0.80
WEIGHT_SOURCE = 0.10
WEIGHT_CONSISTENCY = 0.10
WEIGHT_ML = 0.00
WEIGHT_LANGUAGE = 0.00

HIGH_STANCE_CONFIDENCE_THRESHOLD = 75.0
TIER1_SCORE_FLOOR = 82

# Canonical Tier-1 global news outlets: (canonical_name, domains, substring_patterns, word_boundary_acronyms)
TIER1_NEWS_SPECS: List[Tuple[str, Set[str], Tuple[str, ...], Tuple[str, ...]]] = [
    ("Reuters", {"reuters.com"}, ("reuters",), ()),
    ("Associated Press", {"apnews.com"}, ("associated press", "ap news"), ("ap",)),
    ("AFP", {"afp.com", "factcheck.afp.com"}, ("agence france-presse", "agence france presse"), ("afp",)),
    ("BBC", {"bbc.com", "bbc.co.uk", "bbc.in"}, ("bbc news",), ("bbc",)),
    ("Al Jazeera", {"aljazeera.com", "aljazeera.net"}, ("al jazeera", "aljazeera"), ()),
    ("DPA", {"dpa.com", "dpa-international.com"}, ("deutsche presse-agentur", "dpa international"), ("dpa",)),
    ("DW", {"dw.com"}, ("deutsche welle",), ("dw",)),
    ("The Guardian", {"theguardian.com"}, ("the guardian",), ("guardian",)),
    ("The Hindu", {"thehindu.com"}, ("the hindu", "thehindu"), ()),
    ("The Indian Express", {"indianexpress.com"}, ("indian express",), ()),
    ("Financial Times", {"ft.com"}, ("financial times",), ("ft",)),
    ("The Economist", {"economist.com"}, ("the economist",), ("economist",)),
]

# Accredited IFCN fact-checkers (also count as Tier-1 sources)
IFCN_CHECKER_SPECS: List[Tuple[str, Set[str], Tuple[str, ...], Tuple[str, ...]]] = [
    ("Snopes", {"snopes.com"}, ("snopes",), ()),
    ("PolitiFact", {"politifact.com"}, ("politifact",), ()),
    ("FactCheck.org", {"factcheck.org"}, ("factcheck.org",), ()),
    ("Full Fact", {"fullfact.org"}, ("full fact", "fullfact"), ()),
    ("Alt News", {"altnews.in"}, ("alt news", "altnews"), ()),
    ("BOOM Live", {"boomlive.in"}, ("boom live", "boomlive"), ()),
    ("Factly", {"factly.in"}, ("factly",), ()),
    ("Vishvas News", {"vishvasnews.com"}, ("vishvas news", "vishvasnews"), ()),
]

PARTIAL_OR_NEGATED_TRUE_REGEX = re.compile(
    r'\b(untrue|not\s+true|half[\s-]+true|partly[\s-]+true|partially[\s-]+true|mixed|misleading|false|fake|pants\s+on\s+fire|incorrect|debunked?|hoax|fabricated)\b',
    re.IGNORECASE,
)
EXPLICIT_TRUE_RATING_REGEX = re.compile(
    r'\b(true|mostly\s+true|correct|accurate|verified|confirmed)\b',
    re.IGNORECASE,
)

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

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url:
            return ""
        try:
            netloc = urlparse(url.strip()).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return ""

    @staticmethod
    def _match_specs(
        pub_lower: str,
        domain: str,
        specs: List[Tuple[str, Set[str], Tuple[str, ...], Tuple[str, ...]]]
    ) -> Optional[str]:
        for canonical, domains, substrings, acronyms in specs:
            if domain and any(domain == d or domain.endswith("." + d) for d in domains):
                return canonical
            if pub_lower and any(sub in pub_lower for sub in substrings):
                return canonical
            if pub_lower and any(re.search(rf'\b{re.escape(acr)}\b', pub_lower) for acr in acronyms):
                return canonical
        return None

    def _identify_tier1_or_ifcn(self, item: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """
        Identifies whether an evidence item belongs to a canonical Tier-1 outlet
        or accredited IFCN fact-checker.
        Returns (canonical_name_or_None, is_ifcn).
        """
        pub_raw = str(
            item.get("publisher")
            or item.get("source")
            or item.get("sourceName")
            or item.get("reviewer")
            or ""
        ).strip()
        pub_lower = pub_raw.lower()
        domain = str(item.get("publisherDomain") or "").lower().strip()
        if not domain:
            domain = self._extract_domain(str(item.get("url") or ""))
        elif domain.startswith("www."):
            domain = domain[4:]

        cat_lower = str(item.get("sourceCategory") or item.get("category") or "").lower()

        ifcn_name = self._match_specs(pub_lower, domain, IFCN_CHECKER_SPECS)
        if ifcn_name:
            return ifcn_name, True

        tier1_name = self._match_specs(pub_lower, domain, TIER1_NEWS_SPECS)
        if tier1_name:
            is_ifcn_cat = ("ifcn" in cat_lower) or ("fact check" in pub_lower and "ifcn" in cat_lower)
            return tier1_name, is_ifcn_cat

        if "ifcn" in cat_lower:
            return (pub_raw or domain or "IFCN Fact Checker"), True

        return None, False

    @staticmethod
    def _is_high_stance_confidence(item: Dict[str, Any]) -> bool:
        stance_conf = item.get("stanceConfidence")
        if stance_conf is not None:
            try:
                return float(stance_conf) >= HIGH_STANCE_CONFIDENCE_THRESHOLD
            except (TypeError, ValueError):
                return False
        rel_score = item.get("relevanceScore")
        if rel_score is not None:
            try:
                return float(rel_score) >= 70.0
            except (TypeError, ValueError):
                return False
        return True

    def _is_explicit_false_rating(self, rating: str) -> bool:
        if not rating:
            return False
        r = rating.lower().strip()
        return any(k in r for k in self.FALSE_FACTCHECK_KEYWORDS)

    @staticmethod
    def _is_explicit_true_rating(rating: str) -> bool:
        if not rating:
            return False
        r = rating.strip()
        if PARTIAL_OR_NEGATED_TRUE_REGEX.search(r):
            return False
        return bool(EXPLICIT_TRUE_RATING_REGEX.search(r))

    @staticmethod
    def _build_scoring_factors(
        evi_score: Any,
        ml_score: Any,
        lan_score: Any,
        src_score: Any,
        con_score: Any,
        language_notes: str,
        consistency_notes: str,
        evi_contrib: float = None,
        ml_contrib: float = None,
        lan_contrib: float = None,
        src_contrib: float = None,
        con_contrib: float = None,
        w_evidence: float = WEIGHT_EVIDENCE,
        w_source: float = WEIGHT_SOURCE,
        w_consistency: float = WEIGHT_CONSISTENCY,
        w_ml: float = WEIGHT_ML,
        w_language: float = WEIGHT_LANGUAGE,
        evi_label: str = None,
        ml_label: str = None,
        lan_label: str = None,
        src_label: str = None,
        con_label: str = None
    ) -> Dict[str, Any]:
        if evi_label is None:
            evi_label = f"Evidence Strength ({int(round(w_evidence * 100))}%)"
        if ml_label is None:
            ml_label = f"ML Text Pattern Classifier ({int(round(w_ml * 100))}%)"
        if lan_label is None:
            lan_label = f"Journalistic Language Integrity ({int(round(w_language * 100))}%)"
        if src_label is None:
            src_label = f"Domain & Source Reputation ({int(round(w_source * 100))}%)"
        if con_label is None:
            con_label = f"Factual Claim Consistency ({int(round(w_consistency * 100))}%)"

        return {
            "evidenceStrength": {
                "weight": w_evidence,
                "score": evi_score,
                "contribution": evi_contrib if evi_contrib is not None else (round(w_evidence * float(evi_score), 1) if isinstance(evi_score, (int, float)) else 0.0),
                "label": evi_label
            },
            "sourceReputation": {
                "weight": w_source,
                "score": src_score,
                "contribution": src_contrib if src_contrib is not None else (round(w_source * float(src_score), 1) if isinstance(src_score, (int, float)) else 0.0),
                "label": src_label
            },
            "claimConsistency": {
                "weight": w_consistency,
                "score": con_score,
                "contribution": con_contrib if con_contrib is not None else (round(w_consistency * float(con_score), 1) if isinstance(con_score, (int, float)) else 0.0),
                "label": con_label,
                "notes": consistency_notes
            },
            "mlPrediction": {
                "weight": 0.0,
                "score": ml_score,
                "contribution": 0.0,
                "label": ml_label
            },
            "languageQuality": {
                "weight": 0.0,
                "score": lan_score,
                "contribution": 0.0,
                "label": lan_label,
                "notes": language_notes
            }
        }

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
        top_contradicting: List[Dict[str, Any]] = None,
        is_direct_submission: bool = None,
        source_domain: str = ""
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

        # Determine whether input is direct text submission vs URL/article with known publisher
        if is_direct_submission is None:
            dom = (source_domain or "").lower()
            if not dom and source_quality_assessment:
                dom = str(source_quality_assessment.get("domain", "")).lower()
            is_direct_submission = (
                dom in ("direct-submission", "n/a", "")
                or source_category in ("Unattributed Text Input", "Uploaded Image", "Uploaded Document")
                or (source_tier == "Unverified" and (not dom or dom in ("direct-submission", "n/a")))
            )

        if is_direct_submission:
            w_evidence = WEIGHT_EVIDENCE_DIRECT       # 0.90
            w_source = WEIGHT_SOURCE_DIRECT           # 0.00
            w_consistency = WEIGHT_CONSISTENCY_DIRECT # 0.10
        else:
            w_evidence = WEIGHT_EVIDENCE_URL          # 0.80
            w_source = WEIGHT_SOURCE_URL              # 0.10
            w_consistency = WEIGHT_CONSISTENCY_URL     # 0.10
        w_ml = 0.00
        w_language = 0.00

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
                "scoringFactors": self._build_scoring_factors(
                    evi_score="Satire",
                    ml_score="Satire",
                    lan_score=int(round(lan_s)),
                    src_score=int(round(src_s)),
                    con_score=int(round(con_s)),
                    language_notes=language_notes,
                    consistency_notes=consistency_notes,
                    evi_contrib=0.0,
                    ml_contrib=0.0,
                    lan_contrib=0.0,
                    src_contrib=0.0,
                    con_contrib=0.0,
                    evi_label=f"Evidence Strength ({int(WEIGHT_EVIDENCE * 100)}%) — Satirical publication",
                    ml_label=f"ML Pattern Classifier ({int(WEIGHT_ML * 100)}%) — Satirical publication",
                    lan_label=f"Language Tone & Humor ({int(WEIGHT_LANGUAGE * 100)}%)",
                    src_label=f"Domain & Source Authority ({int(WEIGHT_SOURCE * 100)}%)",
                    con_label=f"Viewpoint Consistency ({int(WEIGHT_CONSISTENCY * 100)}%)"
                )
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
                "scoringFactors": self._build_scoring_factors(
                    evi_score="Not scored",
                    ml_score="Not scored",
                    lan_score=int(round(lan_s)),
                    src_score=int(round(src_s)),
                    con_score=int(round(con_s)),
                    language_notes=language_notes,
                    consistency_notes=consistency_notes,
                    evi_contrib=0.0,
                    ml_contrib=0.0,
                    lan_contrib=0.0,
                    src_contrib=0.0,
                    con_contrib=0.0,
                    evi_label=f"Evidence Strength ({int(WEIGHT_EVIDENCE * 100)}%) — N/A for Opinion",
                    ml_label=f"ML Pattern Classifier ({int(WEIGHT_ML * 100)}%) — N/A for Opinion",
                    lan_label=f"Language Tone & Rhetoric ({int(WEIGHT_LANGUAGE * 100)}%)",
                    src_label=f"Domain & Source Authority ({int(WEIGHT_SOURCE * 100)}%)",
                    con_label=f"Viewpoint Consistency ({int(WEIGHT_CONSISTENCY * 100)}%)"
                )
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
                "scoringFactors": self._build_scoring_factors(
                    evi_score=int(round(evi_s)),
                    ml_score=int(round(ml_s)),
                    lan_score=int(round(lan_s)),
                    src_score=int(round(src_s)),
                    con_score=int(round(con_s)),
                    language_notes=language_notes,
                    consistency_notes=consistency_notes
                )
            }

        # -----------------------------------------------------------------
        # 2. FACTUAL VERIFICATION MULTI-FACTOR SCORING
        # ML Text Pattern Classifier & Journalistic Language Integrity = 0%
        # Known publisher URL: 80% Evidence + 10% Source + 10% Consistency
        # Direct text claim: 90% Evidence + 10% Consistency
        # -----------------------------------------------------------------
        computed_score = round(
            (w_evidence * evi_s) +
            (w_source * src_s) +
            (w_consistency * con_s)
        )
        final_trust_score = max(5, min(98, int(computed_score)))

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
                    elif any(c_val in str(item.get("publisher", "")).lower() for c_val in ["international", "agency", "united nations", "reuters", "ap", "who", "iea", "irena"]):
                        recency = 0.75
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

        # Detect Google Fact Check / IFCN false debunked rating if not explicitly supplied
        if has_false_fact_check is None:
            has_false_fact_check = False
            if fact_check_results:
                for fc in fact_check_results:
                    rating = str(fc.get("rating") or fc.get("textualRating") or "")
                    if self._is_explicit_false_rating(rating):
                        has_false_fact_check = True
                        break

            if not has_false_fact_check and claims:
                for c in claims:
                    for fc in c.get("factCheckMatches", []):
                        rating = str(fc.get("rating") or fc.get("textualRating") or "")
                        if self._is_explicit_false_rating(rating):
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

            if not has_false_fact_check and contra_items:
                for ci in contra_items:
                    fc_rating = str(ci.get("factCheckRating") or "")
                    if self._is_explicit_false_rating(fc_rating):
                        has_false_fact_check = True
                        break
                    _, is_ifcn_ci = self._identify_tier1_or_ifcn(ci)
                    if is_ifcn_ci:
                        text_blob = (
                            str(ci.get("articleTitle") or "")
                            + " "
                            + str(ci.get("evidenceSnippet") or ci.get("snippet") or "")
                        ).lower()
                        cat_ci = str(ci.get("sourceCategory") or "").lower()
                        if "ifcn" in cat_ci or "fact checker" in cat_ci or any(
                            k in text_blob for k in ("fact check", "false", "debunk", "hoax", "pants on fire", "misleading", "fake")
                        ):
                            has_false_fact_check = True
                            break

        # Detect Tier-1 Consensus and IFCN True ratings
        distinct_tier1_supp: List[str] = []
        ifcn_true_sources: List[str] = []
        tier1_contra_sources: Set[str] = set()

        for s in supp_items:
            canonical, is_ifcn_s = self._identify_tier1_or_ifcn(s)
            if canonical and self._is_high_stance_confidence(s):
                if canonical not in distinct_tier1_supp:
                    distinct_tier1_supp.append(canonical)
            if is_ifcn_s and self._is_explicit_true_rating(str(s.get("factCheckRating") or "")):
                if canonical and canonical not in ifcn_true_sources:
                    ifcn_true_sources.append(canonical)

        for fc in (fact_check_results or []):
            rating = str(fc.get("rating") or fc.get("textualRating") or "")
            if self._is_explicit_true_rating(rating):
                reviewer = str(fc.get("reviewer") or "").strip()
                probe = {
                    "publisher": reviewer,
                    "url": fc.get("url", ""),
                    "sourceCategory": "IFCN Fact Checker" if "google fact check" in str(fc.get("source", "")).lower() else "",
                }
                canonical, is_ifcn_fc = self._identify_tier1_or_ifcn(probe)
                if is_ifcn_fc:
                    label = canonical or reviewer or "IFCN Fact Checker"
                    if label not in ifcn_true_sources:
                        ifcn_true_sources.append(label)

        for c in (claims or []):
            for fc in c.get("factCheckMatches", []):
                rating = str(fc.get("rating") or fc.get("textualRating") or "")
                if self._is_explicit_true_rating(rating):
                    reviewer = str(fc.get("reviewer") or "").strip()
                    probe = {
                        "publisher": reviewer,
                        "url": fc.get("url", ""),
                        "sourceCategory": "IFCN Fact Checker" if "google fact check" in str(fc.get("source", "")).lower() else "",
                    }
                    canonical, is_ifcn_fc = self._identify_tier1_or_ifcn(probe)
                    if is_ifcn_fc:
                        label = canonical or reviewer or "IFCN Fact Checker"
                        if label not in ifcn_true_sources:
                            ifcn_true_sources.append(label)

        for ci in contra_items:
            canonical_ci, is_ifcn_ci = self._identify_tier1_or_ifcn(ci)
            cat_ci = str(ci.get("sourceCategory") or "").lower()
            if canonical_ci or is_ifcn_ci or "ifcn" in cat_ci or "fact checker" in cat_ci:
                tier1_contra_sources.add(canonical_ci or str(ci.get("publisher") or ci.get("sourceName") or "Tier-1/IFCN"))

        has_wikidata_living_contra = any(
            "[derived check]" in str(ci.get("evidenceSnippet") or ci.get("snippet") or "").lower()
            or "no date of death recorded" in str(ci.get("evidenceSnippet") or ci.get("snippet") or "").lower()
            or ci.get("retrievalChannel") == "Wikidata API"
            for ci in contra_items
        )
        has_contradicted_subclaim = bool(
            claims and len(claims) > 1 and any(c.get("status") in ("Contradicted", "False") for c in claims)
        )

        tier1_consensus_met = (
            (not has_false_fact_check)
            and (len(tier1_contra_sources) == 0)
            and (not has_wikidata_living_contra)
            and (not has_contradicted_subclaim)
            and (len(distinct_tier1_supp) >= 2 or len(ifcn_true_sources) >= 1)
        )

        total_support_weight = sum(_calc_source_weight(s) for s in supp_items)
        total_contradict_weight = sum(_calc_source_weight(c) for c in contra_items)

        if not supp_items and not contra_items:
            # Fallback when raw evidence counts are supplied directly (e.g. unit test scenarios)
            if supporting_evidence_count > 0 or contradicting_evidence_count > 0:
                total_support_weight = supporting_evidence_count * 0.5
                total_contradict_weight = contradicting_evidence_count * 0.5

        distinct_supp_pubs = {
            (self._identify_tier1_or_ifcn(s)[0] or str(s.get("publisher") or s.get("sourceName") or s.get("source") or s.get("url") or "")).lower().strip()
            for s in supp_items
        }
        distinct_supp_pubs.discard("")

        # Both sides are strong and close (balanced conflict)
        both_sides_strong = (total_support_weight >= 0.20 and total_contradict_weight >= 0.20)
        weights_close = False
        if both_sides_strong:
            ratio = total_support_weight / max(0.001, total_contradict_weight)
            weights_close = (0.5 <= ratio <= 2.0)

        is_needs_review = both_sides_strong and weights_close
        support_clearly_dominates = (total_support_weight >= 0.45 and (total_contradict_weight == 0 or total_support_weight >= 2.0 * total_contradict_weight))
        contradict_clearly_dominates = (
            (total_contradict_weight >= 0.45 and (total_support_weight == 0 or total_contradict_weight >= 2.0 * total_support_weight))
            or (total_support_weight == 0 and (total_contradict_weight >= 0.20 or len(contra_items) >= 1 or contradicting_evidence_count >= 1))
        )

        if has_false_fact_check:
            verdict = "KNOWN FALSE"
            final_trust_score = min(final_trust_score, 12)
            risk_level = "CRITICAL"
            summary_reason = "Conclusively disproven. Flagged as false by certified Google Fact Check organization."

        elif tier1_consensus_met:
            verdict = "VERIFIED"
            final_trust_score = max(TIER1_SCORE_FLOOR, min(98, int(computed_score)))
            risk_level = "LOW"
            if len(distinct_tier1_supp) >= 2:
                tier1_names = ", ".join(distinct_tier1_supp[:3])
                summary_reason = (
                    f"Tier-1 consensus verified across {len(distinct_tier1_supp)} independent authoritative sources ({tier1_names}) "
                    f"with support clearly dominating (Support weight: {round(total_support_weight, 2)}) and zero Tier-1/IFCN contradictions."
                )
            else:
                ifcn_names = ", ".join(ifcn_true_sources[:2])
                summary_reason = (
                    f"IFCN fact-check verified by {ifcn_names} with support clearly dominating "
                    f"(Support weight: {round(total_support_weight, 2)}) and zero Tier-1/IFCN contradictions."
                )

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
            has_multi_distinct_supp = (len(distinct_supp_pubs) >= 2) if supp_items else (supporting_evidence_count >= 2)
            if total_support_weight >= 1.0 and has_multi_distinct_supp:
                verdict = "VERIFIED"
                # Let weighted score show within sensible band [70, 98]
                final_trust_score = max(70, min(98, int(computed_score)))
                risk_level = "LOW"
                summary_reason = f"Corroborated across reliable wire sources (Support weight: {round(total_support_weight, 2)}) with support clearly dominating."
            else:
                verdict = "LIKELY TRUE"
                # Let weighted score show within sensible band [60, 85]
                final_trust_score = max(60, min(85, int(computed_score)))
                risk_level = "LOW"
                summary_reason = f"Supported by reputable reporting (Support weight: {round(total_support_weight, 2)}) without substantial contradiction."

        elif not supp_items and not contra_items and supporting_evidence_count == 0 and contradicting_evidence_count == 0:
            # Return UNVERIFIED with "no live evidence found"
            verdict = "UNVERIFIED"
            final_trust_score = min(50, final_trust_score)
            risk_level = "MEDIUM"
            summary_reason = "No live evidence found across news wire feeds or fact-check registries (no reliable supporting or contradicting citations attached). Low confidence — unverified pending independent evidence."

        elif total_support_weight > total_contradict_weight or supp_items or supporting_evidence_count > 0:
            verdict = "LIKELY TRUE"
            final_trust_score = max(60, min(85, int(computed_score)))
            risk_level = "LOW"
            summary_reason = f"Supported by reporting (Support weight: {round(total_support_weight, 2)} vs Contradict: {round(total_contradict_weight, 2)})."

        else:
            verdict = "UNVERIFIED"
            final_trust_score = min(50, final_trust_score)
            risk_level = "MEDIUM"
            summary_reason = "No live evidence found across news wire feeds or fact-check registries (no reliable supporting or contradicting citations attached). Low confidence — unverified pending independent evidence."

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
            "scoringFactors": self._build_scoring_factors(
                evi_score=int(round(evi_s)),
                ml_score=int(round(ml_s)),
                lan_score=int(round(lan_s)),
                src_score=int(round(src_s)),
                con_score=int(round(con_s)),
                language_notes=language_notes,
                consistency_notes=consistency_notes,
                w_evidence=w_evidence,
                w_source=w_source,
                w_consistency=w_consistency,
                w_ml=w_ml,
                w_language=w_language
            )
        }

scoring_engine = MultiFactorScoringEngine()

def get_scoring_engine() -> MultiFactorScoringEngine:
    return scoring_engine
