"""
Lucid Evidence Ranker & Stance Analysis Engine
Ranks retrieved evidence candidates based on:
1. Semantic Relevance (Neural Embedding Cosine Similarity)
2. Publisher Credibility (Wire/IFCN/Institutional Trust Scores)
3. Recency (Publication Date Timeliness)
4. Claim Stance (Supporting vs. Contradicting vs. Neutral)

Produces:
- Top Supporting Sources
- Top Contradicting Sources
- Per-Claim Evidence Matrix
"""
import re
import logging
import contextvars
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Optional

from services.factcheck_service import is_factcheck_claim_relevant
from services.semantic_embeddings import get_semantic_engine
from services.gemini_stance_classifier import get_gemini_stance_classifier
from services.wikipedia_service import claim_refers_to_relative_or_associate
from utils.url_extractor import is_real_article_url

logger = logging.getLogger("lucid.ranker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [EvidenceRanker] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Debunk and contradiction lexical signals
CONTRADICTION_PATTERNS = [
    # Required by user: conspiracy theory, debunked, false, misleading, no evidence, baseless
    r'\b(conspiracy\s+theor(y|ies|ists?)|conspiracy)\b',
    r'\b(debunk|debunked|debunking|debunks)\b',
    r'\b(false|falsely|falsehood|false\s+claim)\b',
    r'\b(misleading|misinformation|disinformation)\b',
    r'\b(no\s+evidence|no\s+scientific\s+evidence|no\s+proof|lacks\s+evidence|without\s+evidence|no\s+basis)\b',
    r'\b(baseless|baseless\s+fear|baseless\s+claim|unfounded|unsubstantiated|groundless)\b',
    
    # Fact-checking & Hoax markers
    r'\b(fact\s*check|hoax|pants\s+on\s+fire|fake\s+news|fabricated|fake|distorted|manipulated)\b',
    r'\b(refuted|refutes|refuting|disproven|discredited|denies|denied|untrue|inaccurate|incorrect|out\s+of\s+context)\b',

    # Explicit causation & linkage negation (5G, vaccines, health)
    r'\b(does\s+not\s+cause|doesn\'?t\s+cause|did\s+not\s+cause|didn\'?t\s+cause|cannot\s+cause|not\s+cause|no\s+link|not\s+linked|never\s+caused|do\s+not\s+cause|don\'?t\s+cause)\b',
    r'\b(does\s+not\s+spread|doesn\'?t\s+spread|cannot\s+spread|not\s+spread|do\s+not\s+spread|does\s+not\s+transmit|cannot\s+transmit)\b',
    r'\b(bans\s+incitement\s+to\s+attack|burn(ing)?\s+cell\s+towers|burn(ing)?\s+5g\s+towers|towers\s+ablaze|attacks?\s+on\s+(5g|cell)\s+towers)\b',

    # Endorsement negation
    r'\b(does\s+not\s+endorse|doesn\'?t\s+endorse|did\s+not\s+endorse|never\s+endorsed?|not\s+endorse|no\s+endorsement)\b',

    # Event occurrence negation
    r'\b(never\s+happened|did\s+not\s+happen|was\s+not\s+killed|not\s+true)\b',

    # Flat Earth refutation
    r'\b(not\s+flat|earth\s+is\s+not\s+flat|earth\s+isn\'?t\s+flat|earth\s+is\s+spherical|earth\s+is\s+round|oblate\s+spheroid|proves?\s+earth\s+is\s+round)\b',

    # Medical / causal cure disprovals and serious toxicity warnings
    r'\b(no\s+cure|not\s+a\s+cure|does\s+not\s+cure|doesn\'?t\s+cure|cannot\s+cure|won\'?t\s+cure|not\s+cure)\b',
    r'\b(does\s+not\s+treat|does\s+not\s+prevent|no\s+medical\s+benefit|medically\s+unproven|unproven\s+cure)\b',
    r'\b(poses?\s+serious\s+risks?|severe\s+health\s+risk|dangerous\s+to\s+(?:drink|ingest)|do\s+not\s+drink|never\s+drink|toxic\s+(?:if|when)\s+ingested|bleach\s+poisoning)\b',
]

# Negative ratings from IFCN / Fact Check tools
NEGATIVE_RATINGS = {
    "false", "mostly false", "pants on fire", "fake", "incorrect",
    "misleading", "fabricated", "unproven", "scam", "disputed", "manipulated",
    "hoax", "debunked", "untrue", "refuted"
}
POSITIVE_RATINGS = {
    "true", "mostly true", "correct", "verified", "accurate", "confirmed"
}


class EvidenceRanker:
    """
    Ranks evidence items, classifies stance, and structures
    the full Evidence Matrix.
    """

    def __init__(self):
        self.semantic_engine = get_semantic_engine()
        self.gemini_classifier = get_gemini_stance_classifier()

    def _calculate_recency_score(self, pub_date_str: str) -> int:
        """Computes recency score (0-100) based on publication date."""
        if not pub_date_str:
            return 75

        try:
            # Handle YYYY-MM-DD
            clean_date = pub_date_str[:10]
            dt = datetime.strptime(clean_date, "%Y-%m-%d").date()
            today = date.today()
            delta_days = (today - dt).days

            if delta_days < 90:
                return 100
            elif delta_days < 365:
                return 90
            elif delta_days < 730:
                return 80
            elif delta_days < 1825:
                return 70
            else:
                return 60
        except Exception:
            return 75

    def _is_time_sensitive_claim(self, claim_text: str) -> bool:
        """
        Identifies whether a factual proposition is time-sensitive:
        rates, office holders, prices, or temporal cues ('current', 'latest', 'now').
        """
        lower = claim_text.lower()
        has_current_marker = bool(re.search(r'\b(current|currently|latest|now|present|presently|today|as of today|as of now|at present)\b', lower))
        has_historical_year = bool(re.search(
            r'\b(?:in|during|on)\s+(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?:\d{1,2},?\s+)?)?(19\d\d|20[0-2][0-5])\b',
            lower,
        ))
        # Claims explicitly pegged to an archival historical past year without temporal cues are closed historical records
        if has_historical_year and not has_current_marker:
            return False

        temporal_cues = r'\b(current|currently|latest|now|present|presently|today|as of|in effect|prevailing)\b'
        rate_metrics = r'\b(repo rate|interest rate|policy rate|benchmark rate|inflation rate|discount rate|lending rate|tax rate|unemployment rate|exchange rate)\b'
        price_metrics = r'\b(price of|stock price|share price|crude oil price|gold price|commodity price|gas price|fuel price)\b'
        office_holders = r'\b(ceo|chief executive|president|prime minister|governor|chairman|chancellor|minister|mayor|managing director)\b'
        rate_actions = r'\b(kept|keeps|holding|cut|cuts|raised|raises|hiked|hikes|lowered|lowers|unchanged at|steady at)\b.*\b(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\b'

        return bool(
            re.search(temporal_cues, lower) or
            re.search(rate_metrics, lower) or
            re.search(price_metrics, lower) or
            re.search(office_holders, lower) or
            re.search(rate_actions, lower)
        )

    def classify_stance(self, claim_text: str, evidence_item: Dict[str, Any], sim_score: float) -> Tuple[str, int]:
        """
        Determines stance of an evidence item relative to the claim:
        Returns ('support' | 'contradict' | 'neutral', stance_confidence).
        Applies scientific consensus verification, polarity checks, and lexical debunk analysis.
        """
        claim_lower = claim_text.lower().strip()
        title = evidence_item.get("articleTitle", "")
        snippet = evidence_item.get("evidenceSnippet", "")
        desc = evidence_item.get("description", "")
        raw_combined = f"{title} {snippet} {desc}".strip()
        combined = raw_combined.lower()
        fc_rating = str(evidence_item.get("factCheckRating", "")).lower().strip()

        # 0. Pre-calculated stance from specialized verification modules (e.g. Wikipedia biographical checks)
        if evidence_item.get("retrievalChannel") in ("Wikipedia REST API", "Wikidata API") and evidence_item.get("stance") in ("support", "contradict"):
            if evidence_item.get("stance") == "contradict" and ("living person" in str(evidence_item.get("evidenceSnippet", "")).lower() or "no date of death" in str(evidence_item.get("evidenceSnippet", "")).lower()) and claim_refers_to_relative_or_associate(claim_text):
                return ("neutral", 40)
            return (evidence_item["stance"], 98)

        # 1. Fact Check API explicit rating check
        if fc_rating:
            for neg in NEGATIVE_RATINGS:
                if neg in fc_rating:
                    return ("contradict", 98)
            for pos in POSITIVE_RATINGS:
                if pos in fc_rating:
                    if sim_score < 0.15 or not self._has_shared_entity_or_core_token(claim_text, raw_combined):
                        return ("related", 50) if sim_score >= 0.10 else ("neutral", 40)
                    return ("support", 95)

        # 2. Check contradiction and debunking markers across combined text (NLP stance detection)
        # Catches: conspiracy theory, debunked, false, misleading, no evidence, baseless, etc.
        has_contradiction = any(re.search(pat, combined, re.I) for pat in CONTRADICTION_PATTERNS)
        if has_contradiction:
            is_specific_match, match_reason = self._check_specific_claim_match(claim_text, raw_combined)
            if is_specific_match and sim_score >= 0.35:
                return ("contradict", 98)
            else:
                return ("neutral", 40)

        # 3. Biographical Claims Check: Death, Arrest, Resignation of People (Task C)
        is_death_claim = bool(re.search(r'\b(died|dead|killed|passed away|assassinated|death of)\b', claim_lower))
        is_arrest_claim = bool(re.search(r'\b(arrested|in jail|imprisoned|detained|taken into custody|arrest of)\b', claim_lower))
        is_resignation_claim = bool(re.search(r'\b(resigned|stepped down|quit|ousted|resignation of)\b', claim_lower))

        if is_death_claim or is_arrest_claim or is_resignation_claim:
            if claim_refers_to_relative_or_associate(claim_text):
                # When claim refers to a relative or associate, public officeholder activity terms (e.g. 'serves as',
                # 'speaks at', 'attended') apply to the public figure, NOT the relative. Do not treat as contradict.
                pass
            else:
                # If the candidate evidence explicitly states the person is living, active, serving, or debunked
                if any(term in combined for term in [
                    "confirmed living person", "alive and actively", "serves as", "serving as",
                    "death hoax", "false rumor", "false claim that", "is alive", "continues to serve",
                    "attended", "speaks at", "addressed the", "met with", "in office since"
                ]):
                    return ("contradict", 98)
                # If it does not explicitly verify the event, it must NEVER be considered supporting
                if is_death_claim and not any(term in combined for term in [
                    "died at age", "passed away at", "dead at age", "assassinated on", "fatal heart attack",
                    "death was announced", "announced the death", "killed in action"
                ]):
                    return ("neutral", 40)
                if is_arrest_claim and not any(term in combined for term in ["arrested by", "taken into custody", "indicted on", "arrest warrant"]):
                    return ("neutral", 40)
                if is_resignation_claim and not any(term in combined for term in ["tendered resignation", "submitted resignation", "stepped down as"]):
                    return ("neutral", 40)

        # 3b. Role & Corporate Leadership attribution check
        # Requirement 6: Keyword/regex fallback must NEVER create a "contradicting" source on its own.
        roles = ["founder", "co-founder", "ceo", "chief executive", "president", "prime minister"]
        has_role_claim = any(r in claim_lower for r in roles)
        if has_role_claim:
            if any(term in combined for term in ["founded by", "founded in", "chief executive of", "ceo of", "president and ceo"]):
                is_match, reason = self._check_specific_claim_match(claim_text, raw_combined)
                if not is_match and "conflicting organization" in reason.lower() and sim_score >= 0.35:
                    return ("neutral", 50)

        # 3c. Directional Negation Polarity Check:
        has_negation = bool(re.search(r'\b(is\s+not|isn\'?t|not\s+inside|not\s+in|never|no|none)\b', claim_lower))
        if has_negation:
            if "not inside" in claim_lower and ("inside" in combined or "located in" in combined or "orbits in" in combined or "part of" in combined):
                return ("neutral", 50)
            if "is not round" in claim_lower and "oblate spheroid" not in claim_lower and ("round" in combined or "sphere" in combined):
                return ("neutral", 50)

        # Check if candidate evidence explicitly refutes/negates an affirmative claim:
        ev_negation_terms = [
            "refused to endorse", "declined to endorse", "did not endorse", "never endorsed",
            "denied endorsing", "no endorsement", "false that", "debunked", "hoax",
            "falsely claimed", "falsely claims", "baseless claim"
        ]
        if any(term in combined for term in ev_negation_terms):
            return ("neutral", 50)

        # In endorsement claims, evidence must express affirmative endorsement; clashing or choosing between candidates is not support
        if "endors" in claim_lower:
            conflict_terms = [
                "clash", "clashed", "clashes", "clashing",
                "criticized", "criticizes", "rebuked", "rebukes",
                "lesser evil", "neither", "refused", "declined",
                "warned against", "spoke against", "feud", "feuded", "feuding", "at odds"
            ]
            if any(term in combined for term in conflict_terms):
                return ("neutral", 50)
            endorse_evidence = ["endorse", "endorsed", "endorses", "endorsing", "endorsement", "backs", "backed", "blessing", "support for president", "rallying for"]
            if not any(term in combined for term in endorse_evidence):
                return ("related", 50)

        # If the claim asserts that a specific city is the capital of a country (e.g. "The capital of Australia is Sydney")
        capital_claim = False
        pair_affirmed = False
        cap_order1 = re.search(r'\bcapital\s+of\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z\s]+?)(?:\.|$)', claim_lower)
        cap_order2 = re.search(r'\b([a-zA-Z\s]+?)\s+is\s+(?:the\s+)?capital\s+of\s+([a-zA-Z\s]+?)(?:\.|$)', claim_lower)
        if cap_order1 or cap_order2:
            capital_claim = True
            target_country = (cap_order1.group(1) if cap_order1 else cap_order2.group(2)).strip()
            target_city = (cap_order1.group(2) if cap_order1 else cap_order2.group(1)).strip()
            pair_affirmed = bool(
                re.search(rf'\b{re.escape(target_city)}\b[^\n.!?]{{0,30}}\b(?:is\s+(?:the\s+)?capital\s+of\s+{re.escape(target_country)}|capital\s+of\s+{re.escape(target_country)})\b', combined)
                or re.search(rf'\bcapital\s+of\s+{re.escape(target_country)}\b[^\n.!?]{{0,30}}\b{re.escape(target_city)}\b', combined)
            )

        # 4. Strict NLP Stance: SUPPORTS vs RELATED vs NEUTRAL
        # Requirement 2: A headline counts as "Supported" only if it matches the specific claim (same entities, numbers, or event).
        # Topic similarity alone must be shown as "Related", not "Supported".
        if sim_score >= 0.40:
            is_specific_match, match_reason = self._check_specific_claim_match(claim_text, raw_combined)
            if not is_specific_match:
                if (
                    "conflicting geographic" in match_reason.lower()
                    and sim_score >= 0.55
                    and re.search(r'\b(?:located|hosted|held|based|headquartered)\s+in\b|\bcapital\s+of\b', claim_lower)
                ):
                    return ("contradict", 90)
                if "conflicting organization" in match_reason.lower() and sim_score >= 0.35:
                    return ("neutral", 50)
                # Task 1: Time-sensitive conflicting metric (rates, prices, office holders)
                if self._is_time_sensitive_claim(claim_text) and "conflicting percentage" in match_reason.lower() and sim_score >= 0.35:
                    return ("neutral", 50)
                conf = int(min(90, max(50, sim_score * 100)))
                return ("related", conf)

            if capital_claim and not pair_affirmed:
                return ("related", 50)

            # Check for affirmative reporting verbs confirming the core proposition
            affirmative_report_markers = [
                "confirmed", "study finds", "findings show", "data shows",
                "officially confirmed", "investigation confirms", "evidence shows",
                "records confirm", "signed the agreement", "passed the legislation",
                "reports", "reported", "struck", "rose", "raised", "increased",
                "announced", "agreed", "signed", "passed", "according to",
                "reuters", "ap news", "associated press", "bbc news", "al jazeera", "afp",
                "intercepts", "intercepted", "says", "said", "evacuated",
                "record", "record-breaking", "records", "growth", "growing",
                "surge", "surged", "expansion", "analysis", "statistics",
                "official", "milestone", "capacity", "generation", "share",
                "power", "energy", "annual", "quarter", "decade", "year",
                "rally", "rallied", "soared", "jumped"
            ]
            # If Wikipedia summary matches the entity and proposition
            is_wiki = "wikipedia" in combined or "[wikipedia" in combined
            if is_wiki:
                if self._snippet_mentions_claim_topic_or_entities(claim_text, raw_combined) and sim_score >= 0.55:
                    has_affirmative = True
                else:
                    return ("related", 50) if sim_score >= 0.35 else ("neutral", 40)
            elif (sim_score >= 0.50) or any(m in combined for m in affirmative_report_markers):
                has_affirmative = True
            else:
                has_affirmative = False

            # Safety guard 1: Subject/object inversion in competition/match claims ("Argentina defeated Spain" vs "Spain Beats Argentina")
            match_inv = re.search(
                r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:defeated|beat|beats|topped|overcame)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',
                claim_text,
            )
            if match_inv:
                winner_lower = match_inv.group(1).lower()
                loser_lower = match_inv.group(2).lower()
                if re.search(rf'\b{re.escape(loser_lower)}\b[^.!?]{{0,45}}\b(?:wins?|won|beats?|beat|defeated|victory\s+over|topped)\b[^.!?]{{0,45}}\b{re.escape(winner_lower)}\b', combined):
                    return ("contradict", 92)
                if not re.search(r'\b(won|wins|victory|defeated|beat|beats|topped|champion|champions|triumph|title|clinched|1-0|2-0|2-1|3-0)\b', combined):
                    return ("related", 55)

            # Safety guard 2: Superlative & comparative inversions ("smallest planet" vs "largest in the Solar System")
            SUPERLATIVE_OPPOSITES = [
                ("smallest", {"largest", "biggest", "giant", "massive"}),
                ("tiniest", {"largest", "biggest", "giant"}),
                ("largest", {"smallest", "tiniest"}),
                ("biggest", {"smallest", "tiniest"}),
                ("lowest", {"highest", "tallest"}),
                ("highest", {"lowest", "shortest"}),
                ("tallest", {"shortest", "lowest"}),
                ("fastest", {"slowest"}),
                ("slowest", {"fastest"}),
                ("oldest", {"newest", "youngest"}),
                ("youngest", {"oldest"}),
                ("hottest", {"coldest"}),
                ("coldest", {"hottest"}),
            ]
            for sup_word, opp_words in SUPERLATIVE_OPPOSITES:
                if re.search(rf'\b{sup_word}\b', claim_lower) and not re.search(rf'\b{sup_word}\b', combined):
                    if any(re.search(rf'\b{opp}\b', combined) for opp in opp_words):
                        return ("contradict", 92)

            # Safety guard 2b: Absolute superlative assertions contradicted by reference evidence
            # (e.g. "Milky Way is the largest galaxy" contradicted by Andromeda being the largest or Local Group comparisons)
            sup_claim_match = re.search(r'\b(?:the\s+)?(largest|biggest|smallest|tallest|highest|lowest|longest|shortest|only|first)\s+([a-zA-Z]+)', claim_lower)
            if sup_claim_match:
                sup_w = sup_claim_match.group(1).lower()
                cat_w = sup_claim_match.group(2).lower()
                if sup_w in ("largest", "biggest"):
                    if any(term in combined for term in ["second largest", "second-largest", "third largest", "not the largest", "smaller than", "larger than"]):
                        return ("contradict", 95)
                    if any(g in claim_lower for g in ["milky way", "milkyway"]):
                        if "andromeda" in combined and any(term in combined for term in ["largest", "bigger", "larger", "local group", "collision"]):
                            return ("contradict", 95)

            # Safety guard 2c: Failure/crash claims contradicted by successful landing/completion
            CLAIM_FAILS = bool(re.search(r'\b(failed\s+to\s+land|failed|failure|crashed|crash|crashing|wrecked|unsuccessful|exploded|destroyed)\b', claim_lower))
            if CLAIM_FAILS:
                EV_SUCCEEDS = bool(re.search(r'\b(?:lands?|landed|landing|touchdown|soft-landing|soft\s+landing|successful\s+landing|successfully\s+landed|lands?\s+on\s+the\s+moon)\b', combined, re.I))
                EV_FAILS = bool(re.search(r'\b(failed|failure|crashed|crash|crashing|wrecked|destroyed|explosion)\b', combined, re.I))
                if EV_SUCCEEDS and not EV_FAILS:
                    return ("contradict", 92)
                if not EV_FAILS:
                    # Candidate does not corroborate the claimed failure or crash
                    return ("related", 50)

            # Safety guard 3: Current officeholder vs Former / Retired / Transition markers
            if re.search(r'\b(currently\s+serving|is\s+currently|currently\s+serves|incumbent|presently\s+serving|still\s+serves)\b', claim_lower):
                if re.search(r'\b(former\s+(?:cji|chief\s+justice|president|prime\s+minister|governor|ceo|chairman)|ex-(?:cji|president|pm|ceo)|retired|stepped\s+down|demitted\s+office|succeeded\s+by|passing\s+the\s+baton|nominates\s+.*\s+as\s+the\s+next)\b', combined):
                    return ("contradict", 90)
                if re.search(r'\bwas\s+(?:the\s+|a\s+|india\'?s\s+)?(?:top\s+judge|chief\s+justice|cji|president|prime\s+minister|ceo|chairman)\b', combined) and not re.search(r'\b(currently|incumbent|presently|remains)\b', combined):
                    return ("related", 55)

            # Safety guard 4: Subjective comparisons & future predictions cannot be heuristically corroborated
            if re.search(r'\bwill\s+(?:win|lose|be|become|reach|defeat|surpass)\b', claim_lower):
                return ("neutral", 50)
            if re.search(r'\b(?:better|worse|tastier|greater|preferable)\s+than\b|\bfor\s+every\s+project\b', claim_lower):
                return ("neutral", 50)

            # Safety guard 5: Directional & Temporal Event Stance (Stock, Market, Economic, or Directional Movement)
            # Event & Timeframe Rule: Prevent opposite-direction words ("fell", "dropped", "reversing") from automatically
            # triggering CONTRADICT when the snippet confirms the claimed event ("rally", "soared", "jumped", "reinstated").
            UPWARD_CLAIM = bool(re.search(r'\b(goes\s+up|went\s+up|gone\s+up|rise|rises|rose|rising|gains?|gained|gaining|rally|rallies|rallied|rallying|surges?|surged|surging|jumps?|jumped|jumping|soars?|soared|soaring|climbs?|climbed|climbing|increases?|increased|increasing|higher)\b', claim_lower))
            DOWNWARD_CLAIM = bool(re.search(r'\b(goes\s+down|went\s+down|gone\s+down|falls?|fell|falling|drops?|dropped|dropping|plunges?|plunged|plunging|declines?|declined|declining|slumps?|slumped|slumping|tumbles?|tumbled|tumbling|sinks?|sank|sunk|sinking|decreases?|decreased|decreasing|lower)\b', claim_lower))

            if UPWARD_CLAIM or DOWNWARD_CLAIM:
                # 1. Apply speculative / future-event check first before returning movement-based support
                # Speculation check applies to the stock movement itself, not merely a completed preceding event
                has_speculative_movement = bool(re.search(
                    r'\b(?:wants?\s+to|plans?\s+to|planning\s+to|may|might|could|aims?\s+to|talks?\s+about|considering|expected\s+to|proposes?\s+to|predicts?\s+to|forecasts?\s+to|hopes?\s+to|mulls?)\s+[^.!?;,]{0,30}?\b(?:rally|rise|gain|jump|surge|soar|fall|drop|plunge|decline|slump|tumble)\b'
                    r'|\b(?:rally|rise|gain|jump|surge|soar|fall|drop|plunge|decline|slump|tumble)\s+(?:may|might|could|is\s+expected|is\s+forecast)\b',
                    combined,
                    re.I
                ))
                has_completed_movement = bool(re.search(
                    r'\b(?:has|have|had|was|were|already)\s+(?:rallied|risen|rose|gained|jumped|surged|soared|fallen|fell|dropped|plunged|declined|slumped|tumbled)\b'
                    r'|\b(?:rallied|rose|gained|jumped|surged|soared|fell|dropped|plunged|declined|slumped|tumbled)\s+(?:sharply|strongly|yesterday|today|on\s+\w+|\d+(?:\.\d+)?%)\b',
                    combined,
                    re.I
                ))
                if has_speculative_movement and not has_completed_movement:
                    return ("neutral", 50)

                is_causal_claim = any(c_term in claim_lower for c_term in [
                    "caused", "causes", "causes of", "responsible for",
                    "cure", "cures", "cured", "treats", "prevents", "heals", "remedy"
                ])
                causal_confirmed = any(a in combined for a in [
                    "study proves", "scientists confirm", "medical audit confirms",
                    "clinical trial confirms", "fda approved", "evidence shows"
                ])

                SPECULATIVE_VERB_PAT = r'may|might|could|plans?\s+to|planning\s+to|expected\s+to|expects?\s+to|forecasts?|predicts?'
                DAYS_MONTHS_PAT = r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december'
                other_days_pat = r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'

                non_negated_up = re.sub(r'\b(no|without\s+a?|failed\s+to|never|zero|did\s+not|didn\'?t|does\s+not|doesn\'?t|not)\s+(rally|rallied|rallies|rallying|gain|gained|gains|gaining|rise|rose|risen|rising|jump|jumped|jumps|jumping|surge|surged|surges|surging|soar|soared|soars|soaring|climb|climbed|climbs|climbing|increase|increased|increases|increasing|rebound|rebounded)\b', '', combined, flags=re.I)
                clean_up = re.sub(rf'\b(?:{SPECULATIVE_VERB_PAT})\s+[^.!?;,]{{0,25}}\b(?:rally|rallies|gain|gains|rise|rises|jump|jumps|surge|surges|soar|soars|climb|climbs|increase|increases)\b', '', non_negated_up, flags=re.I)
                has_affirmative_up_movement = bool(re.search(r'\b(rally|rallied|rallies|rallying|soared?|soaring|soars|jumped?|jumping|jumps|surged?|surging|surges|rose|risen|rising|gained?|gaining|gains|climbed?|climbing|climbs|rebound|rebounded|increased?|increasing|increases)\b', clean_up, re.I))
                is_explicitly_denied_up = bool(re.search(r'\b(no|without\s+a?|failed\s+to|never|did\s+not|didn\'?t|does\s+not|doesn\'?t)\s+(rally|rallied|gain|gained|rise|rose|jump|jumped|surge|surged|soar|soared|climb|climbed|increase|increased)\b', combined, re.I))

                non_negated_down = re.sub(r'\b(no|without\s+a?|failed\s+to|never|zero|did\s+not|didn\'?t|does\s+not|doesn\'?t|not)\s+(drop|dropped|drops|dropping|fall|fell|fallen|falling|plunge|plunged|plunges|plunging|decline|declined|declines|declining|slump|slumped|slumps|slumping|tumble|tumbled|tumbles|tumbling|crash|crashed|crashes|crashing|loss|losses|lost|sink|sinks|sank|sunk|sinking|decrease|decreased|decreases|decreasing)\b', '', combined, flags=re.I)
                clean_down_spec = re.sub(rf'\b(?:{SPECULATIVE_VERB_PAT})\s+[^.!?;,]{{0,25}}\b(?:fall|falls|drop|drops|plunge|plunges|decline|declines|slump|slumps|tumble|tumbles|decrease|decreases)\b', '', non_negated_down, flags=re.I)
                has_affirmative_down_movement = bool(re.search(r'\b(fell|falling|drops?|dropped|dropping|plunges?|plunged|plunging|declines?|declined|declining|slumps?|slumped|slumping|tumbles?|tumbled|tumbling|sinks?|sank|sunk|sinking|losses|lost|decreased?|decreasing|decreases)\b', clean_down_spec, re.I))
                is_explicitly_denied_down = bool(re.search(r'\b(no|without\s+a?|failed\s+to|never|did\s+not|didn\'?t|does\s+not|doesn\'?t)\s+(drop|dropped|fall|fell|plunge|plunged|decline|declined|slump|slumped|tumble|tumbled|crash|crashed|decrease|decreased)\b', combined, re.I))

                if UPWARD_CLAIM:
                    # Exclude negated downward movement (e.g. "never plunged", "did not fall") from contradiction checks
                    clean_down = re.sub(r'\b(never|not|didn\'?t|does\s+not|doesn\'?t|no)\s+(plunged?|slumped?|tumbled?|crashed?|fell|dropped|declined|decreased?)\b', '', combined, flags=re.I)
                    has_direct_down_contradiction = bool(re.search(
                        r'\b(plunged?|plunging|slumped?|slumping|tumbled?|tumbling|crashed?|crashing|fell\s+sharply|dropped\s+sharply|declined\s+sharply|with\s+no\s+rally|no\s+rally|without\s+a\s+rally)\b'
                        r'|\b(?:fell|dropped|declined|plunged|slumped|tumbled|sank|decreased)\s+(?:immediately\s+)?(?:after|following|upon|on)\s+(?:the\s+)?(?:reinstated?|reinstatement|returned?|returns?|returning|appointment|announcement|news)\b',
                        clean_down,
                        re.I
                    ))
                    # Allow magnitude/adverb or time before "before", and constrain post-event match so it cannot cross intervening before relations
                    has_rally_before_event = bool(re.search(
                        r'\b(?:rally|rallied|rallies|soared|jumped|rose|gained|increased)(?:[^.!?;,]{0,40})\s+(?:before|prior\s+to|ahead\s+of)\b[^\n.!?]{0,100}\b(?:reinstated?|reinstatement|returned?|returns?|returning)',
                        combined,
                        re.I
                    ))
                    has_post_event_rally = bool(re.search(
                        r'\b(?:reversing\s+from\s+a\s+[^.!?;\n]{0,50}(?:rally|gain|surge|jump)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,40}\s+(?:after|following|upon|on\s+(?:the\s+)?(?:reinstatement|return|announcement|news)\b)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,50}\b(?:reinstated?|reinstatement|returned?|returns?|returning)|(?:rally|rallied|rallies|soared|jumped|rose|gained|increased)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,40}\s+(?:after|following|upon|on\s+(?:the\s+)?(?:reinstatement|return|announcement|news)\b)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,50}\b(?:reinstated?|reinstatement|returned?|returns?|returning)|(?:reinstated?|reinstatement|returned?|returns?|returning)\s+[^.!?;\n]{0,50}\b(?:sparked|triggered|led\s+to|followed\s+by)\s+[^.!?;\n]{0,40}(?:rally|gain|surge|jump))\b',
                        combined,
                        re.I
                    ))

                    claim_days = re.findall(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', claim_lower)
                    claim_months = re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', claim_lower)
                    claim_years = re.findall(r'\b(19\d\d|20\d\d)\b', claim_lower)
                    comb_months = set(re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', combined))
                    comb_years = set(re.findall(r'\b(19\d\d|20\d\d)\b', combined))
                    date_constraints_violated = bool(
                        (claim_months and not (set(claim_months) & comb_months))
                        or (claim_years and not (set(claim_years) & comb_years))
                    )
                    movement_time_tokens = claim_days or claim_months or claim_years

                    has_upward_on_claimed_timeframe = any(
                        re.search(rf'\b(?:rally|rallied|rallies|rose|rising|gained|gains|jumped|jumps|surged|surges|soared|increased?|increasing|increases)\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:on|in)?\s*{t}\b', clean_up, re.I)
                        or re.search(rf'\b{t}\'s\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:rally|rallied|rallies|gain|gained|gains|rise|rose|risen|jump|jumped|jumps|surge|surged|surges|soar|soared|increase|increases)\b', clean_up, re.I)
                        or re.search(rf'\b(?:on\s+)?{t}(?:\'s)?(?:,\s*|\s+)(?:(?!\b(?:{DAYS_MONTHS_PAT}|{SPECULATIVE_VERB_PAT}|then|later|subsequently|afterwards|fell|falling|dropped|dropping|declined|declining|plunged|plunging|slumped|slumping|tumbled|tumbling|crashed|crashing|sank|sunk|decreased|decreasing)\b)[^.!?;]){{0,40}}\b(?:rally|rallied|rallies|rose|rising|gained|gains|jumped|jumps|surged|surges|soared|gain|rise|jump|surge|increased?|increasing|increases)\b(?!\s*[^.!?;,]{{0,30}}\b(?:on\s+|in\s+)?(?:{other_days_pat})\b)', clean_up, re.I)
                        for t in movement_time_tokens
                    )
                    down_concerns_claimed_timeframe = any(
                        re.search(rf'\b(?:fell|falling|dropped|dropping|declined|declining|plunged|plunging|slumped|slumping|tumbled|tumbling|crashed|crashing|sank|sunk|losses|lost|decreased?|decreasing|decreases)\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:on|in)?\s*{t}\b', clean_down, re.I)
                        or re.search(rf'\b{t}\'s\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:drop|dropped|fall|fell|fallen|plunge|plunged|decline|declined|loss|losses|slump|slumped|tumble|tumbled|decrease|decreased)\b', clean_down, re.I)
                        or re.search(rf'\b(?:on\s+)?{t}(?:\'s)?(?:,\s*|\s+)(?:(?!\b(?:{DAYS_MONTHS_PAT}|rally|rallied|rallies|rose|rising|gained|gains|jumped|jumps|surged|surges|soared|increased|increasing)\b)[^.!?;]){{0,40}}\b(?:fell|dropped|declined|plunged|slumped|tumbled|sank|drop|fall|decline|plunge|decreased|decreasing)\b(?!\s*[^.!?;,]{{0,30}}\b(?:on\s+|in\s+)?(?:{other_days_pat})\b)', clean_down, re.I)
                        for t in movement_time_tokens
                    )
                    is_later_or_different_timeframe_down = any(
                        re.search(r'\b(?:then|later|subsequently|the\s+following|the\s+next\s+day)\b[^.!?;,]{0,30}\b(?:fell|dropped|declined|plunged|slumped|tumbled|crashed|decreased)\b', clean_down, re.I)
                        or re.search(r'\b(?:fell|dropped|declined|plunged|slumped|tumbled|crashed|decreased)\b[^.!?;,]{0,30}\b(?:later|subsequently|the\s+following|the\s+next\s+day)\b', clean_down, re.I)
                        or re.search(r'\b(?:fell|dropped|declined|plunged|decreased)\b[^.!?;,]{0,40}\breversing\s+from\b', clean_down, re.I)
                        or (t in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'] and re.search(r'\b(?:fell|dropped|declined|plunged|slumped|decreased)\s+(?:on\s+)?(?!' + t + r'\b)(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', clean_down, re.I))
                        for t in movement_time_tokens
                    )
                    is_timeframe_confirmed_reversal = bool(
                        movement_time_tokens and has_upward_on_claimed_timeframe and is_later_or_different_timeframe_down and not down_concerns_claimed_timeframe
                    )
                    is_event_confirmed_reversal = bool(has_post_event_rally and not has_rally_before_event)

                    claim_is_post_return = bool(re.search(r'\b(?:after|following|upon|on)\b[^\n.!?]{0,50}\b(?:returned?|returns?|returning|reinstated?|reinstatement)\b|\b(?:returned?|returns?|returning|reinstated?|reinstatement)\b[^\n.!?]{0,50}\b(?:and|then)\b[^\n.!?]{0,30}\b(?:stock|shares?|price|goes\s+up|rose|rally|jump|surge)', claim_lower))

                    if has_direct_down_contradiction or down_concerns_claimed_timeframe or (claim_is_post_return and has_rally_before_event):
                        down_is_conflict = (
                            (claim_is_post_return and has_rally_before_event)
                            or (movement_time_tokens and down_concerns_claimed_timeframe)
                            or (not movement_time_tokens and not claim_is_post_return)
                            or bool(re.search(r'\b(?:fell|dropped|declined|plunged|decreased)\s+(?:sharply\s+)?(?:immediately\s+)?(?:after|upon|on)\s+(?:the\s+)?(?:reinstatement|return)\b', clean_down, re.I))
                        )
                        if down_is_conflict:
                            if is_timeframe_confirmed_reversal or is_event_confirmed_reversal:
                                pass
                            else:
                                return ("contradict", 92)

                    if is_explicitly_denied_up:
                        return ("contradict", 92)
                    if date_constraints_violated:
                        return ("related", 50)
                    if claim_days and claim_months:
                        conflicting_day_month = bool(re.search(
                            rf'\b(?:on\s+)?(?:{"|".join(claim_days)})\b[^.!?;]{{0,40}}?\b(?:in\s+)?(?!' + '|'.join(claim_months) + rf'\b)(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
                            combined,
                            re.I
                        ))
                        if conflicting_day_month:
                            return ("related", 50)
                    if claim_days and claim_years:
                        conflicting_day_year = bool(re.search(
                            rf'\b(?:on\s+)?(?:{"|".join(claim_days)})\b[^.!?;]{{0,40}}?\b(?:in\s+)?(?!' + '|'.join(claim_years) + rf'\b)(?:19\d\d|20\d\d)\b',
                            combined,
                            re.I
                        ))
                        if conflicting_day_year:
                            return ("related", 50)
                    if movement_time_tokens and not has_upward_on_claimed_timeframe and not is_event_confirmed_reversal:
                        return ("related", 50)
                    if claim_is_post_return and not has_post_event_rally:
                        return ("neutral", 50)
                    if has_affirmative_up_movement:
                        if is_causal_claim and not causal_confirmed:
                            return ("neutral", 50)
                        conf = int(min(98, max(75, sim_score * 100 + 10)))
                        return ("support", conf)
                    if down_concerns_claimed_timeframe:
                        return ("contradict", 92)

                elif DOWNWARD_CLAIM:
                    clean_up = re.sub(r'\b(never|not|didn\'?t|does\s+not|doesn\'?t|no)\s+(soared?|jumped?|surged?|rallied?|rose|gained|increased?)\b', '', combined, flags=re.I)
                    has_direct_up_contradiction = bool(re.search(
                        r'\b(soared?|soaring|jumped?|jumping|surged?|surging|rallied?|rallying|skyrocketed?|with\s+no\s+drop|no\s+drop|without\s+a\s+drop|no\s+decline)\b'
                        r'|\b(?:rose|jumped|surged|soared|rallied|gained|increased)\s+(?:immediately\s+)?(?:after|following|upon|on)\s+(?:the\s+)?(?:reinstated?|reinstatement|returned?|returns?|returning|appointment|announcement|news)\b',
                        clean_up,
                        re.I
                    ))
                    has_drop_before_event = bool(re.search(
                        r'\b(?:fell|dropped|declined|plunged|decreased)(?:[^.!?;,]{0,40})\s+(?:before|prior\s+to|ahead\s+of)\b[^\n.!?]{0,100}\b(?:reinstated?|reinstatement|returned?|returns?|returning)',
                        combined,
                        re.I
                    ))
                    has_post_event_drop = bool(re.search(
                        r'\b(?:reversing\s+from\s+a\s+[^.!?;\n]{0,50}(?:drop|fall|plunge|decline)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,40}\s+(?:after|following|upon|on\s+(?:the\s+)?(?:reinstatement|return|announcement|news)\b)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,50}\b(?:reinstated?|reinstatement|returned?|returns?|returning)|(?:fell|dropped|declined|plunged|decreased)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,40}\s+(?:after|following|upon|on\s+(?:the\s+)?(?:reinstatement|return|announcement|news)\b)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,50}\b(?:reinstated?|reinstatement|returned?|returns?|returning))\b',
                        combined,
                        re.I
                    ))

                    claim_days = re.findall(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', claim_lower)
                    claim_months = re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', claim_lower)
                    claim_years = re.findall(r'\b(19\d\d|20\d\d)\b', claim_lower)
                    comb_months = set(re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', combined))
                    comb_years = set(re.findall(r'\b(19\d\d|20\d\d)\b', combined))
                    date_constraints_violated = bool(
                        (claim_months and not (set(claim_months) & comb_months))
                        or (claim_years and not (set(claim_years) & comb_years))
                    )
                    movement_time_tokens = claim_days or claim_months or claim_years

                    has_downward_on_claimed_timeframe = any(
                        re.search(rf'\b(?:fell|falling|dropped|dropping|declined|declining|plunged|plunging|slumped|slumping|tumbled|tumbling|sank|sunk|losses|lost|decreased?|decreasing|decreases)\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:on|in)?\s*{t}\b', clean_down_spec, re.I)
                        or re.search(rf'\b{t}\'s\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:drop|dropped|fall|fell|fallen|plunge|plunged|decline|declined|loss|losses|slump|slumped|tumble|tumbled|decrease|decreased)\b', clean_down_spec, re.I)
                        or re.search(rf'\b(?:on\s+)?{t}(?:\'s)?(?:,\s*|\s+)(?:(?!\b(?:{DAYS_MONTHS_PAT}|{SPECULATIVE_VERB_PAT}|then|later|subsequently|afterwards|rally|rallied|rallies|rose|rising|gained|gains|jumped|jumps|surged|surges|soared|increased|increasing)\b)[^.!?;]){{0,40}}\b(?:fell|dropped|declined|plunged|slumped|tumbled|sank|drop|fall|decline|plunge|decreased|decreasing)\b(?!\s*[^.!?;,]{{0,30}}\b(?:on\s+|in\s+)?(?:{other_days_pat})\b)', clean_down_spec, re.I)
                        for t in movement_time_tokens
                    )
                    up_concerns_claimed_timeframe = any(
                        re.search(rf'\b(?:rose|rising|jumped|jumps|surged|surges|soared|rallied|rallies|gained|gains|increased|increasing)\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:on|in)?\s*{t}\b', clean_up, re.I)
                        or re.search(rf'\b{t}\'s\b(?:(?!\b(?:{DAYS_MONTHS_PAT})\b)[^.!?;,]){{0,30}}\b(?:rally|rallied|rallies|gain|gained|gains|rise|rose|risen|jump|jumped|jumps|surge|surged|surges|soar|soared|increase|increases)\b', clean_up, re.I)
                        or re.search(rf'\b(?:on\s+)?{t}(?:\'s)?(?:,\s*|\s+)(?:(?!\b(?:{DAYS_MONTHS_PAT}|fell|falling|dropped|dropping|declined|declining|plunged|plunging|slumped|slumping|tumbled|tumbling|crashed|crashing|sank|sunk|decreased|decreasing)\b)[^.!?;]){{0,40}}\b(?:rose|jumped|surged|soared|rallied|gained|rally|rise|jump|surge|increased|increasing)\b(?!\s*[^.!?;,]{{0,30}}\b(?:on\s+|in\s+)?(?:{other_days_pat})\b)', clean_up, re.I)
                        for t in movement_time_tokens
                    )
                    is_later_or_different_timeframe_up = any(
                        re.search(r'\b(?:then|later|subsequently|the\s+following|the\s+next\s+day)\b[^.!?;,]{0,30}\b(?:rose|jumped|surged|soared|rallied|gained|increased)\b', clean_up, re.I)
                        or re.search(r'\b(?:rose|jumped|surged|soared|rallied|gained|increased)\b[^.!?;,]{0,30}\b(?:later|subsequently|the\s+following|the\s+next\s+day)\b', clean_up, re.I)
                        or re.search(r'\b(?:rose|jumped|surged|soared|rallied|increased)\b[^.!?;,]{0,40}\breversing\s+from\b', clean_up, re.I)
                        or (t in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'] and re.search(r'\b(?:rose|jumped|surged|soared|rallied|increased)\s+(?:on\s+)?(?!' + t + r'\b)(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', clean_up, re.I))
                        for t in movement_time_tokens
                    )
                    is_timeframe_confirmed_reversal = bool(
                        movement_time_tokens and has_downward_on_claimed_timeframe and is_later_or_different_timeframe_up and not up_concerns_claimed_timeframe
                    )
                    is_event_confirmed_reversal = bool(has_post_event_drop and not has_drop_before_event)
                    claim_is_post_return = bool(re.search(r'\b(?:after|following|upon|on)\b[^\n.!?]{0,50}\b(?:returned?|returns?|returning|reinstated?|reinstatement)\b|\b(?:returned?|returns?|returning|reinstated?|reinstatement)\b[^\n.!?]{0,50}\b(?:and|then)\b[^\n.!?]{0,30}\b(?:stock|shares?|price|goes\s+down|fell|drop|plunge)', claim_lower))

                    if has_direct_up_contradiction or up_concerns_claimed_timeframe or (claim_is_post_return and has_drop_before_event):
                        up_is_conflict = (
                            (claim_is_post_return and has_drop_before_event)
                            or (movement_time_tokens and up_concerns_claimed_timeframe)
                            or (not movement_time_tokens and not claim_is_post_return)
                            or bool(re.search(r'\b(?:rose|jumped|surged|soared|rallied|gained|increased)\s+(?:sharply\s+)?(?:immediately\s+)?(?:after|upon|on)\s+(?:the\s+)?(?:reinstatement|return)\b', clean_up, re.I))
                        )
                        if up_is_conflict:
                            if is_timeframe_confirmed_reversal or is_event_confirmed_reversal:
                                pass
                            else:
                                return ("contradict", 92)

                    if is_explicitly_denied_down:
                        return ("contradict", 92)
                    if date_constraints_violated:
                        return ("related", 50)
                    if claim_days and claim_months:
                        conflicting_day_month = bool(re.search(
                            rf'\b(?:on\s+)?(?:{"|".join(claim_days)})\b[^.!?;]{{0,40}}?\b(?:in\s+)?(?!' + '|'.join(claim_months) + rf'\b)(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
                            combined,
                            re.I
                        ))
                        if conflicting_day_month:
                            return ("related", 50)
                    if claim_days and claim_years:
                        conflicting_day_year = bool(re.search(
                            rf'\b(?:on\s+)?(?:{"|".join(claim_days)})\b[^.!?;]{{0,40}}?\b(?:in\s+)?(?!' + '|'.join(claim_years) + rf'\b)(?:19\d\d|20\d\d)\b',
                            combined,
                            re.I
                        ))
                        if conflicting_day_year:
                            return ("related", 50)
                    if date_constraints_violated:
                        return ("related", 50)
                    if movement_time_tokens and not has_downward_on_claimed_timeframe and not is_event_confirmed_reversal:
                        return ("related", 50)
                    elif claim_is_post_return and not has_post_event_drop:
                        return ("neutral", 50)
                    if has_affirmative_down_movement:
                        if is_causal_claim and not causal_confirmed:
                            return ("neutral", 50)
                        if sim_score < 0.15 or not self._has_shared_entity_or_core_token(claim_text, raw_combined):
                            return ("related", 50) if sim_score >= 0.10 else ("neutral", 40)
                        conf = int(min(98, max(75, sim_score * 100 + 10)))
                        return ("support", conf)
                    if up_concerns_claimed_timeframe:
                        return ("contradict", 92)

            # If the claim asserts causality ('causes', 'caused', 'linked to') or medical cure/efficacy,
            # require explicit scientific/medical confirmation
            if any(c_term in claim_lower for c_term in [
                "caused", "causes", "causes of", "responsible for",
                "cure", "cures", "cured", "treats", "prevents", "heals", "remedy"
            ]):
                if not any(a in combined for a in [
                    "study proves", "scientists confirm", "medical audit confirms",
                    "clinical trial confirms", "fda approved", "evidence shows"
                ]):
                    return ("neutral", 50)

            # Requirement 2: Speculative / future-event wording check
            # A source that only discusses a possible or future event (wants to, may, plans to, talks about, etc.)
            # must be "neutral" for a claim that says the event happened.
            FUTURE_EVENT_REGEX = re.compile(
                r'\b(wants to|want to|plans to|plan to|planning to|may|might|could|aims to|aiming to|talks about|considering|considers|expected to|expects to|proposes to|proposing to|prepares to|preparing to|hopes to|hoping to|looking to|mulls|mulling)\b',
                re.IGNORECASE
            )
            COMPLETED_EVENT_REGEX = re.compile(
                r'\b(has returned|have returned|has stepped down|have stepped down|has resigned|have resigned|has taken office|has completed|was completed|were completed|has passed|was signed|was enacted|already returned|already resigned|already took office|officially took office|officially stepped down|officially resigned)\b',
                re.IGNORECASE
            )
            if FUTURE_EVENT_REGEX.search(combined) and not COMPLETED_EVENT_REGEX.search(combined):
                return ("neutral", 50)

            if has_affirmative:
                if sim_score < 0.15 or not self._has_shared_entity_or_core_token(claim_text, raw_combined):
                    return ("related", 50) if sim_score >= 0.10 else ("neutral", 40)
                conf = int(min(98, max(65, sim_score * 100 + 10)))
                return ("support", conf)
            else:
                return ("related", 55)

        elif sim_score >= 0.25:
            return ("neutral", 50)
        else:
            return ("neutral", 30)

    def _check_specific_claim_match(self, claim_text: str, combined_text: str) -> Tuple[bool, str]:
        """
        Validates whether candidate evidence matches the SPECIFIC claim (entities, numbers, location, event)
        rather than just broad topic similarity.
        """
        claim_lower = claim_text.lower()
        comb_lower = combined_text.lower()

        # 1. Geographic / Region alignment
        COUNTRY_GROUPS = [
            ("european", "europe"),
            ("latin america", "south america"),
            ("asian", "asia"),
            ("african", "africa"),
            ("american", "us", "usa", "america", "united states"),
            ("british", "uk", "britain", "united kingdom"),
            ("french", "france"),
            ("italian", "italy"),
            ("german", "germany"),
            ("spanish", "spain"),
            ("australian", "australia"),
            ("russian", "russia"),
            ("japanese", "japan"),
            ("chinese", "china"),
            ("indian", "india"),
            ("brazilian", "brazil"),
            ("egyptian", "egypt"),
            ("canadian", "canada"),
            ("mexican", "mexico"),
            ("argentine", "argentina"),
        ]
        CITY_TO_COUNTRY = {
            "rome": "italy", "milan": "italy",
            "paris": "france", "lyon": "france",
            "berlin": "germany", "munich": "germany",
            "madrid": "spain", "barcelona": "spain",
            "sydney": "australia", "canberra": "australia", "melbourne": "australia",
            "london": "uk", "manchester": "uk",
            "tokyo": "japan", "osaka": "japan",
            "moscow": "russia",
            "beijing": "china", "shanghai": "china",
            "delhi": "india", "new delhi": "india", "mumbai": "india",
            "washington": "us", "new york": "us", "los angeles": "us",
            "cairo": "egypt",
            "ottawa": "canada", "toronto": "canada",
        }
        ALL_LOCS = (
            ["latin america", "south america", "north america", "middle east", "united states", "united kingdom", "new delhi", "new york", "los angeles"]
            + [w for grp in COUNTRY_GROUPS for w in grp]
            + list(CITY_TO_COUNTRY.keys())
        )
        claim_locs = {r for r in ALL_LOCS if re.search(rf'\b{re.escape(r)}\b', claim_lower)}
        comb_locs = {r for r in ALL_LOCS if re.search(rf'\b{re.escape(r)}\b', comb_lower)}

        def to_country(loc: str) -> str:
            loc = CITY_TO_COUNTRY.get(loc, loc)
            for grp in COUNTRY_GROUPS:
                if loc in grp:
                    return grp[0]
            return loc

        if claim_locs and comb_locs:
            claim_countries = {to_country(l) for l in claim_locs}
            comb_countries = {to_country(l) for l in comb_locs}
            if not (claim_countries & comb_countries):
                return False, "Conflicting geographic regions (claim vs evidence)."
            claim_cities = claim_locs & set(CITY_TO_COUNTRY.keys())
            comb_cities = comb_locs & set(CITY_TO_COUNTRY.keys())
            if claim_cities and comb_cities and not (claim_cities & comb_cities):
                return False, f"Conflicting geographic cities ({', '.join(sorted(comb_cities))} vs {', '.join(sorted(claim_cities))})."

        # Explicit location predicate requires claimed place to actually appear in evidence
        explicit_loc = re.search(
            r'\b(?:located|hosted|held|based|headquartered)\s+in\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})|\bcapital\s+of\b',
            claim_lower,
        )
        if explicit_loc and claim_locs:
            claim_countries = {to_country(l) for l in claim_locs}
            comb_countries = {to_country(l) for l in comb_locs}
            if not (claim_locs & comb_locs) and not (claim_countries & comb_countries):
                return False, "Evidence does not corroborate claimed geographic location."
            claim_cities = claim_locs & set(CITY_TO_COUNTRY.keys())
            comb_cities = comb_locs & set(CITY_TO_COUNTRY.keys())
            if claim_cities and not (claim_cities & comb_locs):
                return False, f"Evidence does not mention claimed city ({', '.join(sorted(claim_cities))})."
            if claim_cities and (comb_cities - claim_cities) and not re.search(r'\bborder\b', claim_lower):
                return False, f"Conflicting geographic city mentioned ({', '.join(sorted(comb_cities - claim_cities))})."

        # 2. Acronyms & Named Agencies (e.g. WHO, Reuters, IEA, NASA, UN, CDC, FDA)
        claim_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', claim_text))
        stopwords_caps = {
            "THE", "AND", "FOR", "NOT", "YES", "NEW", "ALL", "TOP", "HAS", "HAD", "TWO",
            "CEO", "CFO", "CTO", "COO", "CIO", "CMO", "VP", "SVP", "EVP", "MD"
        }
        if "prime minister" in claim_lower or re.search(r'\b(?:[Pp][Mm]|[Pp]\.[Mm]\.)\s+[A-Z][a-z]', claim_text):
            stopwords_caps.add("PM")
        claim_acronyms -= stopwords_caps
        if claim_acronyms:
            comb_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', combined_text))
            comb_acronyms -= stopwords_caps
            acronym_matched = False
            for acr in claim_acronyms:
                if acr in comb_acronyms or acr.lower() in comb_lower:
                    acronym_matched = True
                    break
            if not acronym_matched:
                spelled_out = {
                    "WHO": "world health organization",
                    "IEA": "international energy agency",
                    "UN": "united nations",
                    "CDC": "centers for disease control",
                    "FDA": "food and drug administration",
                    "NASA": "national aeronautics"
                }
                for acr in claim_acronyms:
                    if acr in spelled_out and spelled_out[acr] in comb_lower:
                        acronym_matched = True
                        break
            if not acronym_matched and len(claim_acronyms) >= 1:
                return False, "Candidate does not corroborate specific agency/entity named in claim."

        # 3. Numeric / Percentage metrics check
        claim_pcts = set(re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)', claim_text, re.I))
        comb_pcts = set(re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)', combined_text, re.I))
        if claim_pcts and not (claim_pcts & comb_pcts):
            if comb_pcts:
                return False, "Candidate cites conflicting percentage metric."
            else:
                return False, "Candidate does not corroborate specific percentage metric in claim."

        claim_years = set(re.findall(r'\b(19\d\d|20\d\d)\b', claim_text))
        comb_years = set(re.findall(r'\b(19\d\d|20\d\d)\b', combined_text))
        if claim_years and comb_years and not (claim_years & comb_years):
            return False, "Candidate cites conflicting year or date metric."

        # Check for numbered versions / mission iterations (e.g. Chandrayaan-3 vs Chandrayaan-2, Apollo 11 vs Apollo 13)
        claim_numbered = re.findall(r'\b([A-Za-z]{3,})[-\s](\d{1,3})\b', claim_text)
        comb_numbered = re.findall(r'\b([A-Za-z]{3,})[-\s](\d{1,3})\b', combined_text)
        for base_name, num in claim_numbered:
            comb_nums_for_base = {n for (nm, n) in comb_numbered if nm.lower() == base_name.lower()}
            if comb_nums_for_base and num not in comb_nums_for_base:
                return False, f"Candidate cites different version/iteration of {base_name} ({', '.join(sorted(comb_nums_for_base))} vs {num})."

        # Monetary / large-scale magnitude check (e.g. 'ten trillion dollars' cannot match routine coverage)
        claim_scales = set(re.findall(r'\b(trillion|billion|million)\b', claim_lower))
        if claim_scales and not (claim_scales & set(re.findall(r'\b(trillion|billion|million)\b', comb_lower))):
            return False, "Candidate does not corroborate financial/numeric magnitude in claim."

        # Conflicting Organization for Leadership Roles (e.g. 'Alphabet CEO' vs 'Nvidia CEO')
        ROLES_REGEX = r'(?:([A-Z][a-zA-Z0-9]+)\s+(?:CEO|founder|president|prime minister))|(?:(?:CEO|founder|president|prime minister)\s+of\s+([A-Z][a-zA-Z0-9]+))'
        claim_role_orgs = {x[0] or x[1] for x in re.findall(ROLES_REGEX, claim_text, re.I)}
        comb_role_orgs = {x[0] or x[1] for x in re.findall(ROLES_REGEX, combined_text, re.I)}
        NON_ORGS = {
            "THE", "A", "AN", "THIS", "THAT", "ITS", "THEIR", "HIS", "HER", "OUR", "MY", "YOUR",
            "AND", "OR", "BUT", "AS", "IS", "WAS", "WERE", "BEEN", "BY", "FOR", "IN", "ON", "AT",
            "TO", "OF", "WITH", "FROM", "FORMER", "CURRENT", "ACTING", "INTERIM", "DEPUTY",
            "CO", "VICE", "EXECUTIVE", "SENIOR", "CHIEF", "NEW", "LONGTIME", "HONORARY"
        }
        claim_role_orgs = {o.upper() for o in claim_role_orgs if o and o.upper() not in NON_ORGS}
        comb_role_orgs = {o.upper() for o in comb_role_orgs if o and o.upper() not in NON_ORGS}
        if claim_role_orgs and comb_role_orgs and not (claim_role_orgs & comb_role_orgs):
            return False, f"Conflicting organization for claimed leadership role ({', '.join(comb_role_orgs)} vs {', '.join(claim_role_orgs)})."

        # 4. Core entity / content noun overlap
        stop_words = {
            "this", "that", "these", "those", "have", "been", "were", "with", "from",
            "about", "after", "before", "reported", "noted", "stated", "according", "similar",
            "findings", "across", "other", "their", "which", "could", "would", "should"
        }
        claim_words = {w for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim_lower) if w not in stop_words}
        comb_words = {w for w in re.findall(r'\b[a-zA-Z]{4,}\b', comb_lower) if w not in stop_words}
        overlap = claim_words & comb_words

        overlap_ratio = len(overlap) / max(len(claim_words), 1)
        if len(claim_words) >= 4 and (len(overlap) < 2 or (len(overlap) == 2 and overlap_ratio < 0.25)):
            return False, "Insufficient entity/event overlap between headline and claim."

        # Check for multiple distinct named entities in relational claims (e.g. 'Pope Francis endorsed Trump')
        stopwords_pe = {
            "The", "A", "An", "In", "On", "At", "To", "For", "With", "By", "From", "About",
            "After", "Before", "During", "Under", "Global", "Recent", "New", "First",
            "According", "Addressing", "Several", "Many", "Some", "Local", "Former", "Current",
            "Major", "Leading", "Following", "Amid", "Despite", "Meanwhile", "Earlier", "Later"
        }
        ROLE_STOPWORDS = {
            "chief executive officer", "chief executive", "executive officer", "ceo",
            "founder", "co-founder", "president", "prime minister", "managing director",
            "vice president", "chairman", "director general", "spokesperson", "minister",
            "secretary", "chief justice", "chief justice of india", "justice", "judge"
        }
        KNOWN_ACRONYMS = {
            "reserve bank of india": "rbi",
            "reserve bank": "rbi",
            "world health organization": "who",
            "international energy agency": "iea",
            "united nations": "un",
            "european central bank": "ecb",
            "centers for disease control": "cdc",
            "food and drug administration": "fda",
            "monetary policy committee": "mpc"
        }
        raw_ents = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim_text)
        ent_phrases = []
        for e in raw_ents:
            clean_e = re.sub(r'^(The|A|An|Justice|Judge)\s+', '', e).strip()
            if clean_e and clean_e not in stopwords_pe and clean_e.lower() not in ROLE_STOPWORDS:
                ent_phrases.append(clean_e)

        def entity_in_evidence(entity_str: str) -> bool:
            e_lower = entity_str.lower()
            if e_lower in comb_lower:
                return True
            words = entity_str.split()
            if len(words) >= 2:
                surname = words[-1]
                given_name = words[0]
                if re.search(rf'\b{re.escape(surname.lower())}\b', comb_lower):
                    conflicting_m = re.search(rf'\b(?!(?:its|the|a|an|their|his|her|former|new|interim|acting)\b)([A-Z][a-z]+)\s+{re.escape(surname)}\b', combined_text)
                    if conflicting_m and conflicting_m.group(1).lower() != given_name.lower():
                        return False
                    return True
            elif len(words) == 1:
                if re.search(rf'\b{re.escape(e_lower)}\b', comb_lower):
                    return True
            acr = KNOWN_ACRONYMS.get(e_lower, "")
            if acr and re.search(rf'\b{acr}\b', comb_lower):
                return True
            return False

        # In relational claims (e.g. endorsements, meetings, acquisitions, sports victories between multiple parties),
        # all named participants must be present in the candidate evidence.
        is_relational = bool(re.search(r'\b(endorsed|endorses|endorsing|endorsement|sued|suing|met with|married|accused|appointed|nominated|acquired|bought|merged|defeated|beat|beats)\b', claim_lower))
        is_movement_claim = bool(re.search(
            r'\b(?:stock|shares?|price|equities)\b.*\b(?:goes?\s+(?:up|down)|went\s+(?:up|down)|rall(?:y|ied|ies)|plung(?:e|ed|es)|surges?|surged|jumps?|jumped|climbs?|climbed|climbing|rose|soars?|soared|drops?|dropped|falls?|fell)\b'
            r'|\b(?:goes?\s+(?:up|down)|went\s+(?:up|down)|rall(?:y|ied|ies)|plung(?:e|ed|es)|surges?|surged|jumps?|jumped|climbs?|climbed|climbing|rose|soars?|soared|drops?|dropped|falls?|fell)\b.*\b(?:stock|shares?|price|equities)\b',
            claim_lower
        ))
        is_officeholder = bool(re.search(r'\b(chief justice|president|prime minister|ceo|chief executive|governor|chairman|chancellor)\b', claim_lower))
        if is_relational and len(ent_phrases) >= 2:
            missing_ents = [e for e in ent_phrases if not entity_in_evidence(e)]
            if missing_ents:
                return False, f"Evidence does not mention participant entity '{missing_ents[0]}' from claim."
        elif is_officeholder and ent_phrases:
            # Extract the person explicitly associated with the role/return clause
            role_person_m = (
                re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:returned|returns?|returning|as\s+(?:the\s+)?ceo|is\s+(?:the\s+)?ceo|appointed|reinstated)\b', claim_text)
                or re.search(r'\b(?:reinstated?|reinstatement\s+of|return\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim_text)
                or re.search(r'\b(?:after|upon|following)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:returned|was\s+reinstated)\b', claim_text)
            )
            if role_person_m:
                claimed_person = role_person_m.group(1)
            else:
                person_candidates = [
                    e for e in ent_phrases
                    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', e)
                    and e.split()[0] not in {"The", "A", "An"}
                    and e.lower() not in {"yahoo finance", "google news"}
                ]
                claimed_person = person_candidates[0] if person_candidates else ent_phrases[0]

            co_subject_m = re.search(r'\b(?:The\s+)?([A-Z][a-zA-Z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)*)\s+(?:stock|shares?|equities)\b', claim_text)
            if co_subject_m:
                claimed_co = co_subject_m.group(1).strip()
                company_ents = [claimed_co]
            else:
                company_ents = [e for e in ent_phrases if e.lower() != claimed_person.lower()]

            # Detect rival returnee/officeholder in evidence
            # e.g., "Nvidia stock rose after the board reinstated Bob Jones as CEO" or "Bob Jones is CEO"
            different_person_match = (
                re.search(r'(?i:\b(?:reinstated?|appointed?|named?|hired?|welcomed?)\s+)(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?i:\s+(?:as\s+(?:the\s+)?ceo|to\s+replace|as\s+chief\s+executive)\b)', combined_text)
                or re.search(r'\b(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?i:\s+(?:returned\s+as\s+(?:the\s+)?ceo|is\s+(?:the\s+)?ceo|as\s+(?:the\s+)?ceo|appointed\s+(?:as\s+)?ceo|replaced|succeeded)\b)', combined_text)
                or re.search(r'(?i:\b(?:after|upon|following)(?:\s+the\s+board)?\s+(?:reinstated?|appointed?)\s+)(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', combined_text)
                or re.search(r'(?i:\b(?:after|upon|following)\s+)(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?i:\s+(?:returned|was\s+reinstated|resigned)\b)', combined_text)
            )
            NON_PERSON_ROLES = {
                "its ceo", "the ceo", "a ceo", "former ceo", "new ceo", "interim ceo", "acting ceo",
                "its founder", "the founder", "its chief", "the chief", "its president", "the president",
                "its leader", "the leader", "him", "her", "them", "the board", "its executives"
            }
            captured_name = different_person_match.group(1).strip() if different_person_match else ""
            is_valid_person_name = bool(
                captured_name
                and captured_name.lower() not in NON_PERSON_ROLES
                and not re.match(r'^(?:its|the|a|an|former|new|interim|acting)?\s*(?:ceo|chief\s+executive|founder|president|leader|official|board)\b', captured_name, re.I)
            )
            has_given_name_conflict = bool(
                is_valid_person_name
                and len(claimed_person.split()) >= 2
                and len(captured_name.split()) >= 2
                and claimed_person.split()[0].lower() != captured_name.split()[0].lower()
            )
            has_competing_person = bool(
                is_valid_person_name and
                (not any(w.lower() in captured_name.lower() for w in claimed_person.split())
                 or has_given_name_conflict)
            )
            if has_competing_person:
                return False, f"Evidence attributes leadership event to different person '{captured_name}' rather than '{claimed_person}'."

            if not entity_in_evidence(claimed_person):
                is_leadership_phrase = bool(
                    re.search(
                        r'\b(?:reinstat\w*|return\w*)\s+(?:as\s+(?:the\s+)?(?:ceo|chief\s+executive|leader|boss)|to\s+(?:the\s+)?helm|to\s+lead|(?:its|the|former)\s+(?:ceo|chief\s+executive|founder|leader)|him|her)\b'
                        r'|\b(?:board|company)\s+reinstated\s+(?:its\s+)?(?:ceo|founder|chief\s+executive|him|her)\b'
                        r'|\b(?:reinstatement|return)\s+of\s+(?:its\s+)?(?:ceo|founder|chief\s+executive)\b'
                        r'|\b(?:after|upon|following|on)\s+(?:the\s+)?(?:reinstatement|return)\s+(?:of\s+(?:its\s+)?(?:ceo|founder|chief\s+executive)|as\s+ceo)?\b'
                        r'|\b(?:after|upon|following|on)\s+(?:the\s+)?reinstatement\b',
                        comb_lower
                    )
                )
                is_corporate_asset_reinstatement = bool(
                    re.search(
                        r'\b(?:reinstat\w*|return\w*)\s+(?:of\s+capital|capital|its\s+dividend|the\s+dividend|dividend|payout|guidance|buyback|distribution|share\s+repurchase|trading|operations)\b',
                        comb_lower
                    )
                )
                is_leadership_return_event = is_leadership_phrase and not is_corporate_asset_reinstatement

                # If this is a movement claim, allow company entity + leadership return/reinstatement when no competing person is named
                if is_movement_claim and any(entity_in_evidence(e) for e in company_ents) and is_leadership_return_event:
                    pass
                else:
                    return False, f"Evidence does not mention subject entity '{claimed_person}' for officeholder claim."
        elif ent_phrases:
            # For non-relational claims, at least one key named entity (or its acronym) must match
            if not any(entity_in_evidence(e) for e in ent_phrases):
                return False, f"Evidence does not mention entity '{ent_phrases[0]}' from claim."

        return True, "Matches specific claim."

    def _snippet_mentions_claim_topic_or_entities(self, claim_text: str, combined_text: str) -> bool:
        """
        Validates that the evidence snippet actually mentions the claim's key entities and specific topic/predicate.
        A generic overview page (e.g. an encyclopedia entry on a whole planet, country, or company)
        that does not address the specific controversy/predicate cannot count as a contradiction.
        """
        claim_lower = claim_text.lower()
        comb_lower = combined_text.lower()

        STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "to", "at", "in", "for",
            "on", "by", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "from", "up", "down",
            "out", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can",
            "will", "just", "should", "now", "it", "its", "this", "that"
        }
        tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', claim_lower) if w not in STOP and len(w) > 2]
        if not tokens:
            return True

        matched = [w for w in tokens if w in comb_lower]

        if len(tokens) >= 2 and len(matched) < 2:
            synonyms = {
                "flat": ["shape", "sphere", "spherical", "round", "curvature", "disk", "oblate"],
                "ceo": ["executive", "chief", "resigned", "leadership", "founded"],
                "return": ["resigned", "stepped down", "left", "tenure", "continuous"],
                "covid": ["coronavirus", "pandemic", "sars-cov-2", "virus", "pathogen"],
                "5g": ["radiation", "cellular", "telecommunications", "radio waves", "electromagnetic"]
            }
            missing = [t for t in tokens if t not in matched]
            covered = 0
            for t in missing:
                if t in synonyms and any(syn in comb_lower for syn in synonyms[t]):
                    covered += 1
            if (len(matched) + covered) < 2:
                return False

        return len(matched) >= 1

    def _has_shared_entity_or_core_token(self, claim_text: str, evidence_text: str) -> bool:
        """
        Requirement 2: Validates that candidate evidence shares at least one named entity
        or core event token with the claim.
        Prevents weak, generic topical overlaps (e.g. 6%-8% similarity) from inflating the trust score.
        """
        if not claim_text or not evidence_text:
            return False

        claim_lower = claim_text.lower()
        ev_lower = evidence_text.lower()

        # 1. Check for specific named entities in the claim
        # Multi-word proper names: e.g. "Ehab Abu Jazar", "Jensen Huang", "Chandrayaan-3"
        named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)+\b', claim_text)
        words = claim_text.split()
        single_caps = [
            w.strip('.,!?:;"\'()') for w in words[1:]
            if w and w[0].isupper() and len(w) > 2 and w.upper() not in {"THE", "AND", "FOR", "NOT", "YES", "NEW", "TOP"}
        ]
        acronyms = re.findall(r'\b[A-Z0-9]{2,6}\b', claim_text)

        all_entities = set(named_entities + single_caps + acronyms)
        # Filter out weekdays from all_entities
        all_entities = {e for e in all_entities if e.lower() not in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}}

        # If claim has named entities, check if any appear in evidence
        for ent in all_entities:
            if ent.lower() in ev_lower:
                return True
            parts = ent.split()
            if len(parts) >= 2:
                # Check surname
                surname = parts[-1]
                if len(surname) > 3 and re.search(rf'\b{re.escape(surname.lower())}\b', ev_lower):
                    return True
                if len(parts) >= 3:
                    last_two = " ".join(parts[-2:])
                    if last_two.lower() in ev_lower:
                        return True

        # 2. Check for core event tokens (content words: key nouns, actions, numbers, locations)
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "to", "at", "in", "for",
            "on", "by", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "from", "up", "down",
            "out", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can",
            "will", "just", "should", "now", "it", "its", "this", "that", "these",
            "those", "stated", "said", "reported", "says", "claims", "according",
            "family", "living", "people", "world", "state", "states", "years", "year",
            "today", "news", "report", "first", "last", "country", "countries"
        }

        claim_tokens = [
            w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', claim_lower)
            if w not in STOPWORDS and len(w) >= 4 and not w.isdigit()
        ]

        # Specific numbers / years
        claim_numbers = re.findall(r'\b(?:\d{1,4}|19\d\d|20\d\d)\b', claim_text)

        for token in claim_tokens:
            if re.search(rf'\b{re.escape(token)}\b', ev_lower):
                return True

        for num in claim_numbers:
            if re.search(rf'\b{re.escape(num)}\b', ev_lower):
                return True

        return False

    def rank_evidence_for_claim(
        self,
        claim_text: str,
        candidate_items: List[Dict[str, Any]],
        allow_gemini: bool = True,
    ) -> Dict[str, Any]:
        """
        Ranks all candidate sources for a claim using embeddings, credibility, and recency.
        Returns categorized supporting, contradicting, related, and neutral items.
        """
        # Requirement 3: Only accept sources with a real article URL (not a homepage or a section page like /technology)
        candidate_items = [it for it in (candidate_items or []) if is_real_article_url(it.get("url", ""))]

        if not candidate_items:
            return {
                "supporting": [],
                "contradicting": [],
                "related": [],
                "neutral": [],
                "supportingCount": 0,
                "contradictingCount": 0,
                "relatedCount": 0,
                "neutralCount": 0
            }

        # 1. Ask Gemini stance classifier first (if available, allowed, and API key present)
        # Batches candidate items for this single claim to minimize latency
        gemini_stances = (
            self.gemini_classifier.classify_candidates_for_claim(claim_text, candidate_items, max_candidates=15)
            if allow_gemini
            else None
        )

        evaluated: List[Dict[str, Any]] = []

        for idx, item in enumerate(candidate_items):
            title = item.get("articleTitle", "")
            snippet = item.get("evidenceSnippet", "")
            doc_text = f"{title}. {snippet}".strip()

            # 1. Semantic embedding similarity
            sim = self.semantic_engine.compute_similarity(claim_text, doc_text)
            
            # 2. Check authoritative fact-check ratings or Wikipedia derived checks first
            fc_rating = str(item.get("factCheckRating", "")).lower().strip()
            reviewed_fc_claim = str(item.get("reviewedClaim", "")).strip()
            fc_relevant = (not fc_rating) or is_factcheck_claim_relevant(claim_text, reviewed_fc_claim)
            wiki_channel = item.get("retrievalChannel") in ("Wikipedia REST API", "Wikidata API")

            if wiki_channel and item.get("stance") in ("support", "contradict"):
                if item.get("stance") == "contradict" and ("living person" in str(item.get("evidenceSnippet", "")).lower() or "no date of death" in str(item.get("evidenceSnippet", "")).lower()) and claim_refers_to_relative_or_associate(claim_text):
                    stance = "neutral"
                    stance_conf = 40
                    stance_reason = "Living-person check skipped: claim refers to relative or associate."
                else:
                    stance = item["stance"]
                    stance_conf = 98
                    stance_reason = item.get("evidenceSnippet", "")
            elif fc_rating and fc_relevant and any(neg in fc_rating for neg in NEGATIVE_RATINGS):
                stance = "contradict"
                stance_conf = 98
                stance_reason = f"Fact-checked as '{item.get('factCheckRating')}' by {item.get('publisher', 'fact-checker')}."
            elif fc_rating and fc_relevant and not any(part in fc_rating for part in ("half true", "half-true", "partly true", "partially true", "not true", "mixed")) and any(pos in fc_rating for pos in POSITIVE_RATINGS):
                stance = "support"
                stance_conf = 95
                stance_reason = f"Fact-checked as '{item.get('factCheckRating')}' by {item.get('publisher', 'fact-checker')}."
            elif gemini_stances is not None and idx in gemini_stances:
                gemini_result = gemini_stances[idx]
                g_stance = gemini_result.get("stance", "neutral")
                stance_conf = gemini_result.get("confidence", 80)
                stance_reason = gemini_result.get("reason", "")
                if g_stance in ("support", "contradict"):
                    stance = g_stance
                    # Event & Timeframe Rule guard:
                    # If snippet confirms the claimed event occurred (e.g. rally after return/reinstatement),
                    # do not allow contradictory later movement/pullback to mislabel it as contradict.
                    # Preserve contradictions caused by conflicting metrics, entities, or event details.
                    if stance == "contradict":
                        combined_check = f"{title} {snippet}".strip()
                        UP_C = bool(re.search(r'\b(goes\s+up|went\s+up|rise|rises|rose|rising|gains?|rally|rallies|rallied|surges?|jumps?|soars?|increases?)\b', claim_text.lower()))
                        claim_pcts = set(re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)', claim_text, re.I))
                        claim_years = set(re.findall(r'\b(19\d\d|20\d\d)\b', claim_text))
                        claim_months = set(re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', claim_text.lower()))
                        claim_day_m = re.findall(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', claim_text.lower())
                        
                        returnee_m = (
                            re.search(r'\b(?:after|upon|following)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:returned|returns?|returning|was\s+reinstated|reinstated)\b', claim_text)
                            or re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:returned|returns?|returning|as\s+(?:the\s+)?ceo|is\s+(?:the\s+)?ceo|appointed|reinstated)\b', claim_text)
                            or re.search(r'\b(?:reinstated?|reinstatement\s+of|return\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim_text)
                        )
                        claim_people = [e for e in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text) if e not in {"Yahoo Finance", "Google News"}]
                        claimed_p = returnee_m.group(1).strip() if returnee_m else (claim_people[0] if claim_people else "")

                        co_subject_m = re.search(r'\b(?:The\s+)?([A-Z][a-zA-Z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)*)\s+(?:stock|shares?|equities)\b', claim_text)
                        if co_subject_m:
                            claimed_co = co_subject_m.group(1).strip()
                        else:
                            claim_companies = [
                                c for c in re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', claim_text)
                                if c not in {"CEO", "CFO", "CTO", "COO", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Yahoo", "Finance", "Google", "News"}
                                and (not claim_people or not any(c.lower() == p.lower() for p in claim_people[0].split()))
                            ]
                            claimed_co = claim_companies[0] if claim_companies else ""

                        if claim_day_m:
                            has_later_reversal = bool(
                                re.search(r'\b(revers(?:ing|ed|al)|pullback|subsequent(?:ly)?\s+fell|later\s+fell|later\s+dropped|pulled\s+back)\b', combined_check, re.I)
                                or any(re.search(r'\bfell\s+(?:on\s+)?(?!' + d + r'\b)(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', combined_check, re.I) for d in claim_day_m)
                            )
                        else:
                            has_later_reversal = bool(re.search(
                                r'\b(revers(?:ing|ed|al)|pullback|fell\s+(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|the\s+following|the\s+next\s+day)|subsequent(?:ly)?\s+fell|later\s+fell|later\s+dropped|pulled\s+back)\b',
                                combined_check,
                                re.I
                            ))

                        # Evaluate movement segments to bind company, person, percentage, and timeframe to the same movement
                        segments = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\s*;\s*|\s*,\s*(?:while|whereas|although|but|and)\s+|\s+and\s+(?=[A-Z])', combined_check) if s.strip()]
                        if not segments:
                            segments = [combined_check]

                        NON_PERSON_ROLES = {
                            "its ceo", "the ceo", "a ceo", "former ceo", "new ceo", "interim ceo", "acting ceo",
                            "its founder", "the founder", "its chief", "the chief", "its president", "the president",
                            "its leader", "the leader", "him", "her", "them", "the board", "its executives"
                        }

                        found_qualifying_movement = False
                        for seg in segments:
                            seg_speculative = bool(re.search(
                                r'\b(?:may|might|could|plans?\s+to|planning\s+to|expected\s+to|expects?\s+to|forecasts?|predicts?)\s+[^.!?;\n]{0,25}\b(?:rally|gain|rise|jump|surge|increase|soar)\b',
                                seg,
                                re.I
                            ))
                            if seg_speculative:
                                continue

                            # 1. Company check in this segment
                            if claimed_co:
                                co_ok = bool(re.search(
                                    rf'\b{re.escape(claimed_co)}\b[^.!?;]{{0,60}}?\b(?:stock|shares?|equities)?\b[^.!?;]{{0,40}}?\b(?:rally|rallied|rallies|rose|rising|jumped|jumps|surged|surges|soared|gained|increased)\b'
                                    rf'|\b(?:rally|rallied|rallies|rose|rising|jumped|jumps|surged|surges|soared|gained|increased)\s+in\s+[^.!?;]{{0,30}}?\b{re.escape(claimed_co)}\b',
                                    seg,
                                    re.I
                                ))
                            else:
                                co_ok = True

                            # 2. Person check in this segment
                            is_asset_reinstatement = bool(re.search(
                                r'\b(?:reinstat\w*|return\w*)\s+(?:of\s+capital|capital|its\s+dividend|the\s+dividend|dividend|payout|guidance|buyback|distribution|share\s+repurchase|trading|operations)\b',
                                seg,
                                re.I
                            ))
                            is_leadership_phrase = bool(re.search(
                                r'\b(?:reinstat\w*|return\w*)\s+(?:as\s+(?:the\s+)?(?:ceo|chief\s+executive|leader|boss)|to\s+(?:the\s+)?helm|to\s+lead|(?:its|the|former)\s+(?:ceo|chief\s+executive|founder|leader)|him|her)\b'
                                r'|\b(?:board|company)\s+reinstated\s+(?:its\s+)?(?:ceo|founder|chief\s+executive|him|her)\b'
                                r'|\b(?:reinstatement|return)\s+of\s+(?:its\s+)?(?:ceo|founder|chief\s+executive)\b'
                                r'|\b(?:after|upon|following|on)\s+(?:the\s+)?(?:reinstatement|return)\s+(?:of\s+(?:its\s+)?(?:ceo|founder|chief\s+executive)|as\s+ceo)?\b'
                                r'|\b(?:after|upon|following|on)\s+(?:the\s+)?reinstatement\b',
                                seg,
                                re.I
                            ))

                            if claimed_p:
                                p_words = claimed_p.split()
                                has_full_p = claimed_p.lower() in seg.lower()
                                has_surname = bool(re.search(rf'\b{re.escape(p_words[-1])}\b', seg, re.I))
                                conflicting_m = re.search(rf'\b(?!(?:its|the|a|an|their|his|her|former|new|interim|acting)\b)([A-Z][a-z]+)\s+{re.escape(p_words[-1])}\b', seg)
                                has_given_name_conflict = bool(conflicting_m and len(p_words) >= 2 and conflicting_m.group(1).lower() != p_words[0].lower())

                                rival_m = (
                                    re.search(r'(?i:\b(?:reinstated?|appointed?|named?|hired?)\s+)(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?i:\s+(?:as\s+(?:the\s+)?ceo|to\s+replace|as\s+chief\s+executive)\b)', seg)
                                    or re.search(r'\b(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?i:\s+(?:returned\s+as\s+(?:the\s+)?ceo|is\s+(?:the\s+)?ceo|as\s+(?:the\s+)?ceo|appointed\s+(?:as\s+)?ceo)\b)', seg)
                                    or re.search(r'(?i:\b(?:after|upon|following)(?:\s+the\s+board)?\s+(?:reinstated?|appointed?)\s+)(?!(?:Its|The|A|An|Their|His|Her|Former|New|Interim|Acting)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', seg)
                                )
                                captured_rival = rival_m.group(1).strip() if rival_m else ""
                                is_rival_valid = bool(
                                    captured_rival
                                    and captured_rival.lower() not in NON_PERSON_ROLES
                                    and not re.match(r'^(?:its|the|a|an|former|new|interim|acting)?\s*(?:ceo|chief\s+executive|founder|president|leader|official|board)\b', captured_rival, re.I)
                                )
                                has_rival = bool(is_rival_valid and (not any(w.lower() in captured_rival.lower() for w in p_words) or (len(p_words) >= 2 and len(captured_rival.split()) >= 2 and p_words[0].lower() != captured_rival.split()[0].lower())))
                                p_ok = ((has_full_p or (has_surname and not has_given_name_conflict) or (is_leadership_phrase and not is_asset_reinstatement)) and not has_given_name_conflict and not has_rival)
                            else:
                                p_ok = True

                            # 3. Percentage check in this segment
                            if claim_pcts:
                                if claimed_co:
                                    co_pct_m = re.findall(
                                        rf'\b{re.escape(claimed_co)}\b[^.!?;]{{0,60}}?\b(?:stock|shares?|equities)?\b[^.!?;]{{0,40}}?\b(?:rally|rallied|rallies|rose|rising|jumped|jumps|surged|surges|soared|gained|increased)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)'
                                        rf'|\b(?:reversing\s+from\s+a\s+[^.!?;\n]{{0,40}}?(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)[^.!?;\n]{{0,40}}?(?:rally|gain|surge|jump)|(?:rally|rallied|rallies|soared|jumped|rose|gained|increased)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)[^.!?;\n]{{0,50}}?(?:after|following|upon|on))',
                                        seg,
                                        re.I
                                    )
                                    seg_pcts = {p for tup in co_pct_m for p in (tup if isinstance(tup, tuple) else (tup,)) if p}
                                    pct_ok = bool(claim_pcts & seg_pcts)
                                else:
                                    seg_rise_pcts = set(re.findall(
                                        r'\b(?:reversing\s+from\s+a\s+[^.!?;\n]{0,40}?(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)[^.!?;\n]{0,40}?(?:rally|gain|surge|jump)|(?:rally|rallied|rallies|soared|jumped|rose|gained|increased)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)[^.!?;\n]{0,50}?(?:after|following|upon|on))',
                                        seg,
                                        re.I
                                    ))
                                    seg_pcts = {p for tup in seg_rise_pcts for p in tup if p}
                                    if not seg_pcts:
                                        seg_pcts = set(re.findall(
                                            r'\b(?:rally|rallied|rallies|rose|rising|gained|gains|jumped|jumps|surged|surges|soared|increased)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)(?!\w)',
                                            seg,
                                            re.I
                                        ))
                                    pct_ok = bool(claim_pcts & seg_pcts)
                            else:
                                pct_ok = True

                            # 4. Year check in this segment
                            if claim_years:
                                seg_years = set(re.findall(
                                    r'\b(?:returned|returns?|returning|reinstated?|reinstatement|rally|rallied|rallies|rose|rising|jumped|surged|in|during)\s+[^.!?;,]{0,25}?(19\d\d|20\d\d)\b',
                                    seg,
                                    re.I
                                ))
                                year_ok = bool(claim_years & seg_years)
                            else:
                                year_ok = True

                            # 5. Month check in this segment
                            if claim_months:
                                seg_months = {m.lower() for m in re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', seg, re.I)}
                                comb_months_all = {m.lower() for m in re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', combined_check, re.I)}
                                month_ok = bool(claim_months & seg_months) if seg_months else (not bool(comb_months_all - claim_months))
                            else:
                                month_ok = True

                            # 6. Day check in this segment
                            if claim_day_m:
                                seg_up_day = any(
                                    re.search(r'\b(?:rally|rallied|rallies|rose|gained|jumped|surged|soared|increased)\b[^.!?;,]{0,40}\b(?:on\s+)?' + d + r'\b|\b' + d + r'\'?s?\b[^.!?;,]{0,40}\b(?:rally|gain|rise|jump|surge)\b', seg, re.I)
                                    for d in claim_day_m
                                )
                                seg_down_day = any(
                                    re.search(r'\b(?:fell|dropped|declined|plunged|slumped|tumbled|crashed|decreased)\b[^.!?;,]{0,30}\b(?:on\s+)?' + d + r'\b', seg, re.I)
                                    for d in claim_day_m
                                )
                                day_ok = seg_up_day and not seg_down_day
                            else:
                                day_ok = True

                            # 7. Non-negated rise check
                            is_negated_rise = bool(re.search(
                                r'\b(?:no|never|failed\s+to|without\s+a?|zero|did\s+not|didn\'?t|not)\s+(?:rally|rallied|rallies|gain|gained|gains|rise|rose|risen|rising|jump|jumped|jumps|surge|surged|surges|soar|soared|soars|increase|increased|increases)\b',
                                seg,
                                re.I
                            ))

                            # 8. Event check in this segment
                            seg_has_post_event_rally = bool(re.search(
                                r'\b(?:reversing\s+from\s+a\s+[^.!?;\n]{0,50}(?:rally|gain|surge|jump)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,40}\s+(?:after|following|upon|on\s+(?:the\s+)?(?:reinstatement|return|announcement|news)\b)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,50}\b(?:reinstated?|reinstatement|returned?|returns?|returning)|(?:rally|rallied|rallies|soared|jumped|rose|gained)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,40}\s+(?:after|following|upon|on\s+(?:the\s+)?(?:reinstatement|return|announcement|news)\b)(?:(?!\bbefore\b|\bprior\b|\bahead\b)[^.!?;\n]){0,50}\b(?:reinstated?|reinstatement|returned?|returns?|returning)|(?:reinstated?|reinstatement|returned?|returns?|returning)\s+[^.!?;\n]{0,50}\b(?:sparked|triggered|led\s+to|followed\s+by)\s+[^.!?;\n]{0,40}(?:rally|gain|surge|jump))\b',
                                seg,
                                re.I
                            ))
                            seg_has_rally_before_event = bool(re.search(
                                r'\b(?:rally|rallied|rallies|soared|jumped|rose|gained)(?:[^.!?;,]{0,40})\s+(?:before|prior\s+to|ahead\s+of)\b[^\n.!?]{0,100}\b(?:reinstated?|reinstatement|returned?|returns?|returning)',
                                seg,
                                re.I
                            ))
                            seg_contradicts_return = bool(re.search(
                                r'\b(?:never\s+left|did\s+not\s+return|not\s+reinstated|refused\s+to\s+return|denied\s+returning|never\s+returned)\b',
                                seg,
                                re.I
                            ))

                            claim_has_return_event = bool(re.search(r'\b(return|reinstat)', claim_text, re.I))
                            if claim_has_return_event:
                                event_ok = seg_has_post_event_rally and not seg_has_rally_before_event and not seg_contradicts_return and not is_asset_reinstatement
                            else:
                                event_ok = (seg_has_post_event_rally or (claim_day_m and day_ok)) and not is_asset_reinstatement

                            if co_ok and p_ok and pct_ok and year_ok and month_ok and day_ok and event_ok and not is_negated_rise:
                                found_qualifying_movement = True
                                break

                        if UP_C and found_qualifying_movement and has_later_reversal:
                            stance = "support"
                            stance_conf = 88
                            stance_reason = "Evidence confirms the claimed event occurred (rally after return/reinstatement), despite later reversal."
                else:  # neutral
                    # If semantic similarity is high (>= 0.35), categorize as "related" topic coverage
                    if sim >= 0.35:
                        stance = "related"
                    else:
                        stance = "neutral"
            else:
                stance, stance_conf = self.classify_stance(claim_text, item, sim)
                stance_reason = ""

            # Requirement 2: Minimum Semantic & Entity Match Floor for SUPPORT stance
            # Require minimum semantic similarity threshold (>= 0.15 / 15%) AND at least one shared
            # named entity or core event token before classifying an article as SUPPORT.
            # Downgrades weak 6%-8% topical overlaps to neutral/related.
            if stance == "support":
                has_entity_match = self._has_shared_entity_or_core_token(
                    claim_text, f"{title} {snippet} {item.get('description', '')}"
                )
                if sim < 0.15 or not has_entity_match:
                    if sim >= 0.10:
                        stance = "related"
                        stance_conf = 50
                        stance_reason = f"Topical overlap ({int(round(sim * 100))}%) below support floor (15%) or lacks entity/event corroboration."
                    else:
                        stance = "neutral"
                        stance_conf = 40
                        stance_reason = f"Weak match ({int(round(sim * 100))}%) below minimum threshold (15%) with no entity overlap."

            # Requirement 5: Require snippet to mention claim's key entities or topic.
            # A generic overview page (like a Wikipedia article on a whole planet) must not count as a contradiction.
            # Explicit IFCN / Fact Check ratings are direct debunks and not generic overview pages.
            combined_snip = f"{title} {snippet}"
            if stance == "contradict" and not (fc_rating and any(neg in fc_rating for neg in NEGATIVE_RATINGS)):
                if not self._snippet_mentions_claim_topic_or_entities(claim_text, combined_snip):
                    stance = "related" if sim >= 0.35 else "neutral"
                    stance_conf = 50
                    stance_reason = "Generic overview page does not address the specific claim proposition (not a contradiction)."

            # 3. Trust score (neutral 50 baseline for unlisted sources)
            trust = item.get("sourceTrustScore", 50)

            # 4. Recency score & Temporal check (Requirement 3: Real publish date & Wikipedia exemption)
            pub_date_val = str(item.get("publicationDate") or item.get("publicationTime") or "").strip()[:10]
            recency = self._calculate_recency_score(pub_date_val)

            is_wiki = "wikipedia" in str(item.get("publisher", "")).lower() or "wikipedia" in str(item.get("sourceName", "")).lower() or "wikipedia.org" in str(item.get("url", "")).lower()
            if is_wiki:
                is_old_source = False
                temporal_status = "reference / evergreen"
            else:
                is_old_source = False
                if pub_date_val:
                    try:
                        p_dt = datetime.strptime(pub_date_val, "%Y-%m-%d").date()
                        if (date.today() - p_dt).days > 180:
                            is_old_source = True
                    except Exception:
                        pass
                temporal_status = "older, may be outdated" if is_old_source else "recent"

            # Composite ranking score (0 - 100)
            composite_score = int(round(
                (sim * 100.0) * 0.40 +
                trust * 0.30 +
                recency * 0.10 +
                stance_conf * 0.20
            ))

            ranked_item = dict(item)
            ranked_item.update({
                "relevanceScore": int(round(sim * 100)),
                "semanticSimilarity": round(sim, 4),
                "compositeRankScore": composite_score,
                "stance": stance,
                "stanceConfidence": stance_conf,
                "stanceReason": stance_reason,
                "matchingClaim": claim_text,
                "temporalStatus": temporal_status,
                "isOlder": is_old_source
            })
            evaluated.append(ranked_item)

        # Sort each stance bucket by composite score descending
        if gemini_stances is not None:
            # Gemini-classified buckets (strictly require semantic similarity >= 0.15 and entity/core token match)
            supporting = [
                it for it in evaluated
                if it["stance"] == "support"
                and it.get("semanticSimilarity", 0.0) >= 0.15
                and self._has_shared_entity_or_core_token(claim_text, f"{it.get('articleTitle', '')} {it.get('evidenceSnippet', '')}")
            ]
            contradicting = [it for it in evaluated if it["stance"] == "contradict"]
            related = [it for it in evaluated if it["stance"] == "related"]
            neutral = [it for it in evaluated if it["stance"] == "neutral"]
        else:
            # Heuristic fallback buckets (strictly require semantic similarity >= 0.15 and entity/core token match)
            supporting = [
                it for it in evaluated
                if it["stance"] == "support"
                and it.get("semanticSimilarity", 0.0) >= 0.15
                and it.get("relevanceScore", 0) >= 30
                and self._has_shared_entity_or_core_token(claim_text, f"{it.get('articleTitle', '')} {it.get('evidenceSnippet', '')}")
            ]
            contradicting = [it for it in evaluated if it["stance"] == "contradict" and (it["relevanceScore"] >= 20 or "Contradicted by" in it.get("evidenceSnippet", "") or "[Derived check]" in it.get("evidenceSnippet", ""))]
            related = [it for it in evaluated if it["stance"] == "related"]
            neutral = [it for it in evaluated if it["stance"] == "neutral"]

        supporting.sort(key=lambda x: (x["compositeRankScore"], x["relevanceScore"]), reverse=True)
        contradicting.sort(key=lambda x: (x["compositeRankScore"], x["relevanceScore"]), reverse=True)
        related.sort(key=lambda x: x["compositeRankScore"], reverse=True)
        neutral.sort(key=lambda x: x["compositeRankScore"], reverse=True)

        return {
            "supporting": supporting,
            "contradicting": contradicting,
            "related": related,
            "neutral": neutral,
            "supportingCount": len(supporting),
            "contradictingCount": len(contradicting),
            "relatedCount": len(related),
            "neutralCount": len(neutral),
            "stanceClassifierStatus": getattr(self.gemini_classifier, "last_status", "available")
        }

    def build_evidence_matrix_for_claims(
        self,
        claims: List[Dict[str, Any]],
        retrieval_map: Dict[str, List[Dict[str, Any]]],
        pre_ranked_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Constructs the comprehensive Evidence Matrix:
        For each claim:
        - Claim
        - Claim Number
        - Is Opinion (or Factual Claim)
        - Supporting Evidence
        - Contradicting Evidence
        - Source Links
        - Status
        - Confidence
        - Grounded Explanation
        """
        matrix: List[Dict[str, Any]] = []

        # Concurrently rank any unranked factual claims with ThreadPoolExecutor(max_workers=3)
        ranked_cache: Dict[str, Dict[str, Any]] = dict(pre_ranked_map or {})
        pending_claims: List[str] = []
        for claim_obj in claims:
            c_text = claim_obj.get("claim", "")
            is_op = claim_obj.get("isOpinion", False) or claim_obj.get("claimType") == "Opinion / Perspective Assertion"
            if not is_op and c_text and c_text not in ranked_cache:
                pending_claims.append(c_text)

        if len(pending_claims) > 1:
            executor = ThreadPoolExecutor(max_workers=min(3, len(pending_claims)))
            future_to_claim = {}
            for c_text in pending_claims:
                ctx = contextvars.copy_context()
                future_to_claim[
                    executor.submit(ctx.run, self.rank_evidence_for_claim, c_text, retrieval_map.get(c_text, []))
                ] = c_text
            try:
                done, _ = wait(list(future_to_claim.keys()), timeout=10.0)
                for fut in done:
                    c_text = future_to_claim[fut]
                    try:
                        ranked_cache[c_text] = fut.result()
                    except Exception as e:
                        logger.warning(f"Error ranking evidence for claim '{c_text[:40]}': {e}")
                        ranked_cache[c_text] = {"supporting": [], "contradicting": [], "related": [], "neutral": []}
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        elif len(pending_claims) == 1:
            c_text = pending_claims[0]
            ranked_cache[c_text] = self.rank_evidence_for_claim(c_text, retrieval_map.get(c_text, []))

        for idx, claim_obj in enumerate(claims):
            claim_text = claim_obj.get("claim", "")
            is_opinion = claim_obj.get("isOpinion", False) or claim_obj.get("claimType") == "Opinion / Perspective Assertion"
            candidates = retrieval_map.get(claim_text, [])

            if is_opinion:
                # Do not treat opinion claims as factual breaking news
                matrix_entry = {
                    "claimNumber": idx + 1,
                    "claim": claim_text,
                    "claimType": "Opinion / Perspective Assertion",
                    "isOpinion": True,
                    "status": "Opinion / Subjective",
                    "confidence": 85,
                    "supportingEvidence": [],
                    "contradictingEvidence": [],
                    "sourceLinks": [],
                    "evidenceSummary": "Opinion and perspective statements express subjective viewpoint arguments and are excluded from strict true/false wire fact-checking."
                }
                matrix.append(matrix_entry)
                continue

            ranked_data = ranked_cache.get(claim_text)
            if ranked_data is None:
                ranked_data = self.rank_evidence_for_claim(claim_text, candidates)
            supp = ranked_data["supporting"]
            contra = ranked_data["contradicting"]
            related = ranked_data.get("related", [])

            # Source links list
            all_links = []
            for it in supp + contra + related:
                u = it.get("url")
                if u and u not in all_links:
                    all_links.append(u)

            # Determine claim status based on grounded evidence
            if contra and len(contra) >= 1 and (not supp or contra[0]["compositeRankScore"] > (supp[0]["compositeRankScore"] if supp else 0)):
                if any(it.get("sourceCategory") == "IFCN Fact Checker" for it in contra) or (len(supp) == 0 and len(contra) >= 1):
                    status = "False"
                else:
                    status = "Contradicted"
                conf = max(75, min(98, contra[0]["compositeRankScore"]))
                reason_detail = contra[0].get("stanceReason")
                if reason_detail:
                    summary = f"Claim contradicted by {contra[0]['publisher']}: {reason_detail}"
                else:
                    summary = f"Claim contradicted by {len(contra)} source(s), including {contra[0]['publisher']}."
            elif len(supp) >= 2:
                status = "Verified"
                conf = max(80, min(98, int(round((supp[0]["compositeRankScore"] + supp[1]["compositeRankScore"]) / 2))))
                reason_detail = supp[0].get("stanceReason")
                if reason_detail:
                    summary = f"Corroborated by {supp[0]['publisher']} ({reason_detail}) and independent sources."
                else:
                    summary = f"Corroborated across multiple independent sources: {supp[0]['publisher']} and {supp[1]['publisher']}."
            elif len(supp) == 1:
                status = "Supported"
                conf = max(70, min(92, supp[0]["compositeRankScore"]))
                reason_detail = supp[0].get("stanceReason")
                if reason_detail:
                    summary = f"Supported by {supp[0]['publisher']}: {reason_detail}"
                else:
                    summary = f"Supported by reporting from {supp[0]['publisher']}."
            elif related:
                status = "Unverified"
                conf = 0
                summary = f"Topic-related coverage found ({related[0]['publisher']}), but does not specifically verify the claim proposition."
            else:
                # Missing evidence must NEVER be treated as false!
                status = "Unverified"
                conf = 0
                summary = "No sufficiently relevant corroborating sources were found. Unverified pending corroboration."

            # Outdated evidence check for time-sensitive claims (Task 1)
            # If supporting evidence is older than 6 months, mark the claim 'may be outdated'
            is_ts = self._is_time_sensitive_claim(claim_text)
            if is_ts and status in ("Verified", "Supported") and supp:
                supp_dates = []
                today_dt = date.today()
                for s_item in supp:
                    pdate_str = s_item.get("publicationDate", "")
                    if pdate_str:
                        try:
                            dt = datetime.strptime(pdate_str[:10], "%Y-%m-%d").date()
                            supp_dates.append((today_dt - dt).days)
                        except Exception:
                            pass
                if supp_dates and all(days > 180 for days in supp_dates):
                    status = "May be outdated"
                    conf = min(conf, 60)
                    summary += " Supporting evidence is older than 6 months for this time-sensitive claim and may be outdated."
                    claim_obj["isOutdated"] = True

            # Update the claim_obj with grounded metadata
            claim_obj["status"] = status
            claim_obj["confidence"] = conf
            claim_obj["supportingEvidenceCount"] = len(supp)
            claim_obj["contradictingEvidenceCount"] = len(contra)
            claim_obj["relatedEvidenceCount"] = len(related)
            claim_obj["supportingSources"] = [s["publisher"] for s in supp[:3]]
            claim_obj["contradictingSources"] = [c["publisher"] for c in contra[:3]]
            claim_obj["relatedSources"] = [r["publisher"] for r in related[:3]]
            claim_obj["analysisNotes"] = summary

            matrix_entry = {
                "claimNumber": idx + 1,
                "claim": claim_text,
                "claimType": claim_obj.get("claimType", "Factual Proposition"),
                "isOpinion": False,
                "status": status,
                "confidence": conf,
                "supportingEvidence": supp[:4],
                "contradictingEvidence": contra[:4],
                "relatedEvidence": related[:4],
                "sourceLinks": all_links,
                "evidenceSummary": summary
            }
            matrix.append(matrix_entry)

        return matrix


_ranker = EvidenceRanker()


def get_evidence_ranker() -> EvidenceRanker:
    return _ranker
