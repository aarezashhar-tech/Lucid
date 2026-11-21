"""
Layer 1b: Article Classifier & Analytical Type Evaluator.
Classifies articles into:
- News Report (Strict factual breaking-news verification)
- Feature Article (Narrative storytelling / contextual credibility)
- Historical Analysis (Source quality assessment over time)
- Opinion Piece (Subjective viewpoints & editorial rhetoric)

Ensures feature articles and opinion pieces are not penalized as uncorroborated breaking news,
and provides Source Quality Assessment for historical and analysis articles.
"""
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("lucid.article_classifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [ArticleClassifier] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =========================================================================
# Pattern Definitions for the 4 Article Types
# =========================================================================

# 1. OPINION PIECE PATTERNS
OPINION_TITLE_PATTERNS = [
    r'\b(?:opinion|editorial|column|perspective|op-ed|commentary|essay)\b',
    r'^(?:in my view|in our view|my take|the case for|the case against|why we must|why we should)\b',
    r'\b(?:letters to the editor|guest essay)\b'
]

OPINION_FIRST_PERSON_PATTERNS = [
    r'\b(?:in my view|in our view|in my opinion|i believe|we believe|i think|my take|to my mind|it seems to me|i argue|we argue|from my perspective|my conclusion is|i am convinced)\b',
    r'\b(?:i write this|i have argued|my view is|we must recognize|let us not forget|as i see it)\b'
]

OPINION_RHETORICAL_PATTERNS = [
    r'\b(?:should|ought to|must not|disgraceful|shameful|unacceptable|hypocrisy|triumph|tragic mistake|flawed reasoning|fundamentally wrong|undeniably|morally|arguably|inexcusable)\b',
    r'\b(?:absurd|outrageous|disastrous decision|folly|shortsighted|we deserve better)\b'
]

# 2. HISTORICAL ANALYSIS PATTERNS
HISTORICAL_TITLE_PATTERNS = [
    r'\b(?:history of|historical analysis|retrospective|the origins of|looking back|how .* changed the world|chronicle of|untold story of the)\b',
    r'\b(?:centuries of|in the \d{1,2}(?:th|st|nd|rd) century|ancient|renaissance|industrial revolution|cold war era)\b'
]

HISTORICAL_ERA_PATTERNS = [
    r'\b(?:centuries ago|in the \d{1,2}(?:th|st|nd|rd) century|during the (?:cold war|middle ages|renaissance|industrial revolution|world war|great depression))\b',
    r'\b(?:ancient greece|roman empire|ottoman empire|byzantine|pre-war|post-war period|decades earlier|centuries earlier)\b',
    r'\b(?:historiography|historians argue|historical archives|archival evidence|treaty of|historical consensus|chronicles of)\b'
]

# 3. FEATURE ARTICLE PATTERNS
FEATURE_TITLE_PATTERNS = [
    r'\b(?:feature|special report|inside the world of|the story of|profile:|life inside|a day in the life|journey into|the hidden world)\b'
]

FEATURE_NARRATIVE_PATTERNS = [
    r'\b(?:the morning sun|sitting in|nestled in|walking through|for years (?:he|she|they)|for months|a journey through|a quiet corner|beneath the surface|behind the scenes|surrounded by|tales of|an intimate look|at first glance)\b',
    r'\b(?:memories of|the aroma of|the sound of|stood motionless|gazed across|every afternoon|a timeless tradition|lived among)\b'
]

# 4. NEWS REPORT PATTERNS
NEWS_WIREDATELINE_PATTERNS = [
    r'^[A-Z\s,]{2,20}\s*(?:\([A-Za-z\s,]+\))?\s*[-–—]\s*',
    r'\b(?:reuters|associated press|ap|afp|bbc news|bloomberg news)\b'
]

NEWS_EVENT_PATTERNS = [
    r'\b(?:breaking|just in|announced today|reported on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|press release|statement issued|official confirmed|according to officials|police said|ministry stated|authorities reported)\b',
    r'\b(?:death toll|casualty|fatalities|emergency meeting|signed an agreement|voted to approve|declared a state of emergency|official data showed)\b'
]


class ArticleClassifier:
    """Classifies articles into 4 distinct journalistic and analytical genres."""

    def classify_article_type(self, text: str, title: str = "", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classifies article into:
        - News Report
        - Feature Article
        - Historical Analysis
        - Opinion Piece
        """
        if not text and not title:
            return {
                "articleType": "News Report",
                "confidence": 60,
                "indicators": ["Default classification for empty text."],
                "evaluationMode": "factual_verification",
                "description": "Factual reporting evaluated against corroborating wire consensus."
            }

        # If title is omitted, derive candidate title from lead line
        if not title:
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            title = first_line[:120].strip()

        full_content = (title + "\n" + text).strip()
        lower_content = full_content.lower()
        title_lower = title.lower()

        scores = {
            "Opinion Piece": 0,
            "Historical Analysis": 0,
            "Feature Article": 0,
            "News Report": 0
        }
        indicators: Dict[str, List[str]] = {
            "Opinion Piece": [],
            "Historical Analysis": [],
            "Feature Article": [],
            "News Report": []
        }

        # Check metadata dictionary if supplied
        if metadata and isinstance(metadata, dict):
            sec = str(metadata.get("section", "")).lower()
            tag = str(metadata.get("tag", "")).lower()
            if any(k in sec or k in tag for k in ["opinion", "editorial", "commentary", "column", "op-ed"]):
                scores["Opinion Piece"] += 50
                indicators["Opinion Piece"].append(f"Metadata section identified as '{sec or tag}'.")
            if any(k in sec or k in tag for k in ["history", "archive", "retrospective", "historical"]):
                scores["Historical Analysis"] += 50
                indicators["Historical Analysis"].append("Metadata section identified historical archive.")
            if any(k in sec or k in tag for k in ["feature", "magazine", "culture", "lifestyle", "profile"]):
                scores["Feature Article"] += 50
                indicators["Feature Article"].append("Metadata section identified narrative feature.")

        # -------------------------------------------------------------
        # 1. OPINION PIECE SCORING
        # -------------------------------------------------------------
        for pat in OPINION_TITLE_PATTERNS:
            if re.search(pat, title_lower):
                scores["Opinion Piece"] += 50
                indicators["Opinion Piece"].append(f"Title / Header contains opinion genre tag ('{pat}').")
                break

        fp_matches = [pat for pat in OPINION_FIRST_PERSON_PATTERNS if re.search(pat, lower_content)]
        if fp_matches:
            weight = min(45, len(fp_matches) * 20 + 15)
            scores["Opinion Piece"] += weight
            indicators["Opinion Piece"].append(f"First-person argumentative perspective detected ({len(fp_matches)} indicator(s)).")

        rhetorical_matches = [pat for pat in OPINION_RHETORICAL_PATTERNS if re.search(pat, lower_content)]
        if rhetorical_matches:
            weight = min(35, len(rhetorical_matches) * 15 + 10)
            scores["Opinion Piece"] += weight
            indicators["Opinion Piece"].append("Subjective evaluative rhetoric and normative framing ('should', 'must', etc.).")

        # -------------------------------------------------------------
        # 2. HISTORICAL ANALYSIS SCORING
        # -------------------------------------------------------------
        for pat in HISTORICAL_TITLE_PATTERNS:
            if re.search(pat, title_lower):
                scores["Historical Analysis"] += 50
                indicators["Historical Analysis"].append("Title / Header frames a historical retrospective.")
                break

        era_matches = [pat for pat in HISTORICAL_ERA_PATTERNS if re.search(pat, lower_content)]
        if era_matches:
            weight = min(50, len(era_matches) * 20 + 15)
            scores["Historical Analysis"] += weight
            indicators["Historical Analysis"].append(f"Historical era references, archival records, or historiographical framing ({len(era_matches)} indicator(s)).")

        # Check for year citations in past centuries (1500s-1980s)
        past_years = re.findall(r'\b(?:1[5-9]\d{2})\b', full_content)
        if len(past_years) >= 2:
            scores["Historical Analysis"] += 30
            indicators["Historical Analysis"].append(f"Multiple historical year citations ({', '.join(past_years[:3])}).")

        # -------------------------------------------------------------
        # 3. FEATURE ARTICLE SCORING
        # -------------------------------------------------------------
        for pat in FEATURE_TITLE_PATTERNS:
            if re.search(pat, title_lower):
                scores["Feature Article"] += 50
                indicators["Feature Article"].append("Title matches feature/profile storytelling format.")
                break

        narrative_matches = [pat for pat in FEATURE_NARRATIVE_PATTERNS if re.search(pat, lower_content)]
        if narrative_matches:
            weight = min(50, len(narrative_matches) * 20 + 15)
            scores["Feature Article"] += weight
            indicators["Feature Article"].append(f"Descriptive narrative storytelling, scene-setting, and human-interest pacing ({len(narrative_matches)} indicator(s)).")

        # Long descriptive paragraphs without breaking news urgency
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        avg_line_len = (sum(len(l.split()) for l in lines) / max(len(lines), 1)) if lines else 0
        if avg_line_len > 25 and scores["Historical Analysis"] < 30 and scores["Opinion Piece"] < 30:
            scores["Feature Article"] += 15

        # -------------------------------------------------------------
        # 4. NEWS REPORT SCORING
        # -------------------------------------------------------------
        for pat in NEWS_WIREDATELINE_PATTERNS:
            if re.search(pat, text[:150]):
                scores["News Report"] += 40
                indicators["News Report"].append("Lead journalistic wire dateline detected.")
                break

        news_event_matches = [pat for pat in NEWS_EVENT_PATTERNS if re.search(pat, lower_content)]
        if news_event_matches:
            weight = min(40, len(news_event_matches) * 15)
            scores["News Report"] += weight
            indicators["News Report"].append(f"Immediate event reporting and official statements ({len(news_event_matches)} indicator(s)).")

        # Baseline news report points if nothing else strongly matched
        scores["News Report"] += 10

        # Determine winner
        best_type = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_type]
        total_score = sum(scores.values()) or 1
        confidence = int(round(min(98, max(70, (best_score / total_score) * 100 + 15))))

        # Fallback if no strong signals
        if best_score <= 10:
            best_type = "News Report"
            confidence = 75
            indicators["News Report"] = ["Standard event reporting format."]

        # Define evaluation modes and descriptions
        if best_type == "Opinion Piece":
            eval_mode = "viewpoint_evaluation"
            desc = "Opinion Piece / Commentary: Evaluated for rhetorical transparency, publisher authority, and viewpoint argumentation without treating opinions as factual breaking-news claims."
        elif best_type == "Feature Article":
            eval_mode = "narrative_credibility"
            desc = "Feature Article: In-depth narrative and human-interest storytelling. Evaluated for thematic credibility, source depth, and context rather than breaking-news wire corroboration."
        elif best_type == "Historical Analysis":
            eval_mode = "source_quality_assessment"
            desc = "Historical Analysis: Retrospective inquiry evaluated via Source Quality Assessment on archival rigor and scholarly authority rather than strict true/false daily wire verification."
        else:
            eval_mode = "factual_verification"
            desc = "News Report: Factual breaking news evaluated against wire corroboration and fact-checking consensus."

        return {
            "articleType": best_type,
            "confidence": confidence,
            "indicators": indicators.get(best_type, []),
            "evaluationMode": eval_mode,
            "description": desc
        }

    def generate_source_quality_assessment(
        self,
        text: str,
        source_data: Dict[str, Any],
        article_type: str = "Historical Analysis"
    ) -> Dict[str, Any]:
        """
        Generates Source Quality Assessment for Historical Analysis & analysis articles
        instead of strict true/false wire verification.
        """
        trust = source_data.get("trustScore", 85)
        tier = source_data.get("tier", "High")
        pub = source_data.get("publisher", source_data.get("sourceName", "Historical Archive"))

        # Evaluate citations and archival depth
        has_citations = bool(re.search(r'\b(?:source|sources|archive|references|bibliography|historian|treaty|records|in 19\d{2}|century)\b', text, re.I))
        has_academic_tone = not bool(re.search(r'\b(?:shocking|insane|click here|miracle|secret revealed)\b', text, re.I))

        if trust >= 85 and has_academic_tone:
            rating = "Authoritative Historical / Academic Standard"
            scholarly_std = "High - Institutional and archival documentation adhering to scholarly standards."
            methodology = "Structured chronological analysis referencing historical records and archival context."
        elif trust >= 65:
            rating = "Reputable Historical Review"
            scholarly_std = "Moderate - Established editorial publisher with general historical review standards."
            methodology = "Informative historical overview with contextual retrospective analysis."
        else:
            rating = "Unverified Historical Commentary"
            scholarly_std = "Unverified - Content lacks independent institutional archival verification."
            methodology = "Retrospective claims without accredited historiographical backing."

        return {
            "overallRating": rating,
            "sourceQualityScore": trust,
            "sourceAuthority": f"{pub} ({tier} Authority Tier)",
            "scholarlyStandards": scholarly_std,
            "methodologicalRigor": methodology,
            "archivalIntegrity": "Archival documentation identified in text context." if has_citations else "Contextual narrative review.",
            "assessmentSummary": f"Evaluated under Source Quality Assessment: {pub} demonstrates {tier.lower()}-tier source quality. Historical and analysis articles are judged on archival depth, scholarly standards, and source credibility rather than daily breaking news wire feeds.",
            "verificationMode": "Source Quality Assessment"
        }


article_classifier = ArticleClassifier()

def get_article_classifier() -> ArticleClassifier:
    return article_classifier
