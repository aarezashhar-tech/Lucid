"""
Gemini Evidence Stance Classifier Service
Uses Google GenAI SDK (gemini-2.5-flash with resilient fallback) strictly as a stance classifier
that compares a single claim against retrieved evidence snippets (title + snippet + date).

Structured Output Schema:
- stance: "support" | "contradict" | "neutral"
- confidence: int (0 - 100)
- reason: 1-sentence explanation

Handles:
- Subject/object inversions (e.g., "A defeated B" vs "B defeated A")
- Wrong locations, capitals, years, and dates
- Comparative and superlative swaps ("smallest" vs "largest")
- Debunked myths and viral hoaxes
- Keyword overlap trap: Never marks "support" simply because words match
- Graceful fallback: Falls back to heuristic stance ranker if API key is missing,
  rate-limited, offline, or errors out.
"""

import os
import json
import time
import threading
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from services.evidence_availability import evidence_tracker

# Ensure environment variables are loaded
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logger = logging.getLogger("lucid.gemini_stance")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [GeminiStance] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai package not found. Stance classifier will use heuristic fallback.")


class GeminiStanceClassifier:
    """
    Evidence stance classifier powered by Google Gemini (gemini-2.5-flash).
    Batches candidate snippets for a single claim into a single prompt for speed and consistency.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client = None
        self.preferred_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        self.active_model = self.preferred_model
        self._disabled = False
        self._state_lock = threading.Lock()
        self._daily_exhausted_until: Dict[str, float] = {}
        self.last_status = "available" if (GENAI_AVAILABLE and self.api_key) else "stance classifier unavailable"

        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=15000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
                logger.info(f"Initialized Gemini Client for stance classification (Target Model: {self.preferred_model}).")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client: {e}. Falling back to heuristic ranker.")
                self.client = None
        else:
            if not self.api_key:
                logger.info("GEMINI_API_KEY not configured. Stance classifier will use heuristic fallback.")

    def is_available(self) -> bool:
        """Returns True if Gemini client is initialized and not permanently disabled."""
        return bool(self.client and self.api_key and not self._disabled)

    def classify_candidates_for_claim(
        self,
        claim_text: str,
        candidate_items: List[Dict[str, Any]],
        max_candidates: int = 15
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Classifies the stance of all candidate evidence items for a single claim in a single batched call.

        Returns:
            Dict[int, Dict[str, Any]] mapping item index (0..N-1) to:
            {
                "stance": "support" | "contradict" | "neutral",
                "confidence": int (0-100),
                "reason": str (1 sentence)
            }
            OR None if Gemini is unavailable or errors out (triggers heuristic fallback).
        """
        if not self.is_available():
            reason = "GEMINI_API_KEY not configured" if not self.api_key else self.last_status
            evidence_tracker.mark_failure("Gemini Stance Classifier", reason)
            return None
        if not candidate_items or not claim_text.strip():
            return None

        # Format candidates into a clean, compact payload for Gemini
        items_payload = []
        for idx, item in enumerate(candidate_items[:max_candidates]):
            title = (item.get("articleTitle") or "").strip()
            snippet = (item.get("evidenceSnippet") or item.get("description") or "").strip()
            # Truncate snippet to 350 chars to keep token count low and latency fast
            snippet = snippet[:350]
            date_str = (item.get("publicationDate") or item.get("publicationTime") or "").strip()[:10]
            publisher = (item.get("publisher") or item.get("sourceName") or "").strip()

            item_dict = {
                "id": idx,
                "publisher": publisher,
                "date": date_str,
                "title": title,
                "snippet": snippet
            }
            fc_rating = (item.get("factCheckRating") or "").strip()
            if fc_rating:
                item_dict["factCheckRating"] = fc_rating
            items_payload.append(item_dict)

        if not items_payload:
            return None

        prompt = (
            "You are an expert fact-checking evidence stance classifier.\n"
            "Compare the single claim below against each retrieved evidence snippet.\n"
            "Determine whether each evidence snippet supports, contradicts, or is neutral towards the claim.\n\n"
            f'Claim: "{claim_text}"\n\n'
            "Retrieved Evidence Snippets:\n"
            f"{json.dumps(items_payload, indent=2)}\n\n"
            "CRITICAL CLASSIFICATION RULES:\n"
            "1. SUBJECT/OBJECT INVERSION: If the claim says 'A defeated B' but the evidence says 'B defeated A' (or vice-versa), the stance is CONTRADICT.\n"
            "2. CONFLICTING LOCATIONS, CAPITALS, YEARS & METRICS: If the claim asserts an incorrect capital city, location, year, date, number, or officeholder that directly CONFLICTS with the evidence (e.g. 'Sydney is the capital of Australia' vs evidence showing Canberra is capital; or 'cut rates by 5%' vs evidence showing 'raised rates by 25 bps'; or a trivial/low growth rate like '2%' when evidence shows 'record pace' or massive expansion), the stance is CONTRADICT or NEUTRAL, NEVER 'support'. However, mere omission of non-conflicting background context or headquarters cities (e.g., 'in Washington' for the Federal Reserve) does not contradict or negate support.\n"
            "3. COMPARATIVE & SUPERLATIVE SWAPS: If the claim asserts 'smallest' and evidence says 'largest' (or vice-versa), the stance is CONTRADICT.\n"
            "4. DEBUNKED MYTHS & FACT-CHECK RATINGS: If the evidence debunks, refutes, or clarifies a known falsehood or myth (e.g. vaccines cause autism, microchips in drinking water), OR if the snippet has an IFCN Fact Check Rating such as 'False' or 'Pants on Fire', the stance is CONTRADICT.\n"
            "5. NO KEYWORD OVERLAP SUPPORT: Never mark 'support' just because words match or the article is about the same general topic/entities. If the evidence is merely on the same general topic without confirming the specific assertion (e.g. an article about Latin American air pollution when the claim is about WHO/European utilities), mark 'neutral'.\n"
            "6. SUPPORT REQUIREMENT: Mark 'support' if the evidence affirmatively confirms the core factual action, event, finding, or growth asserted in the claim (e.g. if reputable news or institutional reporting confirms the primary finding or record growth rate in 2023, mark 'support' even if the headline uses summary phrasing like 'grew at record pace in 2023' without reciting every metric).\n"
            "7. TEMPORAL AWARENESS & EVENT TIMELINE: Pay careful attention to the publication date of each source relative to the claim and today's date. If a claim asserts an event, appointment, decision, resignation, or recent change, a source published BEFORE that event occurred CANNOT contradict it (the source simply reported prior reality before the event happened). In such cases, mark 'neutral', NEVER 'contradict'.\n"
            "8. OLD SOURCES: Check if the source publication date is older than 6 months relative to today. If so, set 'is_older' to true. (For Wikipedia or reference encyclopedias, set 'is_older' to false).\n"
            "9. FUTURE OR SPECULATIVE EVENTS VS OCCURRED EVENTS: If the claim asserts an event, decision, appointment, resignation, acquisition, or milestone HAS HAPPENED (completed past/present fact), but the evidence only discusses a possible, planned, proposed, or future event (using phrasing like 'wants to', 'may', 'might', 'could', 'plans to', 'planning to', 'talks about', 'aims to', 'considering', 'expected to', 'proposing to', 'prepares to'), the stance MUST be 'neutral', NEVER 'support'. Only sources reporting that the event has actually occurred count as 'support'.\n"
            "10. SPECIFIC CLAIM TOPIC & KEY ENTITY REQUIREMENT: An evidence snippet CANNOT contradict a claim unless the snippet explicitly addresses the specific assertion, controversy, or proposition in the claim. A generic overview or background encyclopedia entry (e.g. general article on 'Earth', 'Nvidia', or an entire planet or country) that does not explicitly refute or address the specific claim proposition MUST be 'neutral', NEVER 'contradict'.\n\n"
            "Output must be a JSON array of objects with keys:\n"
            "- id: integer matching the snippet id\n"
            "- stance: exactly one of 'support', 'contradict', or 'neutral'\n"
            "- confidence: integer from 0 to 100\n"
            "- is_older: boolean (true if source is older than 6 months relative to today)\n"
            "- reason: exactly 1 clear sentence explaining why the evidence supports, contradicts, or is neutral towards the claim."
        )

        # Try the configured GEMINI_MODEL first, then stable fallbacks if it is
        # retired or temporarily unavailable (including free-tier daily quota exhaustion).
        with self._state_lock:
            now = time.time()
            candidates = [self.active_model]
            for fallback in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]:
                if fallback not in candidates:
                    candidates.append(fallback)
            models_to_try = [m for m in candidates if self._daily_exhausted_until.get(m, 0.0) <= now]
            if not models_to_try:
                models_to_try = candidates[:1]

        for model_name in models_to_try:
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )

                if model_name != self.active_model:
                    logger.info(f"Updated active Gemini stance model from {self.active_model} to {model_name}.")
                    self.active_model = model_name

                raw_text = response.text.strip()
                parsed = json.loads(raw_text)

                if isinstance(parsed, dict) and "items" in parsed:
                    parsed = parsed["items"]
                elif isinstance(parsed, dict) and "results" in parsed:
                    parsed = parsed["results"]

                if not isinstance(parsed, list):
                    logger.warning(f"Unexpected response structure from Gemini: {raw_text[:200]}")
                    return None

                results: Dict[int, Dict[str, Any]] = {}
                for obj in parsed:
                    item_id = obj.get("id")
                    stance = str(obj.get("stance", "neutral")).lower().strip()
                    if stance not in ("support", "contradict", "neutral"):
                        stance = "neutral"
                    conf = int(obj.get("confidence", 80))
                    conf = max(0, min(100, conf))
                    reason = str(obj.get("reason", "")).strip()
                    matched_payload = next((it for it in items_payload if it.get("id") == item_id), None)
                    is_wiki = False
                    if matched_payload:
                        pub = (matched_payload.get("publisher") or "").lower()
                        if "wikipedia" in pub:
                            is_wiki = True

                    # Requirement 3: Real publish date & Wikipedia rule
                    # For Wikipedia, do not label by page edit date
                    if is_wiki:
                        is_older = False
                    elif matched_payload and matched_payload.get("date"):
                        try:
                            from datetime import date, datetime
                            p_dt = datetime.strptime(matched_payload["date"][:10], "%Y-%m-%d").date()
                            is_older = ((date.today() - p_dt).days > 180)
                        except Exception:
                            is_older = False
                    else:
                        is_older = False

                    temporal_label = "reference / evergreen" if is_wiki else ("older, may be outdated" if is_older else "recent")

                    if isinstance(item_id, int):
                        results[item_id] = {
                            "stance": stance,
                            "confidence": conf,
                            "reason": reason,
                            "is_older": is_older,
                            "temporal_label": temporal_label
                        }

                self.last_status = "available"
                evidence_tracker.mark_success("Gemini Stance Classifier")
                logger.info(f"Gemini stance classification succeeded for claim '{claim_text[:45]}...' across {len(results)} items (Model: {model_name}).")
                return results

            except ClientError as ce:
                err_str = str(ce)
                with self._state_lock:
                    if "PerDay" in err_str or "GenerateRequestsPerDay" in err_str:
                        self._daily_exhausted_until[model_name] = time.time() + 600.0
                    else:
                        self._daily_exhausted_until[model_name] = time.time() + 120.0
                logger.warning(f"Gemini model '{model_name}' returned ClientError ({err_str[:120]}). Trying next fallback model...")
                continue
            except Exception as e:
                err_str = str(e)
                # Cool down models reporting 503 high demand, 504 timeout, or 429 rate limit
                if any(k in err_str for k in ("503", "504", "429", "UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "high demand")):
                    with self._state_lock:
                        self._daily_exhausted_until[model_name] = time.time() + 120.0
                logger.warning(f"Gemini stance classification error with model '{model_name}': ({err_str[:120]}). Trying next fallback model...")
                continue

        self.last_status = "stance classifier unavailable"
        failed_list = models_to_try or candidates
        evidence_tracker.mark_failure(
            "Gemini Stance Classifier",
            f"Stance classification unavailable (models failed: {', '.join(failed_list)})"
        )
        return None

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the stance classifier."""
        return {
            "status": self.last_status,
            "model": self.active_model,
            "preferredModel": self.preferred_model,
            "available": (self.last_status == "available")
        }


# Global singleton instance
_gemini_stance_classifier_instance: Optional[GeminiStanceClassifier] = None


def get_gemini_stance_classifier() -> GeminiStanceClassifier:
    """Returns the singleton instance of GeminiStanceClassifier."""
    global _gemini_stance_classifier_instance
    if _gemini_stance_classifier_instance is None:
        _gemini_stance_classifier_instance = GeminiStanceClassifier()
    return _gemini_stance_classifier_instance
