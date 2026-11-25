"""
Google Fact Check Tools API Service
Integrates with https://factchecktools.googleapis.com/v1alpha1/claims:search
to cross-examine extracted claims against the international IFCN fact-checking network.
"""
import os
import re
import logging
from typing import List, Dict, Any
import requests
from services.evidence_availability import evidence_tracker

logger = logging.getLogger("lucid.factcheck")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [FactCheckAPI] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def query_google_factcheck(query: str) -> List[Dict[str, Any]]:
    """
    Queries Google Fact Check Tools API for a given claim or query string.
    Requirements:
    1. Reads FACTCHECK_API_KEY from environment variables.
    2. Uses requests library.
    3. Query: https://factchecktools.googleapis.com/v1alpha1/claims:search
       Parameters: query=<claim_text>, key=<FACTCHECK_API_KEY>
    4. Parses response and returns:
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
    6. Returns [] if no fact-check exists or on failure.
    11. Backend logging for API responses.
    """
    if not query or not query.strip():
        return []

    api_key = (os.getenv("FACTCHECK_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    clean_query = query.strip()[:180]

    if not api_key:
        logger.info(f"Neither FACTCHECK_API_KEY nor GOOGLE_API_KEY is configured. Skipping live Google Fact Check call for query: '{clean_query}'")
        return []

    params = {
        "query": clean_query,
        "key": api_key,
        "languageCode": "en"
    }

    logger.info(f"Dispatching query to Google Fact Check Tools API: '{clean_query}'")

    try:
        response = requests.get(FACTCHECK_ENDPOINT, params=params, timeout=4)
        logger.info(f"Google Fact Check API response status code: {response.status_code}")

        if response.status_code != 200:
            evidence_tracker.mark_failure("FactCheck", f"HTTP {response.status_code}")
            logger.warning(
                f"Google Fact Check API returned error status {response.status_code}: {response.text[:200]}"
            )
            return []

        data = response.json()
        claims = data.get("claims", [])
        logger.info(f"Google Fact Check API returned {len(claims)} claim result(s) for query: '{clean_query}'")

        results: List[Dict[str, Any]] = []
        for item in claims:
            claim_text = item.get("text", clean_query)
            claimant = item.get("claimant", "Unknown Claimant")
            reviews = item.get("claimReview", [])

            for review in reviews:
                publisher = review.get("publisher", {})
                reviewer = publisher.get("name") or publisher.get("site", "Independent Fact Checker")
                rating = review.get("textualRating", "Unrated")
                url = review.get("url", "")
                review_date = review.get("reviewDate", "") or item.get("claimDate", "")

                result_entry = {
                    "claim": claim_text,
                    "claimant": claimant,
                    "reviewer": reviewer,
                    "rating": rating,
                    "url": url,
                    "review_date": review_date,
                    "source": "Google Fact Check API"
                }
                results.append(result_entry)
                logger.info(f"Parsed fact check record: Reviewer='{reviewer}', Rating='{rating}', URL='{url}'")

        return results

    except requests.exceptions.Timeout:
        evidence_tracker.mark_failure("FactCheck", "request timed out")
        logger.error(f"Google Fact Check API timed out after 6 seconds for query: '{clean_query}'")
        return []
    except requests.exceptions.RequestException as e:
        evidence_tracker.mark_failure("FactCheck", e)
        logger.error(f"Google Fact Check API request failed: {e}")
        return []
    except Exception as e:
        evidence_tracker.mark_failure("FactCheck", e)
        logger.error(f"Unexpected error parsing Google Fact Check API response: {e}")
        return []


_FC_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "after",
    "before", "during", "about", "since", "under", "between", "against", "among",
    "has", "have", "had", "was", "were", "are", "been", "being", "will", "would",
    "could", "should", "may", "might", "can", "did", "does", "not", "his", "her",
    "its", "their", "our", "who", "whom", "whose", "which", "what", "when", "where",
    "why", "how", "also", "more", "most", "very", "ever", "both", "each", "other",
    "than", "then", "only", "still", "such", "some", "many", "all", "new",
    "claimed", "claims", "claim", "says", "said", "according", "report", "reports",
    "reported", "viral", "social", "media", "post", "posts", "shows", "show",
    "shown", "showing", "made", "make", "makes",
}

_FC_TITLE_WORDS = {
    "ceo", "president", "prime", "minister", "chief", "executive", "officer",
    "chairman", "director", "governor", "senator", "ambassador", "king", "queen",
    "prince", "leader", "spokesperson", "official", "officials", "company",
}

_FC_SYNONYM_GROUPS = [
    {"die", "died", "dead", "death", "dying", "killed", "pass", "passed", "passing", "obituary", "funeral", "demise"},
    {"vaccin", "vaccine", "vaccines", "vaccinated", "vaccination", "jab", "jabs", "immunization"},
    {"return", "returns", "returned", "returning", "reinstated", "reappointed", "rejoin", "rejoined"},
    {"found", "founded", "founder", "cofound", "cofounded", "established"},
    {"resign", "resigned", "resigns", "stepped", "ousted", "quit", "fired"},
    {"arrest", "arrested", "detained", "jailed", "imprisoned", "custody", "indicted"},
    {"endors", "endorse", "endorsed", "endorses", "endorsement", "backed"},
]


def _fc_token_matches_text(tok: str, reviewed_lower: str, reviewed_tokens: set) -> bool:
    """Checks direct, prefix/stem, or predicate-family match between a predicate token and reviewed claim."""
    if tok in reviewed_tokens:
        return True
    if len(tok) >= 5 and any(rt.startswith(tok[:5]) or tok.startswith(rt[:5]) for rt in reviewed_tokens if len(rt) >= 5):
        return True
    for group in _FC_SYNONYM_GROUPS:
        if tok in group and (group & reviewed_tokens or any(rt.startswith(g) for g in group if len(g) >= 5 for rt in reviewed_tokens)):
            return True
    return tok in reviewed_lower and len(tok) >= 5


def is_factcheck_claim_relevant(claim_text: str, reviewed_claim: str) -> bool:
    """
    Guards against false-positive keyword matches from Google Fact Check Tools API.
    For example, querying "Jensen Huang co-founded NVIDIA" may return an unrelated AFP check
    for "Vintage picture shows Nvidia CEO Jensen Huang and his wife Lori Huang".
    """
    if not reviewed_claim or not reviewed_claim.strip():
        return True

    claim_clean = claim_text.strip()
    reviewed_clean = reviewed_claim.strip()
    if claim_clean.lower() == reviewed_clean.lower():
        return True

    claim_lower = claim_clean.lower()
    reviewed_lower = reviewed_clean.lower()

    # Identify proper-noun / organization tokens so sharing ONLY entity names ("Jensen Huang", "Nvidia", "Mars")
    # does not treat an unrelated fact-check about a photo/scam as evaluating the user's factual predicate.
    entity_tokens = set()
    for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+)\b', claim_clean):
        for w in re.findall(r'[a-z0-9]+', match.group(1).lower()):
            entity_tokens.add(w)
    for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]{1,})\b', claim_clean):
        w = match.group(1).lower()
        # Treat all-caps acronyms (NVIDIA, PM, FIFA) or mid-sentence capitalized proper nouns as entity tokens
        if match.group(1).isupper() or match.start() > 0:
            entity_tokens.add(w)

    all_claim_tokens = [t for t in re.findall(r'[a-z0-9]+', claim_lower) if len(t) >= 3]
    predicate_tokens = [
        t for t in all_claim_tokens
        if t not in _FC_STOPWORDS and t not in _FC_TITLE_WORDS and t not in entity_tokens
    ]
    # Also exclude sentence-initial proper subject if >= 2 non-entity predicate tokens remain
    # (e.g. "Mars has two moons named Phobos and Deimos" -> prevent "Mars" alone matching "Picture of Earth on Mars")
    if all_claim_tokens and len(predicate_tokens) >= 3 and re.match(r'^[A-Z][a-z]+\b', claim_clean):
        first_tok = all_claim_tokens[0]
        if first_tok in predicate_tokens:
            predicate_tokens = [t for t in predicate_tokens if t != first_tok]
    # Fallback to non-stopword tokens if stripping entity tokens leaves nothing (e.g. short acronym claim)
    if not predicate_tokens:
        predicate_tokens = [t for t in all_claim_tokens if t not in _FC_STOPWORDS and t not in _FC_TITLE_WORDS]
    if not predicate_tokens:
        return True

    reviewed_tokens = set(re.findall(r'[a-z0-9]+', reviewed_lower))
    matched_preds = [tok for tok in predicate_tokens if _fc_token_matches_text(tok, reviewed_lower, reviewed_tokens)]

    is_visual_check = bool(re.search(
        r'\b(vintage picture|this photo|photo shows|photo of|this image|image shows|image of|picture of|screenshot|altered image|fake photo|old photo|edited photo|viral video|video shows|video of|footage of|clip shows|colours of|colors of|fireworks)\b',
        reviewed_lower,
    ))
    claim_mentions_media = bool(re.search(r'\b(photo|picture|image|video|footage|screenshot|clip|fireworks)\b', claim_lower))

    if is_visual_check and not claim_mentions_media:
        if len(matched_preds) == 0 or (len(matched_preds) / len(predicate_tokens)) < 0.5:
            return False

    if not matched_preds:
        return False

    # Guard against low-relevance topic overlap when semantic embeddings are available
    # (e.g. "FIFA World Cup" / "2026" fireworks fact-check sim=0.236 vs game result claim)
    try:
        from services.semantic_embeddings import get_semantic_engine
        sim = get_semantic_engine().compute_similarity(claim_clean, reviewed_clean)
        if sim < 0.42:
            return False
    except Exception:
        pass

    return True
