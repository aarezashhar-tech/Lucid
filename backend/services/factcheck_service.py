"""
Google Fact Check Tools API Service
Integrates with https://factchecktools.googleapis.com/v1alpha1/claims:search
to cross-examine extracted claims against the international IFCN fact-checking network.
"""
import os
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

    api_key = os.getenv("FACTCHECK_API_KEY", "").strip()
    clean_query = query.strip()[:180]

    if not api_key:
        logger.info(f"FACTCHECK_API_KEY not configured. Skipping live Google Fact Check call for query: '{clean_query}'")
        return []

    params = {
        "query": clean_query,
        "key": api_key,
        "languageCode": "en"
    }

    logger.info(f"Dispatching query to Google Fact Check Tools API: '{clean_query}'")

    try:
        response = requests.get(FACTCHECK_ENDPOINT, params=params, timeout=6)
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
