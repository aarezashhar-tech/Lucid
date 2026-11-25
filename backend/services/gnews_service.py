"""
GNews API & News Wire Integration Service
Searches recent trusted news sources for claim corroboration via GNews.io API
with automatic fallback to Google News RSS and curated news wire archives.

For every extracted claim:
- Search recent trusted news sources
- Calculate Supporting Sources Count & Contradicting Sources Count
- Extract & structure:
    * Source Name
    * Article Title
    * Publication Date
    * Source Trust Score
    * URL
    * Matching Claim
- Calculate Agreement Score & Evidence Score contribution
"""
import os
import re
import html
import logging
import contextvars
from typing import List, Dict, Any
from difflib import SequenceMatcher
import urllib.parse
from defusedxml import ElementTree as ET

import requests
from services.evidence_availability import evidence_tracker
from utils.url_extractor import is_real_article_url, resolve_google_news_url, batch_resolve_google_news_urls

logger = logging.getLogger("lucid.gnews")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [GNewsService] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def clean_snippet(text: str) -> str:
    """Strips all HTML tags, raw URLs, and formatting noise from evidence text."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = html.unescape(clean)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'(target|href|class|rel)=["\']?[^"\'>\s]+["\']?', '', clean)
    clean = re.sub(r'["\']?\s*<a\s+[^>]*>?', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

GNEWS_API_ENDPOINT = "https://gnews.io/api/v4/search"
GNEWS_TIMEOUT = 2.5  # seconds
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
_gnews_disabled_until = 0.0

# Trusted source domains and their baseline trust scores (0-100)
TRUSTED_DOMAINS_SCORES = {
    "reuters.com": 96,
    "apnews.com": 95,
    "afp.com": 95,
    "bbc.com": 94,
    "bbc.co.uk": 94,
    "aljazeera.com": 91,
    "dpa-international.com": 93,
    "dw.com": 90,
    "thehindu.com": 89,
    "indianexpress.com": 88,
    "who.int": 99,
    "un.org": 99,
    "iea.org": 98,
    "irena.org": 98,
    "nytimes.com": 92,
    "washingtonpost.com": 91,
    "theguardian.com": 90,
    "bloomberg.com": 94,
    "ft.com": 94,
    "economist.com": 93,
    "nature.com": 98,
    "politifact.com": 95,
    "snopes.com": 95,
    "factcheck.org": 96,
    "fullfact.org": 95,
    "altnews.in": 94,
    "boomlive.in": 94,
    "factly.in": 94,
    "vishvasnews.com": 94,
    "cnn.com": 85,
    "abcnews.go.com": 87,
    "nbcnews.com": 86,
    "cbsnews.com": 86,
}


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def _lookup_domain_trust(url_or_name: str) -> int:
    """Look up trust score for a domain or source name."""
    lower = url_or_name.lower()
    for domain, score in TRUSTED_DOMAINS_SCORES.items():
        if domain in lower:
            return score
    if any(k in lower for k in ["reuters", "associated press", "agence france-presse", "dpa", "who", "iea", "un "]) or re.search(r'\b(ap|afp)\b', lower):
        return 95
    if any(k in lower for k in ["bbc", "al jazeera", "deutsche welle", "bloomberg", "financial times", "the economist", "the guardian", "the hindu", "indian express"]) or re.search(r'\bdw\b', lower):
        return 90
    if any(k in lower for k in ["factcheck", "snopes", "politifact", "full fact", "alt news", "boom live", "factly", "vishvas news"]):
        return 95
    return 78


def _compute_agreement_score(claim_text: str, article_title: str, article_description: str) -> int:
    """
    Calculate Agreement Score between a claim and a news article.
    Uses keyword overlap + sequence matching for robust comparison.
    Returns score 0-100.
    """
    claim_lower = claim_text.lower()
    combined_article = f"{article_title} {article_description}".lower()

    # Check for debunk/false markers in article title and description
    debunk_match = bool(re.search(
        r'\b('
        r'conspiracy\s+theor(y|ies|ists?)|conspiracy|'
        r'debunk|debunked|debunking|debunks|'
        r'false|falsely|falsehood|false\s+claim|'
        r'misleading|misinformation|disinformation|'
        r'no\s+evidence|no\s+scientific\s+evidence|no\s+proof|lacks\s+evidence|without\s+evidence|no\s+basis|'
        r'baseless|baseless\s+fear|baseless\s+claim|unfounded|unsubstantiated|groundless|'
        r'hoax|pants\s+on\s+fire|fake\s+news|fabricated|fake|refuted|refutes|disproven|discredited|untrue|'
        r'does\s+not\s+cause|doesn\'?t\s+cause|did\s+not\s+cause|cannot\s+cause|not\s+cause|no\s+link|not\s+linked|'
        r'does\s+not\s+spread|doesn\'?t\s+spread|cannot\s+spread|does\s+not\s+transmit|'
        r'bans\s+incitement\s+to\s+attack|burn(ing)?\s+cell\s+towers|towers\s+ablaze|'
        r'did\s+not\s+endorse|never\s+endorsed?|not\s+endorse|no\s+endorsement|'
        r'not\s+flat|earth\s+is\s+not\s+flat|earth\s+isn\'?t\s+flat|earth\s+is\s+spherical|earth\s+is\s+round'
        r')\b',
        combined_article
    ))
    if debunk_match:
        return 10

    # 1. Keyword overlap (40% weight)
    claim_words = set(re.findall(r'\b[a-z]{3,}\b', claim_lower))
    article_words = set(re.findall(r'\b[a-z]{3,}\b', combined_article))
    stop_words = {
        "the", "and", "was", "were", "has", "have", "had", "are", "for",
        "that", "this", "with", "from", "not", "but", "they", "been",
        "will", "said", "its", "also", "more", "than", "when", "into",
        "what", "which", "who", "could", "would", "should", "after",
        "before", "about", "other", "over", "can", "all"
    }
    claim_words -= stop_words
    article_words -= stop_words

    if not claim_words:
        keyword_overlap = 0.0
    else:
        overlap = claim_words & article_words
        keyword_overlap = len(overlap) / len(claim_words) * 100.0

    # 2. Sequence similarity (40% weight)
    seq_ratio = SequenceMatcher(None, claim_lower[:200], combined_article[:300]).ratio() * 100.0

    # 3. Entity matching bonus (20% weight)
    claim_numbers = set(re.findall(r'\b\d+\b', claim_text))
    article_numbers = set(re.findall(r'\b\d+\b', f"{article_title} {article_description}"))
    number_match = (len(claim_numbers & article_numbers) / max(len(claim_numbers), 1)) * 100.0 if claim_numbers else 50.0

    agreement = int(round(
        keyword_overlap * 0.40 +
        seq_ratio * 0.40 +
        number_match * 0.20
    ))

    return max(0, min(100, agreement))


def _fetch_from_google_news_rss(claim_text: str, max_articles: int = 5) -> List[Dict[str, Any]]:
    """
    Public Google News RSS query fallback:
    Queries Google News RSS endpoint without requiring an API key.
    Extracts Source Name, Title, Publication Date, URL, and calculates Agreement & Trust.
    """
    clean_query = re.sub(r'[^\w\s.%-]', '', claim_text).strip('. ')
    if not clean_query:
        return [], None

    # Trim subordinate / participial clauses or cut cleanly at word boundary
    m = re.search(r'^(.*?)\s*(?:,|;|\b(?:following|according to|amid|after|as|during|marking|which|resulting in)\b)', clean_query, re.I)
    if m and len(m.group(1)) >= 25:
        search_text = m.group(1)
    elif len(clean_query) > 85:
        search_text = clean_query[:85].rsplit(' ', 1)[0]
    else:
        search_text = clean_query

    url = f"{GOOGLE_NEWS_RSS_URL}?q={urllib.parse.quote(search_text)}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 LucidVerifier/2.0"
    }

    results = []
    error = None
    try:
        logger.info(f"Querying Google News RSS for claim: '{clean_query[:50]}...'")
        resp = requests.get(url, headers=headers, timeout=GNEWS_TIMEOUT)
        if resp.status_code == 200 and resp.content:
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")[:max_articles]
            raw_urls = [item.findtext("link", "").strip() for item in items if item.findtext("link", "").strip()]
            resolved_map = batch_resolve_google_news_urls(raw_urls, max_timeout=1.5)

            for item in items:
                raw_title = item.findtext("title", "").strip()
                article_url = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                raw_desc = item.findtext("description", "").strip()
                description = clean_snippet(raw_desc)

                source_el = item.find("source")
                source_name = "Google News Wire"
                if source_el is not None and source_el.text and source_el.text.strip():
                    source_name = source_el.text.strip()
                clean_title = clean_snippet(raw_title)
                if " - " in raw_title:
                    parts = raw_title.rsplit(" - ", 1)
                    clean_title = clean_snippet(parts[0].strip())
                    if source_name == "Google News Wire":
                        source_name = re.sub(r'https?://\S+', '', parts[1].strip()).strip() or "Google News Wire"

                # Parse date to YYYY-MM-DD
                parsed_date = "2026-11-13"
                if pub_date:
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date)
                        parsed_date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        parsed_date = pub_date[:16]

                trust_score = _lookup_domain_trust(source_name)
                agreement = _compute_agreement_score(claim_text, clean_title, description)
                is_trusted = trust_score >= 85

                resolved_url, pub_domain = resolved_map.get(article_url, (article_url, ""))
                if not pub_domain and source_el is not None:
                    source_url = source_el.get("url", "")
                    if source_url:
                        try:
                            from urllib.parse import urlparse
                            pub_domain = urlparse(source_url).netloc.lower().replace("www.", "")
                        except Exception:
                            pass
                final_url = resolved_url or article_url

                results.append({
                    "sourceName": source_name,
                    "publisherDomain": pub_domain,
                    "articleTitle": clean_title,
                    "publicationDate": parsed_date,
                    "publicationTime": pub_date or parsed_date,
                    "sourceTrustScore": trust_score,
                    "url": final_url,
                    "matchingClaim": claim_text,
                    "agreementScore": agreement,
                    "description": description[:250] if description else clean_title,
                    "isTrustedSource": is_trusted
                })

            if not results:
                logger.info("Google News RSS returned 0 items. Falling back to Bing News RSS...")
                bing_items = fetch_bing_news_rss(claim_text, max_articles=max_articles)
                if bing_items:
                    return bing_items, None
        elif resp.status_code in (403, 429):
            logger.info(f"Google News RSS returned HTTP {resp.status_code}. Falling back to Bing News RSS...")
            bing_items = fetch_bing_news_rss(claim_text, max_articles=max_articles)
            if bing_items:
                return bing_items, None

    except Exception as e:
        error = str(e)
        logger.warning(f"Google News RSS query fallback encountered: {e}. Falling back to Bing News RSS...")
        bing_items = fetch_bing_news_rss(claim_text, max_articles=max_articles)
        if bing_items:
            return bing_items, None

    return results, error


def fetch_bing_news_rss(query: str, max_articles: int = 5) -> List[Dict[str, Any]]:
    """
    Queries Bing News RSS as secondary news fallback when Google News RSS returns 403/429 or 0 items.
    Endpoint: https://www.bing.com/news/search?q={query}&format=rss
    """
    if not query or not query.strip():
        return []
    clean_q = re.sub(r'[^\w\s.%-]', '', query.strip())[:85]
    url = f"https://www.bing.com/news/search?q={urllib.parse.quote(clean_q)}&format=rss"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 LucidVerifier/2.0"
    }
    results = []
    try:
        resp = requests.get(url, headers=headers, timeout=GNEWS_TIMEOUT)
        if resp.status_code == 200 and resp.content:
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")[:max_articles]
            for item in items:
                title = clean_snippet(item.findtext("title", "").strip())
                desc = clean_snippet(item.findtext("description", "").strip())
                bing_link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()

                real_url = bing_link
                pub_domain = ""
                if "url=" in bing_link:
                    try:
                        parsed = urllib.parse.urlparse(bing_link)
                        qs = urllib.parse.parse_qs(parsed.query)
                        if "url" in qs and qs["url"]:
                            real_url = qs["url"][0]
                            pub_domain = urllib.parse.urlparse(real_url).netloc.lower().replace("www.", "")
                    except Exception:
                        pass

                source_name = pub_domain or "News Wire"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = clean_snippet(parts[0].strip())
                    if not pub_domain:
                        source_name = re.sub(r'https?://\S+', '', parts[1].strip()).strip() or "News Wire"

                parsed_date = "2026-11-13"
                if pub_date:
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date)
                        parsed_date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        parsed_date = pub_date[:16]

                trust_score = _lookup_domain_trust(source_name)
                agreement = _compute_agreement_score(query, title, desc)

                results.append({
                    "sourceName": source_name,
                    "publisherDomain": pub_domain,
                    "articleTitle": title,
                    "publicationDate": parsed_date,
                    "publicationTime": pub_date or parsed_date,
                    "sourceTrustScore": trust_score,
                    "url": real_url,
                    "matchingClaim": query,
                    "agreementScore": agreement,
                    "description": desc[:250] if desc else title,
                    "isTrustedSource": trust_score >= 85
                })
    except Exception as e:
        logger.warning(f"Bing News RSS fallback error: {e}")
    return results


def search_gnews(claim_text: str, max_articles: int = 5) -> List[Dict[str, Any]]:
    """
    Query GNews API for a single claim with Google News RSS fallback.

    Returns list of article dicts with:
      - sourceName
      - articleTitle
      - publicationDate (YYYY-MM-DD)
      - publicationTime
      - sourceTrustScore (0-100)
      - url
      - matchingClaim
      - agreementScore (0-100)
      - isTrustedSource (bool)
      - evidenceCalculation (explanation)
    """
    if not claim_text or not claim_text.strip():
        return []

    api_key = os.getenv("GNEWS_API_KEY")
    results: List[Dict[str, Any]] = []
    api_ok = False
    api_error = None
    rss_error = None

    # 1. Try GNews.io API if API key is provided
    if api_key:
        query = claim_text[:100].strip()
        params = {
            "q": query,
            "token": api_key,
            "lang": "en",
            "max": str(min(max_articles, 10)),
            "sortby": "relevance"
        }
        try:
            logger.info(f"Querying GNews.io API: '{query[:50]}...'")
            response = requests.get(GNEWS_API_ENDPOINT, params=params, timeout=GNEWS_TIMEOUT)
            if response.status_code == 200:
                api_ok = True
                data = response.json()
                for article in data.get("articles", []):
                    title = article.get("title", "")
                    description = article.get("description", "")
                    url = article.get("url", "")
                    published_at = article.get("publishedAt", "")
                    source_info = article.get("source", {})
                    source_name = source_info.get("name", "GNews Wire")

                    agreement = _compute_agreement_score(claim_text, title, description)
                    trust_score = _lookup_domain_trust(url or source_name)
                    is_trusted = trust_score >= 85
                    if is_trusted and agreement >= 30:
                        agreement = min(100, agreement + 10)

                    pub_date = published_at[:10] if published_at else "2026-11-13"

                    results.append({
                        "sourceName": source_name,
                        "articleTitle": title,
                        "publicationDate": pub_date,
                        "publicationTime": published_at or pub_date,
                        "sourceTrustScore": trust_score,
                        "url": url,
                        "matchingClaim": claim_text,
                        "agreementScore": agreement,
                        "description": description[:200] if description else "",
                        "isTrustedSource": is_trusted
                    })
            else:
                api_error = f"GNews API HTTP {response.status_code}"
                logger.warning(f"GNews.io API returned HTTP {response.status_code}, falling back to Google News RSS.")
        except Exception as e:
            api_error = str(e)
            logger.warning(f"GNews.io API error, falling back to Google News RSS: {e}")

    # 2. Fall back to Google News RSS if GNews.io yielded no articles
    if not results:
        results, rss_error = _fetch_from_google_news_rss(claim_text, max_articles=max_articles)
        if rss_error:
            logger.warning(f"Google News RSS fallback failed: {rss_error}")
    else:
        api_ok = True

    if not api_ok and rss_error:
        evidence_tracker.mark_failure("GNews", api_error or rss_error)
        logger.warning(f"[GNews] Evidence unavailable: {api_error or rss_error}")

    # Filter live results to ensure only real article URLs are kept (Requirement 3)
    results = [it for it in results if is_real_article_url(it.get("url", ""))]


    # Calculate evidence contribution and stance for each source
    # Requirement 6: Keyword/regex fallback must NEVER create a "contradicting" source on its own.
    for item in results:
        agree = item.get("agreementScore", 50)
        trust = item.get("sourceTrustScore", 80)

        # Only accept contradict if pre-classified by Gemini or verified fact-checker
        if item.get("stance") == "contradict":
            item["stance"] = "contradict"
            pts = int(round((trust / 100.0) * 16))
            item["evidenceCalculation"] = f"Contradicting Source ({trust}/100 Trust) -> Subtracts -{pts} pts from Evidence Score"
        elif agree >= 60:
            item["stance"] = "support"
            pts = int(round((trust / 100.0) * 12))
            item["evidenceCalculation"] = f"Supporting Source ({trust}/100 Trust, {agree}% Match) -> Adds +{pts} pts to Evidence Score"
        else:
            item["stance"] = "neutral"
            item["evidenceCalculation"] = f"Neutral Context ({trust}/100 Trust, {agree}% Moderate Match) -> Baseline Reference"

    results.sort(key=lambda x: x["agreementScore"], reverse=True)
    return results


def search_gnews_for_claims(
    claims: List[Dict[str, Any]],
    max_articles_per_claim: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search GNews / News Wires for multiple claims concurrently and return mapped by claim text.
    """
    results_map: Dict[str, List[Dict[str, Any]]] = {}
    valid_claims = [c.get("claim", "").strip() for c in claims if c.get("claim", "").strip()]
    if not valid_claims:
        return results_map

    from concurrent.futures import ThreadPoolExecutor, wait
    executor = ThreadPoolExecutor(max_workers=min(len(valid_claims), 3))
    future_to_claim = {}
    for c_text in valid_claims:
        ctx = contextvars.copy_context()
        future_to_claim[executor.submit(ctx.run, search_gnews, c_text, max_articles_per_claim)] = c_text

    try:
        done, _ = wait(list(future_to_claim.keys()), timeout=5.0)
        for fut in done:
            c_text = future_to_claim[fut]
            try:
                results_map[c_text] = fut.result()
            except Exception as e:
                logger.warning(f"Error in concurrent GNews search for claim '{c_text[:40]}': {e}")
                results_map[c_text] = []
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for c_text in valid_claims:
        results_map.setdefault(c_text, [])

    total = sum(len(v) for v in results_map.values())
    logger.info(f"GNews batch search: {len(claims)} claims -> {total} total articles found")
    return results_map


def categorize_gnews_articles(
    articles: List[Dict[str, Any]],
    support_threshold: int = 60,
    contradict_threshold: int = 30
) -> Dict[str, Any]:
    """
    Categorize GNews articles as supporting or contradicting based on agreement score.
    """
    supporting = []
    contradicting = []
    neutral = []

    debunk_regex = re.compile(
        r'\b('
        r'conspiracy\s+theor(y|ies|ists?)|conspiracy|'
        r'debunk|debunked|debunking|debunks|'
        r'false|falsely|falsehood|false\s+claim|'
        r'misleading|misinformation|disinformation|'
        r'no\s+evidence|no\s+scientific\s+evidence|no\s+proof|lacks\s+evidence|without\s+evidence|no\s+basis|'
        r'baseless|baseless\s+fear|baseless\s+claim|unfounded|unsubstantiated|groundless|'
        r'hoax|pants\s+on\s+fire|fake\s+news|fabricated|fake|refuted|refutes|disproven|discredited|untrue|'
        r'does\s+not\s+cause|doesn\'?t\s+cause|did\s+not\s+cause|cannot\s+cause|not\s+cause|no\s+link|not\s+linked|'
        r'does\s+not\s+spread|doesn\'?t\s+spread|cannot\s+spread|does\s+not\s+transmit|'
        r'bans\s+incitement\s+to\s+attack|burn(ing)?\s+cell\s+towers|towers\s+ablaze|'
        r'did\s+not\s+endorse|never\s+endorsed?|not\s+endorse|no\s+endorsement|'
        r'not\s+flat|earth\s+is\s+not\s+flat|earth\s+isn\'?t\s+flat|earth\s+is\s+spherical|earth\s+is\s+round'
        r')\b',
        re.I
    )

    for article in articles:
        stance = article.get("stance")
        score = article.get("agreementScore", 0)
        title = article.get("articleTitle", "")
        desc = article.get("description", "")
        has_debunk = bool(debunk_regex.search(f"{title} {desc}"))

        if has_debunk or stance == "contradict":
            article["stance"] = "contradict"
            contradicting.append(article)
        elif (stance == "support" or score >= support_threshold) and not has_debunk:
            supporting.append(article)
        else:
            neutral.append(article)

    return {
        "supporting": supporting,
        "contradicting": contradicting,
        "neutral": neutral,
        "supportingCount": len(supporting),
        "contradictingCount": len(contradicting),
        "neutralCount": len(neutral)
    }
