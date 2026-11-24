from __future__ import annotations
"""
Lucid RAG Retrieval Layer
Performs multi-source claim-by-claim evidence retrieval across:
- Reuters
- AP News
- BBC
- Al Jazeera
- The Guardian
- AFP
- Google Fact Check API
- IFCN Fact Check sources (PolitiFact, FactCheck.org, Snopes, Lead Stories, Full Fact)
- Official government websites (.gov, .gov.uk, etc.)
- UN, WHO, and recognised international institutions

Returns structured evidence items with:
- Article Title
- Publisher
- Publication Date
- URL
- Evidence Snippet
- Publisher Trust Score
- Source Category
"""
import os
import re
import html
import logging
import urllib.parse
from defusedxml import ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import services.factcheck_service as factcheck_service
from services.evidence_availability import evidence_tracker
from services.wikipedia_service import get_wikipedia_service, claim_refers_to_relative_or_associate
from utils.url_extractor import is_real_article_url, resolve_google_news_url

logger = logging.getLogger("lucid.rag_retriever")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [RAGRetriever] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

REQUEST_TIMEOUT = 6
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

def clean_text_snippet(raw_text: str) -> str:
    """Strips all HTML tags, raw URLs, and formatting noise from evidence text."""
    if not raw_text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    text = html.unescape(text)
    # Strip full URLs to prevent layout breaks
    text = re.sub(r'https?://\S+', '', text)
    # Strip leftover HTML attributes e.g. target="_blank"
    text = re.sub(r'(target|href|class|rel)=["\']?[^"\'>\s]+["\']?', '', text)
    text = re.sub(r'["\']?\s*<a\s+[^>]*>?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Standard target repositories audited for transparency
CORE_SOURCES_SEARCHED = [
    "Reuters",
    "AP News",
    "BBC",
    "Al Jazeera",
    "The Guardian",
    "AFP",
    "Google Fact Check API",
    "IFCN Fact Check sources",
    "Wikipedia (Official REST API)",
    "Official government websites",
    "UN, WHO & recognised institutions"
]

# Baseline publisher trust registry
PUBLISHER_TRUST_REGISTRY = {
    # Encyclopedic Reference
    "wikipedia": {"score": 88, "publisher": "Wikipedia", "category": "Encyclopedic Reference"},

    # Wire & Global Agencies (Tier 1)
    "reuters": {"score": 96, "publisher": "Reuters", "category": "Wire Service"},
    "ap news": {"score": 95, "publisher": "AP News", "category": "Wire Service"},
    "associated press": {"score": 95, "publisher": "Associated Press (AP)", "category": "Wire Service"},
    "bbc": {"score": 94, "publisher": "BBC", "category": "Public Broadcaster"},
    "al jazeera": {"score": 91, "publisher": "Al Jazeera", "category": "International News"},
    "the guardian": {"score": 90, "publisher": "The Guardian", "category": "Independent News"},
    "afp": {"score": 95, "publisher": "Agence France-Presse (AFP)", "category": "Wire Service"},
    "agence france-presse": {"score": 95, "publisher": "Agence France-Presse (AFP)", "category": "Wire Service"},
    "bloomberg": {"score": 94, "publisher": "Bloomberg", "category": "Financial News"},
    "financial times": {"score": 94, "publisher": "Financial Times", "category": "Financial News"},
    # National Newspapers & Major Broadcasters
    "the hindu": {"score": 89, "publisher": "The Hindu", "category": "National Newspaper"},
    "the times of india": {"score": 82, "publisher": "The Times of India", "category": "National Newspaper"},
    "times of india": {"score": 82, "publisher": "The Times of India", "category": "National Newspaper"},
    "ndtv": {"score": 83, "publisher": "NDTV", "category": "Broadcast News"},
    "india today": {"score": 82, "publisher": "India Today", "category": "Broadcast News"},
    
    # IFCN & Accredited Fact Checkers
    "politifact": {"score": 95, "publisher": "PolitiFact", "category": "IFCN Fact Checker"},
    "factcheck.org": {"score": 96, "publisher": "FactCheck.org", "category": "IFCN Fact Checker"},
    "snopes": {"score": 95, "publisher": "Snopes", "category": "Fact Checker"},
    "lead stories": {"score": 94, "publisher": "Lead Stories", "category": "IFCN Fact Checker"},
    "full fact": {"score": 95, "publisher": "Full Fact", "category": "IFCN Fact Checker"},

    # International Institutions & Scientific Bodies
    "world health organization": {"score": 99, "publisher": "World Health Organization (WHO)", "category": "International Institution"},
    "who": {"score": 99, "publisher": "World Health Organization (WHO)", "category": "International Institution"},
    "united nations": {"score": 99, "publisher": "United Nations (UN)", "category": "International Institution"},
    "un": {"score": 99, "publisher": "United Nations (UN)", "category": "International Institution"},
    "international energy agency": {"score": 98, "publisher": "International Energy Agency (IEA)", "category": "International Institution"},
    "iea": {"score": 98, "publisher": "International Energy Agency (IEA)", "category": "International Institution"},
    "centers for disease control": {"score": 98, "publisher": "Centers for Disease Control (CDC)", "category": "Government Institution"},
    "cdc": {"score": 98, "publisher": "Centers for Disease Control (CDC)", "category": "Government Institution"},
    "food and drug administration": {"score": 98, "publisher": "Food and Drug Administration (FDA)", "category": "Government Institution"},
    "fda": {"score": 98, "publisher": "Food and Drug Administration (FDA)", "category": "Government Institution"},
    "nasa": {"score": 99, "publisher": "NASA", "category": "Government Institution"},
    "nature": {"score": 98, "publisher": "Nature Journal", "category": "Scientific Journal"},
    "the lancet": {"score": 98, "publisher": "The Lancet", "category": "Medical Journal"},

    # Satire & Parody Outlets
    "the onion": {"score": 15, "publisher": "The Onion", "category": "Satire"},
    "the babylon bee": {"score": 15, "publisher": "The Babylon Bee", "category": "Satire"},
    "babylon bee": {"score": 15, "publisher": "The Babylon Bee", "category": "Satire"},
}


class RAGRetrievalEngine:
    """
    Orchestrates claim-by-claim retrieval across wires, fact-checkers,
    government archives, and institutional databases.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 LucidRAG/2.0"
        })
        self.wikipedia_service = get_wikipedia_service()
        self.session_queried_sources: Dict[str, int] = {}

    def reset_session_audit(self):
        """Clears per-session queried sources audit log."""
        self.session_queried_sources = {}

    def _match_publisher_metadata(self, pub_name: str, url: str = "") -> Dict[str, Any]:
        """
        Resolves publisher string to normalized publisher name, trust score, and category.
        Ensures:
        - Real publisher attribution (never assigns UN or other bodies based on headline text)
        - Matches publisher name against registry using exact / word boundaries for short acronyms
        - Only checks domain if url is a direct authoritative portal (not a Google redirect)
        - Unknown publishers get neutral score 50
        """
        clean_pub = re.sub(r'https?://\S+', '', pub_name or "").strip(' "\'-')
        if not clean_pub or clean_pub.lower() in ("unknown", "news source", "news wire"):
            if url and not any(g in url.lower() for g in ["google.", "news.google", "googleusercontent"]):
                try:
                    netloc = urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
                    if netloc:
                        clean_pub = netloc
                except Exception:
                    pass
        if not clean_pub or clean_pub.lower() in ("unknown", "news source", "news wire"):
            clean_pub = "Independent News Source"

        pub_lower = clean_pub.lower().strip()

        # Check known publishers using exact / word-boundary matches on publisher name
        for key, meta in PUBLISHER_TRUST_REGISTRY.items():
            if len(key) <= 4:
                # Word boundary for short acronyms: \bun\b, \bwho\b, \biea\b, \bafp\b, \bap\b, \bbbc\b
                if re.search(rf'\b{re.escape(key)}\b', pub_lower):
                    return meta
            else:
                if key in pub_lower:
                    return meta

        # Only check domain if it is an authoritative direct non-google URL
        if url and not any(g in url.lower() for g in ["google.", "news.google"]):
            try:
                netloc = urllib.parse.urlparse(url).netloc.lower()
                if netloc.endswith(".gov") or ".gov." in netloc:
                    return {"score": 96, "publisher": clean_pub if clean_pub != "Independent News Source" else "Government Official Portal (.gov)", "category": "Government"}
                if netloc == "un.org" or netloc.endswith(".un.org"):
                    return {"score": 99, "publisher": "United Nations (UN)", "category": "International Institution"}
                if netloc == "who.int" or netloc.endswith(".who.int"):
                    return {"score": 99, "publisher": "World Health Organization (WHO)", "category": "International Institution"}
                if netloc == "iea.org" or netloc.endswith(".iea.org"):
                    return {"score": 98, "publisher": "International Energy Agency (IEA)", "category": "International Institution"}
            except Exception:
                pass

        # Unknown publishers get the neutral 50
        return {"score": 50, "publisher": clean_pub, "category": "News Media"}

    def _fetch_rss_wire_evidence(self, query: str, max_items: int = 6) -> List[Dict[str, Any]]:
        """Queries Google News RSS wire archives for recent published coverage."""
        clean_q = re.sub(r'[^\w\s.-]', '', query).strip('. ')
        if not clean_q or len(clean_q) < 4:
            return []

        # Trim subordinate clauses or cut cleanly at word boundary (avoid mid-word cuts)
        m = re.search(r'^(.*?)\s+(?:following|according to|amid|after|as|during)\b', clean_q, re.I)
        if m and len(m.group(1)) >= 25:
            search_text = m.group(1)
        elif len(clean_q) > 90:
            search_text = clean_q[:90].rsplit(' ', 1)[0]
        else:
            search_text = clean_q

        url = f"{GOOGLE_NEWS_RSS_URL}?q={urllib.parse.quote(search_text)}&hl=en-US&gl=US&ceid=US:en"
        results: List[Dict[str, Any]] = []

        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:max_items]:
                    raw_title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub_date_raw = item.findtext("pubDate", "").strip()
                    desc = item.findtext("description", "").strip()

                    # 1. Extract real publisher from <source> element or title suffix
                    pub_candidate = ""
                    source_el = item.find("source")
                    if source_el is not None and source_el.text and source_el.text.strip():
                        pub_candidate = source_el.text.strip()

                    clean_article_title = raw_title
                    if " - " in raw_title:
                        title_part, suffix_pub = raw_title.rsplit(" - ", 1)
                        if not pub_candidate:
                            pub_candidate = suffix_pub.strip()
                        clean_article_title = title_part.strip()

                    clean_article_title = clean_text_snippet(clean_article_title)
                    snippet = clean_text_snippet(desc)
                    if len(snippet) > 250:
                        snippet = snippet[:250] + "..."

                    # Parse publication date to YYYY-MM-DD
                    parsed_date = datetime.now().strftime("%Y-%m-%d")
                    if pub_date_raw:
                        try:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(pub_date_raw)
                            parsed_date = dt.strftime("%Y-%m-%d")
                        except Exception:
                            parsed_date = pub_date_raw[:10]

                    # Requirement 8: Resolve Google News redirect link to publisher's real article URL and domain
                    resolved_url, pub_domain = resolve_google_news_url(link)
                    source_url = source_el.get("url", "") if source_el is not None else ""
                    if not pub_domain and source_url:
                        try:
                            pub_domain = urllib.parse.urlparse(source_url).netloc.lower().replace("www.", "")
                        except Exception:
                            pass

                    meta = self._match_publisher_metadata(pub_candidate, resolved_url or link)
                    if pub_domain and not meta.get("domain"):
                        meta["domain"] = pub_domain

                    final_url = resolved_url or link
                    results.append({
                        "articleTitle": clean_article_title,
                        "publisher": meta["publisher"],
                        "publisherDomain": pub_domain or meta.get("domain", ""),
                        "publicationDate": parsed_date,
                        "url": final_url,
                        "evidenceSnippet": snippet or clean_article_title,
                        "sourceTrustScore": meta["score"],
                        "sourceCategory": meta["category"],
                        "retrievalChannel": "Wire News Search"
                    })
        except Exception as e:
            logger.debug(f"RSS wire retrieval error: {e}")

        return results

    def _fetch_factcheck_evidence(self, claim_text: str) -> List[Dict[str, Any]]:
        """Queries Google Fact Check API & accredited IFCN signatories."""
        fc_records = factcheck_service.query_google_factcheck(claim_text)
        results: List[Dict[str, Any]] = []

        for fc in fc_records:
            reviewer = fc.get("reviewer", "Fact Check Organization")
            meta = self._match_publisher_metadata(reviewer + " " + fc.get("url", ""))

            rating = fc.get("rating", "Unrated")
            claimant = fc.get("claimant", "")
            claim_desc = fc.get("claim", "")

            snippet = f"Fact Check Verdict: {rating.upper()}. Reviewed claim: \"{claim_desc}\""
            if claimant:
                snippet += f" (attributed to {claimant})"

            results.append({
                "articleTitle": f"{reviewer} Fact Check: {claim_desc[:80]}",
                "publisher": meta["publisher"],
                "publicationDate": fc.get("review_date", datetime.now().strftime("%Y-%m-%d")),
                "url": fc.get("url", ""),
                "evidenceSnippet": snippet,
                "sourceTrustScore": meta["score"],
                "sourceCategory": "IFCN Fact Checker",
                "factCheckRating": rating,
                "retrievalChannel": "Google Fact Check Tools API"
            })

        return results

    def _fetch_institutional_evidence_if_relevant(self, claim_text: str) -> List[Dict[str, Any]]:
        """Queries specialized institutional portals (UN, WHO, IEA, CDC) when topic is relevant."""
        lower = claim_text.lower()
        results: List[Dict[str, Any]] = []

        is_health = bool(re.search(r'\b(vaccine|covid|virus|pandemic|who|health|disease|treatment|fda|cdc)\b', lower))
        is_climate_energy = bool(re.search(r'\b(renewable|solar|wind|energy|iea|emissions|climate|fossil|capacity)\b', lower))
        is_un_geopolitical = bool(re.search(r'\b(un\b|united nations|security council|treaty|resolution|sanctions|icj|icc|genocide|ceasefire)\b', lower))

        if is_health:
            rss_health = self._fetch_rss_wire_evidence(f"{claim_text} site:who.int OR site:cdc.gov OR site:nih.gov", max_items=2)
            results.extend(rss_health)
        if is_climate_energy:
            rss_energy = self._fetch_rss_wire_evidence(f"{claim_text} site:iea.org OR site:un.org", max_items=2)
            results.extend(rss_energy)
        if is_un_geopolitical:
            rss_un = self._fetch_rss_wire_evidence(f"{claim_text} site:un.org", max_items=2)
            results.extend(rss_un)

        return results

    def retrieve_evidence_for_claim(
        self,
        claim_text: str,
        max_items: int = 12,
        wire_articles: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes multi-channel retrieval for an individual claim:
        1. Google Fact Check Tools API & IFCN signatories
        2. Google News targeted wire RSS (Reuters, AP, BBC, Al Jazeera, The Guardian, AFP)
        3. Relevant institutional databases (UN, WHO, IEA, government archives)
        4. Canonical wire records
        """
        if not claim_text or not claim_text.strip():
            return []

        all_candidates: List[Dict[str, Any]] = []
        seen_urls = set()

        def add_candidates(items):
            for item in items:
                u = item.get("url", "").strip()
                # Requirement 3: Only accept sources with a real article URL (not a homepage or a section page like /technology)
                if not is_real_article_url(u):
                    continue
                title = item.get("articleTitle", "").strip().lower()
                key = u.lower() or title
                if key and key not in seen_urls:
                    seen_urls.add(key)
                    all_candidates.append(item)

        # 0. Wikipedia Official API lookup (supporting/contradicting encyclopedic reference)
        try:
            wiki_items = self.wikipedia_service.get_wikipedia_evidence_for_claim(claim_text)
            if claim_refers_to_relative_or_associate(claim_text):
                filtered_wiki = []
                for w in wiki_items:
                    snip = str(w.get("evidenceSnippet", "")).lower()
                    if w.get("stance") == "contradict" and ("living person" in snip or "no date of death" in snip):
                        continue
                    filtered_wiki.append(w)
                wiki_items = filtered_wiki
            self.session_queried_sources["Wikipedia (Official REST API)"] = self.session_queried_sources.get("Wikipedia (Official REST API)", 0) + len(wiki_items)
            add_candidates(wiki_items)
        except Exception as e:
            evidence_tracker.mark_failure("Wikipedia", e)
            self.session_queried_sources["Wikipedia (Official REST API)"] = self.session_queried_sources.get("Wikipedia (Official REST API)", 0)
            logger.warning(f"Wikipedia retrieval error: {e}")

        # 1. Fact-check API query (only queried if API key configured)
        if os.getenv("FACTCHECK_API_KEY", "").strip():
            factchecks = self._fetch_factcheck_evidence(claim_text)
            self.session_queried_sources["Google Fact Check Tools API"] = self.session_queried_sources.get("Google Fact Check Tools API", 0) + len(factchecks)
            add_candidates(factchecks)

        # 2. Wire search with primary news wire priority
        if wire_articles is not None:
            converted = []
            for art in wire_articles:
                raw_pub = art.get("sourceName") or art.get("source") or art.get("publisher") or "News Source"
                if isinstance(raw_pub, dict):
                    pub = raw_pub.get("name", "News Source")
                else:
                    pub = str(raw_pub)
                
                title = clean_text_snippet(art.get("articleTitle") or art.get("title") or "")
                raw_snip = art.get("evidenceSnippet") or art.get("description") or art.get("content") or title
                snippet = clean_text_snippet(raw_snip) or title
                if len(snippet) > 250:
                    snippet = snippet[:250] + "..."
                date_str = art.get("publicationDate") or art.get("publishedAt") or art.get("date") or datetime.now().strftime("%Y-%m-%d")
                url = art.get("url") or ""
                meta = self._match_publisher_metadata(pub, url)
                
                converted.append({
                    "articleTitle": title,
                    "publisher": meta["publisher"],
                    "publicationDate": date_str,
                    "url": url,
                    "evidenceSnippet": snippet,
                    "sourceTrustScore": meta["score"] if meta["score"] != 80 else art.get("sourceTrustScore", 80),
                    "sourceCategory": meta["category"],
                    "retrievalChannel": "Wire News Search"
                })
            add_candidates(converted)
            self.session_queried_sources["Google News RSS Wire Search"] = self.session_queried_sources.get("Google News RSS Wire Search", 0) + len(converted)
        else:
            wire_query = f"{claim_text}"
            wire_results = self._fetch_rss_wire_evidence(wire_query, max_items=6)
            add_candidates(wire_results)

            # 3. Targeted query specifying wire publishers
            targeted_wire_q = f"{claim_text} (Reuters OR AP OR BBC OR \"Al Jazeera\" OR \"The Guardian\")"
            targeted_results = self._fetch_rss_wire_evidence(targeted_wire_q, max_items=4)
            add_candidates(targeted_results)
            self.session_queried_sources["Google News RSS Wire Search"] = self.session_queried_sources.get("Google News RSS Wire Search", 0) + len(wire_results) + len(targeted_results)

        # 4. Institutional archives (UN, WHO, etc.)
        lower = claim_text.lower()
        if any(k in lower for k in ["vaccine", "covid", "virus", "pandemic", "who", "health", "disease", "treatment", "fda", "cdc", "renewable", "solar", "wind", "energy", "iea", "emissions", "climate", "fossil", "capacity", "un", "united nations", "security council", "treaty", "resolution", "sanctions", "icj", "icc", "genocide", "ceasefire"]):
            inst_results = self._fetch_institutional_evidence_if_relevant(claim_text)
            self.session_queried_sources["Institutional Archives (UN/WHO/CDC)"] = self.session_queried_sources.get("Institutional Archives (UN/WHO/CDC)", 0) + len(inst_results)
            add_candidates(inst_results)


        logger.info(f"RAG Retrieval for claim '{claim_text[:60]}...': found {len(all_candidates)} raw candidate sources.")
        return all_candidates[:max_items]

    def get_sources_searched_list(self) -> List[str]:
        """Returns the formal list of sources actually queried with result counts for user transparency."""
        filtered_sources = dict(self.session_queried_sources)
        if not filtered_sources:
            return [
                "Google News RSS Wire Search (0 results)",
                "Wikipedia (Official REST API) (0 results)"
            ]
        return [
            f"{source} ({count} result{'s' if count != 1 else ''})"
            for source, count in filtered_sources.items()
        ]


_retriever = RAGRetrievalEngine()


def get_rag_retriever() -> RAGRetrievalEngine:
    return _retriever
