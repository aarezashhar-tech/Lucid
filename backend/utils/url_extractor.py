import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Tuple

logger = logging.getLogger("lucid.url_extractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [URLExtractor] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
]

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 10  # 10 second timeout required by Task A


def is_ssrf_blocked_ip(ip_str: str) -> bool:
    """Checks if an IP address is a private, loopback, or internal address."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return True
        for net in BLOCKED_IP_NETWORKS:
            if ip in net:
                return True
        return False
    except ValueError:
        return True


def validate_url_security(url: str) -> Tuple[bool, str, str]:
    """
    Validates the URL protocol and ensures target host does not resolve
    to internal or private network IP addresses (SSRF protection).
    Returns (is_valid, error_message, normalized_url).
    """
    if not url:
        return False, "Missing URL parameter.", ""

    url_clean = url.strip()
    if not re.match(r'^https?://', url_clean, re.IGNORECASE):
        url_clean = 'https://' + url_clean

    try:
        parsed = urlparse(url_clean)
        scheme = parsed.scheme.lower()
        if scheme not in ('http', 'https'):
            return False, "Only HTTP and HTTPS URLs are supported.", ""

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL hostname.", ""

        hostname_lower = hostname.lower()
        if hostname_lower in ('localhost', '127.0.0.1', '0.0.0.0', '::1') or hostname_lower.endswith('.local') or hostname_lower.endswith('.internal'):
            return False, "Access to private or internal network addresses is blocked.", ""

        # Resolve hostname to check all destination IPs
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                sockaddr = item[4]
                ip_candidate = sockaddr[0]
                if is_ssrf_blocked_ip(ip_candidate):
                    logger.warning(f"SSRF block triggered for host '{hostname}' resolving to '{ip_candidate}'")
                    return False, "Access to private or internal network addresses is blocked.", ""
        except socket.gaierror:
            return False, "Could not resolve hostname.", ""

        return True, "", url_clean

    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False, f"URL validation failed: {str(e)}", ""


def fetch_and_extract_article(url_input: str) -> Dict[str, Any]:
    """
    Fetches article content from a web URL, strictly enforcing:
    1. 10-second timeout.
    2. Browser-like User-Agent.
    3. SSRF prevention for localhost, private and link-local ranges.
    4. Text extraction of article title and main body.
    5. Clean error message if page cannot be read or has too little text.
    """
    is_valid, err_msg, url = validate_url_security(url_input)
    if not is_valid:
        return {
            "success": False,
            "error": err_msg,
            "url": url_input
        }

    domain = urlparse(url).netloc

    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        logger.info(f"Fetching URL: {url} (Timeout: {REQUEST_TIMEOUT}s)")
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        # Check final redirected URL for SSRF as well
        resp_history = getattr(resp, 'history', None)
        if resp_history and isinstance(resp_history, (list, tuple)):
            for r in resp_history:
                r_url = getattr(r, 'url', None)
                if r_url and isinstance(r_url, str):
                    r_valid, r_err, _ = validate_url_security(r_url)
                    if not r_valid:
                        return {"success": False, "error": r_err, "url": url}
        final_url = getattr(resp, 'url', None)
        if final_url and isinstance(final_url, str):
            final_valid, final_err, _ = validate_url_security(final_url)
            if not final_valid:
                return {"success": False, "error": final_err, "url": url}

        if resp.status_code >= 400:
            logger.warning(f"Failed to fetch {url} - HTTP status {resp.status_code}")
            return {
                "success": False,
                "error": "Could not read this page. Paste the article text instead.",
                "statusCode": resp.status_code
            }

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 1. Extract title (og:title, twitter:title, h1, or title tag)
        title = ""
        og_title = (
            soup.find('meta', property='og:title') or
            soup.find('meta', attrs={'name': 'og:title'}) or
            soup.find('meta', attrs={'name': 'twitter:title'})
        )
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 2. Decompose noise and boilerplate tags
        for tag in soup([
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'form', 'figure', 'figcaption', 'time', 'noscript', 'svg', 'iframe'
        ]):
            tag.decompose()

        # Decompose known social, byline, topic, and ad widgets
        for selector in [
            '.byline', '.author', '[data-component="byline-block"]',
            '.share-tools', '.social-share', '.article-share',
            '.tags', '.topics', '[data-testid="topics-list"]',
            '.related-topics', '.related-content', '.related-articles',
            '.advertisement', '.ad-slot', '.sponsored-content',
            '.cookie-banner', '.newsletter-signup', '.comments-container'
        ]:
            for el in soup.select(selector):
                el.decompose()

        # 3. Extract article body paragraphs
        article_el = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_=re.compile(r'article[-_]?body|story[-_]?body|story[-_]?content|entry[-_]?content', re.I)) or
            soup.body
        )

        paragraphs = []
        boilerplate_triggers = [
            'cookie', 'subscribe', 'sign up', 'all rights reserved',
            'terms of service', 'privacy policy', 'add as preferred',
            'image caption', 'photo caption', 'share this story',
            'follow us on', 'download our app'
        ]

        if article_el:
            for p in article_el.find_all('p'):
                p_text = p.get_text(strip=True)
                p_lower = p_text.lower()
                if len(p_text) > 25 and not any(bp in p_lower for bp in boilerplate_triggers):
                    paragraphs.append(p_text)

        body_text = "\n\n".join(paragraphs).strip()
        full_content = (title + "\n\n" + body_text).strip() if title else body_text

        # 4. Strict text sufficiency check (paywall, blocked, or JS-only check)
        # If fewer than 150 characters or fewer than 30 words, consider unreadable
        word_count = len(full_content.split())
        if len(full_content) < 150 or word_count < 30:
            logger.warning(f"URL extraction returned insufficient text ({len(full_content)} chars, {word_count} words)")
            return {
                "success": False,
                "error": "Could not read this page. Paste the article text instead.",
                "textLength": len(full_content),
                "wordCount": word_count
            }

        logger.info(f"Successfully extracted {len(full_content)} chars ({word_count} words) from {url}")
        return {
            "success": True,
            "title": title,
            "text": full_content,
            "domain": domain,
            "url": url,
            "wordCount": word_count,
            "charCount": len(full_content)
        }

    except requests.exceptions.Timeout:
        logger.warning(f"Request timeout fetching URL: {url}")
        return {
            "success": False,
            "error": "Could not read this page. Paste the article text instead.",
            "reason": "Request timed out after 10 seconds."
        }
    except Exception as e:
        logger.error(f"Error fetching article from {url}: {e}")
        return {
            "success": False,
            "error": "Could not read this page. Paste the article text instead.",
            "reason": str(e)
        }
