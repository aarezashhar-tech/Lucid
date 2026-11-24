import base64
import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse, urljoin
import requests
from requests.adapters import HTTPAdapter
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
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
]

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 10  # 10 second timeout required by Task A
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB maximum response size (SSRF & resource exhaustion protection)


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


def resolve_and_validate_url(url: str) -> Tuple[bool, str, str, str, str, int]:
    """
    Validates the URL protocol, resolves the hostname, and ensures every
    resolved address is public. Returns:
    (is_valid, error_message, normalized_url, validated_ip, hostname, port)
    """
    if not url:
        return False, "Missing URL parameter.", "", "", "", 0

    url_clean = url.strip()
    if not re.match(r'^https?://', url_clean, re.IGNORECASE):
        url_clean = 'https://' + url_clean

    try:
        parsed = urlparse(url_clean)
        scheme = parsed.scheme.lower()
        if scheme not in ('http', 'https'):
            return False, "Only HTTP and HTTPS URLs are supported.", "", "", "", 0

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL hostname.", "", "", "", 0

        port = parsed.port or (443 if scheme == 'https' else 80)
        hostname_lower = hostname.lower()
        if hostname_lower in ('localhost', '127.0.0.1', '0.0.0.0', '::1') or hostname_lower.endswith('.local') or hostname_lower.endswith('.internal'):
            return False, "Access to private or local network resources is forbidden.", "", "", "", port

        try:
            ip_obj = ipaddress.ip_address(hostname_lower)
            if is_ssrf_blocked_ip(str(ip_obj)):
                return False, "Access to private or internal IP addresses is forbidden.", "", "", "", port
            return True, "", url_clean, str(ip_obj), hostname_lower, port
        except ValueError:
            pass

        try:
            addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            logger.warning(f"DNS resolution failure for host {hostname}: {e}")
            return False, "Could not resolve hostname.", "", "", "", port

        resolved_ips = []
        for entry in addr_info:
            ip_addr = entry[4][0]
            if is_ssrf_blocked_ip(ip_addr):
                logger.warning(f"SSRF blocked: {hostname} resolved to private/restricted IP: {ip_addr}")
                return False, "Target host resolves to a private or restricted network address.", "", "", "", port
            if ip_addr not in resolved_ips:
                resolved_ips.append(ip_addr)

        if not resolved_ips:
            return False, "Could not resolve hostname.", "", "", "", port

        # Prefer IPv4 for maximum compatibility, while still supporting IPv6.
        resolved_ips.sort(key=lambda ip: 0 if ':' not in ip else 1)
        return True, "", url_clean, resolved_ips[0], hostname_lower, port

    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False, "Malformed or invalid URL.", "", "", "", 0


def validate_url_security(url: str) -> Tuple[bool, str, str]:
    """
    Backwards-compatible wrapper around resolve_and_validate_url().
    Returns (is_valid, error_message, normalized_url).
    """
    ok, err, normalized_url, _ip, _hostname, _port = resolve_and_validate_url(url)
    return ok, err, normalized_url


class SSRFBlockedError(Exception):
    """Raised when a URL or redirect resolves to a blocked/private address."""


class TooManyRedirectsError(Exception):
    """Raised when a URL exceeds the maximum number of manual redirects."""


class _PinnedHTTPAdapter(HTTPAdapter):
    """
    Requests adapter that connects to an already-validated IP address while
    preserving the original Host header and TLS SNI/certificate hostname.
    This prevents DNS rebinding between validation and connection.
    """

    def __init__(self, ip: str, hostname: str, scheme: str, **kwargs):
        self._pinned_ip = ip
        self._pinned_hostname = hostname
        self._pinned_scheme = scheme.lower()
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        if self._pinned_scheme == 'https':
            pool_kwargs.setdefault('assert_hostname', self._pinned_hostname)
            pool_kwargs.setdefault('server_hostname', self._pinned_hostname)
        return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)

    def send(self, request, **kwargs):
        parsed = urlparse(request.url)
        ip_host = f"[{self._pinned_ip}]" if ':' in self._pinned_ip else self._pinned_ip
        ip_netloc = ip_host
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port and port not in (80, 443):
            ip_netloc = f"{ip_host}:{port}"
        request.url = parsed._replace(netloc=ip_netloc).geturl()
        request.headers['Host'] = parsed.netloc
        return super().send(request, **kwargs)


def _fetch_once(url: str, headers: Dict[str, str], timeout: float, max_bytes: int) -> Dict[str, Any]:
    """
    Performs exactly one HTTP request with redirects disabled. The hostname is
    resolved and validated immediately before connecting, and the connection is
    pinned to that validated IP.
    """
    ok, err, normalized_url, validated_ip, hostname, _port = resolve_and_validate_url(url)
    if not ok:
        raise SSRFBlockedError(err)

    scheme = urlparse(normalized_url).scheme.lower()
    adapter = _PinnedHTTPAdapter(validated_ip, hostname, scheme)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.get(
            normalized_url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False
        )

        content_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            content_bytes.extend(chunk)
            if len(content_bytes) > max_bytes:
                logger.warning(f"Response size exceeded {max_bytes} bytes. Truncating.")
                break

        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': bytes(content_bytes),
            'encoding': response.encoding,
            'apparent_encoding': response.apparent_encoding,
            'url': normalized_url,
        }
    finally:
        session.close()


def safe_get(
    url: str,
    headers: Dict[str, str],
    timeout: float,
    max_redirects: int = 5,
    max_bytes: int = MAX_RESPONSE_BYTES
) -> Dict[str, Any]:
    """
    Follows redirects manually (max 5). Every hop is re-resolved, re-validated,
    and connected to its validated IP. Redirects are never followed automatically.
    """
    current_url = url
    for attempt in range(max_redirects + 1):
        result = _fetch_once(current_url, headers, timeout, max_bytes)
        status = result['status_code']
        location = result['headers'].get('Location') or result['headers'].get('location')

        if status in (301, 302, 303, 307, 308) and location:
            if attempt >= max_redirects:
                raise TooManyRedirectsError(f"Exceeded maximum of {max_redirects} redirects.")
            current_url = urljoin(current_url, location)
            logger.info(f"Following redirect ({attempt + 1}/{max_redirects}) to {current_url}")
            continue

        result['final_url'] = current_url
        result['redirect_count'] = attempt
        return result

    raise TooManyRedirectsError(f"Exceeded maximum of {max_redirects} redirects.")


def is_real_article_url(url: str) -> bool:
    """
    Validates whether a URL points to a specific article or document,
    rather than a generic website homepage, section index, hub page, or search query.
    Only sources with a real article URL are accepted for evidence verification.
    """
    if not url or not isinstance(url, str):
        return False

    url_clean = url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        return False

    try:
        parsed = urlparse(url_clean)
        netloc = (parsed.netloc or "").lower().replace("www.", "")
        path = (parsed.path or "").strip("/")
    except Exception:
        return False

    if not netloc:
        return False

    # Block API endpoints, search queries, and placeholder paths
    if "factchecktools.googleapis.com" in netloc or "googleapis.com" in netloc:
        return False
    if "google.com" in netloc and ("search" in path or not path or path == "search"):
        return False
    if url_clean.endswith("#") or url_clean.endswith("/#"):
        return False

    # A homepage with no path or only whitespace/slash is never an article URL
    if not path:
        return False

    segments = [s.lower() for s in path.split("/") if s]
    if not segments:
        return False

    # Generic top-level section and hub directories
    GENERIC_SECTIONS = {
        "news", "technology", "tech", "business", "politics", "world", "opinion",
        "opinions", "sports", "sport", "science", "health", "features", "hub",
        "topics", "topic", "category", "categories", "section", "sections", "tag",
        "tags", "video", "videos", "live", "about", "contact", "home", "index",
        "default", "middleeast", "middle-east", "us", "asia", "europe", "africa",
        "americas", "latin-america", "national", "international", "lifestyle"
    }

    # Case 1: Single segment path, e.g. /technology, /news, /middleeast
    if len(segments) == 1:
        seg = segments[0]
        if seg in GENERIC_SECTIONS:
            return False
        # If single segment is short and has no slug indicators (dashes, numbers, or file extension)
        if len(seg) < 20 and not re.search(r'[-_0-9.]', seg):
            return False

    # Case 2: Two-segment path that points to a hub or sub-section
    # e.g., /hub/israel-hamas-war, /news/world-middle-east, /category/technology, /section/politics
    if segments[0] in ("hub", "topics", "topic", "category", "categories", "section", "sections", "tag", "tags"):
        return False

    if segments[0] in ("news", "world", "business", "technology", "tech", "politics"):
        if len(segments) == 2 and segments[1] in (
            "world-middle-east", "middleeast", "middle-east", "us", "asia", "europe",
            "africa", "politics", "business", "technology", "tech", "national", "international"
        ):
            return False

    return True


def resolve_google_news_url(url: str, timeout: float = 2.5) -> Tuple[str, str]:
    """
    Attempts to resolve a Google News redirect URL to the publisher's real article URL and domain.
    Returns (resolved_url, publisher_domain).
    If resolution fails or URL is not a Google News redirect, returns (url, extracted_domain).
    """
    if not url or not isinstance(url, str):
        return "", ""

    url_clean = url.strip()
    try:
        parsed = urlparse(url_clean)
        netloc = (parsed.netloc or "").lower().replace("www.", "")
    except Exception:
        return url_clean, ""

    if "news.google.com" not in netloc:
        return url_clean, netloc

    # Method 1: Decode base64 / protobuf token if present in path
    # Google News article URLs: https://news.google.com/rss/articles/<TOKEN>?oc=5
    resolved_url = None
    try:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[-2] == "articles":
            token = parts[-1].split("?")[0]
            padded = token + "=" * (-len(token) % 4)
            decoded = base64.urlsafe_b64decode(padded)
            found_urls = re.findall(rb'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\x00-\x1f\x7f-\xff\s\"\'<>]*', decoded)
            for raw_u in found_urls:
                cand_u = raw_u.decode('utf-8', errors='ignore')
                cand_parsed = urlparse(cand_u)
                cand_netloc = (cand_parsed.netloc or "").lower().replace("www.", "")
                if cand_netloc and "google.com" not in cand_netloc and is_real_article_url(cand_u):
                    resolved_url = cand_u
                    break
    except Exception:
        pass

    if resolved_url:
        cand_domain = urlparse(resolved_url).netloc.lower().replace("www.", "")
        return resolved_url, cand_domain

    # Method 2: HTTP GET with manual, SSRF-validated redirect following
    try:
        result = safe_get(
            url_clean,
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=timeout,
            max_redirects=5,
            max_bytes=200 * 1024
        )
        final_url = result.get("final_url", url_clean)
        if final_url and "news.google.com" not in final_url and is_real_article_url(final_url):
            resolved_domain = urlparse(final_url).netloc.lower().replace("www.", "")
            return final_url, resolved_domain

        # Check HTML for meta refresh or anchor redirect
        content = result.get("content") or b""
        if content:
            soup = BeautifulSoup(content[:10000], "html.parser")
            meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile(r"^refresh$", re.I)})
            if meta_refresh and meta_refresh.get("content"):
                m = re.search(r'url=(https?://[^\s"\']+)', meta_refresh["content"], re.I)
                if m and "google.com" not in m.group(1) and is_real_article_url(m.group(1)):
                    ru = m.group(1)
                    return ru, urlparse(ru).netloc.lower().replace("www.", "")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith("http") and "google.com" not in href and is_real_article_url(href):
                    return href, urlparse(href).netloc.lower().replace("www.", "")
    except Exception:
        pass

    return url_clean, netloc


def fetch_and_extract_article(url_input: str) -> Dict[str, Any]:
    """
    Fetches article content from a web URL, strictly enforcing:
    1. 10-second timeout.
    2. Browser-like User-Agent.
    3. SSRF prevention for localhost, private and link-local ranges.
    4. Text sufficiency check (>= 150 chars, >= 30 words).
    5. Clean text parsing (removing script, style, nav, ads, boilerplate).
    """
    is_valid, err, validated_url = validate_url_security(url_input)
    if not is_valid:
        logger.warning(f"URL validation failed for {url_input}: {err}")
        return {
            "success": False,
            "error": err,
            "reason": "URL security validation failed (SSRF or malformed URL)."
        }

    url = validated_url
    domain = urlparse(url).netloc.lower().replace("www.", "")

    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        logger.info(f"Fetching URL with 10s timeout: {url}")
        result = safe_get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            max_redirects=5,
            max_bytes=MAX_RESPONSE_BYTES
        )

        final_url = result.get("final_url", url)
        content_bytes = result.get("content") or b""
        if not content_bytes:
            return {
                "success": False,
                "error": "Could not read this page. Paste the article text instead.",
                "reason": "Empty HTTP response received."
            }

        encoding = result.get("encoding") or result.get("apparent_encoding") or 'utf-8'
        html_content = content_bytes.decode(encoding, errors='replace')

        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Extract title
        title = ""
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove site name suffix from title if present
        if " | " in title:
            title = title.split(" | ")[0].strip()
        elif " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()

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

    except SSRFBlockedError as e:
        logger.warning(f"SSRF protection blocked request/redirect for {url}: {e}")
        return {
            "success": False,
            "error": "Redirected to forbidden or internal address.",
            "reason": "SSRF protection blocked request."
        }
    except TooManyRedirectsError as e:
        logger.warning(f"Too many redirects fetching {url}: {e}")
        return {
            "success": False,
            "error": "Could not read this page. Paste the article text instead.",
            "reason": str(e)
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
