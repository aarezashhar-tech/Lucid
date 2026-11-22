"""
Layer 1: Source Reputation Engine
Evaluates source authority, domain trust scores, and publisher credibility.
"""
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse

class DomainOwnershipResult(dict):
    """
    Represents the verified domain ownership result.
    Supports dictionary access, attribute access, string conversion, and case-insensitive equality
    against publisher name, canonical organization name, or domain.
    """
    def __init__(self, publisher: str, name: str, domain: str, **kwargs):
        super().__init__(
            publisher=publisher,
            name=name,
            sourceName=name,
            domain=domain,
            **kwargs
        )
        self.publisher = publisher
        self.name = name
        self.sourceName = name
        self.domain = domain

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'DomainOwnershipResult' object has no attribute '{item}'")

    def __eq__(self, other):
        if isinstance(other, str):
            other_clean = other.strip().lower()
            return (
                self.publisher.lower() == other_clean
                or self.name.lower() == other_clean
                or self.domain.lower() == other_clean
            )
        return super().__eq__(other)

    def __str__(self):
        return self.publisher

    def __repr__(self):
        return f"<DomainOwnershipResult publisher='{self.publisher}' name='{self.name}' domain='{self.domain}'>"

# Canonical Domain Ownership Mapping
DOMAIN_OWNERSHIP_MAP: Dict[str, Dict[str, Any]] = {
    # BBC
    "bbc.com": {
        "publisher": "BBC",
        "name": "BBC",
        "domain": "bbc.com",
        "trust": 92,
        "tier": "High",
        "category": "Public Broadcaster",
        "description": "British Broadcasting Corporation - Established public service broadcaster with strict editorial guidelines and FactCheck desk."
    },
    "bbc.co.uk": {
        "publisher": "BBC",
        "name": "BBC",
        "domain": "bbc.co.uk",
        "trust": 92,
        "tier": "High",
        "category": "Public Broadcaster",
        "description": "British Broadcasting Corporation - Established UK public service broadcaster."
    },
    "bbc.in": {
        "publisher": "BBC",
        "name": "BBC",
        "domain": "bbc.in",
        "trust": 92,
        "tier": "High",
        "category": "Public Broadcaster",
        "description": "BBC India."
    },

    # Reuters
    "reuters.com": {
        "publisher": "Reuters",
        "name": "Reuters",
        "domain": "reuters.com",
        "trust": 95,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Top-tier international news organization known for rigorous attribution and impartiality."
    },

    # Al Jazeera
    "aljazeera.com": {
        "publisher": "Al Jazeera",
        "name": "Al Jazeera",
        "domain": "aljazeera.com",
        "trust": 85,
        "tier": "High",
        "category": "International News Network",
        "description": "Major international news organization with extensive Middle East and global reporting."
    },
    "aljazeera.net": {
        "publisher": "Al Jazeera",
        "name": "Al Jazeera",
        "domain": "aljazeera.net",
        "trust": 85,
        "tier": "High",
        "category": "International News Network",
        "description": "Al Jazeera Arabic Network."
    },

    # International Energy Agency
    "iea.org": {
        "publisher": "IEA",
        "name": "International Energy Agency",
        "domain": "iea.org",
        "trust": 98,
        "tier": "International Authority",
        "category": "Intergovernmental Energy Authority",
        "description": "Autonomous intergovernmental agency providing authoritative energy statistics and global market reports."
    },

    # Associated Press
    "apnews.com": {
        "publisher": "Associated Press",
        "name": "Associated Press (AP)",
        "domain": "apnews.com",
        "trust": 95,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Foundational primary news cooperative setting global standards for journalistic verification."
    },

    # World Health Organization
    "who.int": {
        "publisher": "WHO",
        "name": "World Health Organization (WHO)",
        "domain": "who.int",
        "trust": 100,
        "tier": "International Authority",
        "category": "Intergovernmental Health Agency",
        "description": "Primary international public health agency of the United Nations."
    },

    # United Nations
    "un.org": {
        "publisher": "United Nations",
        "name": "United Nations (UN)",
        "domain": "un.org",
        "trust": 100,
        "tier": "International Authority",
        "category": "Intergovernmental Global Organization",
        "description": "Official worldwide diplomatic body with vetted statistical and treaties archives."
    },

    # Major networks and publications
    "cnn.com": {
        "publisher": "CNN",
        "name": "CNN",
        "domain": "cnn.com",
        "trust": 88,
        "tier": "High",
        "category": "International News Network",
        "description": "Global 24-hour news network."
    },
    "theguardian.com": {
        "publisher": "The Guardian",
        "name": "The Guardian",
        "domain": "theguardian.com",
        "trust": 88,
        "tier": "High",
        "category": "Independent Investigative Press",
        "description": "British daily newspaper known for independent investigative reporting."
    },
    "bloomberg.com": {
        "publisher": "Bloomberg",
        "name": "Bloomberg News",
        "domain": "bloomberg.com",
        "trust": 93,
        "tier": "High",
        "category": "Financial News & Terminal Wire",
        "description": "Global business and financial news agency."
    },
    "nytimes.com": {
        "publisher": "The New York Times",
        "name": "The New York Times",
        "domain": "nytimes.com",
        "trust": 90,
        "tier": "High",
        "category": "Major Newspaper of Record",
        "description": "American daily newspaper of record."
    },
    "washingtonpost.com": {
        "publisher": "The Washington Post",
        "name": "The Washington Post",
        "domain": "washingtonpost.com",
        "trust": 88,
        "tier": "High",
        "category": "National Newspaper of Record",
        "description": "Major American daily newspaper."
    }
}

def validate_domain_ownership(url_or_domain: str) -> Optional[DomainOwnershipResult]:
    """
    Validates domain ownership and returns verified publisher identity.
    Enforces canonical mapping:
      bbc.com / bbc.co.uk -> BBC
      reuters.com -> Reuters
      aljazeera.com / aljazeera.net -> Al Jazeera
      iea.org -> International Energy Agency / IEA
    """
    if not url_or_domain or not isinstance(url_or_domain, str):
        return None

    clean = url_or_domain.strip().lower()
    # Strip protocol if present
    if clean.startswith(("http://", "https://")):
        try:
            parsed = urlparse(clean)
            clean = parsed.netloc.lower()
        except Exception:
            clean = re.sub(r'^https?://', '', clean)

    # Strip port
    if ":" in clean:
        clean = clean.split(":")[0]

    # Strip path or query if any remained
    clean = clean.split("/")[0].split("?")[0].split("#")[0]

    # Strip leading subdomains
    subdomain_prefixes = ("www.", "m.", "news.", "amp.", "edition.", "arabic.", "live.", "uk.", "us.")
    for prefix in subdomain_prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    # 1. Exact match in DOMAIN_OWNERSHIP_MAP
    if clean in DOMAIN_OWNERSHIP_MAP:
        data = DOMAIN_OWNERSHIP_MAP[clean]
        return DomainOwnershipResult(
            publisher=data["publisher"],
            name=data["name"],
            domain=data["domain"],
            trust=data["trust"],
            tier=data["tier"],
            category=data["category"],
            description=data["description"],
            isTrusted=data["trust"] >= 75,
            isPrimary=True
        )

    # 2. Suffix match in DOMAIN_OWNERSHIP_MAP
    for root_domain, data in DOMAIN_OWNERSHIP_MAP.items():
        if clean == root_domain or clean.endswith("." + root_domain):
            return DomainOwnershipResult(
                publisher=data["publisher"],
                name=data["name"],
                domain=data["domain"],
                trust=data["trust"],
                tier=data["tier"],
                category=data["category"],
                description=data["description"],
                isTrusted=data["trust"] >= 75,
                isPrimary=True
            )

    # 3. Check SOURCE_DATABASE for matches
    for key, data in SOURCE_DATABASE.items():
        src_domain = data.get("domain", "").lower()
        if src_domain and (clean == src_domain or clean.endswith("." + src_domain)):
            pub = data.get("publisher") or data.get("name", key.title())
            return DomainOwnershipResult(
                publisher=pub,
                name=data["name"],
                domain=src_domain,
                trust=data["trust"],
                tier=data["tier"],
                category=data["category"],
                description=data["description"],
                isTrusted=data["trust"] >= 75,
                isPrimary=True
            )

    return None

# Source reputation database with domain trust scores (0-100)
SOURCE_DATABASE: Dict[str, Dict[str, Any]] = {
    # International & Health Authorities (100)
    "who": {
        "name": "World Health Organization (WHO)",
        "domain": "who.int",
        "trust": 100,
        "tier": "International Authority",
        "category": "Intergovernmental Health Agency",
        "description": "Primary international public health agency of the United Nations."
    },
    "un": {
        "name": "United Nations (UN)",
        "domain": "un.org",
        "trust": 100,
        "tier": "International Authority",
        "category": "Intergovernmental Global Organization",
        "description": "Official worldwide diplomatic body with vetted statistical and treaties archives."
    },

    # Major Primary Global News Wires (95)
    "reuters": {
        "name": "Reuters",
        "domain": "reuters.com",
        "trust": 95,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Top-tier international news organization known for rigorous attribution and impartiality."
    },
    "associated press": {
        "name": "Associated Press (AP)",
        "domain": "apnews.com",
        "trust": 95,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Foundational primary news cooperative setting global standards for journalistic verification."
    },
    "ap": {
        "name": "Associated Press (AP)",
        "domain": "apnews.com",
        "trust": 95,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Foundational primary news cooperative setting global standards for journalistic verification."
    },
    "ap news": {
        "name": "Associated Press (AP)",
        "domain": "apnews.com",
        "trust": 95,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Foundational primary news cooperative setting global standards for journalistic verification."
    },
    "afp": {
        "name": "Agence France-Presse (AFP)",
        "domain": "afp.com",
        "trust": 94,
        "tier": "High",
        "category": "Global News Wire",
        "description": "Major international news agency with worldwide verification and bureau network."
    },

    # Broadcasters & Major Publishers (85 - 92)
    "bbc": {
        "publisher": "BBC",
        "name": "BBC News",
        "domain": "bbc.com",
        "trust": 92,
        "tier": "High",
        "category": "Public Broadcaster",
        "description": "Established public service broadcaster with strict editorial guidelines and FactCheck desk."
    },
    "bbc news": {
        "publisher": "BBC",
        "name": "BBC News",
        "domain": "bbc.com",
        "trust": 92,
        "tier": "High",
        "category": "Public Broadcaster",
        "description": "Established public service broadcaster with strict editorial guidelines and FactCheck desk."
    },
    "al jazeera": {
        "publisher": "Al Jazeera",
        "name": "Al Jazeera",
        "domain": "aljazeera.com",
        "trust": 85,
        "tier": "High",
        "category": "International News Network",
        "description": "Major international news organization with extensive Middle East and global reporting."
    },
    "aljazeera": {
        "publisher": "Al Jazeera",
        "name": "Al Jazeera",
        "domain": "aljazeera.com",
        "trust": 85,
        "tier": "High",
        "category": "International News Network",
        "description": "Major international news organization with extensive Middle East and global reporting."
    },
    "cnn": {
        "publisher": "CNN",
        "name": "CNN",
        "domain": "cnn.com",
        "trust": 88,
        "tier": "High",
        "category": "International News Network",
        "description": "Global 24-hour news network with extensive international correspondent bureaus and verification desks."
    },
    "iea": {
        "publisher": "IEA",
        "name": "International Energy Agency (IEA)",
        "domain": "iea.org",
        "trust": 98,
        "tier": "International Authority",
        "category": "Intergovernmental Energy Authority",
        "description": "Autonomous intergovernmental agency providing authoritative energy statistics and global market reports."
    },
    "international energy agency": {
        "publisher": "IEA",
        "name": "International Energy Agency (IEA)",
        "domain": "iea.org",
        "trust": 98,
        "tier": "International Authority",
        "category": "Intergovernmental Energy Authority",
        "description": "Autonomous intergovernmental agency providing authoritative energy statistics and global market reports."
    },
    "the hindu": {
        "name": "The Hindu",
        "domain": "thehindu.com",
        "trust": 85,
        "tier": "High",
        "category": "National Newspaper of Record",
        "description": "Premier Indian English-language daily respected for deep reporting and fact-checking."
    },
    "the guardian": {
        "name": "The Guardian",
        "domain": "theguardian.com",
        "trust": 90,
        "tier": "High",
        "category": "Independent Investigative Press",
        "description": "Global newspaper known for investigative reporting, environmental coverage, and fact-checking."
    },
    "new york times": {
        "name": "The New York Times",
        "domain": "nytimes.com",
        "trust": 92,
        "tier": "High",
        "category": "Major Newspaper of Record",
        "description": "Leading global daily newspaper with multiple Pulitzer-winning investigations."
    },
    "wall street journal": {
        "name": "The Wall Street Journal",
        "domain": "wsj.com",
        "trust": 91,
        "tier": "High",
        "category": "Financial News & Analysis",
        "description": "Authoritative financial and national reporting with rigorous corporate fact-checking."
    },
    "washington post": {
        "name": "The Washington Post",
        "domain": "washingtonpost.com",
        "trust": 90,
        "tier": "High",
        "category": "National Daily Newspaper",
        "description": "Major investigative news organization running the renowned Fact Checker database."
    },
    "dw": {
        "name": "Deutsche Welle (DW)",
        "domain": "dw.com",
        "trust": 90,
        "tier": "High",
        "category": "International Public Broadcaster",
        "description": "German public broadcaster with dedicated multilingual fact-checking team."
    },
    "npr": {
        "name": "National Public Radio (NPR)",
        "domain": "npr.org",
        "trust": 91,
        "tier": "High",
        "category": "Public Broadcaster",
        "description": "Non-profit public radio service recognized for transparent source attribution."
    },
    "bloomberg": {
        "name": "Bloomberg News",
        "domain": "bloomberg.com",
        "trust": 93,
        "tier": "High",
        "category": "Financial & Market Wire",
        "description": "High-speed financial and geopolitical news with verified market data feeds."
    },
    "financial times": {
        "name": "Financial Times",
        "domain": "ft.com",
        "trust": 93,
        "tier": "High",
        "category": "Global Business Daily",
        "description": "International daily newspaper with high reputation for accuracy and economic reporting."
    },
    "ft": {
        "name": "Financial Times",
        "domain": "ft.com",
        "trust": 93,
        "tier": "High",
        "category": "Global Business Daily",
        "description": "International daily newspaper with high reputation for accuracy and economic reporting."
    },
    "sky news": {
        "name": "Sky News",
        "domain": "sky.com",
        "trust": 88,
        "tier": "High",
        "category": "International News Broadcaster",
        "description": "British 24-hour television news network and online publication."
    },
    "abc news": {
        "name": "ABC News",
        "domain": "abcnews.go.com",
        "trust": 88,
        "tier": "High",
        "category": "Broadcast News Network",
        "description": "Major American commercial broadcast television news division."
    },
    "cbs news": {
        "name": "CBS News",
        "domain": "cbsnews.com",
        "trust": 88,
        "tier": "High",
        "category": "Broadcast News Network",
        "description": "American commercial broadcast television and radio network."
    },
    "nbc news": {
        "name": "NBC News",
        "domain": "nbcnews.com",
        "trust": 88,
        "tier": "High",
        "category": "Broadcast News Network",
        "description": "Major American broadcast news organization with verified national reporting."
    },
    "cnbc": {
        "name": "CNBC",
        "domain": "cnbc.com",
        "trust": 88,
        "tier": "High",
        "category": "Business & Financial News",
        "description": "Global business and financial market coverage network."
    },
    "politico": {
        "name": "Politico",
        "domain": "politico.com",
        "trust": 88,
        "tier": "High",
        "category": "Political Journalism",
        "description": "Nonpartisan political journalism company focusing on politics and policy."
    },
    "axios": {
        "name": "Axios",
        "domain": "axios.com",
        "trust": 88,
        "tier": "High",
        "category": "Digital News Publisher",
        "description": "American news website known for brevity and transparent sourcing."
    },
    "time": {
        "name": "Time Magazine",
        "domain": "time.com",
        "trust": 88,
        "tier": "High",
        "category": "National Weekly Magazine",
        "description": "Historic American news magazine and digital news publication."
    },
    "forbes": {
        "name": "Forbes",
        "domain": "forbes.com",
        "trust": 85,
        "tier": "High",
        "category": "Business & Technology Magazine",
        "description": "American business magazine featuring finance, industry, and leadership."
    },
    "usa today": {
        "name": "USA Today",
        "domain": "usatoday.com",
        "trust": 86,
        "tier": "High",
        "category": "National Newspaper",
        "description": "American daily newspaper with verified national coverage."
    },
    "times of india": {
        "name": "The Times of India",
        "domain": "timesofindia.indiatimes.com",
        "trust": 80,
        "tier": "High",
        "category": "National Daily Newspaper",
        "description": "Indian English-language daily newspaper with nationwide reporting network."
    },
    "indian express": {
        "name": "The Indian Express",
        "domain": "indianexpress.com",
        "trust": 84,
        "tier": "High",
        "category": "National Investigative Daily",
        "description": "Indian English-language news daily renowned for investigative journalism."
    },
    "ndtv": {
        "name": "NDTV",
        "domain": "ndtv.com",
        "trust": 82,
        "tier": "High",
        "category": "National Broadcast News",
        "description": "Pioneer Indian commercial television network with active fact-checking division."
    },

    # Scientific Journals & Registries (98)
    "nature": {
        "name": "Nature Journal",
        "domain": "nature.com",
        "trust": 98,
        "tier": "Scientific Consensus",
        "category": "Peer-Reviewed Scientific Journal",
        "description": "Premier multidisciplinary peer-reviewed scientific journal."
    },
    "science": {
        "name": "Science Magazine (AAAS)",
        "domain": "science.org",
        "trust": 98,
        "tier": "Scientific Consensus",
        "category": "Peer-Reviewed Scientific Journal",
        "description": "Authoritative peer-reviewed research and global scientific news."
    },
    "lancet": {
        "name": "The Lancet",
        "domain": "thelancet.com",
        "trust": 97,
        "tier": "Scientific Consensus",
        "category": "Medical Research Journal",
        "description": "World-leading peer-reviewed medical publication."
    },

    # Dedicated Fact-Checking Organizations (95)
    "snopes": {
        "name": "Snopes Fact Check",
        "domain": "snopes.com",
        "trust": 95,
        "tier": "Fact Checking Desk",
        "category": "Independent Fact Checker (IFCN)",
        "description": "Pioneer myth-busting and viral rumor debunking platform."
    },
    "politifact": {
        "name": "PolitiFact",
        "domain": "politifact.com",
        "trust": 95,
        "tier": "Fact Checking Desk",
        "category": "Independent Fact Checker (IFCN)",
        "description": "Pulitzer Prize-winning political fact-checking organization."
    },
    "factcheck.org": {
        "name": "FactCheck.org",
        "domain": "factcheck.org",
        "trust": 95,
        "tier": "Fact Checking Desk",
        "category": "Annenberg Public Policy Center",
        "description": "Nonpartisan consumer advocate for voters verifying political and viral claims."
    },

    # Low Credibility / Disinformation / Tabloids (< 40)
    "infowars": {
        "name": "InfoWars",
        "domain": "infowars.com",
        "trust": 10,
        "tier": "Low",
        "category": "Flagged Disinformation Platform",
        "description": "Repeatedly flagged for promoting unproven conspiracy theories and false claims."
    },
    "natural news": {
        "name": "Natural News",
        "domain": "naturalnews.com",
        "trust": 12,
        "tier": "Low",
        "category": "Health Misinformation / Conspiracy",
        "description": "Documented history of promoting medical falsehoods and debunked hoaxes."
    },
    "beforeitsnews": {
        "name": "Before It's News",
        "domain": "beforeitsnews.com",
        "trust": 15,
        "tier": "Low",
        "category": "Uncurated User Blog / Viral Hoaxes",
        "description": "Unchecked clickbait aggregation network frequently spreading unsubstantiated rumors."
    },
    "rt": {
        "name": "RT (Russia Today)",
        "domain": "rt.com",
        "trust": 35,
        "tier": "Low",
        "category": "State-Controlled Media",
        "description": "State-funded media outlet frequently flagged for state propaganda and selective omission."
    },
    "sputnik": {
        "name": "Sputnik News",
        "domain": "sputniknews.com",
        "trust": 30,
        "tier": "Low",
        "category": "State-Controlled Media",
        "description": "State-controlled international media network with low editorial independence."
    },
    "daily mail": {
        "name": "Daily Mail",
        "domain": "dailymail.co.uk",
        "trust": 55,
        "tier": "Medium-Low",
        "category": "Tabloid Press",
        "description": "High sensation tabloid reporting; unreliable for scientific and complex geopolitical consensus."
    },
    "the onion": {
        "name": "The Onion",
        "domain": "theonion.com",
        "trust": 10,
        "tier": "Satire",
        "category": "Satirical Publication",
        "description": "Intentionally fictional satirical publication not intended as factual news."
    },
    "babylon bee": {
        "name": "The Babylon Bee",
        "domain": "babylonbee.com",
        "trust": 10,
        "tier": "Satire",
        "category": "Satirical Publication",
        "description": "Intentionally satirical comedy outlet creating fictional parody commentary."
    }
}

class SourceReputationEngine:
    """
    Evaluates source credibility based on domain analysis, known news registries,
    government TLD rules, and automatic publisher attribution detection from article text.
    """

    def extract_domain(self, url_or_hint: str) -> str:
        """Extracts clean hostname domain from a URL or source hint."""
        if not url_or_hint:
            return ""
        
        clean = url_or_hint.strip()
        clean_lower = clean.lower()

        # Filenames (.jpg, .jpeg, .png, .pdf, etc.) must NEVER be treated as domains
        if re.search(r'\.(jpe?g|png|pdf|txt|gif|webp|bmp|tiff|svg)$', clean_lower):
            # Only allow if it has a real URL scheme with a path (e.g. https://bbc.com/pic.jpg)
            if not clean_lower.startswith(("http://", "https://")):
                return ""

        if not clean.startswith(("http://", "https://")) and "." in clean:
            clean = "https://" + clean
            
        try:
            parsed = urlparse(clean)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            # Strip port if present
            if ":" in domain:
                domain = domain.split(":")[0]
            # If domain itself ends in an image/document file extension, it was a filename, not a domain
            if re.search(r'\.(jpe?g|png|pdf|txt|gif|webp|bmp|tiff|svg)$', domain):
                return ""
            return domain
        except Exception:
            if re.search(r'\.(jpe?g|png|pdf|txt|gif|webp|bmp|tiff|svg)$', clean_lower):
                return ""
            return url_or_hint.lower().strip()

    def detect_url(self, url_or_domain: str = "", text: str = "") -> Optional[str]:
        """
        Extracts news URL if provided directly or embedded as an explicit URL header at the start of text.
        Never extracts arbitrary URLs from body text to prevent evidence contamination.
        Never treats image or document filenames as URLs.
        """
        if url_or_domain and url_or_domain.strip():
            raw = url_or_domain.strip()
            # Do not treat image/document filenames as URLs
            raw_lower = raw.lower()
            if re.search(r'\.(jpe?g|png|pdf|txt|gif|webp|bmp|tiff|svg)$', raw_lower):
                if not raw_lower.startswith(("http://", "https://")):
                    return None
            if "." in raw:
                # Also verify netloc does not end in file extension
                clean = raw if raw.startswith(("http://", "https://")) else "https://" + raw
                try:
                    netloc = urlparse(clean).netloc.lower()
                    if re.search(r'\.(jpe?g|png|pdf|txt|gif|webp|bmp|tiff|svg)$', netloc):
                        return None
                except Exception:
                    pass
                return raw

        if not text:
            return None

        # Inspect ONLY the first 3 lines of text for explicit article URL or URL header
        for line in text.strip().splitlines()[:3]:
            line = line.strip()
            if not line:
                continue
            # Match explicit URL line
            if re.match(r'^https?://[^\s<>"\'\)\]]+', line):
                return line.split()[0].rstrip('.,;:!?')
            if re.match(r'^www\.[^\s<>"\'\)\]]+', line):
                return "https://" + line.split()[0].rstrip('.,;:!?')
            # Match explicit URL metadata tag
            url_meta = re.match(r'^(?:url|link|article\s+url|source\s+url)\s*:\s*(https?://\S+)', line, re.I)
            if url_meta:
                return url_meta.group(1).rstrip('.,;:!?')
            # If line is regular article text/headline, do NOT treat body text as URL
            break

        return None

    def validate_domain_ownership(self, url_or_domain: str) -> Optional[DomainOwnershipResult]:
        """Validates domain ownership and returns verified publisher identity."""
        return validate_domain_ownership(url_or_domain)

    def _match_source_key(self, raw_str: str) -> Optional[str]:
        """Normalizes and matches raw string against SOURCE_DATABASE keys and aliases."""
        if not raw_str:
            return None
        cleaned = raw_str.lower().strip()
        cleaned = re.sub(r'^(?:the|an|a)\s+', '', cleaned)
        cleaned = re.sub(r'\s+(?:news|wire|agency|network|press|post|times|journal|desk)$', '', cleaned).strip()

        if cleaned in SOURCE_DATABASE:
            return cleaned

        lower_orig = raw_str.lower().strip()
        if lower_orig in SOURCE_DATABASE:
            return lower_orig

        # Match against database names or domains
        for key, entry in SOURCE_DATABASE.items():
            entry_name_clean = re.sub(r'^(?:the|an|a)\s+', '', entry["name"].lower())
            if cleaned == key or cleaned in key or key in cleaned:
                return key
            if cleaned == entry_name_clean or cleaned in entry_name_clean:
                return key
            domain_name = entry["domain"].split('.')[0]
            if domain_name == cleaned:
                return key

        return None

    def detect_metadata_header(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects explicit publisher metadata tags in text headers (e.g. 'Publisher: BBC', 'Source: Reuters').
        """
        if not text:
            return None

        # Check first 5 lines for explicit metadata
        header_text = "\n".join(text.strip().splitlines()[:5])

        # 1. Source / Publisher tags
        meta_match = re.search(
            r'^(?:source|publisher|published\s+by|credit|via|wire|agency)\s*:\s*(?:the\s+)?([A-Za-z0-9\s\.\-&]{2,35}?)(?:$|\n|\r|,|\.)',
            header_text,
            re.IGNORECASE | re.MULTILINE
        )
        if meta_match:
            raw_pub = meta_match.group(1).strip()
            key = self._match_source_key(raw_pub)
            if key and key in SOURCE_DATABASE:
                entry = SOURCE_DATABASE[key]
                return {
                    "key": key,
                    "entry": entry,
                    "matchedPattern": f"Publisher Metadata: {entry['name']}",
                    "rawSnippet": meta_match.group(0).strip(),
                    "attributionType": f"Publisher Metadata ('{meta_match.group(0).strip()}')",
                    "confidence": 97
                }

        # 2. Copyright metadata
        cr_match = re.search(
            r'(?:copyright|©)\s*(?:\(c\)|c)?\s*(?:\d{4})?\s*(?:the\s+)?([A-Za-z0-9\s\.\-&]{2,35}?)(?:$|\n|\r|\.|\s*all\s+rights)',
            header_text,
            re.IGNORECASE
        )
        if cr_match:
            raw_pub = cr_match.group(1).strip()
            key = self._match_source_key(raw_pub)
            if key and key in SOURCE_DATABASE:
                entry = SOURCE_DATABASE[key]
                return {
                    "key": key,
                    "entry": entry,
                    "matchedPattern": f"Copyright Notice: {entry['name']}",
                    "rawSnippet": cr_match.group(0).strip(),
                    "attributionType": f"Publisher Copyright Metadata ('{cr_match.group(0).strip()}')",
                    "confidence": 95
                }

        return None

    def detect_byline(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects journalistic bylines and lead wire datelines at the beginning of the article.
        Does NOT scan body text for quotations or citations.
        """
        if not text:
            return None

        header_text = "\n".join(text.strip().splitlines()[:4])

        # 1. Look for Bylines: e.g. "By John Smith, BBC News" or "Written by Jane Doe for Reuters"
        byline_match = re.search(
            r'\b(?:by|written\s+by)\s+([A-Za-z\s\.\-]+?)\s*(?:,\s*|\|\s*|\/\s*|for\s+)(?:the\s+)?([A-Za-z0-9\s\.\-&]{2,35}?)(?:$|\n|\r|\.|\s*[-—–])',
            header_text,
            re.IGNORECASE
        )
        if byline_match:
            raw_pub = byline_match.group(2).strip()
            key = self._match_source_key(raw_pub)
            if key and key in SOURCE_DATABASE:
                entry = SOURCE_DATABASE[key]
                return {
                    "key": key,
                    "entry": entry,
                    "matchedPattern": f"Byline: {byline_match.group(1).strip()}, {entry['name']}",
                    "rawSnippet": byline_match.group(0).strip(),
                    "attributionType": f"Journalistic Byline ('{byline_match.group(0).strip()}')",
                    "confidence": 96
                }

        # 2. Look for Datelines at the very start: e.g. "(Reuters) —", "WASHINGTON (AP) —", "DOHA (Al Jazeera) —"
        dateline_match = re.search(
            r'^(?:[A-Z\s,]{2,25}\s*)?\(([A-Za-z0-9\s\.\-&]+?)\)\s*[-—–]',
            text.strip()
        )
        if dateline_match:
            raw_pub = dateline_match.group(1).strip()
            key = self._match_source_key(raw_pub)
            if key and key in SOURCE_DATABASE:
                entry = SOURCE_DATABASE[key]
                return {
                    "key": key,
                    "entry": entry,
                    "matchedPattern": f"Dateline wire: ({entry['name']})",
                    "rawSnippet": dateline_match.group(0).strip(),
                    "attributionType": f"Wire Dispatch Dateline ('{dateline_match.group(0).strip()}')",
                    "confidence": 98
                }

        # 3. Leading wire header: "REUTERS -", "BBC -", "AL JAZEERA -"
        lead_wire_match = re.search(
            r'^(?:[A-Z\s,]{2,20}\s+)?\b(REUTERS|AP|AFP|BBC|AL JAZEERA|BLOOMBERG)\b\s*[-—–/]',
            text.strip(),
            re.IGNORECASE
        )
        if lead_wire_match:
            raw_pub = lead_wire_match.group(1).strip()
            key = self._match_source_key(raw_pub)
            if key and key in SOURCE_DATABASE:
                entry = SOURCE_DATABASE[key]
                return {
                    "key": key,
                    "entry": entry,
                    "matchedPattern": f"Wire Header: {entry['name']}",
                    "rawSnippet": lead_wire_match.group(0).strip(),
                    "attributionType": f"Primary News Wire Header ('{lead_wire_match.group(0).strip()}')",
                    "confidence": 98
                }

        return None

    def detect_copied_article_publisher(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects publisher metadata from copied articles in raw text submissions.
        Recognizes publisher signatures:
        - BBC article patterns:
            * BBC
            * BBC News
            * bbc.com
            * bbc.co.uk
            * BBC topic links (e.g. bbc.com/news/..., bbc.co.uk/news/..., bbc.in/...)
        - Al Jazeera article patterns:
            * Al Jazeera
            * aljazeera.com (or aljazeera.net)
        - Reuters article patterns:
            * Reuters
            * (Reuters)

        If publisher confidence > 90%, returns structured detection details.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
        header_text = "\n".join(lines[:6])

        # -------------------------------------------------------------
        # 1. BBC Article Patterns
        # -------------------------------------------------------------
        # BBC topic links or URLs anywhere in copied text
        bbc_link = re.search(r'\b(?:https?://)?(?:www\.)?bbc\.(?:com|co\.uk|in)(?:/[^\s]*)?', clean_text, re.IGNORECASE)
        if bbc_link:
            return {
                "publisher": "BBC",
                "name": "BBC",
                "domain": "bbc.com",
                "confidence": 98,
                "confidenceLevel": "High",
                "pattern": f"BBC URL / Topic Link ('{bbc_link.group(0)[:40]}')"
            }

        # Explicit BBC News branding
        if re.search(r'\bBBC\s+News\b', clean_text, re.IGNORECASE):
            return {
                "publisher": "BBC",
                "name": "BBC",
                "domain": "bbc.com",
                "confidence": 96,
                "confidenceLevel": "High",
                "pattern": "BBC News Branding"
            }

        # BBC Wire dateline / header in leading lines
        bbc_dateline = re.search(r'(?:\([A-Za-z\s,]*BBC[A-Za-z\s,]*\)|\bBBC\b\s*[-—–:])', header_text, re.IGNORECASE)
        if bbc_dateline:
            return {
                "publisher": "BBC",
                "name": "BBC",
                "domain": "bbc.com",
                "confidence": 97,
                "confidenceLevel": "High",
                "pattern": f"BBC Wire Dateline ('{bbc_dateline.group(0)}')"
            }

        # Standalone uppercase BBC in header/title or multiple occurrences
        if re.search(r'\bBBC\b', header_text) or len(re.findall(r'\bBBC\b', clean_text)) >= 2:
            return {
                "publisher": "BBC",
                "name": "BBC",
                "domain": "bbc.com",
                "confidence": 94,
                "confidenceLevel": "High",
                "pattern": "BBC Publisher Identification"
            }

        # -------------------------------------------------------------
        # 2. Al Jazeera Article Patterns
        # -------------------------------------------------------------
        # aljazeera.com or aljazeera.net domain links
        aj_link = re.search(r'\b(?:https?://)?(?:www\.)?aljazeera\.(?:com|net)(?:/[^\s]*)?', clean_text, re.IGNORECASE)
        if aj_link:
            return {
                "publisher": "Al Jazeera",
                "name": "Al Jazeera",
                "domain": "aljazeera.com",
                "confidence": 98,
                "confidenceLevel": "High",
                "pattern": f"Al Jazeera Domain Link ('{aj_link.group(0)[:40]}')"
            }

        # Al Jazeera branding / dateline / byline
        if re.search(r'\bAl\s+Jazeera\b', clean_text, re.IGNORECASE):
            return {
                "publisher": "Al Jazeera",
                "name": "Al Jazeera",
                "domain": "aljazeera.com",
                "confidence": 96,
                "confidenceLevel": "High",
                "pattern": "Al Jazeera Publisher Marker"
            }

        # -------------------------------------------------------------
        # 3. Reuters Article Patterns
        # -------------------------------------------------------------
        # (Reuters) dateline e.g. "(Reuters)", "(Reuters) -", "LONDON (Reuters) —"
        reuters_dateline = re.search(r'\(\s*Reuters\s*\)', clean_text, re.IGNORECASE)
        if reuters_dateline:
            return {
                "publisher": "Reuters",
                "name": "Reuters",
                "domain": "reuters.com",
                "confidence": 98,
                "confidenceLevel": "High",
                "pattern": "Reuters Dateline '(Reuters)'"
            }

        # reuters.com domain link
        reuters_link = re.search(r'\b(?:https?://)?(?:www\.)?reuters\.com(?:/[^\s]*)?', clean_text, re.IGNORECASE)
        if reuters_link:
            return {
                "publisher": "Reuters",
                "name": "Reuters",
                "domain": "reuters.com",
                "confidence": 98,
                "confidenceLevel": "High",
                "pattern": f"Reuters Domain Link ('{reuters_link.group(0)[:40]}')"
            }

        # Reuters wire header: "Reuters -", "REUTERS —"
        reuters_header = re.search(r'^(?:[A-Z\s,]{2,20}\s+)?\bReuters\b\s*[-—–:]', header_text, re.IGNORECASE | re.MULTILINE)
        if reuters_header:
            return {
                "publisher": "Reuters",
                "name": "Reuters",
                "domain": "reuters.com",
                "confidence": 96,
                "confidenceLevel": "High",
                "pattern": "Reuters Wire Header"
            }

        # Prominent Reuters in header line (check it is NOT merely an internal citation like 'According to Reuters')
        if re.search(r'\bReuters\b', header_text, re.IGNORECASE):
            is_just_body_citation = bool(re.search(
                r'\b(?:according\s+to|reported\s+by|cited\s+by|quoted\s+by|told\s+by)\s+reuters\b',
                header_text,
                re.IGNORECASE
            ))
            if not is_just_body_citation:
                return {
                    "publisher": "Reuters",
                    "name": "Reuters",
                    "domain": "reuters.com",
                    "confidence": 95,
                    "confidenceLevel": "High",
                    "pattern": "Reuters Publisher Marker"
                }

        return None

    def detect_publisher_attribution(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Backwards-compatible attribution helper. Strictly restricts attribution
        to publisher metadata and journalistic bylines. Body-text citations are ignored.
        """
        meta = self.detect_metadata_header(text)
        if meta:
            return meta
        return self.detect_byline(text)

    def evaluate_source(
        self,
        url_or_domain: str = "",
        text: str = "",
        source_hint: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_image: bool = False,
        file_type: str = "",
        image_publisher: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates source credibility based ONLY on:
        1. URL domain (via domain ownership validation)
        2. Publisher metadata (explicit tags or metadata dict)
        3. Article byline (byline lines and lead datelines)
        4. Image watermark / logo / publisher detection for image uploads

        Never treats .jpg, .jpeg, .png, .pdf filenames as domains.
        External corroboration sources in body text must NEVER replace primary source.
        """
        # Determine upload type
        if not is_image:
            is_image = (
                file_type in ("image", "img", "photo") or
                bool(re.search(r'\.(jpe?g|png|gif|webp|bmp)$', str(source_hint).lower().strip())) or
                bool(re.search(r'\.(jpe?g|png|gif|webp|bmp)$', str(url_or_domain).lower().strip()))
            )
        is_pdf = (
            file_type == "pdf" or
            bool(re.search(r'\.pdf$', str(source_hint).lower().strip())) or
            bool(re.search(r'\.pdf$', str(url_or_domain).lower().strip()))
        )

        # -------------------------------------------------------------
        # TIER 0: Image Watermark / Logo / Publisher Detection
        # -------------------------------------------------------------
        if is_image:
            det_pub = ""
            det_method = ""
            if image_publisher and image_publisher.get("detected"):
                det_pub = image_publisher.get("publisher", "")
                det_method = image_publisher.get("method", "Image Watermark/Logo Detection")
            elif metadata and isinstance(metadata, dict) and metadata.get("detectedPublisher"):
                det_pub = metadata.get("detectedPublisher", "")
                det_method = "Image Watermark/Logo Detection"

            if det_pub and det_pub != "Unknown":
                key = self._match_source_key(det_pub)
                entry = SOURCE_DATABASE.get(key) if key else None
                pub_domain = DOMAIN_OWNERSHIP_MAP.get(det_pub.lower(), {}).get("domain") or (entry.get("domain") if entry else "N/A")
                canonical_name = entry["name"] if entry else det_pub
                trust_score = entry["trust"] if entry else 92
                tier = entry["tier"] if entry else "High"
                cat = entry["category"] if entry else "Verified News Organization"

                return {
                    "publisher": canonical_name,
                    "sourceName": canonical_name,
                    "name": canonical_name,
                    "domain": pub_domain,
                    "sourceType": "Uploaded Image",
                    "trustScore": trust_score,
                    "tier": tier,
                    "publisherConfidence": "High",
                    "confidence": 96,
                    "category": cat,
                    "description": f"Verified publisher identified via image watermark, logo, or stamp ({canonical_name}).",
                    "attributionType": f"Detected Watermark / Logo ({canonical_name})",
                    "isTrusted": True,
                    "isPrimary": True,
                    "detectedFromImage": True
                }

        # -------------------------------------------------------------
        # TIER 1: URL Domain Ownership Validation
        # -------------------------------------------------------------
        detected_url = self.detect_url(url_or_domain=url_or_domain, text=text)
        domain = self.extract_domain(detected_url) if detected_url else (self.extract_domain(source_hint) if "." in source_hint else "")

        if domain:
            ownership = validate_domain_ownership(domain)
            if ownership:
                return {
                    "publisher": ownership.publisher,
                    "sourceName": ownership.name,
                    "name": ownership.name,
                    "domain": ownership.domain,
                    "trustScore": ownership.get("trust", 90),
                    "tier": ownership.get("tier", "High"),
                    "category": ownership.get("category", "Verified News Organization"),
                    "description": f"{ownership.get('description', '')} (Verified via Domain Ownership: {ownership.domain} → {ownership.publisher}).",
                    "attributionType": f"Domain Ownership Validation ({ownership.domain} → {ownership.publisher})",
                    "isTrusted": ownership.get("isTrusted", True),
                    "isPrimary": True
                }

            # Check government domain (.gov, .mil)
            if re.search(r"\.gov(\.[a-z]{2})?$", domain) or domain.endswith(".gov") or domain.endswith(".mil"):
                return {
                    "publisher": f"Government Official ({domain})",
                    "sourceName": f"Government Official Domain ({domain})",
                    "name": f"Government Official Domain ({domain})",
                    "domain": domain,
                    "trustScore": 95,
                    "tier": "High",
                    "category": "Official Government Portal (.gov)",
                    "description": "Verified official government authority subject to legal and statutory reporting accountability.",
                    "attributionType": "Domain TLD Verification (.gov)",
                    "isTrusted": True,
                    "isPrimary": True
                }

            # Check higher education / academic
            if domain.endswith(".edu") or domain.endswith(".ac.uk"):
                return {
                    "publisher": f"Academic Institution ({domain})",
                    "sourceName": f"Academic Institution ({domain})",
                    "name": f"Academic Institution ({domain})",
                    "domain": domain,
                    "trustScore": 90,
                    "tier": "High",
                    "category": "Higher Education & Research",
                    "description": "Accredited educational research domain with institutional peer-reviewed publishing standards.",
                    "attributionType": f"Actual Domain TLD Verification ({domain})",
                    "isTrusted": True,
                    "isPrimary": True
                }

            # Check non-profit (.org)
            if domain.endswith(".org"):
                return {
                    "publisher": f"Non-Profit Organization ({domain})",
                    "sourceName": f"Non-Profit Organization ({domain})",
                    "name": f"Non-Profit Organization ({domain})",
                    "domain": domain,
                    "trustScore": 75,
                    "tier": "Medium",
                    "category": "Organization Domain (.org)",
                    "description": "Non-profit organizational web entity; evaluated on publisher transparency.",
                    "attributionType": f"Actual Domain Analysis ({domain})",
                    "isTrusted": True,
                    "isPrimary": True
                }

            # General web domain evaluation (Neutral value for unlisted domains)
            return {
                "publisher": domain.split(".")[0].upper(),
                "sourceName": f"Web Publisher ({domain})",
                "name": f"Web Publisher ({domain})",
                "domain": domain,
                "trustScore": 50,
                "tier": "Neutral",
                "category": "Unlisted Web Domain",
                "description": f"Source reputation evaluated from web domain: {domain}. Neutral baseline assigned for unlisted domain.",
                "attributionType": f"Domain Baseline Evaluation ({domain})",
                "isTrusted": False,
                "isPrimary": True
            }

        # -------------------------------------------------------------
        # TIER 2: Publisher Metadata
        # -------------------------------------------------------------
        # Check explicit metadata dictionary
        if metadata and isinstance(metadata, dict):
            pub_meta = metadata.get("publisher") or metadata.get("source") or metadata.get("byline")
            if pub_meta:
                key = self._match_source_key(str(pub_meta))
                if key and key in SOURCE_DATABASE:
                    entry = SOURCE_DATABASE[key]
                    pub_name = DOMAIN_OWNERSHIP_MAP.get(entry.get("domain", ""), {}).get("publisher") or entry.get("publisher", entry["name"])
                    return {
                        "publisher": pub_name,
                        "sourceName": entry["name"],
                        "name": entry["name"],
                        "domain": entry["domain"],
                        "trustScore": entry["trust"],
                        "tier": entry["tier"],
                        "category": entry["category"],
                        "description": f"{entry['description']} (Verified from publisher metadata: {pub_meta}).",
                        "attributionType": f"Publisher Metadata ('{pub_meta}')",
                        "isTrusted": entry["trust"] >= 75,
                        "isPrimary": True
                    }

        # Check explicit metadata in text (Publisher: ..., Source: ..., Copyright ...)
        meta_info = self.detect_metadata_header(text)
        if meta_info:
            entry = meta_info["entry"]
            pub_name = DOMAIN_OWNERSHIP_MAP.get(entry.get("domain", ""), {}).get("publisher") or entry.get("publisher", entry["name"])
            return {
                "publisher": pub_name,
                "sourceName": entry["name"],
                "name": entry["name"],
                "domain": entry["domain"],
                "trustScore": entry["trust"],
                "tier": entry["tier"],
                "category": entry["category"],
                "description": f"{entry['description']} (Verified via {meta_info['matchedPattern']}).",
                "attributionType": meta_info["attributionType"],
                "isTrusted": entry["trust"] >= 75,
                "isPrimary": True
            }

        # Check source_hint if provided and not domain
        if source_hint and not "." in source_hint:
            key = self._match_source_key(source_hint)
            if key and key in SOURCE_DATABASE:
                entry = SOURCE_DATABASE[key]
                pub_name = DOMAIN_OWNERSHIP_MAP.get(entry.get("domain", ""), {}).get("publisher") or entry.get("publisher", entry["name"])
                return {
                    "publisher": pub_name,
                    "sourceName": entry["name"],
                    "name": entry["name"],
                    "domain": entry["domain"],
                    "trustScore": entry["trust"],
                    "tier": entry["tier"],
                    "category": entry["category"],
                    "description": entry["description"],
                    "attributionType": f"Publisher Metadata Hint ('{source_hint}')",
                    "isTrusted": entry["trust"] >= 75,
                    "isPrimary": True
                }

        # -------------------------------------------------------------
        # TIER 3: Article Byline
        # -------------------------------------------------------------
        byline_info = self.detect_byline(text)
        if byline_info:
            entry = byline_info["entry"]
            pub_name = DOMAIN_OWNERSHIP_MAP.get(entry.get("domain", ""), {}).get("publisher") or entry.get("publisher", entry["name"])
            return {
                "publisher": pub_name,
                "sourceName": f"Possible Publisher: {pub_name}",
                "name": f"Possible Publisher: {pub_name}",
                "domain": entry["domain"],
                "trustScore": entry["trust"],
                "tier": entry["tier"],
                "publisherConfidence": "High",
                "confidenceLevel": "High",
                "confidence": byline_info.get("confidence", 96),
                "category": entry["category"],
                "description": f"{entry['description']} (Verified via {byline_info['matchedPattern']}).",
                "attributionType": f"Possible Publisher: {pub_name} (Confidence: High)",
                "isTrusted": entry["trust"] >= 75,
                "isPrimary": True
            }

        # -------------------------------------------------------------
        # TIER 4: Copied Article Publisher Metadata Detection
        # -------------------------------------------------------------
        copied_match = self.detect_copied_article_publisher(text)
        if copied_match and copied_match.get("confidence", 0) > 90:
            pub_name = copied_match["publisher"]
            pub_domain = copied_match["domain"]
            entry = DOMAIN_OWNERSHIP_MAP.get(pub_domain) or SOURCE_DATABASE.get(pub_name.lower(), {})
            trust_score = entry.get("trust", 90)
            return {
                "publisher": pub_name,
                "sourceName": f"Possible Publisher: {pub_name}",
                "name": f"Possible Publisher: {pub_name}",
                "domain": pub_domain,
                "trustScore": trust_score,
                "tier": "High",
                "publisherConfidence": copied_match.get("confidenceLevel", "High"),
                "confidenceLevel": copied_match.get("confidenceLevel", "High"),
                "confidence": copied_match["confidence"],
                "category": entry.get("category", "Verified News Organization"),
                "description": f"Detected publisher metadata from copied article ({copied_match['pattern']}).",
                "attributionType": f"Possible Publisher: {pub_name} (Confidence: High)",
                "isTrusted": True,
                "isPrimary": True,
                "detectedFromCopiedArticle": True
            }

        # -------------------------------------------------------------
        # Fallback for Image Uploads: No publisher detected
        # -------------------------------------------------------------
        if is_image:
            return {
                "publisher": "Unknown",
                "sourceName": "Unknown",
                "name": "Unknown",
                "domain": "N/A",
                "sourceType": "Uploaded Image",
                "trustScore": 50,
                "tier": "Unverified",
                "category": "Uploaded Image",
                "description": "Uploaded image file content; no recognized publisher watermark, logo, or metadata detected.",
                "attributionType": "Uploaded Image (No Publisher Detected)",
                "isTrusted": False,
                "isPrimary": True
            }

        # -------------------------------------------------------------
        # Fallback for Document / PDF Uploads: No publisher detected
        # -------------------------------------------------------------
        if is_pdf:
            return {
                "publisher": "Unknown",
                "sourceName": "Unknown",
                "name": "Unknown",
                "domain": "N/A",
                "sourceType": "Uploaded Document",
                "trustScore": 50,
                "tier": "Unverified",
                "category": "Uploaded Document",
                "description": "Uploaded PDF document content; no recognized publisher metadata detected.",
                "attributionType": "Uploaded Document (No Publisher Detected)",
                "isTrusted": False,
                "isPrimary": True
            }

        # -------------------------------------------------------------
        # Fallback: Unspecified / Direct Submission
        # -------------------------------------------------------------
        return {
            "publisher": "Unknown",
            "sourceName": "Unspecified / Self-Published Content",
            "name": "Unspecified / Self-Published Content",
            "domain": "direct-submission",
            "trustScore": 50,
            "tier": "Unverified",
            "category": "Unattributed Text Input",
            "description": "No recognized authoritative news wire, government domain, or verified publisher attributed.",
            "attributionType": "Missing Source Attribution",
            "isTrusted": False,
            "isPrimary": True
        }

source_engine = SourceReputationEngine()

def get_source_engine() -> SourceReputationEngine:
    return source_engine
