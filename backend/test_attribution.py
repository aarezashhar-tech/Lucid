import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.source_reputation import get_source_engine

engine = get_source_engine()

test_cases = [
    ("According to Reuters, global renewable energy capacity increased by 50% in 2023.", ""),
    ("According to BBC, the peace delegates arrived in Geneva this morning.", ""),
    ("Al Jazeera reports that Saudi Arabia intercepted three missiles near the southern border.", ""),
    ("By Sarah Jones, Associated Press - New environmental guidelines were published today.", ""),
    ("Source: Bloomberg News\nTreasury yields fell sharply on Friday following job market data.", ""),
    ("LONDON (Reuters) — British inflation cooled to 2% in the latest official figures.", ""),
    ("Read the full investigation at https://www.theguardian.com/world/2026/nov/14/climate-report", ""),
    ("General breaking news story", "https://edition.cnn.com/2026/11/14/politics/index.html"),
    ("A random user blog post without any sources or links.", "")
]

for text, url in test_cases:
    res = engine.evaluate_source(url_or_domain=url, text=text)
    print(f"INPUT: {text[:55]} | URL: {url}")
    print(f"  -> SOURCE: {res['sourceName']} (Trust: {res['trustScore']}/100, Tier: {res['tier']})")
    print(f"  -> DOMAIN: {res['domain']} | TRUSTED: {res['isTrusted']}")
    print(f"  -> ATTR:   {res['attributionType']}")
    print("-" * 60)
