import sys
import os
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from utils.url_extractor import fetch_and_extract_article
from services.article_classifier import get_article_classifier
from app import run_shared_verification_pipeline

urls = [
    ("BBC News Article 1", "https://www.bbc.co.uk/news/articles/cvj6l8y3x1x4o"),
    ("The Hindu Article", "https://www.thehindu.com/news/national/people-bought-less-gold-prices-fell-modi-recalls-year-old-appeal/article71731156.ece"),
    ("The Hindu Article 2", "https://www.thehindu.com/news/national/india-bulgaria-discuss-bilateral-ties-imec-black-sea-situation/article71729472.ece")
]

ac = get_article_classifier()

print("=" * 80)
print("LIVE NEWS URL vs PASTED TEXT PARITY EVALUATION (3 LIVE URLS)")
print("=" * 80)

for label, url in urls:
    print(f"\n--- {label} ---")
    print(f"URL: {url}")
    ext = fetch_and_extract_article(url)
    assert ext.get("success"), f"Extraction failed: {ext.get('error')}"
    text = ext.get("text", "")
    title = ext.get("title", "")
    char_len = len(text)
    word_len = len(text.split())
    print(f"Title: {title}")
    print(f"Extracted Length: {char_len} chars ({word_len} words)")
    print(f"First 300 chars:\n{text[:300]}")
    
    cls = ac.classify_article_type(text, title=title)
    print(f"Classification: {cls['articleType']} (Confidence: {cls['confidence']}%)")
    print(f"Markers/Indicators: {cls.get('indicators')}")
    
    # Run URL pipeline
    res_url = run_shared_verification_pipeline(
        text=text,
        url=url,
        title=title,
        input_type="url",
        source_hint=ext.get("domain", "")
    )
    
    # Run Pasted Text pipeline
    res_txt = run_shared_verification_pipeline(
        text=text,
        title=title,
        input_type="text",
        source_hint=ext.get("domain", "")
    )
    
    url_score = res_url["trustScore"]
    txt_score = res_txt["trustScore"]
    print(f"URL Input Result: Trust Score = {url_score}, Verdict = {res_url['verdict']}")
    print(f"Pasted Text Result: Trust Score = {txt_score}, Verdict = {res_txt['verdict']}")
    assert isinstance(url_score, (int, float)), f"Expected numeric score, got {url_score}"
    assert isinstance(txt_score, (int, float)), f"Expected numeric score, got {txt_score}"
    delta = abs(int(url_score) - int(txt_score))
    print(f"Parity Delta: {delta} points (Requirement: <= 5 points)")
    assert delta <= 5, f"Delta too high: {delta}"
    assert res_url["verdict"] == res_txt["verdict"], f"Verdict mismatch: {res_url['verdict']} vs {res_txt['verdict']}"
    print(f"STATUS: PASS (URL vs Text Parity Verified)")

print("\nAll 3 live news articles verified with numeric scores and delta <= 5!")
