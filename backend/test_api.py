import io
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from services.ai_service import get_verification_engine
from services.factcheck_service import query_google_factcheck

def test_endpoints():
    client = app.test_client()
    engine = get_verification_engine()

    print("1. Testing GET /api/health...")
    resp = client.get('/api/health')
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    health_data = resp.get_json()
    print("Health response service:", health_data.get("service"))
    assert health_data["modelLoaded"] is True
    assert health_data["vectorizerLoaded"] is True
    assert "disclaimer" in health_data

    print("\n2. Testing Google Fact Check Tools API query_external_factcheck without API key...")
    empty_res = engine.query_external_factcheck("Test claim without key")
    assert isinstance(empty_res, list)
    assert empty_res == [], f"Expected [], got {empty_res}"
    print("Correctly returned [] when FACTCHECK_API_KEY is not set.")

    print("\n3. Testing Google Fact Check Tools API query with mocked API response...")
    mock_payload = {
        "claims": [
            {
                "text": "Secret 5G mind control microchips in drinking water",
                "claimant": "Social Media Posts",
                "claimDate": "2024-01-15T00:00:00Z",
                "claimReview": [
                    {
                        "publisher": {"name": "PolitiFact", "site": "politifact.com"},
                        "textualRating": "Pants on Fire",
                        "url": "https://www.politifact.com/factchecks/5g-water-hoax/",
                        "reviewDate": "2024-01-16T12:00:00Z"
                    }
                ]
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.dict(os.environ, {"FACTCHECK_API_KEY": "test-mock-key"}):
        with patch("requests.get", return_value=mock_resp):
            parsed_fc = engine.query_external_factcheck("5G mind control microchips in water")
            print("Parsed Google Fact Check output:", parsed_fc)
            assert len(parsed_fc) == 1
            assert parsed_fc[0]["reviewer"] == "PolitiFact"
            assert parsed_fc[0]["rating"] == "Pants on Fire"
            assert parsed_fc[0]["source"] == "Google Fact Check API"
            assert parsed_fc[0]["url"] == "https://www.politifact.com/factchecks/5g-water-hoax/"

    print("\n4. Testing Pipeline with Google Fact Check False Label (increases false-evidence score)...")
    with patch.dict(os.environ, {"FACTCHECK_API_KEY": "test-mock-key"}):
        with patch("services.factcheck_service.requests.get", return_value=mock_resp):
            fc_analysis = engine.analyze("Secret 5G mind control microchips in drinking water.")
            print(f"Verdict: {fc_analysis['verdict']}, Trust: {fc_analysis['trustScore']}, Contra: {fc_analysis['evidence']['contradictingCount']}")
            assert fc_analysis['verdict'] in ("KNOWN FALSE", "LIKELY FALSE", "FALSE")
            assert fc_analysis['evidence']['contradictingCount'] > 0
            assert len(fc_analysis['factCheckResults']) > 0

    print("\n5. Testing POST /api/analyze with Prompt Example (Al Jazeera - Saudi Arabia Missile Interception)...")
    prompt_example = {
        "text": "Al Jazeera reported on Friday that Saudi Arabia intercepted three missiles near the southern border. Officials confirmed 33 people were safely evacuated. The attack occurred near Sanaa without causing airport disruptions.",
        "source": "Al Jazeera"
    }
    resp = client.post('/api/analyze', json=prompt_example)
    assert resp.status_code == 200, f"Prompt example analyze failed: {resp.status_code}"
    res = resp.get_json()
    
    print("\n--- RESULTS FOR AL JAZEERA PROMPT EXAMPLE ---")
    print(f"Source Reputation: {res['sourceReputation']['trustScore']}/100 ({res['sourceReputation']['tier']})")
    print(f"ML Classification: {res['prediction']} ({res['scoringFactors']['mlPrediction']['score']}%)")
    print(f"Evidence Score: {res['evidence']['evidenceScore']}/100")
    print(f"Supporting Evidence Count: {res['evidence']['supportingCount']}")
    print(f"Contradicting Evidence Count: {res['evidence']['contradictingCount']}")
    print(f"Final Trust Score: {res['trustScore']}/100")
    print(f"Final Verdict: {res['verdict']}")
    print(f"Summary Reason: {res['summaryReason']}")
    print(f"Transparency Disclaimer: {res['disclaimer']}")

    assert res['sourceReputation']['trustScore'] == 85, f"Expected 85, got {res['sourceReputation']['trustScore']}"
    assert res['evidence']['supportingCount'] >= 3, f"Expected >= 3 supporting evidence count, got {res['evidence']['supportingCount']}"
    assert res['evidence']['contradictingCount'] == 0, f"Expected 0 contradicting evidence, got {res['evidence']['contradictingCount']}"
    assert res['verdict'] in ("LIKELY TRUE", "VERIFIED"), f"Expected LIKELY TRUE or VERIFIED, got {res['verdict']}"
    assert "disclaimer" in res
    assert "whyAppearsTrue" in res["reasoning"]
    assert "whyAppearsFalse" in res["reasoning"]
    assert "evidenceFound" in res["reasoning"]
    assert "evidenceMissing" in res["reasoning"]
    assert len(res["claims"]) >= 3, f"Expected at least 3 extracted claims, got {len(res['claims'])}"

    print("\n6. Testing POST /api/analyze with Fake / Hoax News (5G drinking water)...")
    fake_sample = {
        "text": "SHOCKING BOMBSHELL: Government secretly installs 5G mind control microchips in drinking water! Leaked documents expose big pharma coverup. Share before deleted!"
    }
    resp = client.post('/api/analyze', json=fake_sample)
    assert resp.status_code == 200, f"Fake news analyze failed: {resp.status_code}"
    fake_res = resp.get_json()
    print(f"Hoax Result: Verdict={fake_res['verdict']}, TrustScore={fake_res['trustScore']}, ContraCount={fake_res['evidence']['contradictingCount']}")
    assert fake_res['verdict'] in ("KNOWN FALSE", "LIKELY FALSE", "FALSE")
    assert fake_res['riskLevel'] in ("HIGH", "CRITICAL")
    assert fake_res['evidence']['contradictingCount'] > 0

    print("\n7. Testing POST /api/analyze-file with text file...")
    txt_data = (io.BytesIO(b"Supreme Court delivers unanimous landmark ruling clarifying environmental standards."), "test_article.txt")
    resp = client.post('/api/analyze-file', data={'file': txt_data}, content_type='multipart/form-data')
    assert resp.status_code == 200, f"File analyze failed: {resp.status_code}"
    file_res = resp.get_json()
    print("File Upload Result: Verdict =", file_res["verdict"], "Trust =", file_res["trustScore"])

    print("\n8. Testing GNews API query without API key (Google News RSS)...")
    empty_gnews = engine.query_gnews("Saudi Arabia intercepted missiles")
    assert isinstance(empty_gnews, list)
    print(f"Correctly handled GNews query without API key (retrieved {len(empty_gnews)} RSS articles).")

    print("\n9. Testing GNews API with mocked API response and Agreement Score...")
    mock_gnews_payload = {
        "totalArticles": 2,
        "articles": [
            {
                "title": "Saudi Arabia intercepted three missiles near southern border",
                "description": "Saudi defense forces intercepted missiles fired toward southern regions with no casualties reported.",
                "content": "Full article content from Reuters wire...",
                "url": "https://www.reuters.com/world/middle-east/saudi-intercepts-missiles-2026",
                "image": "https://reuters.com/img.jpg",
                "publishedAt": "2026-11-13T10:30:00Z",
                "source": {
                    "name": "Reuters",
                    "url": "https://www.reuters.com"
                }
            },
            {
                "title": "Unrelated economic summit opens in Geneva",
                "description": "World trade representatives gathered in Geneva to discuss tariff reductions.",
                "content": "Geneva trade report...",
                "url": "https://www.bloomberg.com/news/articles/geneva-trade",
                "image": "https://bloomberg.com/img.jpg",
                "publishedAt": "2026-11-13T09:00:00Z",
                "source": {
                    "name": "Bloomberg",
                    "url": "https://www.bloomberg.com"
                }
            }
        ]
    }
    mock_gn_resp = MagicMock()
    mock_gn_resp.status_code = 200
    mock_gn_resp.json.return_value = mock_gnews_payload

    with patch.dict(os.environ, {"GNEWS_API_KEY": "test-gnews-key"}):
        with patch("requests.get", return_value=mock_gn_resp):
            gnews_results = engine.query_gnews("Saudi Arabia intercepted three missiles near border")
            print("GNews Parsed output:", gnews_results)
            assert len(gnews_results) == 2
            assert gnews_results[0]["sourceName"] == "Reuters"
            assert gnews_results[0]["agreementScore"] >= 60
            assert "url" in gnews_results[0]
            assert "publicationTime" in gnews_results[0]
            print(f"Verified GNews fields: Source={gnews_results[0]['sourceName']}, AgreementScore={gnews_results[0]['agreementScore']}%")

    print("\n10. Testing Pipeline with GNews Evidence Boosting Evidence Strength...")
    with patch.dict(os.environ, {"GNEWS_API_KEY": "test-gnews-key"}):
        with patch("services.gnews_service.requests.get", return_value=mock_gn_resp):
            pipeline_gnews = engine.analyze(
                "Saudi Arabia intercepted three missiles near the southern border. Officials confirmed safety.",
                source_hint="Reuters"
            )
            print(f"Pipeline with GNews: Verdict={pipeline_gnews['verdict']}, Trust={pipeline_gnews['trustScore']}, EvidenceScore={pipeline_gnews['evidence']['evidenceScore']}")
            assert "gnewsArticles" in pipeline_gnews
            assert len(pipeline_gnews["gnewsArticles"]) > 0
            assert pipeline_gnews["timeline"][2]["step"] == "GNews API Corroboration"
            assert pipeline_gnews["aiIntegrations"]["gnewsConfigured"] is True
            print("GNews articles successfully integrated into verification pipeline!")

    print("\n11. Testing Verdict Escalation Safeguards (True with no citations, False fact check, Mixed evidence, Strong corroboration)...")
    from test_verdict_escalation import (
        test_true_article_with_no_citations_is_unverified,
        test_false_factcheck_result_is_known_false,
        test_mixed_evidence_is_needs_review,
        test_strong_corroboration_is_likely_true_or_verified,
        test_contradiction_threshold_and_likely_false
    )
    from services.multi_factor_scorer import get_scoring_engine
    scoring_eng = get_scoring_engine()

    test_true_article_with_no_citations_is_unverified(scoring_eng, engine, client)
    print("  -> Passed: True article with no citations -> UNVERIFIED (missing evidence != false)")
    test_false_factcheck_result_is_known_false(scoring_eng, engine)
    print("  -> Passed: False fact-check result -> KNOWN FALSE")
    test_mixed_evidence_is_needs_review(scoring_eng)
    print("  -> Passed: Mixed evidence -> NEEDS REVIEW")
    test_strong_corroboration_is_likely_true_or_verified(scoring_eng, engine)
    print("  -> Passed: Strong corroboration -> LIKELY TRUE / VERIFIED")
    test_contradiction_threshold_and_likely_false(scoring_eng)
    print("  -> Passed: Contradiction threshold & LIKELY FALSE rules verified")

    print("\n12. Testing Source Ownership and Domain Ownership Validation (BBC, IEA, Al Jazeera)...")
    from test_source_attribution import (
        test_domain_ownership_validation,
        test_publisher_from_bbc_url,
        test_publisher_from_iea_url,
        test_publisher_from_aljazeera_url,
        test_publisher_determined_from_metadata,
        test_publisher_determined_from_byline,
        test_body_citations_do_not_determine_publisher,
        test_prevent_evidence_contamination_on_bbc_article
    )
    from services.source_reputation import get_source_engine
    src_eng = get_source_engine()

    test_domain_ownership_validation()
    print("  -> Passed: Domain ownership validation (bbc.com -> BBC, reuters.com -> Reuters, aljazeera.com -> Al Jazeera, iea.org -> IEA / International Energy Agency)")
    test_publisher_from_bbc_url(src_eng, engine)
    print("  -> Passed: BBC URL -> Publisher = BBC")
    test_publisher_from_iea_url(src_eng, engine)
    print("  -> Passed: IEA URL -> Publisher = IEA")
    test_publisher_from_aljazeera_url(src_eng, engine)
    print("  -> Passed: Al Jazeera URL -> Publisher = Al Jazeera")
    test_publisher_determined_from_metadata(src_eng)
    print("  -> Passed: Publisher determined from metadata")
    test_publisher_determined_from_byline(src_eng)
    print("  -> Passed: Publisher determined from byline")
    test_body_citations_do_not_determine_publisher(src_eng)
    print("  -> Passed: Body citations do not determine primary publisher")
    test_prevent_evidence_contamination_on_bbc_article(engine)
    print("  -> Passed: Evidence contamination prevented (IEA/WHO/Reuters never replace primary BBC publisher)")

    print("\n13. Testing Automatic Article Summarization (Executive Summary, Key Takeaways, TL;DR)...")
    from test_summarizer import (
        test_extractive_summarization_structure,
        test_summary_explains_content_not_verdict,
        test_ai_summarization_when_gemini_available,
        test_url_submission_scrapes_title_body_and_summarizes
    )
    from services.summarizer import get_article_summarizer
    art_summarizer = get_article_summarizer()

    test_extractive_summarization_structure(art_summarizer)
    print("  -> Passed: Extractive summarization (3-5 sentences Executive Summary, <=5 Key Takeaways, 1 sentence TL;DR)")
    test_summary_explains_content_not_verdict(art_summarizer)
    print("  -> Passed: Summary explains article content, never injects verdict labels")
    test_ai_summarization_when_gemini_available()
    print("  -> Passed: AI summarization executes when Gemini/OpenAI configured")
    test_url_submission_scrapes_title_body_and_summarizes()
    print("  -> Passed: URL submission extracts title/body and returns structured summary")

    print("\n=== ALL ARCHITECTURAL LAYERS, GOOGLE FACT CHECK & GNEWS INTEGRATIONS VERIFIED! ===")

if __name__ == '__main__':
    test_endpoints()
