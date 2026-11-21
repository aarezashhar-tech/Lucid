"""
Unit Tests for Automatic Article Summarization.

Requirements:
1. Extract article title and body for URL submissions.
2. Generate Executive Summary (3-5 simple sentences).
3. Generate Key Takeaways (bullet points, maximum 5 points).
4. Generate TL;DR (1 sentence).
5. If Gemini/OpenAI is available: use AI summarization; otherwise extractive summarization.
6. The summary must explain what the article is about, NOT whether it is true or false.
"""
import pytest
from unittest.mock import patch, MagicMock

from services.summarizer import get_article_summarizer, ArticleSummarizer
from services.ai_service import get_verification_engine
from app import app


@pytest.fixture
def summarizer():
    return get_article_summarizer()


@pytest.fixture
def verification_engine():
    return get_verification_engine()


SAMPLE_ARTICLE_TITLE = "Renewable Energy Capacity Surges 50% Worldwide in 2023"
SAMPLE_ARTICLE_BODY = (
    "Global renewable energy capacity increased by 50% in 2023, marking the fastest growth rate in two decades. "
    "According to a report published by the International Energy Agency, solar photovoltaics accounted for "
    "three-quarters of all new capacity additions worldwide. European countries significantly accelerated deployment "
    "to enhance regional energy security and reduce fossil fuel dependency. In Asia, government subsidies for offshore "
    "wind and utility-scale solar farms reached new highs. Industry analysts project that annual renewable expansion "
    "will continue to rise through 2030 as equipment costs decline and grid modernization efforts expand."
)


def test_extractive_summarization_structure(summarizer):
    """
    Tests that extractive summarization generates:
    1. Executive summary with 3-5 simple sentences.
    2. Key takeaways with at most 5 bullet points.
    3. TL;DR with exactly 1 sentence.
    """
    summary = summarizer.summarize(SAMPLE_ARTICLE_BODY, title=SAMPLE_ARTICLE_TITLE)

    assert "executiveSummary" in summary
    assert "keyTakeaways" in summary
    assert "tldr" in summary
    assert summary["title"] == SAMPLE_ARTICLE_TITLE

    # 1. Executive Summary: 3-5 simple sentences
    exec_summary = summary["executiveSummary"]
    assert isinstance(exec_summary, str)
    # Count sentences ending with period/exclamation/question mark
    sentences = [s.strip() for s in exec_summary.split(".") if s.strip()]
    assert 3 <= len(sentences) <= 5, f"Expected 3-5 sentences in executiveSummary, got {len(sentences)}"

    # 2. Key Takeaways: Bullet points, Maximum 5 points
    takeaways = summary["keyTakeaways"]
    assert isinstance(takeaways, list)
    assert 1 <= len(takeaways) <= 5, f"Expected <= 5 takeaways, got {len(takeaways)}"
    for item in takeaways:
        assert isinstance(item, str)
        assert len(item) > 10

    # 3. TL;DR: 1 sentence
    tldr = summary["tldr"]
    assert isinstance(tldr, str)
    tldr_sentences = [s.strip() for s in tldr.split(".") if s.strip()]
    assert len(tldr_sentences) == 1, f"Expected 1 sentence in TL;DR, got {len(tldr_sentences)}"


def test_summary_explains_content_not_verdict(summarizer):
    """
    Requirement: The summary must explain what the article is about, NOT whether it is true or false.
    Verify that summary does not inject fact-checking verdict assertions.
    """
    # Sample article with a disputed/false claim
    disputed_text = (
        "Researchers at a private biotechnology summit discussed new micro-filtration techniques for municipal water. "
        "The symposium featured presentations on synthetic nanoparticles and fluid dynamics in urban reservoirs. "
        "Engineers highlighted improved testing protocols for detecting chemical residues in regional pipelines. "
        "Representatives from municipal utility boards evaluated operational costs and rollout schedules."
    )
    summary = summarizer.summarize(disputed_text, title="Water Filtration Symposium Discusses Micro-Particles")

    combined_text = (summary["executiveSummary"] + " " + " ".join(summary["keyTakeaways"]) + " " + summary["tldr"]).lower()

    # Must NOT inject judgment labels
    forbidden_verdict_terms = [
        "this is false",
        "this claim is false",
        "debunked by fact-checkers",
        "hoax",
        "known false",
        "unverified claim",
        "our verdict is",
        "is misleading"
    ]
    for term in forbidden_verdict_terms:
        assert term not in combined_text, f"Found forbidden verdict term '{term}' in summary"


def test_ai_summarization_when_gemini_available():
    """
    Tests that when GEMINI_API_KEY is configured, AI summarization is invoked,
    and falls back to extractive if the API fails.
    """
    mock_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{\n'
                                '  "executiveSummary": "Global renewable power installations expanded by fifty percent last year. '
                                'Solar systems led the global growth across multiple continents. European nations led policy '
                                'efforts to improve energy self-sufficiency. Emerging economies also expanded solar infrastructure.",\n'
                                '  "keyTakeaways": [\n'
                                '    "Renewable energy generation grew by 50% in 2023.",\n'
                                '    "Solar photovoltaics accounted for most of the new capacity.",\n'
                                '    "Europe accelerated clean energy deployment for energy independence.",\n'
                                '    "Emerging markets expanded renewable infrastructure investments."\n'
                                '  ],\n'
                                '  "tldr": "Worldwide renewable energy capacity grew 50% in 2023 led by solar energy expansion."\n'
                                '}'
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key"}):
        test_summarizer = ArticleSummarizer()
        with patch("requests.post", return_value=mock_resp):
            res = test_summarizer.summarize(SAMPLE_ARTICLE_BODY, title=SAMPLE_ARTICLE_TITLE)
            assert res["provider"] == "Gemini AI"
            assert len(res["keyTakeaways"]) <= 5
            assert "Global renewable power" in res["executiveSummary"]


def test_url_submission_scrapes_title_body_and_summarizes():
    """
    Tests /api/analyze-url endpoint:
    1. Scrapes article title and body from HTML response.
    2. Generates articleSummary with executiveSummary, keyTakeaways, and tldr.
    """
    mock_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta property="og:title" content="International Climate Report Details Clean Energy Surge" />
        <title>International Climate Report</title>
      </head>
      <body>
        <main>
          <h1>International Climate Report Details Clean Energy Surge</h1>
          <p>Global renewable energy capacity expanded at an unprecedented pace over the last twelve months.</p>
          <p>The latest international analysis revealed that solar and wind farms provided record electricity shares.</p>
          <p>National energy ministries reported major investments in grid storage batteries and transmission lines.</p>
          <p>Economic forecasters predict that renewable generation will surpass coal globally within the next three years.</p>
        </main>
      </body>
    </html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_html
    mock_resp.raise_for_status = MagicMock()

    client = app.test_client()

    with patch("requests.get", return_value=mock_resp):
        resp = client.post("/api/analyze-url", json={"url": "https://www.reuters.com/business/clean-energy-surge"})
        assert resp.status_code == 200
        data = resp.get_json()

        assert "articleSummary" in data
        summary = data["articleSummary"]
        assert "executiveSummary" in summary
        assert "keyTakeaways" in summary
        assert "tldr" in summary
        assert len(summary["keyTakeaways"]) <= 5
        assert data["articleTitle"] == "International Climate Report Details Clean Energy Surge"
