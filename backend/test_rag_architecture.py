"""
Unit and Integration Tests for Lucid RAG (Retrieval-Augmented Generation) Architecture
Validates:
1. Retrieval Layer across wires, fact checkers, government & UN/WHO institutions
2. Dense semantic embeddings & similarity on paraphrased claims (e.g. IDF / Palestinian boy)
3. Evidence ranking by relevance, credibility, recency, and stance
4. Grounded reasoning & strict source transparency
5. Evidence Matrix per claim
6. Strict verdict logic: VERIFIED, LIKELY TRUE, UNVERIFIED, LIKELY FALSE, FALSE
   (Ensuring missing evidence is NEVER treated as false)
7. Opinion statement exclusion from factual wire checks
8. End-to-end VerificationEngine RAG pipeline
"""
import pytest
from services.semantic_embeddings import get_semantic_engine
from services.rag_retriever import get_rag_retriever
from services.evidence_ranker import get_evidence_ranker
from services.grounded_reasoner import get_grounded_reasoner
from services.claim_extractor import get_claim_engine
from services.ai_service import get_verification_engine


def test_semantic_embedding_paraphrase_matching():
    """
    Requirement 2: Semantic Search.
    Example given by user:
    Claim: "Israeli forces shot a Palestinian boy."
    Should retrieve semantically similar reports even if wording differs.
    """
    engine = get_semantic_engine()
    claim = "Israeli forces shot a Palestinian boy."
    candidate_paraphrase = "IDF gunfire killed youth in West Bank during military raid."
    unrelated_text = "Apple reported record quarterly iPhone revenues in Cupertino today."

    sim_match = engine.compute_similarity(claim, candidate_paraphrase)
    sim_unrelated = engine.compute_similarity(claim, unrelated_text)

    # Paraphrased reporting should have substantially higher semantic similarity than unrelated text
    assert sim_match > 0.45
    assert sim_match > sim_unrelated * 2.0


def test_rag_retrieval_layer():
    """
    Requirement 1 & 5: Retrieval Layer & Source Transparency
    Searches Reuters, AP News, BBC, Al Jazeera, The Guardian, AFP, Fact Check, etc.
    """
    retriever = get_rag_retriever()
    sources = retriever.get_sources_searched_list()

    # Honest sources actually queried by Lucid RAG engine
    assert any("Google News RSS" in s for s in sources)
    assert any("Wikipedia" in s for s in sources)

    # Retrieve for Palestinian boy conflict claim
    results = retriever.retrieve_evidence_for_claim("Israeli forces shot a Palestinian boy.")
    assert len(results) > 0

    first = results[0]
    assert "articleTitle" in first
    assert "publisher" in first
    assert "publicationDate" in first
    assert "url" in first
    assert "evidenceSnippet" in first
    assert "sourceTrustScore" in first


def test_evidence_ranking_and_stance_classification():
    """
    Requirement 3: Evidence Ranking by relevance, credibility, recency, stance.
    """
    ranker = get_evidence_ranker()
    claim = "Israeli forces shot a Palestinian boy."
    candidates = [
        {
            "articleTitle": "West Bank raid: Israeli troops fatally shoot Palestinian boy",
            "publisher": "Associated Press (AP)",
            "publicationDate": "2024-02-15",
            "url": "https://apnews.com/article/1",
            "evidenceSnippet": "Palestinian health officials confirmed youth died from gunfire.",
            "sourceTrustScore": 95,
            "sourceCategory": "Wire Service"
        },
        {
            "articleTitle": "Fact Check: Viral claim about hospital attack is fabricated and false",
            "publisher": "Reuters Fact Check",
            "publicationDate": "2024-02-16",
            "url": "https://reuters.com/fact-check/1",
            "evidenceSnippet": "No evidence that military struck hospital; footage was debunked as false claim.",
            "sourceTrustScore": 96,
            "sourceCategory": "IFCN Fact Checker",
            "factCheckRating": "False"
        }
    ]

    ranked = ranker.rank_evidence_for_claim(claim, candidates)
    assert ranked["supportingCount"] >= 1
    assert ranked["contradictingCount"] >= 1
    assert ranked["supporting"][0]["stance"] == "support"
    assert ranked["contradicting"][0]["stance"] == "contradict"


def test_evidence_matrix_and_opinion_handling():
    """
    Requirement 6 & 9: Evidence Matrix & Opinion statement exclusion.
    """
    ranker = get_evidence_ranker()
    claim_engine = get_claim_engine()

    text = (
        "In my opinion, current economic governance is completely unacceptable. "
        "The Federal Reserve raised interest rates by 25 basis points in Washington."
    )
    claims = claim_engine.extract_claims(text, article_type="News Report")

    # One claim should be identified as opinion, the other as factual
    assert any(c.get("isOpinion") for c in claims)
    assert any(not c.get("isOpinion") for c in claims)

    retrieval_map = {
        claims[1]["claim"]: [
            {
                "articleTitle": "Federal Reserve raises interest rates by quarter percentage point",
                "publisher": "Reuters",
                "publicationDate": "2024-03-20",
                "url": "https://reuters.com/rates",
                "evidenceSnippet": "The US central bank raised its benchmark rate by 25 basis points.",
                "sourceTrustScore": 96,
                "sourceCategory": "Wire Service"
            }
        ]
    }

    matrix = ranker.build_evidence_matrix_for_claims(claims, retrieval_map)
    assert len(matrix) >= 2

    opinion_row = next(r for r in matrix if r["isOpinion"])
    factual_row = next(r for r in matrix if not r["isOpinion"])

    assert opinion_row["status"] == "Opinion / Subjective"
    assert "Opinion" in opinion_row["claimType"]
    assert factual_row["status"] in ("Supported", "Verified")


def test_strict_verdict_logic_and_missing_evidence():
    """
    Requirement 7: Verdict Logic & Rule: Missing evidence must NEVER be treated as false!
    """
    reasoner = get_grounded_reasoner()
    sources_searched = ["Reuters", "AP News", "BBC", "Google Fact Check API"]

    # Case A: Missing evidence (0 citations found) -> Must be UNVERIFIED, NEVER FALSE!
    unverified_res = reasoner.synthesize_rag_verdict_and_reasoning(
        claims=[{"claim": "Subterranean aliens built pyramid under lunar pole", "isOpinion": False}],
        evidence_matrix=[],
        top_supporting=[],
        top_contradicting=[],
        sources_searched=sources_searched,
        article_type="News Report"
    )
    assert unverified_res["verdict"] == "UNVERIFIED"
    assert unverified_res["verdict"] != "FALSE"
    assert unverified_res["verdict"] != "LIKELY FALSE"

    # Case B: Multiple independent supporting sources -> VERIFIED
    verified_res = reasoner.synthesize_rag_verdict_and_reasoning(
        claims=[{"claim": "Renewable capacity rose 50% in 2023", "isOpinion": False}],
        evidence_matrix=[],
        top_supporting=[
            {"publisher": "Reuters", "articleTitle": "Renewables surged 50%", "evidenceSnippet": "Record solar additions.", "sourceTrustScore": 96},
            {"publisher": "BBC", "articleTitle": "Global green energy grew 50% in 2023", "evidenceSnippet": "Clean energy additions reached historic high.", "sourceTrustScore": 94}
        ],
        top_contradicting=[],
        sources_searched=sources_searched,
        article_type="News Report"
    )
    assert verified_res["verdict"] == "VERIFIED"

    # Case C: Fact-check rated False -> FALSE
    false_res = reasoner.synthesize_rag_verdict_and_reasoning(
        claims=[{"claim": "5G chips are dissolved in drinking water", "isOpinion": False}],
        evidence_matrix=[],
        top_supporting=[],
        top_contradicting=[
            {"publisher": "PolitiFact", "articleTitle": "Fact Check: No microchips in drinking water", "evidenceSnippet": "Pants on Fire: impossible to dissolve microchips.", "sourceCategory": "IFCN Fact Checker", "sourceTrustScore": 95}
        ],
        sources_searched=sources_searched,
        article_type="News Report"
    )
    assert false_res["verdict"] == "FALSE"


def test_end_to_end_rag_verification_engine():
    """
    Requirement 10: End-to-end RAG verification returning sources, evidence matrix, grounded reasoning.
    """
    engine = get_verification_engine()
    article = (
        "Global renewable capacity additions increased by nearly 50% in 2023, "
        "marking the fastest growth rate in two decades according to official energy monitoring."
    )

    result = engine.analyze(article, source_hint="reuters.com")

    assert result["ragEnabled"] is True
    assert "sourcesSearched" in result
    assert "resultsFound" in result
    assert "evidenceMatrix" in result
    assert "topSupportingSources" in result
    assert "groundedReasoning" in result
    assert result["verdict"] in ("VERIFIED", "LIKELY TRUE")
    assert result["trustScore"] >= 80


def test_pope_francis_trump_endorsement_verdict_false():
    """
    Critical RAG reasoning test:
    Input claim: 'Pope Francis endorsed Trump'
    Expected:
    - Contradicting Sources > 0
    - Supporting Sources == 0
    - Final Verdict == 'FALSE'
    """
    engine = get_verification_engine()
    result = engine.analyze("Pope Francis endorsed Trump")

    supporting_count = result.get("evidence", {}).get("supportingCount", 0)
    contradicting_count = result.get("evidence", {}).get("contradictingCount", 0)
    top_supporting = result.get("topSupportingSources", [])
    top_contradicting = result.get("topContradictingSources", [])

    assert len(top_supporting) == 0, f"Expected 0 top supporting sources, got {len(top_supporting)}"
    assert supporting_count == 0, f"Expected supportingCount 0, got {supporting_count}"
    assert len(top_contradicting) > 0, f"Expected >0 top contradicting sources, got {len(top_contradicting)}"
    assert contradicting_count > 0, f"Expected contradictingCount > 0, got {contradicting_count}"
    assert result["verdict"] == "FALSE", f"Expected verdict FALSE, got {result['verdict']}"
