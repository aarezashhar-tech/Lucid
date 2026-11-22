"""
Comprehensive Evaluation & Evidence Suite for Lucid 'scoring-consistency' Branch.
Tests all requirements:
1. Publisher attribution before/after (renewable-energy-industry.com, Carbon Brief, Fair Observer)
2. Evidence matching: topic similarity vs specific claim match ('related' vs 'support')
3. Wikipedia snippet: raw extracts and Wikidata checks, with [Derived check] label
4. No evidence: score capped at 50, low confidence label
5. Language: short claims (< 20 chars) default to en, no Afrikaans, no template praise
6. Honest evaluation of 6 sources:
   - Source 1: BBC News (Text)
   - Source 2: The Hindu (Live URL vs Pasted Text side-by-side, extracted char/word count, score delta <= 5)
   - Source 3: NDTV (TV website text, plus live URL 403 handling)
   - Source 4: Reuters (Text)
   - Source 5: Unknown personal blog (Domain trust 50, score <= 50, low confidence)
   - Source 6: Known satire (The Onion)
   - Extra: BBC News (Live URL vs Pasted Text side-by-side, extracted char/word count)
   - Extra: Opinion piece ("Not scored", "Opinion - not fact-checked")
7. New tests on 3 real articles with 1 number or 1 name changed showing score drop without tuning.
"""
import os
import sys
import json
import requests

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import run_shared_verification_pipeline
from services.rag_retriever import get_rag_retriever
from services.evidence_ranker import get_evidence_ranker
from services.wikipedia_service import get_wikipedia_service
from services.language_service import get_language_service

def evaluate_all():
    print("=" * 80)
    print("LUCID EVALUATION SUITE: SCORING CONSISTENCY & ACCURACY VERIFICATION")
    print("=" * 80)

    rag = get_rag_retriever()
    ranker = get_evidence_ranker()
    wiki = get_wikipedia_service()
    lang = get_language_service()

    # =========================================================================
    # REQUIREMENT 1: PUBLISHER ATTRIBUTION BEFORE / AFTER
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 1: PUBLISHER ATTRIBUTION BEFORE / AFTER")
    print("=" * 80)
    test_publishers = [
        ("renewable-energy-industry.com", "https://news.google.com/rss/articles/CBMiRGh0dHBzOi8vd3d3LnJlbmV3YWJsZS1lbmVyZ3ktaW5kdXN0cnkuY29tL25ld3MvcmVuZXdhYmxlLWVuZXJneS1pZWEtcmVwb3J00gEA"),
        ("Carbon Brief", "https://news.google.com/rss/articles/CBMiP2h0dHBzOi8vd3d3LmNhcmJvbmJyaWVmLm9yZy9kZXRhaWxlZC1xLWEtaWVhLWZhc3Rlc3QtcmVuZXdhYmxlLXJlcG9ydC_SAQA"),
        ("Fair Observer", "https://news.google.com/rss/articles/CBMiSWh0dHBzOi8vd3d3LmZhaXJvYnNlcnZlci5jb20vd29ybGQtbmV3cy9pZWEtcmVuZXdhYmxlcy1nbG9iYWwtY2FwYWNpdHktMjAyMy_SAQA")
    ]
    pub_table = []
    for pub_name, test_url in test_publishers:
        after_meta = rag._match_publisher_metadata(pub_name, test_url)
        pub_table.append({
            "Source String": pub_name,
            "Before Publisher": "United Nations (UN)",
            "Before Score": 99,
            "Before Category": "International Institution",
            "After Publisher": after_meta["publisher"],
            "After Score": after_meta["score"],
            "After Category": after_meta["category"]
        })
        print(f"Publisher: '{pub_name}'")
        print(f"  BEFORE -> Assigned: 'United Nations (UN)', Trust Score: 99, Category: 'International Institution'")
        print(f"  AFTER  -> Assigned: '{after_meta['publisher']}', Trust Score: {after_meta['score']}, Category: '{after_meta['category']}'")
        assert after_meta["score"] == 50, f"Expected neutral 50 for {pub_name}, got {after_meta['score']}"
        assert after_meta["publisher"] == pub_name, f"Expected publisher to equal {pub_name}"

    # =========================================================================
    # REQUIREMENT 2: EVIDENCE MATCHING (SPECIFIC CLAIM MATCH vs RELATED)
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 2: EVIDENCE MATCHING - SPECIFIC CLAIM MATCH vs RELATED")
    print("=" * 80)
    claim_text = "The World Health Organization (WHO) noted health benefits from reduced air pollution, while Reuters reported similar findings across European utilities."
    unrelated_headline = "Government approves action plan to reduce air pollution in Latin American cities - Reuters"
    
    ranked_res = ranker.rank_evidence_for_claim(
        claim_text=claim_text,
        candidate_items=[{
            "articleTitle": unrelated_headline,
            "publisher": "Reuters",
            "url": "https://reuters.com/latin-america-air-pollution",
            "evidenceSnippet": "Ministers in Latin America approved a regional pact targeting urban emissions.",
            "sourceTrustScore": 92,
            "sourceCategory": "News Agency / Wire"
        }]
    )
    assert ranked_res["supportingCount"] == 0, "Specific claim mismatch must NOT be marked as 'support'!"
    assert ranked_res["relatedCount"] == 1, f"Expected 1 related item, got {ranked_res['relatedCount']}"
    assigned_stance = ranked_res["related"][0].get("stance")
    print(f"Claim: '{claim_text}'")
    print(f"Candidate Headline: '{unrelated_headline}'")
    print(f"Assigned Stance: '{assigned_stance}' (Expected: 'related', NOT 'support')")
    assert assigned_stance == "related", f"Expected 'related', got {assigned_stance}"
    print(">>> PASS: Geographic/institutional mismatch correctly marked as 'related'!")

    # =========================================================================
    # REQUIREMENT 3: WIKIPEDIA SNIPPET (AUTHENTIC API EXTRACT, NO SYNTHETIC QUOTES)
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 3: WIKIPEDIA SNIPPET")
    print("=" * 80)
    wiki_res = wiki.check_biographical_or_encyclopedic_claim("PM Modi died")
    print(f"Claim: 'PM Modi died'")
    print(f"Wikipedia Check Result:")
    print(f"  Contradicted: {wiki_res.get('isContradicted')}")
    print(f"  Reason: {wiki_res.get('reason')}")
    ev_item = wiki_res.get("evidenceItem", {})
    print(f"  Snippet: {ev_item.get('evidenceSnippet')}")
    assert "[Derived check]" in wiki_res.get("reason", "")
    assert "[Derived check]" in ev_item.get("evidenceSnippet", "")
    assert "no date of death recorded" in ev_item.get("evidenceSnippet", "")
    print(">>> PASS: Wikipedia snippet contains authentic API/Wikidata text with [Derived check] prefix!")

    # =========================================================================
    # REQUIREMENT 4: NO EVIDENCE CAPPED AT 50 & LOW CONFIDENCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 4: NO EVIDENCE CAP AT 50 & LOW CONFIDENCE")
    print("=" * 80)
    res_no_evidence = run_shared_verification_pipeline(
        text="A local resident observed three stray cats crossing the alley behind Elm Street on Tuesday evening.",
        title="Cats crossing Elm Street alley",
        input_type="text"
    )
    print(f"Trust Score: {res_no_evidence['trustScore']} (Capped at <= 50)")
    print(f"Confidence: {res_no_evidence['confidence']}%")
    print(f"Summary Reason: {res_no_evidence['summaryReason']}")
    assert res_no_evidence['trustScore'] <= 50
    assert "Low confidence" in res_no_evidence['summaryReason']
    assert res_no_evidence['confidence'] <= 40
    print(">>> PASS: Zero evidence capped at <= 50 and flagged as low confidence!")

    # =========================================================================
    # REQUIREMENT 5: LANGUAGE (< 20 CHARACTERS -> NO AFRIKAANS, NO JOURNALISTIC PROSE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 5: LANGUAGE HANDLING")
    print("=" * 80)
    short_claim = "PM Modi died"
    detected_lang = lang.detect_language(short_claim)
    print(f"Short Text: '{short_claim}' ({len(short_claim)} chars)")
    print(f"Detected Language: {detected_lang['language']} ({detected_lang['languageCode']})")
    assert detected_lang['languageCode'] == "en", f"Expected 'en', got {detected_lang['languageCode']}"
    
    res_modi = run_shared_verification_pipeline(
        text=short_claim,
        title=short_claim,
        input_type="text"
    )
    has_prose_template = any("Objective, attributable journalistic prose detected" in exp for exp in res_modi['explanation'])
    print(f"Contains 'Objective, attributable journalistic prose detected': {has_prose_template}")
    assert not has_prose_template, "Template line should NOT be present for short text!"
    print(">>> PASS: Language detected as English, template praise removed!")

    # =========================================================================
    # REQUIREMENT 7: 3 DIFFERENT REAL ARTICLES WITH 1 NUMBER / 1 NAME CHANGED
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 7: 3 REAL ARTICLES WITH 1 NUMBER OR 1 NAME CHANGED")
    print("=" * 80)

    # Article 1: IEA Renewable Energy (Number Change: 50% -> 2%)
    print("\n--- Test Article 1: IEA Renewable Energy (Number Change: 50% -> 2%) ---")
    art1_original = (
        "Global renewable energy capacity additions increased by nearly 50% in 2023, "
        "marking the fastest growth rate in two decades according to the International Energy Agency. "
        "The annual market report showed that solar photovoltaic systems accounted for three-quarters "
        "of additions worldwide, driven by substantial policy support and economic momentum in global markets."
    )
    art1_altered = (
        "Global renewable energy capacity additions increased by nearly 2% in 2023, "
        "marking the fastest growth rate in two decades according to the International Energy Agency. "
        "The annual market report showed that solar photovoltaic systems accounted for three-quarters "
        "of additions worldwide, driven by substantial policy support and economic momentum in global markets."
    )
    res_art1_orig = run_shared_verification_pipeline(
        text=art1_original,
        title="IEA: Renewables capacity additions surged by 50% in 2023",
        input_type="text"
    )
    res_art1_alt = run_shared_verification_pipeline(
        text=art1_altered,
        title="IEA: Renewables capacity additions surged by 2% in 2023",
        input_type="text"
    )
    score1_orig = res_art1_orig['trustScore']
    score1_alt = res_art1_alt['trustScore']
    drop1 = score1_orig - score1_alt
    print(f"Original Article 1 Score: {score1_orig} (Verdict: {res_art1_orig['verdict']})")
    print(f"Altered  Article 1 Score: {score1_alt} (Verdict: {res_art1_alt['verdict']})")
    print(f"Score Drop: -{drop1} points")
    assert drop1 > 0, f"Expected score drop, got {drop1}"

    # Article 2: NVIDIA Leadership (Name Change: Jensen Huang -> Sundar Pichai)
    print("\n--- Test Article 2: NVIDIA Leadership (Name Change: Jensen Huang -> Sundar Pichai) ---")
    art2_original = (
        "Jensen Huang continues as NVIDIA founder and Chief Executive Officer, leading the company "
        "in artificial intelligence chip development and data center computing infrastructure. "
        "Under his leadership, the firm has introduced major graphic processing architectures "
        "that power high-performance deep learning models across the technology sector."
    )
    art2_altered = (
        "Sundar Pichai continues as NVIDIA founder and Chief Executive Officer, leading the company "
        "in artificial intelligence chip development and data center computing infrastructure. "
        "Under his leadership, the firm has introduced major graphic processing architectures "
        "that power high-performance deep learning models across the technology sector."
    )
    res_art2_orig = run_shared_verification_pipeline(
        text=art2_original,
        title="Jensen Huang leads NVIDIA in AI chip development",
        input_type="text"
    )
    res_art2_alt = run_shared_verification_pipeline(
        text=art2_altered,
        title="Sundar Pichai leads NVIDIA in AI chip development",
        input_type="text"
    )
    score2_orig = res_art2_orig['trustScore']
    score2_alt = res_art2_alt['trustScore']
    drop2 = score2_orig - score2_alt
    print(f"Original Article 2 Score: {score2_orig} (Verdict: {res_art2_orig['verdict']})")
    print(f"Altered  Article 2 Score: {score2_alt} (Verdict: {res_art2_alt['verdict']})")
    print(f"Score Drop: -{drop2} points")
    assert drop2 > 0, f"Expected score drop, got {drop2}"

    # Article 3: NDTV / RBI Monetary Policy (Number Change: 6.5% -> 1.2%)
    print("\n--- Test Article 3: NDTV / RBI Monetary Policy (Number Change: 6.5% -> 1.2%) ---")
    art3_original = (
        "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent following the conclusion "
        "of its Monetary Policy Committee meeting. The Monetary Policy Committee announced that the policy "
        "repo rate remains at 6.50% to align inflation with target."
    )
    art3_altered = (
        "The Reserve Bank of India kept the repo rate unchanged at 1.2 percent following the conclusion "
        "of its Monetary Policy Committee meeting. The Monetary Policy Committee announced that the policy "
        "repo rate remains at 1.20% to align inflation with target."
    )
    res_art3_orig = run_shared_verification_pipeline(
        text=art3_original,
        title="RBI keeps repo rate unchanged at 6.5 percent",
        source_hint="ndtv.com",
        input_type="text"
    )
    res_art3_alt = run_shared_verification_pipeline(
        text=art3_altered,
        title="RBI keeps repo rate unchanged at 1.2 percent",
        source_hint="ndtv.com",
        input_type="text"
    )
    score3_orig = res_art3_orig['trustScore']
    score3_alt = res_art3_alt['trustScore']
    drop3 = score3_orig - score3_alt
    print(f"Original Article 3 Score: {score3_orig} (Verdict: {res_art3_orig['verdict']})")
    print(f"Altered  Article 3 Score: {score3_alt} (Verdict: {res_art3_alt['verdict']})")
    print(f"Score Drop: -{drop3} points")
    assert drop3 > 0, f"Expected score drop, got {drop3}"

    # =========================================================================
    # REQUIREMENT 6: HONEST EVALUATION OF 6 DIFFERENT SOURCES + LIVE URL TESTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("REQUIREMENT 6: HONEST EVALUATION OF 6 DIFFERENT SOURCES & LIVE URL TESTS")
    print("=" * 80)

    # 1. BBC Article (Text)
    print("\n[Source 1: BBC News (Text)]")
    bbc_text = (
        "The UK government has pledged 22 billion pounds of funding over 25 years for carbon capture "
        "and storage projects. Prime Minister Keir Starmer and Energy Secretary Ed Miliband announced the funding "
        "for two sites in Teesside and Merseyside, aiming to cut carbon emissions while creating thousands of green jobs."
    )
    res_bbc = run_shared_verification_pipeline(
        text=bbc_text,
        source_hint="bbc.co.uk",
        title="UK pledges 22bn pounds for carbon capture projects",
        input_type="text"
    )
    print(f"BBC News (Text): Score={res_bbc['trustScore']}, Verdict={res_bbc['verdict']}")
    print(f"Scoring Factors: {json.dumps(res_bbc['scoringFactors'], indent=2)}")

    # 2. The Hindu: Live URL Fetch vs Pasted Text Comparison
    print("\n[Source 2: The Hindu - Live URL vs Pasted Text Comparison]")
    from utils.url_extractor import fetch_and_extract_article
    hindu_live_url = "https://www.thehindu.com/news/national/people-bought-less-gold-prices-fell-modi-recalls-year-old-appeal/article71731156.ece"
    ext_hindu = fetch_and_extract_article(hindu_live_url)
    assert ext_hindu.get("success"), f"Failed to extract live Hindu URL: {ext_hindu.get('error')}"
    extracted_hindu_text = ext_hindu["text"]
    char_len_hindu = len(extracted_hindu_text)
    word_len_hindu = len(extracted_hindu_text.split())
    print(f"Live URL: {hindu_live_url}")
    print(f"Extracted Article Title: '{ext_hindu.get('title')}'")
    print(f"Extracted Text Length: {char_len_hindu} characters ({word_len_hindu} words)")
    
    # Process as URL input via shared pipeline
    res_hindu_live_url = run_shared_verification_pipeline(
        text=extracted_hindu_text,
        source_hint=ext_hindu.get("domain", "thehindu.com"),
        url=hindu_live_url,
        title=ext_hindu.get("title", ""),
        input_type="url"
    )
    print(f"The Hindu (Live URL): Score={res_hindu_live_url['trustScore']}, Verdict={res_hindu_live_url['verdict']}")
    print(f"Scoring Factors (URL): {json.dumps(res_hindu_live_url['scoringFactors'], indent=2)}")
    
    # Run same extracted text as pasted text via shared pipeline
    res_hindu_pasted = run_shared_verification_pipeline(
        text=extracted_hindu_text,
        source_hint=ext_hindu.get("domain", "thehindu.com"),
        title=ext_hindu.get("title", ""),
        input_type="text"
    )
    print(f"The Hindu (Pasted Text): Score={res_hindu_pasted['trustScore']}, Verdict={res_hindu_pasted['verdict']}")
    print(f"Scoring Factors (Text): {json.dumps(res_hindu_pasted['scoringFactors'], indent=2)}")
    if res_hindu_live_url['trustScore'] == "Not scored" and res_hindu_pasted['trustScore'] == "Not scored":
        hindu_delta = 0
    else:
        hindu_delta = abs(int(res_hindu_live_url['trustScore']) - int(res_hindu_pasted['trustScore']))
    print(f"Score Delta (URL vs Text): {hindu_delta} points (Requirement: <= 5 points)")
    assert hindu_delta <= 5, f"Expected <= 5 points difference, got {hindu_delta}"
    assert res_hindu_live_url['verdict'] == res_hindu_pasted['verdict'], "Expected same verdict type for URL and pasted text"

    # Live URL Test 2: BBC News Live URL
    print("\n[Additional Live URL Test: BBC News Live URL vs Pasted Text]")
    bbc_live_url = "https://www.bbc.co.uk/news/articles/cvj6l8y3x1x4o"
    ext_bbc = fetch_and_extract_article(bbc_live_url)
    assert ext_bbc.get("success"), f"Failed to extract live BBC URL: {ext_bbc.get('error')}"
    extracted_bbc_text = ext_bbc["text"]
    char_len_bbc = len(extracted_bbc_text)
    word_len_bbc = len(extracted_bbc_text.split())
    print(f"Live URL: {bbc_live_url}")
    print(f"Extracted Article Title: '{ext_bbc.get('title')}'")
    print(f"Extracted Text Length: {char_len_bbc} characters ({word_len_bbc} words)")
    
    res_bbc_live_url = run_shared_verification_pipeline(
        text=extracted_bbc_text,
        source_hint=ext_bbc.get("domain", "bbc.co.uk"),
        url=bbc_live_url,
        title=ext_bbc.get("title", ""),
        input_type="url"
    )
    res_bbc_pasted = run_shared_verification_pipeline(
        text=extracted_bbc_text,
        source_hint=ext_bbc.get("domain", "bbc.co.uk"),
        title=ext_bbc.get("title", ""),
        input_type="text"
    )
    if res_bbc_live_url['trustScore'] == "Not scored" and res_bbc_pasted['trustScore'] == "Not scored":
        bbc_delta = 0
    else:
        bbc_delta = abs(int(res_bbc_live_url['trustScore']) - int(res_bbc_pasted['trustScore']))
    print(f"BBC (Live URL): Score={res_bbc_live_url['trustScore']}, Verdict={res_bbc_live_url['verdict']}")
    print(f"BBC (Pasted Text): Score={res_bbc_pasted['trustScore']}, Verdict={res_bbc_pasted['verdict']}")
    print(f"Score Delta (URL vs Text): {bbc_delta} points (Requirement: <= 5 points)")
    assert bbc_delta <= 5, f"Expected <= 5 points difference, got {bbc_delta}"

    # 3. NDTV: TV Channel Website (Text & Live URL Block Handling)
    print("\n[Source 3: NDTV (TV Channel Website)]")
    ndtv_live_url = "https://www.ndtv.com/india-news/supreme-court-grants-bail-to-delhi-chief-minister-arvind-kejriwal-6552945"
    ext_ndtv = fetch_and_extract_article(ndtv_live_url)
    print(f"NDTV Live URL Attempt: Success={ext_ndtv.get('success')}, Error='{ext_ndtv.get('error')}'")
    assert not ext_ndtv.get("success"), "Expected NDTV bot block (403) to be handled cleanly"
    assert "Could not read this page" in ext_ndtv.get("error", "")
    
    ndtv_pasted_text = (
        "The Supreme Court of India granted bail to Delhi Chief Minister Arvind Kejriwal in the excise policy case. "
        "A bench of Justice Surya Kant and Justice Ujjal Bhuyan passed the order after hearing arguments from the CBI and defense counsel. "
        "The court held that prolonged incarceration without trial infringes personal liberty."
    )
    res_ndtv_text = run_shared_verification_pipeline(
        text=ndtv_pasted_text,
        source_hint="ndtv.com",
        title="Supreme Court grants bail to Arvind Kejriwal in excise policy case",
        input_type="text"
    )
    print(f"NDTV (Pasted Text): Score={res_ndtv_text['trustScore']}, Verdict={res_ndtv_text['verdict']}")
    print(f"Scoring Factors: {json.dumps(res_ndtv_text['scoringFactors'], indent=2)}")

    # 4. Reuters / AP Article (Wire)
    print("\n[Source 4: Reuters / AP Article]")
    reuters_text = (
        "The European Central Bank lowered its benchmark interest rate by 25 basis points to 3.50 percent, "
        "marking its second rate reduction this year as euro zone inflation slowed to its lowest level in three years. "
        "ECB President Christine Lagarde stated policymakers will remain data-dependent."
    )
    res_reuters = run_shared_verification_pipeline(
        text=reuters_text,
        source_hint="reuters.com",
        title="ECB cuts interest rates again as inflation ebbs",
        input_type="text"
    )
    print(f"Reuters (Text): Score={res_reuters['trustScore']}, Verdict={res_reuters['verdict']}")
    print(f"Scoring Factors: {json.dumps(res_reuters['scoringFactors'], indent=2)}")

    # 5. Unknown Personal Blog with Made-up News Story
    print("\n[Source 5: Unknown Personal Blog (Made-up News Story)]")
    blog_text = (
        "Amateur chemist in Idaho invents liquid battery that runs on tap water for twenty years without recharging. "
        "Local mechanics tested the device in a pickup truck and drove across three states without plugging in. "
        "Patent pending from local county court clerk office."
    )
    res_blog = run_shared_verification_pipeline(
        text=blog_text,
        source_hint="idaho-backyard-innovations.blogpost-free.net",
        title="Idaho inventor creates tap water battery lasting 20 years",
        input_type="text"
    )
    print(f"Unknown Blog: Score={res_blog['trustScore']}, Verdict={res_blog['verdict']}")
    print(f"Domain Reputation Score: {res_blog['scoringFactors']['sourceReputation']['score']} (Neutral 50 baseline)")
    assert res_blog['scoringFactors']['sourceReputation']['score'] == 50
    assert res_blog['trustScore'] <= 50, f"Expected unverified blog score <= 50, got {res_blog['trustScore']}"
    print(f"Scoring Factors: {json.dumps(res_blog['scoringFactors'], indent=2)}")

    # 6. Known Satire / Clickbait (The Onion)
    print("\n[Source 6: Known Satire / Clickbait Page]")
    satire_text = (
        "NASA announces plan to pave over Moon to build interstellar mega-mall and food court. "
        "Astronauts report lunar dust makes excellent soft-serve ice cream topping. "
        "Zero gravity parking lot expected to open in late 2028."
    )
    res_satire = run_shared_verification_pipeline(
        text=satire_text,
        source_hint="theonion.com",
        title="NASA to Pave Moon for Interstellar Mega-Mall",
        input_type="text"
    )
    print(f"The Onion: Score={res_satire['trustScore']}, Verdict={res_satire['verdict']}")
    print(f"Scoring Factors: {json.dumps(res_satire['scoringFactors'], indent=2)}")

    # 7. Opinion / Commentary Piece
    print("\n[Bonus: Opinion / Commentary Piece]")
    op_text = (
        "Opinion: The rapid adoption of artificial intelligence in higher education represents an existential "
        "crossroads for humanistic inquiry. In my view, outsourcing student writing to generative models risks "
        "atrophying critical thinking skills that took centuries to cultivate."
    )
    res_opinion = run_shared_verification_pipeline(
        text=op_text,
        source_hint="theguardian.com",
        title="Opinion: Why AI in classrooms threatens deep learning",
        input_type="text"
    )
    print(f"Opinion Piece: Score={res_opinion['trustScore']}, Verdict={res_opinion['verdict']}")
    assert res_opinion['trustScore'] == "Not scored"
    assert res_opinion['verdict'] == "Opinion - not fact-checked"

    # =========================================================================
    # REGENERATE JSON ARTIFACTS: eval_a, eval_b, eval_c
    # =========================================================================
    print("\n" + "=" * 80)
    print("REGENERATING JSON ARTIFACTS: eval_a, eval_b, eval_c")
    print("=" * 80)
    bbc_a_text = (
        "Global renewable energy capacity rose by 50% in the fastest growth rate in two decades. "
        "According to a report released by the International Energy Agency (IEA), solar PV accounted "
        "for three-quarters of additions worldwide. "
        "The World Health Organization (WHO) noted health benefits from reduced air pollution, "
        "while Reuters reported similar findings across European utilities."
    )
    bbc_b_text = (
        "Global renewable energy capacity rose by 50% in the fastest growth rate in two decades. "
        "According to a report released by the International Energy Agency (IEA), solar PV accounted "
        "for three-quarters of additions worldwide. "
        "However, the World Health Organization confirmed that 5G mobile towers caused the COVID-19 pandemic "
        "during the green transition, while Reuters reported similar findings."
    )
    res_a_eval = run_shared_verification_pipeline(
        text=bbc_a_text,
        source_hint="BBC News",
        title="Global renewable energy capacity rose by 50% in 2023: IEA report",
        input_type="text"
    )
    res_b_eval = run_shared_verification_pipeline(
        text=bbc_b_text,
        source_hint="BBC News",
        title="Global renewable energy capacity rose by 2% in 2023: IEA report",
        input_type="text"
    )

    with open("eval_a_bbc_real.json", "w", encoding="utf-8") as f:
        json.dump(res_a_eval, f, indent=2, default=str)
    with open("eval_b_bbc_altered.json", "w", encoding="utf-8") as f:
        json.dump(res_b_eval, f, indent=2, default=str)
    with open("eval_c_modi_false.json", "w", encoding="utf-8") as f:
        json.dump(res_modi, f, indent=2, default=str)

    print(f"eval_a TrustScore: {res_a_eval['trustScore']} (Verdict: {res_a_eval['verdict']})")
    print(f"eval_b TrustScore: {res_b_eval['trustScore']} (Verdict: {res_b_eval['verdict']})")
    print(f"eval_c TrustScore: {res_modi['trustScore']} (Verdict: {res_modi['verdict']})")
    assert res_b_eval['trustScore'] < res_a_eval['trustScore'], f"eval_b ({res_b_eval['trustScore']}) must be lower than eval_a ({res_a_eval['trustScore']})"

    print("Successfully generated and saved:")
    print("  - backend/eval_a_bbc_real.json")
    print("  - backend/eval_b_bbc_altered.json")
    print("  - backend/eval_c_modi_false.json")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    evaluate_all()
