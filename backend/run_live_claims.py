"""
Script to run two live claims through VerificationEngine.analyze():
1. Truly disputed claim (where recent reputable sources disagree, landing on NEEDS REVIEW)
2. Health claim
Displays verdict, trust score, and real source URLs.
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from services.ai_service import get_verification_engine

def run_claims():
    engine = get_verification_engine()

    # Claim 1: Disputed claim where reputable sources have conflicting stances
    # Example: "Federal Reserve interest rate cuts expected in 2024" or "Coffee consumption increases cardiovascular disease risk"
    # Or "Does intermittent fasting cause heart disease" or "Electric vehicles produce more lifecycle emissions than gas cars"
    claims_to_test = [
        {
            "category": "Disputed / Conflicted Claim (NEEDS REVIEW target)",
            "text": "Electric vehicles produce higher lifetime carbon emissions than gasoline powered vehicles."
        },
        {
            "category": "Health Claim",
            "text": "Drinking green tea daily reduces the risk of heart disease and stroke."
        }
    ]

    for item in claims_to_test:
        category = item["category"]
        text = item["text"]
        print("=" * 80)
        print(f"RUNNING LIVE VERIFICATION: [{category}]")
        print(f"Claim: \"{text}\"")
        print("=" * 80)

        try:
            res = engine.analyze(text=text)
            verdict = res.get("verdict")
            score = res.get("trustScore")
            risk = res.get("riskLevel")
            reason = res.get("summaryReason")
            classifier_status = res.get("stanceClassifierStatus")

            print(f"\n>>> VERDICT: {verdict}")
            print(f">>> TRUST SCORE: {score}/100")
            print(f">>> RISK LEVEL: {risk}")
            print(f">>> STANCE CLASSIFIER STATUS: {classifier_status}")
            print(f">>> SUMMARY REASON: {reason}\n")

            supp_sources = res.get("topSupportingSources", [])
            contra_sources = res.get("topContradictingSources", [])

            print(f"--- SUPPORTING SOURCES ({len(supp_sources)}) ---")
            for idx, s in enumerate(supp_sources[:5], 1):
                print(f"  {idx}. [{s.get('publisher', 'Unknown')}] {s.get('articleTitle', 'No Title')}")
                print(f"     URL: {s.get('url', 'N/A')}")
                print(f"     Domain: {s.get('publisherDomain', 'N/A')} | Date: {s.get('publicationDate', 'N/A')} | Status: {s.get('temporalStatus', 'N/A')}")
                print(f"     Stance: {s.get('stance')} (Confidence: {s.get('stanceConfidence', s.get('agreementScore', 'N/A'))}%)")

            print(f"\n--- CONTRADICTING SOURCES ({len(contra_sources)}) ---")
            for idx, c in enumerate(contra_sources[:5], 1):
                print(f"  {idx}. [{c.get('publisher', 'Unknown')}] {c.get('articleTitle', 'No Title')}")
                print(f"     URL: {c.get('url', 'N/A')}")
                print(f"     Domain: {c.get('publisherDomain', 'N/A')} | Date: {c.get('publicationDate', 'N/A')} | Status: {c.get('temporalStatus', 'N/A')}")
                print(f"     Stance: {c.get('stance')} (Confidence: {c.get('stanceConfidence', c.get('agreementScore', 'N/A'))}%)")

        except Exception as e:
            print(f"ERROR running claim: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_claims()
