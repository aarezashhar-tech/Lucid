"""
Evaluation Runner for Lucid 60-Row Evaluation Dataset (eval_set.csv)
Requirement 7:
- Evaluates 60 real, varied, subtle examples (20 TRUE, 20 FALSE, 20 UNVERIFIED).
- Computes overall accuracy.
- Computes and prints 3x3 confusion matrix.
- Lists every misclassified row with full diagnostics.
- Does NOT tune code on this set.
"""
import os
import sys
import csv
from collections import defaultdict

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.ai_service import VerificationEngine

def map_verdict_to_label(verdict: str) -> str:
    """Maps Lucid system verdict to standard benchmark evaluation label."""
    v_upper = str(verdict).upper()
    if "FALSE" in v_upper:
        return "FALSE"
    if "VERIFIED" in v_upper or "TRUE" in v_upper:
        if "UNVERIFIED" in v_upper:
            return "UNVERIFIED"
        return "TRUE"
    if "NEEDS REVIEW" in v_upper:
        return "UNVERIFIED"
    if "OPINION" in v_upper:
        return "UNVERIFIED"
    if "SATIRE" in v_upper:
        return "FALSE"
    return "UNVERIFIED"

def run_evaluation():
    csv_path = os.path.join(backend_dir, "eval_set.csv")
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("=" * 80)
    print(f"RUNNING EVALUATION ON {len(rows)} DATASET ROWS (backend/eval_set.csv)")
    print("=" * 80)

    engine = VerificationEngine()
    results = []
    correct_count = 0
    confusion = defaultdict(lambda: defaultdict(int))
    wrong_rows = []

    for idx, row in enumerate(rows, 1):
        item_id = row.get("id", str(idx))
        text = row.get("text", "")
        expected = row.get("expected_label", "").strip()
        notes = row.get("notes", "")

        analysis = engine.analyze(text=text)
        system_verdict = analysis.get("verdict", "")
        trust_score = analysis.get("trustScore", 0)
        predicted = map_verdict_to_label(system_verdict)

        is_correct = (predicted == expected)
        if is_correct:
            correct_count += 1
        else:
            wrong_rows.append({
                "id": item_id,
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "system_verdict": system_verdict,
                "trust_score": trust_score,
                "notes": notes,
                "supporting_count": len(analysis.get("evidence", {}).get("supporting", [])),
                "contradicting_count": len(analysis.get("evidence", {}).get("contradicting", []))
            })

        confusion[expected][predicted] += 1
        print(f"[{item_id}/60] Expected: {expected:<10} | Predicted: {predicted:<10} ({system_verdict}, {trust_score}) | {'✓' if is_correct else '✗'} | {text[:55]}...")

    accuracy = (correct_count / len(rows)) * 100.0

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated: {len(rows)}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Overall Accuracy: {accuracy:.1f}%\n")

    print("CONFUSION MATRIX:")
    labels = ["TRUE", "FALSE", "UNVERIFIED"]
    col_width = 14
    header = f"{'Actual \\ Pred':<{col_width}}" + "".join(f"{lbl:>{col_width}}" for lbl in labels) + f"{'Total':>{col_width}}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for actual in labels:
        row_str = f"{actual:<{col_width}}"
        row_total = 0
        for pred in labels:
            val = confusion[actual][pred]
            row_total += val
            row_str += f"{val:>{col_width}}"
        row_str += f"{row_total:>{col_width}}"
        print(row_str)
    print("-" * len(header))

    print(f"\nMISCLASSIFIED ROWS ({len(wrong_rows)} total):")
    if not wrong_rows:
        print("None! All rows correctly classified.")
    else:
        for w in wrong_rows:
            print(f"  • Row #{w['id']}: Expected [{w['expected']}], Predicted [{w['predicted']}] (Verdict: '{w['system_verdict']}', Score: {w['trust_score']})")
            print(f"    Claim: \"{w['text']}\"")
            print(f"    Evidence: {w['supporting_count']} supporting, {w['contradicting_count']} contradicting")
            print(f"    Notes: {w['notes']}\n")

if __name__ == "__main__":
    run_evaluation()
