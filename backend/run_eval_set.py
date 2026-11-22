"""
Evaluation Runner for Lucid Evaluation Datasets (eval_set.csv and eval_holdout.csv)

Requirements:
- Evaluates real, varied, subtle examples (TRUE, FALSE, UNVERIFIED).
- Computes overall accuracy.
- Computes and prints 3x3 confusion matrix.
- Computes per-class Precision, Recall, and F1-score:
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
- Computes wrong-direction error rates:
    TRUE -> FALSE (Critical false-negative error)
    FALSE -> TRUE (Critical false-positive error)
- Supports running eval_set.csv and eval_holdout.csv separately or together.
- Lists every misclassified row with full diagnostics.
- Does NOT tune code on evaluation rows.
"""
import os
import sys
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Any

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

def evaluate_file(csv_path: str, engine: VerificationEngine = None) -> Dict[str, Any]:
    """Runs verification evaluation on a specified CSV file and returns detailed stats."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation file not found: {csv_path}")

    if engine is None:
        engine = VerificationEngine()

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dataset_name = os.path.basename(csv_path)
    print("=" * 80)
    print(f"RUNNING EVALUATION ON {len(rows)} ROWS: {dataset_name}")
    print("=" * 80)

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
        print(f"[{item_id}/{len(rows)}] Expected: {expected:<10} | Predicted: {predicted:<10} ({system_verdict}, {trust_score}) | {'✓' if is_correct else '✗'} | {text[:55]}...")

    accuracy = (correct_count / len(rows) * 100.0) if rows else 0.0

    # Print summary
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {dataset_name}")
    print("=" * 80)
    print(f"Total Evaluated:     {len(rows)}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Overall Accuracy:    {accuracy:.2f}%\n")

    # Confusion matrix
    labels = ["TRUE", "FALSE", "UNVERIFIED"]
    col_width = 14
    header = f"{'Actual \\ Pred':<{col_width}}" + "".join(f"{lbl:>{col_width}}" for lbl in labels) + f"{'Total':>{col_width}}"
    print("CONFUSION MATRIX:")
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

    # Per-Class Metrics: Precision, Recall, F1
    print("\nPER-CLASS METRICS (Precision & Recall):")
    p_header = f"{'Class':<14}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Support':>12}"
    print("-" * len(p_header))
    print(p_header)
    print("-" * len(p_header))
    per_class = {}
    for cls in labels:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in labels if other != cls)
        fn = sum(confusion[cls][other] for other in labels if other != cls)
        support = tp + fn

        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": precision, "recall": recall, "f1": f1
        }
        print(f"{cls:<14}{precision:>11.2f}%{recall:>11.2f}%{f1:>11.2f}%{support:>12}")
    print("-" * len(p_header))

    # Wrong-Direction Error Rates (TRUE -> FALSE, FALSE -> TRUE)
    true_to_false = confusion["TRUE"]["FALSE"]
    false_to_true = confusion["FALSE"]["TRUE"]
    total_true = sum(confusion["TRUE"][k] for k in labels)
    total_false = sum(confusion["FALSE"][k] for k in labels)

    rate_true_to_false = (true_to_false / total_true * 100.0) if total_true > 0 else 0.0
    rate_false_to_true = (false_to_true / total_false * 100.0) if total_false > 0 else 0.0

    print("\nWRONG-DIRECTION ERROR ANALYSIS (Critical Safety Rates):")
    print("-" * 65)
    print(f"• TRUE -> FALSE (Actual True called False):  {true_to_false}/{total_true} ({rate_true_to_false:.2f}%)")
    print(f"• FALSE -> TRUE (Actual False called True):  {false_to_true}/{total_false} ({rate_false_to_true:.2f}%)")
    print(f"• Overall Wrong-Direction Error Rate:        {true_to_false + false_to_true}/{len(rows)} ({((true_to_false + false_to_true)/len(rows)*100.0) if rows else 0.0:.2f}%)")
    print("-" * 65)

    # Misclassified Rows
    print(f"\nMISCLASSIFIED ROWS ({len(wrong_rows)} total):")
    if not wrong_rows:
        print("None! All rows correctly classified.")
    else:
        for w in wrong_rows:
            print(f"  • Row #{w['id']}: Expected [{w['expected']}], Predicted [{w['predicted']}] (Verdict: '{w['system_verdict']}', Score: {w['trust_score']})")
            print(f"    Claim: \"{w['text']}\"")
            print(f"    Evidence: {w['supporting_count']} supporting, {w['contradicting_count']} contradicting")
            print(f"    Notes: {w['notes']}\n")

    return {
        "dataset": dataset_name,
        "rows": len(rows),
        "accuracy": accuracy,
        "confusion": dict(confusion),
        "per_class": per_class,
        "true_to_false": true_to_false,
        "false_to_true": false_to_true,
        "rate_true_to_false": rate_true_to_false,
        "rate_false_to_true": rate_false_to_true,
        "wrong_rows": wrong_rows
    }

def main():
    parser = argparse.ArgumentParser(description="Lucid Fact-Checking Evaluation Runner")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to evaluation CSV file (e.g., eval_set.csv or eval_holdout.csv)")
    parser.add_argument("--holdout", action="store_true", help="Run holdout dataset eval_holdout.csv specifically")
    args = parser.parse_args()

    engine = VerificationEngine()

    eval_set_path = os.path.join(backend_dir, "eval_set.csv")
    eval_holdout_path = os.path.join(backend_dir, "eval_holdout.csv")

    if args.csv_path:
        # Run explicitly requested file
        evaluate_file(args.csv_path, engine=engine)
    elif args.holdout:
        if not os.path.exists(eval_holdout_path):
            print(f"Holdout file not found: {eval_holdout_path}. Please create backend/eval_holdout.csv with 40 rows.")
            sys.exit(1)
        evaluate_file(eval_holdout_path, engine=engine)
    else:
        # Default behavior: run eval_set.csv and, if eval_holdout.csv exists, run that too!
        if os.path.exists(eval_set_path):
            evaluate_file(eval_set_path, engine=engine)
        else:
            print(f"eval_set.csv not found at {eval_set_path}")

        if os.path.exists(eval_holdout_path):
            print("\n" + "#" * 80)
            print("FOUND HOLD-OUT SET: Running eval_holdout.csv separately...")
            print("#" * 80 + "\n")
            evaluate_file(eval_holdout_path, engine=engine)
        else:
            print("\n[INFO] eval_holdout.csv not yet present. When added to backend/eval_holdout.csv, it will be evaluated automatically.")

if __name__ == "__main__":
    main()
