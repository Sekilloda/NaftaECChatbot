import os
import sys
import json
import re
import time
from datetime import datetime
from dotenv import load_dotenv

# Ensure we can import from core
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

from core.ocr import process_receipt_image

DATASET_DIR = os.path.join(BASE_DIR, "OCRTestDataSet", "dataset")
RECEIPTS_DIR = os.path.join(BASE_DIR, "OCRTestDataSet", "receipts")

def normalize_value(val):
    if val is None: return ""
    return str(val).strip().lower()

def normalize_date(date_str):
    if not date_str: return ""
    # Dataset is often YYYY-MM-DD, Bot is DD/MM/YYYY
    try:
        if '-' in date_str:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
    except:
        pass
    return str(date_str).strip()

def run_benchmark():
    if not os.path.exists(RECEIPTS_DIR):
        print(f"Error: Receipts directory not found at {RECEIPTS_DIR}")
        return

    results = []
    receipt_files = [f for f in os.listdir(RECEIPTS_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    receipt_files.sort()

    print(f"Starting OCR Benchmark on {len(receipt_files)} receipts...")
    
    metrics = {
        "banco": {"correct": 0, "total": 0},
        "monto": {"correct": 0, "total": 0},
        "fecha": {"correct": 0, "total": 0},
        "numero_comprobante": {"correct": 0, "total": 0},
        "cuenta_origen": {"correct": 0, "total": 0}
    }

    for filename in receipt_files:
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(DATASET_DIR, f"{base_name}.json")
        
        if not os.path.exists(json_path):
            print(f"Skipping {filename}: No annotation found.")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            expected = json.load(f)

        image_path = os.path.join(RECEIPTS_DIR, filename)
        print(f"Processing {filename}...", end=" ", flush=True)

        actual = process_receipt_image(image_path, filename)

        if actual:
            print("Done.")
        else:
            print("FAILED OCR.")
            actual = {}

        comparison = {
            "banco": (normalize_value(expected.get("banco_origen")), normalize_value(actual.get("banco"))),
            "monto": (normalize_value(expected.get("monto")), normalize_value(actual.get("monto"))),
            "fecha": (normalize_date(expected.get("fecha")), normalize_date(actual.get("fecha"))),
            "numero_comprobante": (normalize_value(expected.get("numero_comprobante")), normalize_value(actual.get("numero_comprobante"))),
            "cuenta_origen": (normalize_value(expected.get("cuenta_origen")), normalize_value(actual.get("cuenta_origen")))
        }

        match_row = {"filename": filename}
        for field, (exp_val, act_val) in comparison.items():
            is_correct = False
            if not exp_val and not act_val:
                is_correct = True
            elif exp_val and act_val:
                # Partial match for banco or exact for others
                if field == "banco":
                    is_correct = exp_val in act_val or act_val in exp_val
                else:
                    is_correct = exp_val == act_val

            metrics[field]["total"] += 1
            if is_correct:
                metrics[field]["correct"] += 1

            match_row[field] = "PASS" if is_correct else f"FAIL (Exp: {exp_val}, Act: {act_val})"

        results.append(match_row)

        # Delay between requests to reduce upstream rate limiting pressure.
        time.sleep(15)

    print("\n" + "="*50)
    print("OCR ACCURACY REPORT")
    print("="*50)
    for field, data in metrics.items():
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"{field.replace('_', ' ').title():<20}: {acc:6.2f}% ({data['correct']}/{data['total']})")
    print("="*50)

    # Save detailed results to a file
    with open(os.path.join(BASE_DIR, "ocr_benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2)
    print(f"Detailed results saved to {os.path.join(BASE_DIR, 'ocr_benchmark_results.json')}")

if __name__ == "__main__":
    run_benchmark()
