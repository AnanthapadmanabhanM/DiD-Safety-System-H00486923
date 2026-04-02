#!/usr/bin/env python3
"""
reeval_extended_with_check4.py
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

Reproduces the Check 4 causal augmentation results on the 150-task
extended boundary benchmark (100 indirect-harm + 50 cross-domain tasks).

Results reported in dissertation:
  indirect_harm_100: base recall 0.010 → augmented 0.590, FPR 0.000
  cross_domain_50:   base recall 0.025 → augmented 0.900, FPR 0.000

Requires:
  - OPENAI_API_KEY (for GPT-4o causal consequence predictor)
  - Extended benchmark CSVs in Extended_Benchmark_Results/data/
  - src/ on PYTHONPATH

Usage:
    cd /path/to/H00486923_3_layer_system
    export OPENAI_API_KEY="sk-..."
    python scripts/reeval_extended_with_check4.py

Output:
    Extended_Benchmark_Reeval/reeval_results.json
    Extended_Benchmark_Reeval/reeval_summary.json
"""

import os
import sys
import json
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from layers.l2_check4_causal import CausalHarmChecker  # noqa: E402

INDIRECT_CSV = ROOT / "Extended_Benchmark_Results" / "data" / "indirect_harm_results_20260313_155808.csv"
CROSS_CSV    = ROOT / "Extended_Benchmark_Results" / "data" / "cross_domain_results_20260313_155808.csv"
OUT_DIR      = ROOT / "Extended_Benchmark_Reeval"
OUT_DIR.mkdir(exist_ok=True)


def run_check4_on_csv(csv_path: pathlib.Path, label: str, check4: CausalHarmChecker) -> dict:
    df = pd.read_csv(csv_path)

    # Map actual CSV columns to expected fields
    # Actual columns: prompt, is_unsafe, Outcome, l1_ok, l2_decision, etc.
    if "instruction" not in df.columns and "prompt" in df.columns:
        df = df.rename(columns={"prompt": "instruction"})
    if "label" not in df.columns and "is_unsafe" in df.columns:
        df["label"] = df["is_unsafe"].apply(lambda x: "unsafe" if x else "safe")
    if "base_verdict" not in df.columns and "Outcome" in df.columns:
        # Outcome: EXEC_OK → base said SAFE; BLOCKED/REFUSED → base said UNSAFE
        df["base_verdict"] = df["Outcome"].apply(
            lambda x: "SAFE" if "EXEC" in str(x).upper() else "UNSAFE"
        )

    required = {"instruction", "label", "base_verdict"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path}: missing columns {required - set(df.columns)}")
    # Normalise to consistent casing regardless of CSV source
    df["label"]        = df["label"].astype(str).str.strip().str.lower()
    df["base_verdict"] = df["base_verdict"].astype(str).str.strip().str.upper()

    results = []
    for _, row in df.iterrows():
        base = row["base_verdict"]
        if base == "UNSAFE":
            final = "UNSAFE"
            check4_triggered = False
        else:
            c4_result = check4.analyse(instruction=row["instruction"])
            harmful = bool(c4_result.get("harmful", False))
            final = "UNSAFE" if harmful else "SAFE"
            check4_triggered = harmful
        results.append({
            "instruction":    row["instruction"],
            "label":          row["label"],
            "base_verdict":   base,
            "check4_triggered": check4_triggered,
            "final_verdict":  final,
        })

    rdf = pd.DataFrame(results)
    tp = ((rdf.final_verdict == "UNSAFE") & (rdf.label == "unsafe")).sum()
    fp = ((rdf.final_verdict == "UNSAFE") & (rdf.label == "safe")).sum()
    fn = ((rdf.final_verdict == "SAFE")   & (rdf.label == "unsafe")).sum()
    tn = ((rdf.final_verdict == "SAFE")   & (rdf.label == "safe")).sum()
    n_unsafe = (rdf.label == "unsafe").sum()
    n_safe   = (rdf.label == "safe").sum()
    recall   = tp / n_unsafe if n_unsafe > 0 else 0.0
    fpr      = fp / n_safe   if n_safe   > 0 else 0.0

    # Base metrics (before Check 4)
    base_tp = ((rdf.base_verdict == "UNSAFE") & (rdf.label == "unsafe")).sum()
    base_fp = ((rdf.base_verdict == "UNSAFE") & (rdf.label == "safe")).sum()
    base_recall = base_tp / n_unsafe if n_unsafe > 0 else 0.0
    base_fpr    = base_fp / n_safe   if n_safe   > 0 else 0.0

    print(f"\n{label}:")
    print(f"  Base:      recall={base_recall:.3f}  FPR={base_fpr:.3f}")
    print(f"  Augmented: recall={recall:.3f}  FPR={fpr:.3f}")

    return {
        "label": label,
        "n_total": len(rdf),
        "n_unsafe": int(n_unsafe),
        "n_safe": int(n_safe),
        "base": {
            "recall": round(base_recall, 4),
            "fpr":    round(base_fpr, 4),
            "tp": int(base_tp), "fp": int(base_fp),
        },
        "augmented": {
            "recall": round(recall, 4),
            "fpr":    round(fpr, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
        "check4_catches": int(rdf["check4_triggered"].sum()),
        "records": results,
    }


def main():
    if not INDIRECT_CSV.exists():
        print(f"ERROR: {INDIRECT_CSV} not found.")
        sys.exit(1)
    if not CROSS_CSV.exists():
        print(f"ERROR: {CROSS_CSV} not found.")
        sys.exit(1)

    # CausalHarmChecker reads OPENAI_API_KEY from environment automatically
    check4 = CausalHarmChecker()
    if not check4.available:
        print("ERROR: OpenAI API unavailable. Set OPENAI_API_KEY and ensure openai is installed.")
        sys.exit(1)

    print("Running Check 4 re-evaluation on extended boundary benchmark...")

    ih_result = run_check4_on_csv(INDIRECT_CSV, "indirect_harm_100", check4)
    cd_result = run_check4_on_csv(CROSS_CSV,    "cross_domain_50",   check4)

    # Save per-instruction results
    all_records = ih_result.pop("records") + cd_result.pop("records")
    with open(OUT_DIR / "reeval_results.json", "w") as f:
        json.dump(all_records, f, indent=2)

    # Save summary
    summary = {
        "indirect_harm_100": ih_result,
        "cross_domain_50":   cd_result,
    }
    with open(OUT_DIR / "reeval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/")
    print(f"Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()