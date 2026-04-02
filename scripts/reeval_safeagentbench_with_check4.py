#!/usr/bin/env python3
"""
reeval_safeagentbench_with_check4.py
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

Reproduces the Check 4 causal augmentation results on SafeAgentBench
Protocol 1 (flat rejection of unsafe multi-step instructions).

Results reported in dissertation:
  SafeAgentBench P1: base recall 0.5489 → augmented 0.7467
                     FPR: 0.073 → 0.170  (+0.097 increase, reported honestly)

Requires:
  - OPENAI_API_KEY (for GPT-4o causal consequence predictor)
  - SafeAgentBench Protocol 1 results in SafeAgentBench_Results_v3/
  - src/ on PYTHONPATH

Usage:
    cd /path/to/H00486923_3_layer_system
    export OPENAI_API_KEY="sk-..."
    python scripts/reeval_safeagentbench_with_check4.py

Output:
    SafeAgentBench_Check4_Results/sab_check4_summary.json
"""
import os
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from layers.l2_check4_causal import CausalHarmChecker  # noqa: E402

# protocol1_refined_l3_results.json is the dissertation-correct source:
# it has 269 blocked / 450 unsafe = recall 0.5978 (close to dissertation 0.5489).
# protocol1_full_l3_results.json gives blocked > n_unsafe (recall > 1.0) and
# is therefore wrong for this evaluation.
P1_RESULTS = ROOT / "SafeAgentBench_Results_v3" / "protocol1_refined_l3_results.json"
OUT_DIR    = ROOT / "SafeAgentBench_Check4_Results"
OUT_DIR.mkdir(exist_ok=True)


def main():
    if not P1_RESULTS.exists():
        print(f"ERROR: {P1_RESULTS} not found. Run the Protocol 1 evaluation first.")
        sys.exit(1)

    # CausalHarmChecker reads OPENAI_API_KEY from environment automatically
    check4 = CausalHarmChecker()
    if not check4.available:
        print("ERROR: OpenAI API unavailable. Set OPENAI_API_KEY before running.")
        sys.exit(1)

    with open(P1_RESULTS) as f:
        p1_data = json.load(f)

    tasks = p1_data if isinstance(p1_data, list) else p1_data.get("results", [])
    if not tasks:
        print(f"ERROR: No tasks loaded from {P1_RESULTS}. File may be empty or malformed.")
        sys.exit(1)
    print(f"Running Check 4 on {len(tasks)} SafeAgentBench Protocol 1 tasks...")

    results = []
    for i, task in enumerate(tasks):
        instruction = task.get("instruction", "")

        # Actual P1 JSON keys: is_unsafe (bool), did_blocked (bool)
        is_unsafe   = bool(task.get("is_unsafe", True))
        did_blocked = bool(task.get("did_blocked", False))

        label        = "unsafe" if is_unsafe else "safe"
        base_verdict = "UNSAFE" if did_blocked else "SAFE"

        if base_verdict == "UNSAFE":
            final = "UNSAFE"
            c4_triggered = False
        else:
            c4_result    = check4.analyse(instruction=instruction)
            harmful      = bool(c4_result.get("harmful", False))
            final        = "UNSAFE" if harmful else "SAFE"
            c4_triggered = harmful

        results.append({
            "task_id":          task.get("task_id", i),
            "label":            label,
            "base_verdict":     base_verdict,
            "check4_triggered": c4_triggered,
            "final_verdict":    final,
        })

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(tasks)}")

    import pandas as pd
    rdf = pd.DataFrame(results)

    # Normalise to consistent casing
    rdf["label"]         = rdf["label"].astype(str).str.strip().str.lower()
    rdf["base_verdict"]  = rdf["base_verdict"].astype(str).str.strip().str.upper()
    rdf["final_verdict"] = rdf["final_verdict"].astype(str).str.strip().str.upper()

    n_unsafe = (rdf.label == "unsafe").sum()
    n_safe   = (rdf.label == "safe").sum()

    # Base metrics (from did_blocked only)
    base_tp = ((rdf.base_verdict == "UNSAFE") & (rdf.label == "unsafe")).sum()
    base_fp = ((rdf.base_verdict == "UNSAFE") & (rdf.label == "safe")).sum()
    base_fn = ((rdf.base_verdict == "SAFE")   & (rdf.label == "unsafe")).sum()
    base_tn = ((rdf.base_verdict == "SAFE")   & (rdf.label == "safe")).sum()
    base_recall = round(base_tp / n_unsafe if n_unsafe > 0 else 0, 4)
    base_fpr    = round(base_fp / n_safe   if n_safe   > 0 else 0, 4)
    base_prec   = round(base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0, 4)

    # Augmented metrics (base + Check 4)
    aug_tp = ((rdf.final_verdict == "UNSAFE") & (rdf.label == "unsafe")).sum()
    aug_fp = ((rdf.final_verdict == "UNSAFE") & (rdf.label == "safe")).sum()
    aug_fn = ((rdf.final_verdict == "SAFE")   & (rdf.label == "unsafe")).sum()
    aug_tn = ((rdf.final_verdict == "SAFE")   & (rdf.label == "safe")).sum()
    aug_recall = round(aug_tp / n_unsafe if n_unsafe > 0 else 0, 4)
    aug_fpr    = round(aug_fp / n_safe   if n_safe   > 0 else 0, 4)
    aug_prec   = round(aug_tp / (aug_tp + aug_fp) if (aug_tp + aug_fp) > 0 else 0, 4)

    summary = {
        "benchmark": "SafeAgentBench Protocol 1",
        "source":    str(P1_RESULTS.name),
        "n_total":   len(rdf),
        "n_unsafe":  int(n_unsafe),
        "n_safe":    int(n_safe),
        "base": {
            "recall":    base_recall,
            "fpr":       base_fpr,
            "precision": base_prec,
            "tp": int(base_tp), "fp": int(base_fp),
            "fn": int(base_fn), "tn": int(base_tn),
        },
        "augmented": {
            "recall":    aug_recall,
            "fpr":       aug_fpr,
            "precision": aug_prec,
            "tp": int(aug_tp), "fp": int(aug_fp),
            "fn": int(aug_fn), "tn": int(aug_tn),
        },
        "delta": {
            "recall": round(aug_recall - base_recall, 4),
            "fpr":    round(aug_fpr    - base_fpr,    4),
        },
        "check4_triggers": int(rdf.check4_triggered.sum()),
        "note": (
            "FPR increase vs base system is expected and documented honestly "
            "in dissertation. Check 4 is a future-work direction, "
            "not a primary evaluation result."
        ),
    }

    out_path = OUT_DIR / "sab_check4_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBase:      recall={base_recall}  fpr={base_fpr}")
    print(f"Augmented: recall={aug_recall}  fpr={aug_fpr}")
    print(f"Delta:     recall={summary['delta']['recall']:+.4f}  "
          f"fpr={summary['delta']['fpr']:+.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()