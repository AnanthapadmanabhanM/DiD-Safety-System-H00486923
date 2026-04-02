#!/usr/bin/env python3
"""
gap3_adversarial_breakdown_v2.py — Adversarial robustness summary (RQ3)
H00486923 | MSc Robotics Dissertation | Heriot-Watt University
Loads the pre-computed adversarial evaluation results from
Gap_Analysis/adversarial_breakdown.json and prints a formatted summary
matching the adversarial results table in dissertation Section 5.5.
The underlying evaluation was conducted using bimodal_benchmark_results.csv
(February 2026). This script summarises those pre-computed results.
Usage:
    python scripts/gap3_adversarial_breakdown_v2.py
"""
import json, pathlib
ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "Gap_Analysis" / "adversarial_breakdown.json"
def main() -> None:
    with open(DATA) as f:
        d = json.load(f)
    print("\n=== Adversarial Robustness Results (RQ3) ===")
    print(f"{'Attack Family':<22} {'Variants':>8} {'Baseline ASR':>14} {'DiD ASR':>10} {'Reduction':>12}")
    print("-" * 70)
    for family, stats in d["family_summary"].items():
        print(f"{family:<22} {stats['n_variants']:>8} "
              f"{stats['baseline_mean_asr']:>14.1%} "
              f"{stats['did_mean_asr']:>10.1%} "
              f"{stats['reduction_pct']:>11.1f}%")
    overall = d["overall"]
    print("-" * 70)
    print(f"{'OVERALL':<22} {overall['total_variants']:>8} "
          f"{overall['mean_baseline_asr']:>14.1%} "
          f"{overall['mean_did_asr']:>10.1%} "
          f"{overall['overall_reduction_pct']:>11.1f}%")
    print(f"\nProvenance: {d.get('provenance', 'See Gap_Analysis/adversarial_breakdown.json')[:120]}")
if __name__ == "__main__":
    main()