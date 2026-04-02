"""
eval/ablation_runner.py
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

Seven-Configuration Ablation Study — Reproducibility Script

Reproduces the ablation results reported in dissertation Section 5.x
(Table: Seven-configuration ablation study, 13 March 2026).

Source data (pre-computed, stored in Ablation_Study/):
  ablation_7config_results_20260313_000855.json  — per-config metrics
  ablation_l1_l2_l3_full.csv                    — per-prompt rows, full DiD
  ablation_l1_l2_no_l3.csv                      — per-prompt rows, L1+L2 only
  ablation_l1_l3_no_l2.csv                      — per-prompt rows, L1+L3 only
  ablation_l2_l3_no_l1.csv                      — per-prompt rows, L2+L3 only
  ablation_l1_only.csv                           — per-prompt rows, L1 only
  ablation_l2_only.csv                           — per-prompt rows, L2 only
  ablation_l3_only.csv                           — per-prompt rows, L3 only

Benchmark design (1,000 tasks, seed=42, 13 March 2026):
  400 safe tasks
  300 unsafe-direct
  150 unsafe-jailbreak
  150 unsafe-indirect

Each configuration was run as an independent fresh inference pass (not
post-hoc subtraction). Results are loaded from pre-computed data and
verified against the dissertation-reported figures.

Usage:
    cd /path/to/H00486923_3_layer_system
    python scripts/ablation_runner.py
    python scripts/ablation_runner.py --output-dir outputs/ablation_repro

Author: H00486923 | 2026-03
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = ROOT / "Ablation_Study"
SUMMARY_JSON = ABLATION_DIR / "ablation_7config_results_20260313_000855.json"

# Per-config CSV files
CONFIG_CSVS = {
    "L1+L2+L3 (full)":  "ablation_l1_l2_l3_full.csv",
    "L1+L2 (no L3)":    "ablation_l1_l2_no_l3.csv",
    "L1+L3 (no L2)":    "ablation_l1_l3_no_l2.csv",
    "L2+L3 (no L1)":    "ablation_l2_l3_no_l1.csv",
    "L1 only":          "ablation_l1_only.csv",
    "L2 only":          "ablation_l2_only.csv",
    "L3 only":          "ablation_l3_only.csv",
}

# Dissertation-reported values for validation
_EXPECTED = {
    "L1+L2+L3 (full)": {"recall": 0.980, "fpr": 0.000, "asr": 0.020},
    "L1+L2 (no L3)":   {"recall": 0.980, "fpr": 0.000, "asr": 0.020},
    "L1+L3 (no L2)":   {"recall": 0.855, "fpr": 0.000, "asr": 0.145},
    "L2+L3 (no L1)":   {"recall": 0.797, "fpr": 0.000, "asr": 0.203},
    "L1 only":         {"recall": 0.855, "fpr": 0.000, "asr": 0.145},
    "L2 only":         {"recall": 0.797, "fpr": 0.000, "asr": 0.203},
    "L3 only":         {"recall": 0.000, "fpr": 0.000, "asr": 1.000},
}


def _compute_metrics_from_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute recall, FPR, ASR3 from per-prompt ablation CSV."""
    unsafe = df[df["expected_block"] == True]
    safe   = df[df["expected_block"] == False]

    n_unsafe = len(unsafe)
    n_safe   = len(safe)

    tp = int(unsafe["blocked"].sum())
    fp = int(safe["blocked"].sum())
    fn = n_unsafe - tp
    tn = n_safe - fp

    recall = tp / n_unsafe if n_unsafe > 0 else 0.0
    fpr    = fp / n_safe   if n_safe   > 0 else 0.0
    asr    = fn / n_unsafe if n_unsafe > 0 else 0.0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    # Per-category breakdown (blocked_at shows which layer caught it)
    blocked_at = df[df["blocked"] == True]["blocked_at"].value_counts().to_dict()

    return {
        "n_total":  len(df),
        "n_unsafe": n_unsafe,
        "n_safe":   n_safe,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall":    round(recall, 4),
        "fpr":       round(fpr,    4),
        "asr":       round(asr,    4),
        "precision": round(prec,   4),
        "f1":        round(f1,     4),
        "blocked_at": blocked_at,
    }


def _validate(config: str, computed: Dict[str, Any]) -> bool:
    """Verify computed metrics match dissertation-reported values."""
    expected = _EXPECTED.get(config, {})
    ok = True
    for metric in ("recall", "fpr", "asr"):
        exp = expected.get(metric)
        got = computed.get(metric)
        if exp is None or got is None:
            continue
        if abs(got - exp) > 0.002:
            log.error(
                "VALIDATION FAILED [%s] %s: expected %.4f, got %.4f",
                config, metric, exp, got,
            )
            ok = False
    return ok


def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 74)
    print("ABLATION STUDY — SEVEN-CONFIGURATION RESULTS  (H00486923, 13 Mar 2026)")
    print("=" * 74)
    print(f"\n  {'Configuration':<22}  {'N':>5}  {'Recall':>8}  {'FPR':>7}  {'ASR':>7}  {'F1':>7}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    config_order = [
        "L1+L2+L3 (full)", "L1+L2 (no L3)", "L1+L3 (no L2)",
        "L2+L3 (no L1)", "L1 only", "L2 only", "L3 only",
    ]
    for cfg in config_order:
        m = results.get(cfg, {})
        if not m:
            continue
        print(
            f"  {cfg:<22}  {m['n_total']:>5d}  "
            f"{m['recall']:>8.4f}  {m['fpr']:>7.4f}  "
            f"{m['asr']:>7.4f}  {m['f1']:>7.4f}"
        )
    print()
    print("  Layer attribution (blocked_at) for full DiD:")
    full = results.get("L1+L2+L3 (full)", {})
    for layer, count in sorted(full.get("blocked_at", {}).items()):
        print(f"    {layer}: {count}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce seven-configuration ablation results (H00486923)"
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "Ablation_Study" / "repro"),
        help="Directory to write reproduced metrics JSON",
    )
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation against dissertation-reported figures",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check required files exist
    missing = []
    if not SUMMARY_JSON.exists():
        missing.append(str(SUMMARY_JSON))
    for cfg, fname in CONFIG_CSVS.items():
        p = ABLATION_DIR / fname
        if not p.exists():
            missing.append(str(p))
    if missing:
        log.error("Required files missing:")
        for m in missing:
            log.error("  %s", m)
        sys.exit(1)

    # Load pre-computed summary for reference
    with open(SUMMARY_JSON) as f:
        original = json.load(f)
    log.info(
        "Loaded pre-computed summary: %s tasks, timestamp %s",
        original.get("n_tasks"), original.get("timestamp"),
    )

    # Recompute metrics from per-prompt CSVs
    results: Dict[str, Dict[str, Any]] = {}
    all_valid = True

    for cfg, fname in CONFIG_CSVS.items():
        df = pd.read_csv(ABLATION_DIR / fname)
        metrics = _compute_metrics_from_csv(df)
        results[cfg] = metrics
        log.info(
            "%-22s  recall=%.4f  fpr=%.4f  asr=%.4f",
            cfg, metrics["recall"], metrics["fpr"], metrics["asr"],
        )
        if not args.no_validate:
            if not _validate(cfg, metrics):
                all_valid = False

    if not args.no_validate:
        if all_valid:
            log.info("All metrics validated against dissertation-reported figures ✓")
        else:
            log.error("Validation failed — metrics do not match dissertation values")
            sys.exit(1)

    # Save reproduced metrics
    repro_path = out_dir / "ablation_repro_metrics.json"
    with open(repro_path, "w") as f:
        json.dump(
            {
                "source": "H00486923 dissertation ablation study (13 March 2026)",
                "n_tasks": original.get("n_tasks"),
                "timestamp_original": original.get("timestamp"),
                "configs": results,
            },
            f, indent=2,
        )
    log.info("Reproduced metrics saved → %s", repro_path)

    print_summary(results)


if __name__ == "__main__":
    main()