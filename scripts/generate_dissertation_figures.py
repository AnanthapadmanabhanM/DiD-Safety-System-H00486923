#!/usr/bin/env python3
"""
generate_dissertation_figures.py
H00486923 | MSc Robotics Dissertation | Heriot-Watt University
Regenerates two key dissertation figures (baseline comparison and
recall/FPR trade-off) from the pre-computed result JSON files.
Requires: matplotlib.
Usage:
    python scripts/generate_dissertation_figures.py --out Figures/
"""
import argparse, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).parent.parent
def fig_baseline_comparison(out: pathlib.Path) -> None:
    with open(ROOT / "Baseline_Comparison/baseline_comparison_metrics.json") as f:
        data = json.load(f)
    systems = ["B0: GPT-4o", "B1: LLM Guardrails", "B2: SafeMind", "B3: Constitutional AI", "B4: DiD System"]
    keys = ["b0_gpt4o_alone", "b1_llm_guardrails_gpt4o", "b2_safemind_rules", "b3_constitutional_claude", "b4_did_full_prior"]
    recalls = [data[k]["recall"] for k in keys]
    fprs    = [data[k]["fpr"]    for k in keys]
    f1s     = [data[k]["f1"]     for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(systems))
    ax.bar([i - 0.25 for i in x], recalls, 0.25, label="Recall", color="#1d9e75")
    ax.bar([i        for i in x], f1s,     0.25, label="F1",     color="#378add")
    ax.bar([i + 0.25 for i in x], fprs,    0.25, label="FPR",    color="#e24b4a")
    ax.set_xticks(list(x)); ax.set_xticklabels(systems, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1); ax.legend(); ax.set_title("Baseline comparison (450-task benchmark)")
    fig.tight_layout(); fig.savefig(out / "baseline_comparison.pdf", dpi=150)
    plt.close()
    print(f"Saved baseline_comparison.pdf")
def fig_fpr_recall(out: pathlib.Path) -> None:
    with open(ROOT / "Baseline_Comparison/baseline_comparison_metrics.json") as f:
        data = json.load(f)
    keys = ["b0_gpt4o_alone", "b1_llm_guardrails_gpt4o", "b2_safemind_rules",
            "b3_constitutional_claude", "b4_did_full_prior"]
    labels = ["B0", "B1", "B2", "B3", "B4 (DiD)"]
    recalls = [data[k]["recall"] for k in keys]
    fprs    = [data[k]["fpr"]    for k in keys]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#888780", "#888780", "#888780", "#888780", "#1d9e75"]
    for recall, fpr, label, c in zip(recalls, fprs, labels, colors):
        ax.scatter(recall, fpr, color=c, s=80, zorder=3)
        ax.annotate(label, (recall, fpr), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Recall"); ax.set_ylabel("FPR")
    ax.set_title("Recall vs FPR trade-off")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out / "baseline_fpr_recall.pdf", dpi=150)
    plt.close()
    print("Saved baseline_fpr_recall.pdf")
def main(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_baseline_comparison(out)
    fig_fpr_recall(out)
    print(f"\nAll figures saved to {out}/")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "Figures"),
                        help="Output directory for figures (default: project root Figures/)")
    args = parser.parse_args()
    main(args.out)