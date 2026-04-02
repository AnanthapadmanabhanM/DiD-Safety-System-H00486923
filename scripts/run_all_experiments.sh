#!/bin/bash
# ============================================================
# run_all_experiments.sh
# Triple-Layer DiD Safety System — Experiment Pipeline
# H00486923 — Heriot-Watt University MSc Dissertation, 2026
#
# USAGE (from project root):
#   bash scripts/run_all_experiments.sh
#
# REQUIREMENTS:
#   pip install -r scripts/requirements.txt
#   Set OPENAI_API_KEY for B0/B1 baselines (GPT-4o)
#   Set ANTHROPIC_API_KEY for B3 Constitutional AI (Claude Sonnet 4)
#   Note: B2 SafeMind baseline is deterministic and requires no API key
#   Pre-computed results are in Baseline_Comparison/ — re-running requires API keys
#
# For GPU-dependent evaluations (SAB, bimodal BAP), use:
#   scripts/Run_Full_Benchmarks_Colab.ipynb  (requires A100 GPU)
# ============================================================
set -e

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  DiD Safety System — Experiment Pipeline"
echo "  H00486923 — Heriot-Watt University, 2026"
echo "============================================================"
echo ""

# ── STEP 1: Baseline comparison (B0–B4, 450-task benchmark) ───────
echo "[1/3] Running baseline comparison B0-B4 (450 tasks)..."
echo "      B0/B1 require OPENAI_API_KEY (GPT-4o)."
echo "      B2 is deterministic (no API key required)."
echo "      B3 requires ANTHROPIC_API_KEY."
echo "      Without Anthropic key, B3 uses SafeMind heuristic fallback."
echo "      Pre-computed results are in Baseline_Comparison/ if keys unavailable."
python "$SCRIPTS_DIR/baseline_comparison.py"
echo "      Done. Results: Baseline_Comparison/"
echo ""

# ── STEP 2: Extended human-written benchmark ───────────────────────
echo "[2/3] Running human-written benchmark (50 tasks, Check 4)..."
echo "      Set OPENAI_API_KEY for Check 4 causal layer."
python "$SCRIPTS_DIR/human_written_benchmark.py"
echo "      Done. Results: Human_Written_Benchmark_Results/"
echo ""

# ── STEP 3: Demo suite (text-only, no GPU required) ───────────────
echo "[3/3] Running demo suite (rule-based mode, CPU only)..."
python "$SCRIPTS_DIR/demo.py" --demo
echo "      Done."
echo ""

echo "============================================================"
echo "  NON-GPU EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "  Key pre-computed results (from dissertation evaluation):"
echo "    Primary benchmark F1:       0.9952"
echo "    Primary benchmark FPR:      0.000"
echo "    Adversarial ASR reduction:  94.5%"
echo "    Formal verification PCR:    1.000 (30/30 core, NuSMV 2.6.0)"
echo "    Full pipeline latency:      7.8 ms (CLIP path, A100 GPU)"
echo "    Rule-based latency:         0.14 ms (CPU)"
echo ""
echo "  For GPU evaluations (SAB, bimodal BAP, ablation):"
echo "    scripts/Run_Full_Benchmarks_Colab.ipynb  (requires A100 GPU)"
echo "============================================================"