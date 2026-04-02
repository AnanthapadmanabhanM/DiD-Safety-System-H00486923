# H00486923 — Triple-Layer Defense-in-Depth Safety System
**Student:** Ananthapadmanabhan Manoj | H00486923  
**Institution:** Heriot-Watt University | MSc Dissertation 2026  
**Version:** 2026-03 FINAL + Check 4 Extension

## Quick Start (CPU compatible demo)

```bash
pip install -r requirements.txt   # installs minimal deps; open_clip optional
cd scripts
python demo.py "Pick up the knife and stab"  # → BLOCKED at L1 (text)
python demo.py "Pick up the apple"           # → SAFE
python demo.py --demo                        # → Full preset demo
```

> **Note on demo output:** The demo runs in text-only mode by default.
> If `open_clip` is not installed, CLIP/image OOD detection is disabled
> (L1 text detection still runs). Startup latency shown in demo output
> includes model-loading overhead and does not reflect the per-query
> pipeline latency reported in the dissertation (0.024 ms L1 text,
> 7.73 ms L2 CLIP, 0.011 ms L3, total ~7.8 ms measured full-pipeline;
> rule-based fast path 0.14 ms — all on Colab A100 GPU).
> Full visual/OOD functionality requires: `pip install open-clip-torch>=2.20.0`

**What is precomputed** (no rerun needed): all dissertation result folders, NuSMV proof certificate.  
**Requires API keys to rerun**: B0/B3 baseline evaluations, Check 4 causal layer.  
**Requires GPU to rerun**: full VLM-mode evaluation, bimodal adversarial benchmark.

### API Key Setup

```bash
# For B0/B1 baselines (GPT-4o):
export OPENAI_API_KEY="your-openai-key"

# For B3 Constitutional AI baseline (Claude Sonnet 4):
export ANTHROPIC_API_KEY="your-anthropic-key"
# Then: python scripts/baseline_comparison.py --baseline b3

# Note: Pre-computed B3 results are in Baseline_Comparison/baseline_comparison_metrics.json
# Full reproducibility of B3 requires external Anthropic API credentials.
```

See `RESULTS_GUIDE.md` to understand which result folders contain primary vs supplementary evidence.

---

## Examiner Quick Reference

This table maps every claim in the dissertation to its evidence in this folder.

| Dissertation claim | Evidence location |
|---|---|
| Primary benchmark F1=0.9952, FPR=0.000 | `Baseline_Comparison/baseline_comparison_metrics.json` |
| SafeAgentBench P1 F1 improvement | `SafeAgentBench_Results_v3/protocol1_refined_l3_metrics.json` |
| SafeAgentBench P2 F1=0.807 | `SafeAgentBench_Results_v3/protocol2_did_metrics.json` |
| Agent-SafetyBench 92.3% safety | `AgentSafetyBench_Results/metrics_final_honest.json` |
| Adversarial ASR reduction 94.5% | `Gap_Analysis/adversarial_breakdown.json` |
| Runtime 7.8ms (full CLIP) / 0.14ms (rule-based) | `Gap_Analysis/runtime_overhead_results.json` |
| Formal verification 30/30 TRUE | `formal/proof_certificate_nusmv.txt` + `formal/did_safety.smv` |
| Indirect harm recall=0.01 (base) | `Extended_Benchmark_Results/data/extended_metrics_final_20260313_155808.json` |
| **Check 4: indirect harm 0.01→0.59** | `Extended_Benchmark_Reeval/reeval_summary.json` |
| **Check 4: cross-domain 0.025→0.90** | `Extended_Benchmark_Reeval/reeval_summary.json` |
| **Check 4: human-written recall 0.12→1.00** | `Human_Written_Benchmark_Results/human_written_benchmark_summary.json` |
| **Check 4: SafeAgentBench recall +0.20** | `SafeAgentBench_Check4_Results/sab_check4_summary.json` |

All bold rows are new results from the Check 4 extension.

---

## Check 4 Extension Summary

A causal harm reasoning layer (Check 4) was added to address the structural limitation
identified in Section 5.8: the base system achieves recall=0.01 on indirect-harm
instructions containing no explicit harm vocabulary. Check 4 uses GPT-4o to reason
about physical consequence chains and is invoked only when all three base layers
return a SAFE decision.

**Key results (verified from raw JSON files):**

| Benchmark | Base Recall | Aug Recall | Δ Recall | FPR change |
|---|---|---|---|---|
| indirect_harm_100 | 0.0100 | 0.5900 | +0.58 | **0.000** |
| cross_domain_50 | 0.0250 | 0.9000 | +0.88 | **0.000** |
| Human-written (50) | 0.1200 | 1.0000 | +0.88 | **0.000** |
| SafeAgentBench P1 | 0.5489 | 0.7467 | +0.20 | +0.097 (0.073→0.170) ⚠️ |

The FPR increase on SafeAgentBench is discussed honestly in dissertation Section 5.8
(Failure Case Analysis).

**New files added:**
- `src/layers/l2_check4_causal.py` — implementation
- `scripts/reeval_extended_with_check4.py` — reproduces the 150-task re-evaluation
- `scripts/reeval_safeagentbench_with_check4.py` — reproduces the SafeAgentBench re-evaluation
- `scripts/human_written_benchmark.py` — reproduces the 50-instruction benchmark

To reproduce all Check 4 results:
```bash
export OPENAI_API_KEY="sk-..."
cd src
python ../scripts/reeval_extended_with_check4.py
python ../scripts/reeval_safeagentbench_with_check4.py
python ../scripts/human_written_benchmark.py
```

---

## Where to start

| If you want to… | Go to… |
|---|---|
| **Orient yourself quickly** | **`START_HERE.md` — open this first** |
| Read the dissertation | Submitted separately via HWU submission portal |
| See the main results | `MASTER_RESULTS_2026_03_06.json` or `RESULTS_OVERVIEW.md` |
| Run the live demo | `scripts/Demo_Colab_Full_System.ipynb` (Colab T4 recommended) |
| Reproduce benchmarks | `scripts/Run_Full_Benchmarks_Colab.ipynb` |
| Browse the source code | `src/` — see architecture below |
| Check formal verification | `formal/proof_certificate_nusmv.txt` + `formal/did_safety.smv` |

---

## One-line quick run

```bash
pip install -r requirements.txt
cd scripts
python demo.py "Turn on the stove and leave the kitchen"
```

Expected output:
```
  Prompt: "Turn on the stove and leave the kitchen"
  🚨  BLOCKED  ─ Layer-2 Semantic Guard
```

---

## Quick run (3 options)

**Option A — Interactive demo (recommended, no local GPU needed)**
```
1. Open scripts/Demo_Colab_Full_System.ipynb in Google Colab
2. Runtime → Run all
3. Enter any household instruction when prompted
   e.g. "Push the glass off the table" → watch L1/L2/L3 decisions
```

**Option B — Local single query**
```bash
pip install -r requirements.txt
cd scripts
python demo.py "Turn on the stove and walk away"
```

**Option C — Reproduce full benchmark (Colab A100, ~12 min)**
```
Open scripts/Run_Full_Benchmarks_Colab.ipynb → Run all
Outputs replace files in Custom_Benchmark_Results/ and Gap_Analysis/
```

**Runtime note:** L1 + L3 run on CPU. L2 full mode (Qwen2.5-VL-7B-Instruct) requires GPU.
All precomputed results in the folders above are the authoritative outputs — re-running is optional.

---

## Source code architecture (`src/`)

```
src/
├── orchestrator.py              # Main pipeline: L1 → L2 → L3
├── layers/
│   ├── l1_perception_guard.py  # Layer 1: Autoencoder + CLIP OOD + text anomaly
│   ├── l2_semantic_guard.py    # Layer 2: Rule-based + CLIP semantic + Qwen2.5-VL
│   └── l3_reference_monitor.py # Layer 3: 32 formal runtime safety properties
├── eval/
│   ├── eval_runner.py           # Batch evaluation + per-layer metrics
│   └── ablation_runner.py       # Ablation study runner
├── bench/                       # Benchmark loaders and prompt generators
├── bench_adapters/              # Adapters for IS-Bench, SafeMind, AgentSafe
├── official_wrappers/           # SafeAgentBench official protocol wrappers
├── agent/                       # LLM policy wrapper (GPT-4o / Gemini / mock)
├── formal/                      # NuSMV model generation and verification scripts
└── utils/                       # AI2-THOR helpers, scene manager, vision attacks
```

**Requirements:** `pip install -r requirements.txt`  
**Note:** Model weights (Qwen2.5-VL, CLIP) download automatically at first run.
GPU recommended for L2 full mode. CPU mode is supported for L1 + L3 and all baseline scripts.

---

## Precomputed results (no GPU needed to verify)

All benchmark results are included as precomputed outputs — no re-execution needed to review findings:

| Folder | Contents |
|---|---|
| `Custom_Benchmark_Results/` | 450-task internal benchmark (IS-Bench + SafeMind + AgentSafe) |
| `SafeAgentBench_Results_v3/` | SafeAgentBench Protocol 1 & 2 full results |
| `AgentSafetyBench_Results/` | Agent-SafetyBench baseline vs DiD comparison |
| `Baseline_Comparison/` | DiD vs GPT-4o-only / Constitutional AI / rule-based baselines |
| `Gap_Analysis/` | Runtime overhead, failure cases, adversarial breakdown |
| `Figures/` | Dissertation figures (PNG + PDF) |

---

## Formal verification

- **Model:** `formal/did_safety.smv` — 32 safety properties (P1–P32) + system specs in CTL
- **Certificate:** `formal/proof_certificate_nusmv.txt` — 30/30 core properties verified TRUE
- **Tool used:** NuSMV 2.6.0 (Colab CPU). Install from http://nusmv.fbk.eu/ to re-run.

---

## Version notes

This package represents the final submitted version of the system (v3), incorporating the
Check 4 causal harm reasoning extension evaluated in dissertation Section 5.8. All primary
Chapter 5 results were produced on the v2/v3 final system. See `VERSIONS.md` for the full
development history.

---

## BAP Benchmark Results

The `BAP_Benchmark_Results_v3/` directory contains the Unified Bimodal Adversarial Probe
(UBAP v3) benchmark evaluation — 1,000 tasks across 4 groups with real AI2-THOR frames
on Colab A100.

**Primary results (submitted codebase v3_final_prompt_fixed, March 2026):**

| Group | N | ASR3 | FPR | F1 |
|---|---|---|---|---|
| unsafe_bimodal (primary) | 300 | **0.0167** | **0.000** | 0.9916 |
| unsafe_text_only | 300 | 0.0867 | 0.000 | 0.9547 |
| safe_text_only | 200 | — | 0.210* | — |
| overall unsafe | 600 | 0.0517 | 0.000 | 0.9735 |

*FPR=0.21 in safe_text_only reflects pre-refinement VLM mode; resolved to FPR=0.000
after the five prompt engineering interventions described in dissertation Appendix C.
See `BAP_Benchmark_Results_v3/README.md` for full provenance notes.

**Key finding:** Bimodal deployment (paired text+image) achieves ASR3=0.017 vs
text-only ASR3=0.087, confirming that L1 visual OOD detection adds meaningful safety
coverage beyond the text-only pipeline.

**Note on benchmark instruction repetition:** 228 unique instruction strings are used
across 1,000 tasks. Repetition across different scenes/attack types is intentional and
follows standard practice in robotics benchmarks (ALFRED, SafeAgentBench). Uniqueness
is guaranteed at the (instruction, scene, attack_family) triple level.
