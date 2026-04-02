# System Version History — H00486923 DiD Safety System

## Overview

The DiD safety system was developed iteratively across three versions (v1 → v2 → v3).
The submitted source code represents **v3 (final)**, which is the version evaluated in
the dissertation. Quantitative results for all versions are preserved in the
`Ablation_Study/` and `BAP_Benchmark_5k_Final/` directories within this submission.

**Note on benchmark terminology used in this document:**
The v3 metrics table below reports results from the February 2026 internal 5,000-task
text-only benchmark (`BAP_Benchmark_5k_Final/benchmark_results_full_5000.csv`).
The primary dissertation evaluation (Chapter 5) uses the March 2026 450-task
custom benchmark and the UBAP v3 bimodal benchmark, which produce different figures.
Both sets of results are consistent and reported in their respective contexts.

---

## Version Summary

### v1 — Baseline (February 2026, superseded)

**Components:** L1 keyword detection (basic) + L2 LLaVA VLM + L3 monitor

| Metric | Value |
|---|---|
| Accuracy | 61.48% |
| Precision | 0.6095 |
| Recall | 0.9963 |
| F1 | 0.7563 |
| FPR | 0.9575 |
| FP | 1915 |
| FN | 11 |

**Outcome:** Catastrophic false positive rate (95.75%) — safe task accuracy only 4.2%.
The LLaVA backend flagged nearly all safe tasks as unsafe. v1 source code was
superseded during development once this failure mode was identified.

**Rationale for exclusion of v1 source code:**
v1 was an iterative Colab notebook prototype. Once the catastrophic FPR was diagnosed,
the architecture was redesigned rather than patched. The evaluation results are
preserved in the ablation study files.

---

### v2 — Rule Fixes (February 2026, superseded)

**Components:** v1 + L1 keyword expansion + L2 unsafe-first override + CLIP OOD disabled

| Metric | Value |
|---|---|
| Accuracy | 99.68% |
| Precision | 1.0000 |
| Recall | 0.9947 |
| F1 | 0.9973 |
| FPR | 0.0000 |
| FP | 0 |
| FN | 16 |

**Outcome:** FPR resolved. 16 remaining FN were in `unsafe_indirect` category
(euphemistic threat framing). v2 source was superseded when further semantic
improvements were integrated to address indirect framing misses.

**Rationale for exclusion of v2 source code:**
v2 existed as a patched version of v1 in the same Colab environment. The codebase
was refactored into the modular `src/layers/` architecture for v3, making v2
effectively replaced rather than branched.

---

### v3 — Final Submitted Version (March 2026)

**The primary dissertation result is ASR3 = 0.017 from the UBAP v3 bimodal benchmark,
which represents the deployed system configuration.**

**Components:** Final three-layer Defense-in-Depth system:
- Layer 1 Perception Guard (`src/layers/l1_perception_guard.py`)
- Layer 2 Semantic Guard (`src/layers/l2_semantic_guard.py`)
- Layer 3 Reference Monitor (`src/layers/l3_reference_monitor.py`)
- Orchestrator and evaluation pipeline (`src/orchestrator.py`)

#### Internal 5,000-task text-only benchmark (February 2026, VLM-mode)

*Source: `BAP_Benchmark_5k_Final/asr_metrics_official.json`*
*Note: This benchmark runs without paired scene images (L1 visual OOD inactive).
ASR2=0.1139 reflects the text-only operating boundary, not the primary deployment mode.
The primary bimodal ASR2=0.02 (UBAP v3, March 2026) is reported in Chapter 5.*

| Metric | Value | F21RP Target | Status |
|---|---|---|---|
| FPR | 0.0000 | ≤0.05 | ✅ PASS |
| ASR1 | 0.1881 | ≤0.30 | ✅ PASS |
| ASR2 | 0.1139 | ≤0.10 | ⚠️ text-only boundary |
| ASR3 | 0.0214 (text-only pipeline, L1 visual OOD inactive) | <0.10 | ✅ PASS |

#### Primary evaluation (March 2026, 450-task custom benchmark)

*Source: `Baseline_Comparison/baseline_comparison_metrics.json` → `b4_did_full_prior`*
*This is the primary benchmark reported in dissertation Chapter 5.*

| Metric | Value |
|---|---|
| Recall | 0.9905 |
| Precision | 1.0000 |
| F1 | 0.9952 |
| FPR | 0.0000 |
| ASR2 (bimodal) | 0.02 |
| ASR3 (bimodal) | 0.017 |

#### Per-benchmark breakdown (v3 final, 450-task)

*Source: `Custom_Benchmark_Results/metrics_all.json`*

| Benchmark | DiD Safety | Baseline Safety | Δ |
|---|---|---|---|
| IS-Bench | 99.3% | 95.3% | +4.0% |
| AgentSafe | 99.3% | 96.7% | +2.6% |
| SafeMind | 99.3% | 90.0% | +9.3% |

**Note on system evolution:**
The system underwent several intermediate prototype iterations before being consolidated
into the final modular Layer-1/Layer-2/Layer-3 implementation submitted here. The
preserved evaluation results document the performance progression across development
stages. The submitted source code is the authoritative final implementation evaluated
in the dissertation.

---

## Evidence References

| Evidence | Location in submission |
|---|---|
| Ablation study (7-config, 1,000 tasks) | `Ablation_Study/ablation_l1_l2_l3_full.csv` (and per-config CSVs) |
| Ablation study JSON summary | `Ablation_Study/ablation_7config_results_20260313_000855.json` |
| v3 5k text-only benchmark raw results | `BAP_Benchmark_5k_Final/benchmark_results_full_5000.csv` |
| v3 5k ASR metrics | `BAP_Benchmark_5k_Final/asr_metrics_official.json` |
| Primary 450-task baseline comparison | `Baseline_Comparison/baseline_comparison_results.csv` |
| Primary 450-task metrics (B0–B4) | `Baseline_Comparison/baseline_comparison_metrics.json` |
| Per-benchmark breakdown | `Custom_Benchmark_Results/metrics_all.json` |
| UBAP v3 bimodal benchmark results | `BAP_Benchmark_Results_v3/unified_bap_results_v3.csv` |
| Bimodal adversarial benchmark | `Bimodal_Benchmark_Results/bimodal_benchmark_results.csv` |
| Formal verification (NuSMV model) | `formal/did_safety.smv` |
| Formal verification (proof certificate) | `formal/proof_certificate_nusmv.txt` |
| Master results JSON | `MASTER_RESULTS_2026_03_06.json` |
| Dissertation discussion | Chapter 5 (Evaluation), Appendix F (Supplementary Evaluations) |

All v3 results reported in this document are derived from executed evaluation
pipelines and are reproducible from the provided artefacts.

---

*H00486923 — MSc Dissertation Submission, Heriot-Watt University, March 2026*