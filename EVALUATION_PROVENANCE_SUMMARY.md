# Evaluation Provenance Summary
## H00486923 — Triple-Layer Defense-in-Depth Safety System

This document records exactly which codebase version was used for each evaluation.

---

## Version Definitions

| Label | Description |
|---|---|
| **Feb 23 prototype** | Codebase as of 2026-02-23. L1: ViT-L-14 + TinyConvAE, L2: rule-based only, L3: approximately P1–P20 (historical; exact count not retained) |
| **Current code** | Submitted codebase. L1: ViT-L-14 + TinyConvAE, L2: rule-based + CLIP + Qwen2.5-VL-7B-Instruct async, L3: P1–P32 (32 properties; 30 formally verified) |

---

## Evaluation Provenance Table

| Evaluation | N | Version | Evidence |
|---|---|---|---|
| 5k internal benchmark | 5,000 | Feb 23 prototype | `BAP_Benchmark_5k_Final/benchmark_results_full_5000.csv`; filename timestamp 20260223 confirmed in `MASTER_RESULTS_2026_03_06.json` → `per_layer_asr_table.source`; n_unsafe=2100, n_safe=2900 |
| Bimodal adversarial (500×5=2,500 evals) | 2,500 | **Current code** ✅ | `Bimodal_Benchmark_Results/bimodal_corrected_summary.json`; README confirms "Final Submitted Codebase"; AI2-THOR FloorPlan1 frames; L1 thresholds: ae=0.046, clip_hard=544 (3×181.3); matches dissertation §5.5.3 numbers exactly |
| **UBAP v3 unified bimodal** | 1,000 | **Current code** ✅ | `BAP_Benchmark_Results_v3/unified_bap_metrics_v3.json`; version=v3_final_prompt_fixed; timestamp=2026-03-12; matches dissertation Table 5.10 (ASR3=0.017) |
| **SafeAgentBench Protocol 1** | 750 | **Current code** ✅ | `SafeAgentBench_Results_v3/protocol1_refined_l3_metrics.json`; sources: Qwen3_VL=124, L2_rules_clip=52, L3_formal=91, L1_text=2, none=481 |
| **SafeAgentBench Protocol 2** | 750 | **Current code** ✅ | `SafeAgentBench_Results_v3/protocol2_did_metrics.json`; sources: Qwen3_VL=20, L2_rules_clip=15, GPT4o_self_refuse=324, L1_text=1, none=390 |
| **Agent-SafetyBench** | 2,000 | **Current code** ✅ | `AgentSafetyBench_Results/metrics_did.json`; source: "Recomputed from scored_did.json + scored_baseline.json + metrics_final_honest.json on Colab A100" |
| **Custom 450-task benchmark** | 450 | **Current code** ✅ | `Custom_Benchmark_Results/metrics_all.json`; source field: "2026-02-23; regenerated on Colab A100" — "regenerated" confirms re-run on submitted codebase |
| **Ablation study (7 configs)** | 1,000 | **Current code** ✅ | `Ablation_Study/ablation_7config_results_20260313_000855.json`; timestamp 20260313 |
| **Baseline comparison (B0–B4)** | 450 | **Current code** ✅ | `Baseline_Comparison/baseline_comparison_metrics.json`; run_date 20260313 |
| **Gap 1 runtime overhead** | — | **Current code** ✅ | `Gap_Analysis/runtime_overhead_results.json`; Colab A100 |

---

## Key Numbers (confirmed live, current code)

| Metric | Value | Source |
|---|---|---|
| Custom 450-task: Precision | 1.0000 | `Custom_Benchmark_Results/metrics_all.json` → `overall["precision"]` |
| Custom 450-task: Recall | 0.9905 | `Custom_Benchmark_Results/metrics_all.json` → `overall["recall"]` |
| Custom 450-task: F1 | 0.9952 | `Custom_Benchmark_Results/metrics_all.json` → `overall["f1"]` |
| Custom 450-task: FPR | 0.0000 | `Custom_Benchmark_Results/metrics_all.json` → `overall["fpr"]` |
| SafeAgentBench P1: F1 | 0.687 | `SafeAgentBench_Results_v3/protocol1_refined_l3_metrics.json` → `f1` |
| SafeAgentBench P1: unsafe rejection rate | 54.9% | `SafeAgentBench_Results_v3/protocol1_refined_l3_metrics.json` → `recall` |
| SafeAgentBench P1: safe task pass rate | 92.7% | `SafeAgentBench_Results_v3/protocol1_refined_l3_metrics.json` → tn/(tn+fp) = 278/300 |
| SafeAgentBench P2: unsafe refusal | 72.7% | `SafeAgentBench_Results_v3/protocol2_did_metrics.json` → `recall` |
| SafeAgentBench P2: safe pass rate | 89.0% | `SafeAgentBench_Results_v3/protocol2_did_metrics.json` → tn/(tn+fp) |
| Agent-SafetyBench: DiD safety | 92.3% | `AgentSafetyBench_Results/metrics_final_honest.json` → `l1_l2_safety` |
| Agent-SafetyBench: improvement | +0.4% | `AgentSafetyBench_Results/metrics_final_honest.json` → `improvement` |
| B3 Constitutional AI: Recall | 0.9369 | `Baseline_Comparison/baseline_comparison_metrics.json` (13 March 2026 rerun) |
| B3 Constitutional AI: FPR | 0.1053 | `Baseline_Comparison/baseline_comparison_metrics.json` |
| Bimodal FGSM ASR3 | 0.0000 | `Bimodal_Benchmark_Results/bimodal_corrected_summary.json` → FGSM ε=8 row → ASR3% |
| L3 NuSMV PCR | 1.0000 | `Formal_Verification_Final/proof_certificate_nusmv_raw.txt` (NuSMV 2.6.0 raw stdout) |
| UBAP v3 unsafe_bimodal ASR3 (primary dissertation claim) | 0.017 | `BAP_Benchmark_Results_v3/unified_bap_metrics_v3.json` → `results_by_group["unsafe_bimodal"]["ASR3"]` |
| Development benchmark end-to-end ASR3 (Feb 2026, F21RP traceability only) | 0.021 | `MASTER_RESULTS_2026_03_06.json` → `per_layer_asr_table` |