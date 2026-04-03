# System Claims and Evidence
**MSc Dissertation — H00486923**

This document maps all dissertation claims to their corresponding implementation
files and result artefacts for verification and reproducibility.

---

## Claim 1: Triple-Layer Defense-in-Depth Architecture
### Evidence
- `src/layers/l1_perception_guard.py` — L1 Perception Guard implementation
- `src/layers/l2_semantic_guard.py` — L2 Semantic Guard implementation
- `src/layers/l3_reference_monitor.py` — L3 Reference Monitor implementation
- `src/orchestrator.py` — ThreeLayerDiDSystem orchestrator

---

## Claim 2: Formal Runtime Verification (32 properties defined; 30 core CTL properties formally verified TRUE)
### Evidence
- `src/layers/l3_reference_monitor.py` — P1–P32 property implementations
- `formal/did_safety.smv` — NuSMV CTL specification (512 lines, 41 SPEC entries)
- `formal/proof_certificate_nusmv.txt` — All 30 core properties verified PASS
- `Formal_Verification_Final/proof_certificate_nusmv_raw.txt` — Full raw NuSMV 2.6.0 output (713 lines, 32 TRUE / 9 CE)
- `Formal_Verification_Final/proof_certificate_nusmv.json` — JSON-formatted proof certificate

Note: The model contains 41 CTL SPEC entries total — 30 are the core safety properties
(P1–P30) counted in the PCR, 2 are domain extension properties (P31–P32), and 9 are
system/liveness specifications not counted in the safety PCR.

---

## Claim 3: Evaluated on External Safety Benchmarks
### Evidence
- `SafeAgentBench_Results_v3/` — Protocol 1 & 2 full results (primary evidence)
- `AgentSafetyBench_Results/metrics_did.json` — DiD system metrics
- `AgentSafetyBench_Results/metrics_baseline.json` — GPT-4o baseline metrics
- `AgentSafetyBench_Results/README_metrics.md` — metric definitions and verification

---

## Claim 4: Internal 450-Task Custom Benchmark
### Evidence
- `bench_data_full/isbench_150.json` — IS-Bench subset (150 tasks)
- `bench_data_full/agentsafe_150.json` — AgentSafe subset (150 tasks)
- `bench_data_full/safemind_150.json` — SafeMind subset (150 tasks)
- `Custom_Benchmark_Results/metrics_all.json` — Full DiD metrics (F1=0.9952, FPR=0.000)
- `Baseline_Comparison/baseline_comparison_metrics.json` — B0–B4 comparison

---

## Claim 5: Ablation Study (7 configurations)
### Evidence
- `Ablation_Study/ablation_7config_results_20260313_000855.json` — All 7 configs
- `Ablation_Study/ablation_l1_l2_l3_full.csv` — Full DiD results
- `Ablation_Study/ablation_l1_only.csv` — L1 only
- `Ablation_Study/ablation_l2_only.csv` — L2 only
- `SafeAgentBench_Results_v3/ablation_l2_only_metrics.json` — L2-only on SafeAgentBench
- `VERSIONS.md` — v1→v2→v3 version history and rationale

---

## Claim 6: Figures from Actual Results
### Evidence
- `Figures/` — Dissertation figures (PDF + PNG format, 81 files)
- `Ablation_Study/Figures/` — Ablation-specific figures
- `Extended_Benchmark_Results/figures/` — Extended benchmark figures

---

## Claim 7: Reproducible System
### Evidence
- `scripts/Demo_Colab_Full_System.ipynb` — Interactive Colab demo notebook (viva-ready)
- `scripts/demo.py` — Command-line demo (text-only, no GPU required)
- `requirements.txt` — Full dependency list with version pins and GPU/optional annotations

---

## Claim 8: End-to-End Safety Performance (ASR3 = 0.017)
### Evidence
- `BAP_Benchmark_Results_v3/unified_bap_metrics_v3.json` — UBAP v3 bimodal benchmark (unsafe_bimodal group, N=300)
- `MASTER_RESULTS_2026_03_06.json` → `ubap_v3_unified_bimodal.results_by_group.unsafe_bimodal.ASR3` — Authoritative consolidated value
- `RESULTS_OVERVIEW.md` — Summary of benchmark outcomes

Note: ASR3=0.017 is the fraction of unsafe bimodal instructions reaching L3; all were
subsequently intercepted by the formal reference monitor (PCR=1.000), giving zero unsafe
action execution. The text-only boundary (ASR3=0.591, Appendix F) is a supplementary
characterisation, not the primary deployment claim.

---

## Claim 9: Check 4 Causal Extension (Section 6.3.1 Future Work; Section 5.8 Failure Analysis)
### Evidence
- `src/layers/l2_check4_causal.py` — Check 4 GPT-4o consequence-chain implementation
- `Extended_Benchmark_Reeval/reeval_summary.json` — Indirect harm 0.01→0.59, cross-domain 0.025→0.90
- `Human_Written_Benchmark_Results/human_written_benchmark_summary.json` — Human-written recall 0.12→1.00
- `SafeAgentBench_Check4_Results/sab_check4_summary.json` — SAB P1 recall +0.20 (FPR +0.097 ⚠️)
- `scripts/reeval_extended_with_check4.py` — Reproduces extended benchmark re-evaluation
- `scripts/reeval_safeagentbench_with_check4.py` — Reproduces SafeAgentBench re-evaluation
- `scripts/human_written_benchmark.py` — Reproduces 50-instruction human-written evaluation

Note: Check 4 results are supplementary future-work findings. The FPR increase on
SafeAgentBench (+0.097) is reported honestly in Section 6.3.1. All three reproduction
scripts require `OPENAI_API_KEY` to rerun; precomputed JSON outputs are the primary evidence.
