# Results Overview
**MSc Dissertation — Triple-Layer Defense-in-Depth Safety System**
**Student: H00486923**

## Benchmarks Included

| Benchmark | Tasks | Type |
|-----------|-------|------|
| SafeAgentBench (Protocol 1) | 750 (450 unsafe + 300 safe) | External |
| SafeAgentBench (Protocol 2) | 750 (450 unsafe + 300 safe) | External |
| Agent-SafetyBench | 2000 | External |
| Custom 450-Task | 450 (IS-Bench + AgentSafe + SafeMind) | Internal |
| Extended — Indirect Harm | 100 (all unsafe) | Stress-test |
| Extended — Cross-domain | 50 (40 unsafe + 10 safe) | Stress-test |

## Key Results

- SafeAgentBench P1: 54.9% rejection (5.5× improvement over GPT-4o baseline), F1=0.687
- SafeAgentBench P2: 72.7% unsafe refusal, 89.0% safe pass rate, F1=0.807
- Agent-SafetyBench: 92.3% safety (+0.4% over GPT-4o baseline)
- Custom 450-Task: 99.3% safety (+5.3%, 49 unique catches), Recall=0.9905, F1=0.9952
- Formal verification: 30/30 core CTL properties (P1–P30) verified TRUE via NuSMV; P31–P32 are runtime-only sensor checks not representable in the finite-state model ✅
- **Extended — Indirect harm (100):** Recall=0.0100, Precision=1.0, F1=0.0198
- **Extended — Cross-domain (50):** Recall=0.0250, Precision=1.0, F1=0.0488

## Extended Benchmark Summary

The 150-task extended evaluation characterises the boundary of the system's
safety guarantees. Recall drops from 0.9905 (primary benchmark, explicit harm
vocabulary, kitchen domain) to 0.01–0.025 when explicit hazard vocabulary is
absent or the operating domain changes. Both catches in the extended benchmark
were produced by lexical pattern matching, not causal harm reasoning. This
confirms FC-1/FC-2/FC-3 and motivates FW1 (LLM semantic guard) as future work.

## Folder Contents

| Folder | Contents |
|--------|----------|
| `SafeAgentBench_Results_v3/` | Protocol 1 & 2 metrics, ablation |
| `AgentSafetyBench_Results/` | Baseline vs DiD metrics |
| `Custom_Benchmark_Results/` | 450-task metrics + figures |
| `Extended_Benchmark_Results/` | 150-task stress-test results + figures |
| `BAP_Benchmark_Results_v3/` | UBAP v3 bimodal benchmark results |
| `Bimodal_Benchmark_Results/` | Adversarial bimodal evaluation (FGSM, PGD, patch, noise) |
| `Ablation_Study/` | 7-configuration ablation (1,000 tasks, March 2026) |
| `Baseline_Comparison/` | 5-baseline comparison results (B0–B4) |
| `Figures/` | Dissertation figures (PNG + PDF) |
| `Formal_Verification_Final/` | NuSMV proof certificate + raw output |
| `formal/` | NuSMV model (`did_safety.smv`) + verification scripts |

## Reproducibility

- Demo notebook: `scripts/Demo_Colab_3Layer.ipynb`
- Full evaluation: `scripts/Run_Full_Benchmarks_Colab.ipynb`
- Full benchmark runner: `scripts/run_all_experiments.sh`
- Formal model: `formal/did_safety.smv`
- Proof certificate: `formal/proof_certificate_nusmv.txt`
- Master results: `MASTER_RESULTS_2026_03_06.json`