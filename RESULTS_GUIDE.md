# Results Directory Guide — H00486923
## Primary Dissertation Results (cited in Chapter 5)

| Folder | Contents | Dissertation section |
|--------|----------|----------------------|
| `Baseline_Comparison/` | B0–B4 comparison on 450-task benchmark | §5.2 |
| `Custom_Benchmark_Results/` | DiD system metrics on 450-task benchmark (TP=314, FPR=0.000) | §5.2 |
| `Ablation_Study/` | 7-configuration ablation (1,000 tasks, March 2026) | §5.3 |
| `SafeAgentBench_Results_v3/` | Protocol 1 & 2 results | §5.4 |
| `AgentSafetyBench_Results/` | Agent-SafetyBench evaluation | §5.4 |
| `Bimodal_Benchmark_Results/` | BAP bimodal benchmark (500×5 attacks) | §5.5 |
| `BAP_Benchmark_Results_v3/` | UBAP v3 bimodal benchmark — primary ASR3=0.017 claim | §5.5 |
| `Formal_Verification_Final/` | NuSMV proof certificate (PCR=1.000, 30/30 properties) | §5.6 |
| `Gap_Analysis/` | Runtime, failure analysis, adversarial breakdown | §5.5–§5.7 |
| `Extended_Benchmark_Results/` | Boundary evaluation (indirect harm + cross-domain) | §5.9 |
| `BAP_Benchmark_5k_Final/` | Text-only 5k boundary evaluation | Appendix F |

## Supplementary / Check 4 Results

| Folder | Contents |
|--------|----------|
| `Extended_Benchmark_Reeval/` | Check 4 augmentation on 150-task extended benchmark |
| `Human_Written_Benchmark_Results/` | Check 4 on 50 human-written instructions |
| `SafeAgentBench_Check4_Results/` | Check 4 on SafeAgentBench Protocol 1 |

Authoritative metric values for all primary claims: `MASTER_RESULTS_2026_03_06.json`