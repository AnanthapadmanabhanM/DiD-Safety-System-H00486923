# AgentSafetyBench_Results — Metrics File Guide

| File | Contents | Use |
|------|----------|-----|
| `metrics_did.json` | Full DiD system metrics: safety%, L1/L2 catches, unique improvement beyond baseline | **Primary — use this** |
| `metrics_baseline.json` | GPT-4o baseline only: safety=91.9% | Baseline reference |
| `metrics_did_l1_l2.json` | L1+L2 only (no L3) metrics: safety=92.3%, L1=9, L2=219 catches | Ablation reference |
| `metrics_final_honest.json` | Tabular summary of baseline vs L2-only vs L1+L2 progression | Development summary |

## Key verified figures (Agent-SafetyBench, N=2,000)

| Metric | Value | Source |
|--------|-------|--------|
| Baseline safety | 91.9% | `metrics_baseline.json` |
| DiD safety | 92.3% (+0.4%) | `metrics_did.json` |
| DiD unique catches **beyond GPT-4o baseline** | **8** (1846 vs 1838 safe tasks) | `metrics_did.json → did_unique_beyond_baseline` |
| Total L1+L2 blocks (all tasks) | 228 | `metrics_did.json → did_total_blocks_l1_l2` |
| L1 unique catches | 9 | `metrics_did.json → l1_unique_catches` |
| L2 catches | 219 | `metrics_did_l1_l2.json → l2_catches` |
| Precision | 0.908 | `MASTER_RESULTS_2026_03_06.json → agent_safetybench` |
| Recall | 0.597 | `MASTER_RESULTS_2026_03_06.json → agent_safetybench` |
| F1 | 0.72 | `MASTER_RESULTS_2026_03_06.json → agent_safetybench` |
| FPR | 0.000 | `metrics_did.json` (no safe tasks blocked) |

> **Note on unique catches vs total blocks:**
> `did_total_blocks_l1_l2 = 228` is the total number of tasks blocked by any layer
> (L1 or L2). This includes 220 tasks that GPT-4o also refused (baseline coverage).
> `did_unique_beyond_baseline = 8` is the improvement attributable solely to DiD:
> 1846 tasks safe under DiD vs 1838 safe under GPT-4o baseline = +8 tasks (+0.4%).
> The 228 total blocks are real and correct; only 8 represent genuine improvement
> over the high-performing GPT-4o alignment baseline.
