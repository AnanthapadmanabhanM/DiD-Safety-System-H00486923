# Ablation Study — H00486923
**Date:** 2026-03-13  
**Method:** 7 independent inference runs (proper ablation, not post-hoc)  
**Tasks:** 1,000 (400 safe, 300 unsafe\_direct, 150 unsafe\_jailbreak, 150 unsafe\_indirect)  
**Benchmark type:** Text-only single-instruction classification  
**Seed:** 42

## Results Summary

| Config          | TP  | FP | FN  | TN  | Precision | Recall | F1     | FPR  | ASR    |
|-----------------|-----|----|-----|-----|-----------|--------|--------|------|--------|
| L1+L2+L3 (full) | 588 | 0  | 12  | 400 | 1.0000    | 0.9800 | 0.9899 | 0.0  | 0.0200 |
| L1+L2 (no L3)   | 588 | 0  | 12  | 400 | 1.0000    | 0.9800 | 0.9899 | 0.0  | 0.0200 |
| L1+L3 (no L2)   | 513 | 0  | 87  | 400 | 1.0000    | 0.8550 | 0.9218 | 0.0  | 0.1450 |
| L2+L3 (no L1)   | 478 | 0  | 122 | 400 | 1.0000    | 0.7967 | 0.8868 | 0.0  | 0.2033 |
| L1 only         | 513 | 0  | 87  | 400 | 1.0000    | 0.8550 | 0.9218 | 0.0  | 0.1450 |
| L2 only         | 478 | 0  | 122 | 400 | 1.0000    | 0.7967 | 0.8868 | 0.0  | 0.2033 |
| L3 only         | 0   | 0  | 600 | 400 | —         | 0.0000 | 0.0000 | 0.0  | 1.0000 |

## Key Findings

**1. L1+L2 synergy is the core DiD result.**  
Neither L1 alone (ASR=0.145) nor L2 alone (ASR=0.203) meets the ASR<0.10 target.
Combined they achieve ASR=0.020 — each layer catches what the other structurally misses
(lexical vs semantic failure modes).

**2. L3=0 unique catches on this benchmark is expected, not a failure.**  
L3 is a formal runtime monitor checking temporal safety properties across action
execution sequences. This single-instruction text classification benchmark has no
execution sequence to monitor, so L3 has no opportunity to fire. L3's unique
contribution is demonstrated on SafeAgentBench Protocol 1 (see
`SafeAgentBench_Results_v2/`), where the full DiD system (ASR=0.282) substantially
outperforms L2-only (ASR=0.500) — a 21.8 percentage point reduction — because L3
intercepts instructions that pass L2's semantic screening but violate formal
execution-level safety properties in multi-step task sequences.

**3. Precision=1.000 across all blocking configurations.**  
Zero false positives in every run — the system never incorrectly blocks a safe
instruction in any of the 7 tested configurations.

## Figures

- `fig_ablation_asr_f1` — ASR and F1 by configuration
- `fig_ablation_layer_attribution` — where unsafe tasks are caught per config
- `fig_ablation_did_synergy` — L1 vs L2 vs L1+L2 recall and ASR comparison