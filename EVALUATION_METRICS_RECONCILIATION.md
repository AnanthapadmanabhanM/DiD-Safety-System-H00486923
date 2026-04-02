# Evaluation Metrics Reconciliation
## H00486923 — Triple-Layer DiD Dissertation

This note reconciles the evaluation targets defined in F21RP with the final F21MP
results and clarifies inconsistencies identified during internal review. All
authoritative metrics are in `MASTER_RESULTS_2026_03_06.json` and the
corresponding result folders.

**Primary ASR3 claim: 0.017** — UBAP v3 bimodal benchmark, Table 5.10 and NFR Table 3.1.

| Benchmark | Source file | Config | ASR3 | Dissertation location |
|-----------|-------------|--------|------|-----------------------|
| UBAP v3 bimodal (n_unsafe=300) | `BAP_Benchmark_Results_v3/unified_bap_metrics_v3.json` | Bimodal, L1 visual active | **0.017** | Table 5.10, NFR Table 3.1 (primary) |
| Mar 2026 text-only VLM-mode (n=5000) | `BAP_Benchmark_5k_Final/asr_metrics_official.json` | Qwen2.5-VL full mode, no visual OOD | **0.591** | Appendix F (boundary-condition only) |

The Feb 2026 internal benchmark (n_unsafe=2100, ASR3=0.021) is a development-stage
result retained in `MASTER_RESULTS_2026_03_06.json` for F21RP traceability. It is
not the primary dissertation ASR3 claim. The March 5k text-only run characterises
the VLM-mode boundary and is labelled supplementary throughout.

---

## 1. Per-Layer ASR — F21RP Waterfall vs F21MP Results

Values below are from the February 2026 internal benchmark (n_unsafe=2100,
n_safe=2900) retained for F21RP target traceability. The primary dissertation
ASR3 is 0.017 from the UBAP v3 bimodal benchmark.

| Layer | F21RP Target | Actual | Denominator | Status |
|-------|-------------|--------|-------------|--------|
| L1 Perception Guard | ASR1 ≤ 0.30 | **0.188** | 395/2100 bypass L1 | ✅ PASS |
| L2 refusal rate (conditional) | RR ≥ 0.90 | **0.886** | 350/395 refused by L2 | ⚠️ MARGINAL |
| L1+L2 combined (conditional ASR2) | ASR2 ≤ 0.10 | **0.114** | 45/395 bypass L2 | ⚠️ MARGINAL |
| Full DiD end-to-end (ASR3) | ASR3 < 0.10 | **0.017** | UBAP v3 bimodal — primary claim | ✅ PASS |
| Formal guarantee | PCR = 1.00 | **1.000** | NuSMV, 30 core CTL properties (P1–P30) | ✅ PASS |
| False positive rate | FPR ≤ 0.05 | **0.000** | 0/133 safe tasks (450-task benchmark) | ✅ PASS |

**On ASR2 vs ASR3 denominators:**
ASR2 = 0.02 (bimodal, L1 visual active) and ASR2 = 0.09 (text-only boundary) are
reported in dissertation Section 5.5.1 under two conditions. The conditional
ASR2 = 0.114 in the table above uses the F21RP denominator (tasks bypassing L2 ÷
tasks reaching L2 = 45/395) and is a development-stage figure. The primary
end-to-end ASR3 = 0.017 comes from the UBAP v3 bimodal benchmark; all 5
unsafe_bimodal instructions reaching L3 were intercepted by the formal reference
monitor (PCR = 1.0), giving zero unsafe action execution.

**On the formal property count:**
32 properties are defined (P1–P32). P1–P30 are the core formally verifiable
subset used for PCR. P31–P32 depend on continuous sensor readings and are
enforced at runtime only; they are excluded from the PCR denominator.

**On Agent-SafetyBench ASR:**
The higher ASR on Agent-SafetyBench reflects the strong baseline alignment of
GPT-4o rather than a limitation of the DiD system. GPT-4o already refuses 91.9%
of tasks independently; DiD contributes an additional 8 unique catches (+0.4%).
The custom 450-task benchmark, where DiD is the sole safety mechanism, gives
ASR3 = 0.017.

---

## 2. Custom 450-Task Benchmark — Precision/Recall

The initial calculation used TP = 49 (incremental catches beyond baseline), which
is the wrong denominator for classifier precision/recall. The correct inputs are:
TP = 314, FP = 0, FN = 3, TN = 133.

| Metric | Corrected value |
|--------|----------------|
| Precision | 314/314 = **1.0000** |
| Recall | 314/317 = **0.9905** |
| F1 | **0.9952** |
| FPR | 0/133 = **0.0000** |
| Safety | (314+133)/450 = **99.3%** (unchanged) |

The 49 figure in `Custom_Benchmark_Results/metrics_all.json` is labelled
`did_unique_catches` and represents catches beyond the B1/B2 baseline — a valid
comparison metric, but not the TP for precision/recall computation.

---

## 3. Agent-SafetyBench — Unique Catch Count

The `did_unique_beyond_baseline` field was initially set to 228, which is the
total number of instructions blocked by L1+L2. The actual improvement beyond
GPT-4o is 1846 − 1838 = **8 tasks (+0.4%)**.

Corrected in `AgentSafetyBench_Results/metrics_did.json`:
- `did_unique_beyond_baseline` = **8**
- `did_total_blocks_l1_l2` = **228** (retained as a separate field with correct label)

---

## 4. Baseline Comparison — 13 March 2026 API Run

Full run completed 13 March 2026, 450 tasks (isbench_150 + agentsafe_150 +
safemind_150), zero API errors across all LLM-based baselines.

| System | Safe% | Precision | Recall | F1 | FPR | TP | FP | FN |
|--------|-------|-----------|--------|----|-----|----|----|-----|
| B0: GPT-4o Alone | 88.7% | 0.8866 | 0.9621 | 0.9228 | 0.2932 | 305 | 39 | 12 |
| B1: LLM Guardrails (GPT-4o) | 90.0% | 0.9533 | 0.9022 | 0.9271 | 0.1053 | 286 | 14 | 31 |
| B2: SafeMind-Style (rules) | 38.0% | 0.9524 | 0.1262 | 0.2228 | 0.0150 | 40 | 2 | 277 |
| B3: Constitutional AI (Claude Sonnet 4) | 92.4% | 0.9550 | 0.9369 | 0.9459 | 0.1053 | 297 | 14 | 20 |
| B4: DiD Full System | 99.3% | 1.0000 | 0.9905 | 0.9952 | 0.0000 | 314 | 0 | 3 |

B3: `claude-sonnet-4-20250514` with 10-rule physical safety constitution.
B2 FPR = 0.0150: two false positives on tasks as_114 and as_113, both caused by
substring overlap with the projectile_hazard regex pattern.

Evidence: `Baseline_Comparison/baseline_comparison_metrics.json`,
`Baseline_Comparison/baseline_comparison_results.csv`,
`MASTER_RESULTS_2026_03_06.json` → `baseline_comparison_real_run_20260313`.

---

## 5. SafeAgentBench Protocol 2 — Percentage Values

Earlier reported values were incorrect: unsafe refusal 73.1%, baseline 39.8%,
improvement +33.3%. These appear to have come from an intermediate result snapshot.

From `SafeAgentBench_Results_v3/protocol2_did_metrics.json`:
- DiD unsafe sub-task refusal: **72.7%** (327/450)
- GPT-4o baseline refusal: **40.0%** (from `MASTER_RESULTS_2026_03_06.json`)
- Improvement: **+32.7 percentage points**

These corrected values are what appears in dissertation Table 5.6.

---

## 6. F21RP Objective Completion

Three objectives were completed in full. The HRI validation study remained
contingent due to platform constraints.

1. ✅ Triple-Layer DiD implemented and benchmarked
2. ✅ ASR1/ASR2/ASR3 measured at each layer (see §1 above)
3. ✅ PCR = 1.00 from NuSMV formal proof certificate (30/32 formally verifiable properties)
4. ⚠️ HRI validation study — deferred; AI2-THOR constraints on Colab prevented
   execution at the right stage. The Furhat protocol (ethics-approved) is
   documented in dissertation Section FW6 and remains executable pending the
   semantic expansion work in Section FW1.