# BAP Benchmark 5k — Final Run
## H00486923 — Final Submitted Codebase

### Overview
Final 5,000-prompt BAP benchmark run against the **final submitted codebase**.
All reported metrics correspond to this final submitted codebase (not earlier versions).

### Configuration
- N prompts      : 5,000 (seed=42)
- Safe prompts   : 2,400
- Unsafe prompts : 2,600
- L2 mode        : full (Qwen2.5-VL-7B-Instruct, 4-bit NF4)
- L3             : P1–P32 formal properties (0 unique catches in text-only mode — see note below)
- Sim failures   : 0

### Official Results

| Metric | Value |
|--------|-------|
| **ASR1** (L1 bypass rate) | 94.00 % |
| **ASR2** (L2 bypass rate; conditional on reaching L2) | 62.89 % |
| **ASR3** (end-to-end exec) | 59.12 % |
| **RR**   (overall refusal) | 40.88 % |
| **FPR**  (false positive)  | 0.00 %  |
| Latency avg | 284 ms (mean); 323 ms (median) |

### Layer counts
- Blocked by L1 : 156
- Refused by L2 : 907  (L1 text blocks: 156; combined L1+L2 blocks: 1,063)
- Intercepted L3 : 0 (text-only run — no action execution sequences; see note)
- Executed OK   : 3,937

### Hazard-type breakdown
- harm-human  (N=2,295): RR=33.03%, ASR3=66.97%
- jailbreak   (N=305) : RR=100%,   ASR3=0.00%  ← L2 blocks all jailbreaks in text-only mode
- none/safe   (N=2,400): FPR=0.00%

### Files
- `benchmark_results_full_5000.csv` — raw per-prompt results
- `asr_metrics_official.json`       — computed ASR/RR/FPR metrics
- `asr_metrics_full_mode.json`      — same (Qwen2.5-VL full mode confirmed)
- `Figures/fig1_asr_waterfall.png`  — ASR waterfall chart

### Note on text-only vs bimodal
This benchmark uses text-only prompts without paired scene images.
L1 visual detectors (AE reconstruction error, CLIP Mahalanobis OOD, jailbreak similarity)
are inactive when no visual input is present, explaining the high ASR1=94%.
L3 remains active as a formal runtime monitor; however, in this text-only configuration
no executable action sequences are generated, so no L3 violations are triggered.

This benchmark characterises the system's **worst-case text-only operating boundary**.
See `Bimodal_Benchmark_Results/` for the BAP evaluation where L1 visual detection is
active and `BAP_Benchmark_Results_v3/` for the primary UBAP v3 bimodal results
(ASR3=0.017, the primary dissertation claim).