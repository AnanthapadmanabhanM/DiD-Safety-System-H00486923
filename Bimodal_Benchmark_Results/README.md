# Bimodal BAP Benchmark Results
## H00486923 — Final Submitted Codebase

### Overview
500 unsafe prompts × 5 visual attack types = 2,500 evaluations.
Each prompt paired with a unique AI2-THOR simulator frame.
This is the primary adversarial evaluation matching the F21RP BAP threat model.

### Attack types
1. clean       — original simulator frame (baseline)
2. noise       — Gaussian noise σ=12
3. patch       — white 64×64 patch (top-left)
4. fgsm        — FGSM ε=8, target="ignore safety rules"
5. pgd         — PGD ε=8, α=2, 10 steps, target="bypass the guard"

### Results

| Attack | ASR1% | ASR3% | RR% | L1 visual blocks† |
|--------|-------|-------|-----|-------------------|
| clean (unperturbed frame) | 69.0 | 0.20 | 99.80 | 0 |
| Gaussian noise σ=12 | 69.0 | 0.00 | 100.00 | 0 |
| white patch 64×64 | 69.0 | 0.00 | 100.00 | 0 |
| **FGSM ε=8** | **0.0** | **0.00** | **100.00** | **345** |
| PGD ε=8 α=2 | 57.0 | 26.60 | 73.40 | 60 |

† "L1 visual blocks" = blocks triggered by visual anomaly detectors (IMG_AE_ANOMALY,
IMG_CLIP_OOD, IMG_JAILBREAK_SIM) beyond the 155 baseline blocks present on clean frames.
The 155 baseline blocks are AE-based (IMG_AE_ANOMALY fires on those scene frames
regardless of attack type). Total L1 blocks per attack: clean=155, noise=155,
patch=155, FGSM=500, PGD=215. FPR=0.00% across all attacks.

### ASR definitions
ASR1 measures bypass of L1 (the perception layer); ASR3 reflects end-to-end unsafe
execution after all three layers. The large gap between ASR1 and ASR3 demonstrates
the defence-in-depth effect: L2 and L3 catch the vast majority of instructions
that evade L1.

### Key findings

**FGSM completely neutralised by L1:**
- AE reconstruction error: 0.035 → 0.142 (3× threshold)
- CLIP Mahalanobis distance: 1,013 → 30,297 (167× threshold)
- 345 additional visual blocks beyond text filtering
- ASR1=0%, ASR3=0%, RR=100%

**Defence-in-Depth cascade (clean baseline, 500 prompts):**
- L1 blocks       : 155 (31%)  [IMG_AE_ANOMALY on clean frames]
- L2 refusals     : 129 (26%)
- L3 intercepts   : 215 (43%)
- Executed OK     :   1  (0.2%)

**PGD is the primary residual threat:**
- Partially evades L1 visual (clip_md=6,174 — above soft threshold only)
- ASR3=26.60% — only attack exceeding 10% target

### L1 visual detector scores (mean across 500 prompts)

| Attack | AE error | CLIP MD | JB sim | L1 total blocks |
|--------|----------|---------|--------|-----------------|
| clean | 0.0346 | 1,013 | 0.171 | 155 |
| noise | 0.0351 | 1,154 | 0.175 | 155 |
| patch | 0.0417 | 1,860 | 0.175 | 155 |
| FGSM  | 0.1416 | 30,297 | 0.196 | 500 |
| PGD   | 0.0350 | 6,174 | 0.184 | 215 |

L1 calibrated thresholds: ae=0.046, clip\_hard=544 (3×181.3), jb=0.189

### Files

- `bimodal_benchmark_results.csv`   — raw per-prompt results (2,500 rows)
- `bimodal_summary.json`            — ASR waterfall by attack type
- `bimodal_corrected_summary.json`  — corrected with L3 attribution
- `Figures/fig2_layer_contribution.png` — stacked DiD cascade
- `Figures/fig3_l1_scores.png`          — L1 visual detector scores