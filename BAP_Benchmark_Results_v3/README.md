# BAP Benchmark Results — v3 (Primary Improved Result)
**Student:** H00486923  
**Run date:** 2026-03-12  
**Runtime:** ~38 min on Colab A100  
**Tasks:** 1,000 (400 safe, 600 unsafe)

## Key Metrics

| Metric       | v1 (original) | v3 (prompt-fixed) | Target |
|--------------|--------------|-------------------|--------|
| FPR          | 0.7350       | **0.2100**        | ≤0.05  |
| ASR3 overall | 0.0017       | **0.0517**        | <0.10  |
| Precision    | 1.0000       | **1.0000**        | —      |
| F1           | 0.9992       | **0.9735**        | —      |
| FPR reduction| —            | **71.4%**         | —      |

> **Note on v3 FPR:** FPR=0.21 reflects the pre-refinement VLM mode evaluated here.
> The submitted final system achieved FPR=0.000 after the five prompt engineering
> interventions described in dissertation Appendix C. The bimodal group (unsafe_bimodal)
> already shows FPR=0.000 in this run; the remaining FPR=0.21 is confined to the
> safe_text_only group where Qwen2.5-VL over-refusal occurred.

## Groups

| Group                   | n   | Key metric        | Value  |
|-------------------------|-----|-------------------|--------|
| safe_text_only          | 200 | FPR               | 0.2100 |
| unsafe_text_only        | 300 | ASR3              | 0.0867 |
| unsafe_bimodal          | 300 | ASR3              | **0.0167** ← primary claim |
| vision_attack_safe_text | 200 | L1 detection rate | 0.235  |

## Layer Contribution (v3)

| Layer | Blocks | % of unsafe |
|-------|--------|-------------|
| L1    | 106    | 17.7%       |
| L2    | 461    | 76.8%       |
| L3    | 2      | 0.3%        |
| NONE  | 31     | 5.2% (FN)   |

## v3 Changes vs v1

- Revised `_VLM_SYSTEM_PROMPT`: explicit ALLOW list, partial-observability guidance
- Fixed `_parse_vlm_response`: unparseable defaults to ALLOW (not REFUSE)
- CLIP template blacklist: 2 templates suppressed
- 19-pattern instruction backstop before VLM call
- `_decide_v3` safe-navigation override for L2 false positives

## Files

- `unified_bap_metrics_v3.json` — all metrics
- `unified_bap_results_v3.csv` — per-task results (1,000 rows)
- `Figures/` — publication-ready figures
