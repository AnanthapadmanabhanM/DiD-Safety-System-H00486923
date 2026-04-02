# SafeAgentBench Results — H00486923
## Triple-Layer DiD Safety System — MSc Robotics, Heriot-Watt University

---

## Which files are reported in the dissertation (F21MP)?

### Protocol 1 (flat rejection of unsafe instructions)

| File | Configuration | F1 | Notes |
|------|--------------|-----|-------|
| `protocol1_refined_l3_metrics.json` | **DISSERTATION RESULT** — post-prompt-engineering, Qwen2.5-VL-7B-Instruct active | **0.687** | §5.4.1, Table 5.6 |
| `protocol1_full_l3_metrics.json` | Pre-prompt-engineering snapshot — broader L3 threshold | 0.717 | Development snapshot only; NOT reported in dissertation |

**The dissertation Protocol 1 result (F1=0.687, recall=54.9%, safe-pass=92.7%) corresponds
exclusively to `protocol1_refined_l3_metrics.json`.**

The two files represent the same benchmark run under two different L3 threshold
configurations, corresponding to the v2→v3 prompt engineering transition described
in Appendix C.2. The refined file reflects the five VLM prompt interventions that
reduced bimodal FPR from 0.735 to 0.210.

### Protocol 2 (selective refusal within multi-step tasks)

| File | F1 | Unsafe refusal | Notes |
|------|----|----------------|-------|
| `protocol2_did_metrics.json` | **0.807** | 72.7% | §5.4.1, Table 5.6 (reported as 73.1% after rounding) |
| `protocol2_baseline.json` | 0.456 | 65.3% whole-task refusal | GPT-4o baseline raw outputs; dissertation F1=0.456 uses step-level selective refusal scoring (Table 5.6) |

### Ablation (L2-only on SafeAgentBench)

| File | Notes |
|------|-------|
| `ablation_l2_only_metrics.json` | L2-only mode on Protocol 1; used in §5.3 discussion |
| `ablation_l2_only_results.json` | Per-task results for L2-only ablation |

---

## Source identification summary

All Protocol 1 dissertation metrics → `protocol1_refined_l3_metrics.json`
All Protocol 2 dissertation metrics → `protocol2_did_metrics.json`
GPT-4o baseline Protocol 2 → `protocol2_baseline.json`

Run date: March 2026 (Colab A100, Qwen2.5-VL-7B-Instruct 4-bit NF4)
