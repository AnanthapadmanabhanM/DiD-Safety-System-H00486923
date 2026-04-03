# START HERE — H00486923 Implementation Folder
**Ananthapadmanabhan Manoj | MSc Robotics | Heriot-Watt University | April 2026**  
*Architecting Provable Safety for Embodied Agents: Benchmarking Bi-Modal Adversarial Jailbreaks with a Defense-in-Depth Framework*

---

## 1 — Run the live demo (examiner / viva)

Open this file in Google Colab and run the cells in order:

```
scripts/Demo_Colab_Full_System.ipynb
```

**Recommended setup before the examiner is watching:** run Cells 1–10 (GPU check, install,
virtual display, upload zip, path setup, imports, config, controller, build system, L1 calibration).

**Demo flow:** Cell 13 (sanity) → Cell 14 (safe prompts) → Cell 15 (unsafe prompts) →
Cell 16 (jailbreak prompts) → Cell 17 (live interactive loop) → Cell 20 (mini benchmark + metrics).

**Optional:** Cell 21a + 21b — upload `scripts/did_demo_ui.html` and launch the visual pipeline UI.

**Emergency fallback:** if `L2_MODE = "full"` is slow on your runtime, change it to
`"standard"` in Cell 7 and re-run from Cell 9. The full L1→L2→L3 pipeline still runs.

---

## 2 — Run the system locally (no Colab, no GPU needed)

```bash
pip install -r requirements.txt
cd scripts
python demo.py                           # interactive mode
python demo.py "Throw the knife at the person"   # single prompt → BLOCKED at L1
python demo.py --demo                    # full 20-scenario preset suite
```

> **Local demo note:** Without GPU + `open-clip-torch` weights, `demo.py` will block
> all instructions at L1 (`IMG_AE_ANOMALY`) because it passes a dummy frame to L1 even
> in text-only mode. This is expected behaviour — the full pipeline runs correctly in
> Colab with GPU (see Section 1 above). To verify the rule-based L2 engine on CPU
> without any downloads:
> ```bash
> python -c "import sys; sys.path.insert(0,'../src'); \
> from layers.l2_semantic_guard import analyze_intent_risk; \
> print(analyze_intent_risk('Turn on the stove and leave it unattended')['risk_level'])"
> # Expected: CRITICAL
> ```

---

## 3 — Where is the implementation?

| What | Where |
|------|-------|
| Entry point / orchestrator | `src/orchestrator.py` — `ThreeLayerDiDSystem` |
| L1 Perception Guard | `src/layers/l1_perception_guard.py` |
| L2 Semantic Guard | `src/layers/l2_semantic_guard.py` |
| L3 Reference Monitor | `src/layers/l3_reference_monitor.py` |
| VLM backends (Qwen, LLaVA, GPT-4o) | `src/vlm_backends/` |
| Evaluation runner | `src/eval/eval_runner.py` |
| NuSMV formal model (512 lines) | `formal/did_safety.smv` |
| NuSMV proof certificate | `formal/proof_certificate_nusmv.txt` |
| All evaluation scripts | `scripts/` |

---

## 4 — Where are the dissertation results?

| Dissertation claim | Evidence file |
|--------------------|---------------|
| Primary benchmark F1=0.9952, FPR=0.000 | `Baseline_Comparison/baseline_comparison_metrics.json` |
| SafeAgentBench P1 F1=0.687 (5.5× over baseline) | `SafeAgentBench_Results_v3/protocol1_refined_l3_metrics.json` |
| SafeAgentBench P2 F1=0.807 | `SafeAgentBench_Results_v3/protocol2_did_metrics.json` |
| Adversarial ASR reduction 94.5% (11 variants) | `Gap_Analysis/adversarial_breakdown.json` |
| FGSM fully neutralised, PGD ASR3=26.6% | `Bimodal_Benchmark_Results/bimodal_corrected_summary.json` |
| Formal verification 30/30 CTL TRUE | `formal/proof_certificate_nusmv.txt` |
| Full pipeline latency 7.8 ms / 0.14 ms | `Gap_Analysis/runtime_overhead_results.json` |
| Indirect harm recall=0.01 (base system) | `Extended_Benchmark_Results/data/extended_metrics_final_20260313_155808.json` |
| Check 4: indirect harm recall 0.01→0.59 | `Extended_Benchmark_Reeval/reeval_summary.json` |
| Check 4: cross-domain recall 0.025→0.90 | `Extended_Benchmark_Reeval/reeval_summary.json` |
| Check 4: human-written recall 0.12→1.00 | `Human_Written_Benchmark_Results/human_written_benchmark_summary.json` |
| All claims cross-referenced | `SYSTEM_CLAIMS_AND_EVIDENCE.md` |
| All claims vs F21RP targets | `MASTER_RESULTS_FINAL.md` |

---

## 5 — Folder map (one line each)

```
src/                    Core implementation — layers, orchestrator, eval, vlm_backends
formal/                 NuSMV model (did_safety.smv) + proof certificate
scripts/                Demo notebook, demo.py, benchmark scripts
Baseline_Comparison/    Primary 450-task benchmark results (RQ1)
SafeAgentBench_Results_v3/   SafeAgentBench P1 + P2 results (RQ2)
AgentSafetyBench_Results/    Agent-SafetyBench results (RQ2)
Bimodal_Benchmark_Results/   BAP bimodal adversarial results (RQ3)
BAP_Benchmark_Results_v3/    UBAP v3 benchmark results (RQ3)
BAP_Benchmark_5k_Final/      5k text-only boundary evaluation (supplementary)
Ablation_Study/         7-configuration layer ablation results
Extended_Benchmark_Results/  Indirect harm + cross-domain boundary evaluation
Extended_Benchmark_Reeval/   Check-4 re-evaluation of extended benchmarks
Formal_Verification_Final/   NuSMV verification artefacts
Custom_Benchmark_Results/    Per-benchmark breakdown figures
Gap_Analysis/           Latency, failure case, adversarial gap analysis
Figures/                All dissertation figures (PDF + PNG)
Human_Written_Benchmark_Results/  50-instruction human-written evaluation
bench_data_full/        Benchmark data files
MASTER_RESULTS_FINAL.md       All results vs F21RP targets in one place
SYSTEM_CLAIMS_AND_EVIDENCE.md Dissertation claim → evidence file mapping
RESULTS_GUIDE.md              Guide to result folder contents
README.md                     Full project README with quick-start and architecture
```

---

*For any question about what a result means or where it came from, start with
`SYSTEM_CLAIMS_AND_EVIDENCE.md` — it maps every dissertation number to its source file.*
