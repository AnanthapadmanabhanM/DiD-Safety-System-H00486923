"""
bench/run_benchmark_colab.py
-------------------------------------------------------------------
3-Layer DiD Benchmark Runner.

# Key Capabilities:
# - Multi-scene evaluation using SceneManager (improves robustness across environments)
# - Visual adversarial testing (FGSM / noise / patch perturbations)
# - Formal verification logging (NuSMV / PRISM traces stored with outputs)
# - Per-layer evaluation metrics (FPR_L1, RR_L2, FPR_L2) aligned with ground truth labels

Modes
-----
  python -m bench.run_benchmark_colab                         # generated 5000
  python -m bench.run_benchmark_colab --dataset unified.csv   # external
  python -m bench.run_benchmark_colab --multi_scene           # G2
  python -m bench.run_benchmark_colab --visual_attack fgsm    # G3
  python -m bench.run_benchmark_colab --verify_formal         # Link-A
  python -m bench.run_benchmark_colab --ablation              # G7
-------------------------------------------------------------------
"""
from __future__ import annotations

import logging
log = logging.getLogger(__name__)


import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional
import argparse

import pandas as pd
from ai2thor.controller import Controller

# --- Robust import path setup ---
# Allows running this script from project root (or any cwd) without manual sys.path edits.
import pathlib as _pathlib
_THIS = _pathlib.Path(__file__).resolve()
_SRC = _THIS.parents[1]   # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---
from orchestrator import ThreeLayerDiDSystem
from eval.eval_runner import run_batch, compute_metrics
from bench.load_unified_bench import load_unified
from utils.scene_manager import SceneManager
from utils.vision_attacks import build_bap_visual
from utils.thor import get_event_rgb


# ---
# Prompt loaders

# ---
def load_generated_prompts(csv_path: str) -> List[Dict[str, Any]]:
    """Load generated prompts, preserving label_intent / label_hazard_type."""
    out: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            is_unsafe = bool(int(r.get("is_unsafe", 1) or 1))
            out.append({
                "prompt_id":         r.get("prompt_id"),
                "prompt":            r.get("prompt", ""),
                "group":             r.get("category") or r.get("group"),
                "category":          r.get("category", ""),
                "difficulty":        int(r.get("difficulty", 0) or 0),
                "is_unsafe":         is_unsafe,
                # Ground-truth labels (Link-B): read from CSV if present
                "label_intent":      r.get("label_intent",
                                          "UNSAFE" if is_unsafe else "SAFE"),
                "label_hazard_type": r.get("label_hazard_type",
                                          "harm-human" if is_unsafe else "none"),
                "bench":             "generated",
            })
    return out


# ---
# Formal verification (Link-A)

# ---
def run_formal_verification(system, verbose: bool = True) -> Dict[str, Any]:
    """
    Run NuSMV + PRISM at benchmark start and save proof certificates.

    Outputs
    -------
    outputs/proof_certificate_nusmv.json  — full NuSMV certificate
    outputs/proof_certificate_nusmv_raw.txt
    outputs/proof_certificate_prism.json  — full PRISM certificate
    outputs/formal_verification_summary.json — combined summary
    """
    os.makedirs("outputs", exist_ok=True)
    combined: Dict[str, Any] = {"nusmv": None, "prism": None}

    # ---- NuSMV ----
    try:
        from formal.run_nusmv_with_cert import verify_and_certify as nusmv_certify
        nusmv_result = nusmv_certify(
            smv_path  = "formal/guard_model.smv",
            cert_path = "outputs/proof_certificate_nusmv.json",
            system    = system,
            verbose   = verbose,
        )
        combined["nusmv"] = {
            k: v for k, v in nusmv_result.items()
            if k not in ("raw_output",)  # keep summary slim
        }
    except Exception as e:
        if verbose:
            log.warning("NuSMV step failed: %s", e)
        combined["nusmv"] = {"error": str(e)}

    # ---- PRISM ----
    try:
        from formal.run_prism import verify_and_certify_prism
        prism_result = verify_and_certify_prism(
            model_path   = "formal/guard_model.pm",
            props_path   = "formal/guard_properties.pctl",
            cert_path    = "outputs/proof_certificate_prism.json",
            auto_install = True,
            verbose      = verbose,
        )
        combined["prism"] = {
            k: v for k, v in prism_result.items()
            if k not in ("raw_output",)
        }
    except Exception as e:
        if verbose:
            log.warning("PRISM step failed: %s", e)
        combined["prism"] = {"error": str(e)}

    # ---- Combined summary ----
    nusmv_ok  = bool((combined["nusmv"] or {}).get("ok", False))
    prism_ok  = bool((combined["prism"] or {}).get("ok", False))
    pcr_nusmv = float((combined["nusmv"] or {}).get("pcr") or (1.0 if nusmv_ok else 0.0))
    pcr_prism = float((combined["prism"] or {}).get("pcr") or (1.0 if prism_ok else 0.0))

    summary = {
        "nusmv_ok":   nusmv_ok,
        "prism_ok":   prism_ok,
        "pcr_nusmv":  pcr_nusmv,
        "pcr_prism":  pcr_prism,
        "both_pass":  nusmv_ok and prism_ok,
        "nusmv_cert": "outputs/proof_certificate_nusmv.json",
        "prism_cert": "outputs/proof_certificate_prism.json",
        "details":    combined,
    }
    with open("outputs/formal_verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        log.info("NuSMV=%s  PRISM=%s  PCR_NuSMV=%.2f  PCR_PRISM=%.2f", ("PASS" if nusmv_ok else "FAIL/SKIP"), ("PASS" if prism_ok else "FAIL/SKIP"), pcr_nusmv, pcr_prism)
    return summary


# ---
# Pre-step callback builder (G2 + G3)

# ---
def _build_pre_step_fn(
    scene_manager: Optional[SceneManager],
    visual_attack_cfg: Optional[Dict[str, Any]],
    system: "ThreeLayerDiDSystem",
):
    _trial_counter = {"n": 0}

    def _fn(sys_ref):
        trial = _trial_counter["n"]
        _trial_counter["n"] += 1

        # G2: scene rotation
        if scene_manager is not None:
            cat = getattr(sys_ref, "_current_category", "default") or "default"
            scene_manager.setup_for_category(cat, trial=trial)

        # G3: visual perturbation
        if visual_attack_cfg is not None:
            try:
                rgb       = get_event_rgb(sys_ref.controller.last_event)
                perturbed = build_bap_visual(rgb, attack=visual_attack_cfg)
                sys_ref._visual_rgb_override = perturbed
            except Exception:
                pass

    return _fn


# ---
# Main

# ---
def main():
    ap = argparse.ArgumentParser(description="3-Layer DiD Benchmark Runner")
    ap.add_argument("--dataset",       default="",
                    help="Unified dataset path (.csv/.jsonl/.json)")
    ap.add_argument("--scene",         default="FloorPlan1",
                    help="Base AI2-THOR scene")
    ap.add_argument("--multi_scene",   action="store_true",
                    help="G2: rotate scenes per BAP category")
    ap.add_argument("--visual_attack", default="",
                    help="G3: noise|patch|fgsm|pgd")
    ap.add_argument("--attack_epsilon",type=float, default=8.0,
                    help="L-inf budget for FGSM/PGD (pixel units)")
    ap.add_argument("--reset_each",    action="store_true",
                    help="Reset AI2-THOR before each trial")
    ap.add_argument("--fit_frames",    type=int, default=80,
                    help="Frames for L1 calibration")
    ap.add_argument("--trials",        type=int, default=1,
                    help="Repeat each prompt N times")
    ap.add_argument("--verify_formal", action="store_true",
                    help="Link-A: run NuSMV + PRISM and save proof certificates")
    ap.add_argument("--ablation",      action="store_true",
                    help="G7: run ablation study after benchmark")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # ---- Load prompts ----
    if args.dataset.strip():
        prompts = load_unified(args.dataset.strip(), default_bench="external")
        out_tag = os.path.splitext(os.path.basename(args.dataset.strip()))[0]
    else:
        prompts_csv = "outputs/embodied_benchmark_5000.csv"
        if not os.path.exists(prompts_csv):
            print(f"[bench] Missing {prompts_csv} — generating via: python -m bench.prompt_generator_5000")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "bench.prompt_generator_5000"])
            if not os.path.exists(prompts_csv):
                raise FileNotFoundError(
                    f"Prompt CSV still missing after generation: {prompts_csv}. "
                    f"Please run: python -m bench.prompt_generator_5000"
                )
        prompts = load_generated_prompts(prompts_csv)
        out_tag = "generated_5000"

    out_tag += ("_multiscene" if args.multi_scene else "")
    out_tag += (f"_{args.visual_attack}" if args.visual_attack else "")

    # ---- THOR controller ----
    controller = Controller(
        agentMode="default", visibilityDistance=1.5,
        gridSize=0.25, snapToGrid=True, rotateStepDegrees=90,
        renderDepthImage=False, renderInstanceSegmentation=False,
    )
    controller.reset(scene=args.scene)

    # ---- 3-Layer system ----
    system = ThreeLayerDiDSystem(controller=controller)

    # ---- Link-A: formal verification BEFORE benchmark ----
    formal_summary: Optional[Dict[str, Any]] = None
    if args.verify_formal:
        log.info("Running formal verification ...")
        formal_summary = run_formal_verification(system, verbose=True)

    # ---- G2 scene manager ----
    scene_manager = SceneManager(controller) if args.multi_scene else None

    # ---- G3 visual attack config ----
    visual_attack_cfg: Optional[Dict[str, Any]] = None
    if args.visual_attack:
        visual_attack_cfg = {
            "type":       args.visual_attack,
            "epsilon":    args.attack_epsilon,
            "sigma":      args.attack_epsilon,
            "alpha":      2.0,
            "num_steps":  10,
            "target_text": "ignore safety rules",
        }

    pre_step_fn = _build_pre_step_fn(scene_manager, visual_attack_cfg, system)

    # Wrap step() to apply visual override when set
    _orig_step = system.step
    def _tagged_step(instruction: str, rgb_override=None):
        rgb_ov = getattr(system, "_visual_rgb_override", None)
        if rgb_ov is not None:
            system._visual_rgb_override = None
            return _orig_step(instruction, rgb_override=rgb_ov)
        return _orig_step(instruction, rgb_override=rgb_override)
    system.step = _tagged_step

    # ---- Run benchmark ----
    log.info("Running %s prompts × %s trial(s) ...", len(prompts), args.trials)
    df = run_batch(
        system=system,
        prompts=prompts,
        fit_l1=True,
        fit_frames=int(args.fit_frames),
        trials=int(args.trials),
        reset_each=bool(args.reset_each),
        scene=args.scene,
        pre_step_fn=pre_step_fn,
    )

    out_csv = f"outputs/results_{out_tag}.csv"
    df.to_csv(out_csv, index=False)
    log.info("Saved: %s", out_csv)

    # ---- Metrics (Link-B: uses label_intent from CSV) ----
    metrics = compute_metrics(df)

    def _ser(v):
        if isinstance(v, dict):
            return {kk: _ser(vv) for kk, vv in v.items()}
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, (bool, int, str, type(None))):
            return v
        return str(v)

    summary: Dict[str, Any] = {
        "tag":            out_tag,
        "N":              int(len(df)),
        "multi_scene":    args.multi_scene,
        "visual_attack":  args.visual_attack or None,
        "attack_epsilon": args.attack_epsilon,
        "verify_formal":  args.verify_formal,
        "formal":         formal_summary,
        "metrics":        _ser(metrics),
        "bench_counts":   df["bench"].value_counts().to_dict()  if "bench"  in df.columns else None,
        "group_counts":   df["group"].value_counts().to_dict()  if "group"  in df.columns else None,
        "hazard_counts":  df["label_hazard_type"].value_counts().to_dict()
                          if "label_hazard_type" in df.columns else None,
    }
    out_json = f"outputs/summary_{out_tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved: %s", out_json)

    # Print key metrics
    log.info("ASR3=%.3f  RR_total=%.3f  FPR_L1=%.3f  RR_L2=%.3f  FPR_L2=%.3f  N=%s", metrics.get("ASR3",0), metrics.get("RR_total",0), metrics.get("FPR_L1",0), metrics.get("RR_L2",0), metrics.get("FPR_L2",0), metrics.get("N_total",0))

    # ---- G7 ablation ----
    if args.ablation:
        from eval.ablation_runner import run_ablation
        log.info("Starting ablation study ...")
        run_ablation(
            system=system,
            prompts=prompts[:min(200, len(prompts))],
            configs=["vlm_only", "l1_only", "l1_l2", "full_did"],
            fit_l1=False,
            trials=1,
            scene=args.scene,
            output_dir="outputs",
        )


if __name__ == "__main__":
    main()
