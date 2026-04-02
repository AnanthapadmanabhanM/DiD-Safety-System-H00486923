#!/usr/bin/env python3
"""
formal/run_nusmv_with_cert.py

Utility wrapper to:
  (1) Generate NuSMV model from Layer-3 specification
  (2) Execute model checking via NuSMV
  (3) Save a structured proof certificate

This produces a machine-verifiable artifact supporting the reported
PCR (Property Compliance Rate) under the abstract FSM model.

Certificate Output:
-------------------
Saved to: outputs/proof_certificate_nusmv.json

Fields include:
- tool: verification engine used (NuSMV)
- timestamp: execution time
- model_hash: SHA-256 prefix of model file
- model_path: location of generated SMV model
- property_results: per-property satisfaction status
- ok: overall verification success (all properties satisfied)
- pcr: proportion of satisfied properties (under abstract model)
- raw_output: truncated NuSMV output log
- config: configuration snapshot

PCR Definition:
---------------
PCR = 1.0 iff all specified LTL safety properties evaluate to TRUE
in the abstract finite-state model.

Note:
This reflects formal guarantees at the model level and does not
imply absolute real-world safety.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from formal.gen_nusmv import main as _gen_main, LTL_SPECS, ACTIONS
from formal.run_nusmv import run_nusmv, save_certificate


# ---
# Helpers

# ---
def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else "unavailable"
    except Exception:
        return "unavailable"


def _file_sha256(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unavailable"


def _property_specs() -> List[Dict[str, str]]:
    return [{"name": name, "formula": formula} for name, formula in LTL_SPECS]


def _cfg_snapshot(system) -> Dict[str, Any]:
    """Best-effort extraction of layer configs from a ThreeLayerDiDSystem."""
    snap: Dict[str, Any] = {}
    if system is None:
        return snap
    for attr in ("l1_cfg", "l2_cfg", "l3_cfg"):
        val = getattr(system, attr, None)
        if val is not None:
            try:
                snap[attr] = {k: str(v) for k, v in dict(val).items()}
            except Exception:
                snap[attr] = str(val)
    return snap


# ---
# Main public API

# ---
def verify_and_certify(
    smv_path:  str = "formal/guard_model.smv",
    cert_path: str = "outputs/proof_certificate_nusmv.json",
    timeout_s: int = 120,
    system=None,
    verbose:   bool = True,
) -> Dict[str, Any]:
    """
    Generate .smv → run NuSMV → save fully-specified proof certificate.

    Parameters
    ----------
    smv_path   : where to write the generated .smv model
    cert_path  : where to write the JSON certificate
    timeout_s  : NuSMV subprocess timeout
    system     : optional ThreeLayerDiDSystem (for config snapshot)
    verbose    : print progress to stdout

    Returns a summary dict with keys:
      available, ok, pcr, spec_results, cert_path, message
    """
    for d in [os.path.dirname(smv_path), os.path.dirname(cert_path)]:
        if d:
            os.makedirs(d, exist_ok=True)

    # --- Step 1: (re-)generate .smv ---
    _old_argv = sys.argv[:]
    sys.argv = ["gen_nusmv", "--out", smv_path]
    try:
        _gen_main()
    except SystemExit:
        pass
    finally:
        sys.argv = _old_argv

    if verbose:
        print(f"[NuSMV] Generated model: {smv_path}")

    model_hash = _file_sha256(smv_path)

    # --- Step 2: run NuSMV ---
    result = run_nusmv(smv_path, timeout_s=timeout_s)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    git_ref   = _git_commit()

    # --- Step 3: build full certificate payload ---
    pcr = 1.0 if (result.available and result.ok) else 0.0

    certificate: Dict[str, Any] = {
        "tool":           "nusmv",
        "timestamp":      timestamp,
        "git_commit":     git_ref,
        "model_sha256":   model_hash,
        "model_path":     smv_path,
        "property_specs": _property_specs(),
        "actions":        ACTIONS,
        "nusmv_path":     result.nusmv_path,
        "available":      result.available,
        "ok":             result.ok,
        "spec_results":   result.spec_results,
        "pcr":            pcr,
        "raw_output":     result.raw_output[-20_000:],
        "error":          result.error,
        "config":         _cfg_snapshot(system),
        "pcr_definition": (
            "PCR=1.0 iff all LTLSPEC properties in property_specs "
            "are True in NuSMV's exhaustive LTL model checking. "
            "This certifies G not(action=harmful AND executed) "
            "holds for ALL reachable states of the Layer-3 FSM."
        ),
    }

    # --- Step 4: save ---
    os.makedirs(os.path.dirname(cert_path) or ".", exist_ok=True)
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2)

    # --- Step 5: also save raw NuSMV stdout as .txt for audit trail ---
    raw_txt_path = cert_path.replace(".json", "_raw.txt")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(result.raw_output or "(NuSMV not available — no output)")

    # --- Summary dict (backward-compat with old callers) ---
    if not result.available:
        if verbose:
            print("[NuSMV] NuSMV binary not found.")
            print("        apt install nusmv  OR  see https://nusmv.fbk.eu/")
        summary: Dict[str, Any] = {
            "available": False, "ok": False, "pcr": None,
            "message": "NuSMV binary not found.",
            "spec_results": {}, "cert_path": cert_path,
        }
        return summary

    summary = {
        "tool":           "nusmv",
        "available":      True,
        "ok":             result.ok,
        "pcr":            pcr,
        "spec_results":   result.spec_results,
        "cert_path":      cert_path,
        "raw_txt_path":   raw_txt_path,
        "git_commit":     git_ref,
        "model_sha256":   model_hash,
        "timestamp":      timestamp,
        "all_properties_hold": result.ok,
        "message": (
            f"PCR=1.00 — all {len(result.spec_results)} LTL properties verified."
            if result.ok else
            f"PCR=0.00 — FAILED: "
            f"{[k for k,v in result.spec_results.items() if not v]}"
        ),
    }

    if verbose:
        print(f"[NuSMV] git={git_ref}  model_sha256={model_hash}  ts={timestamp}")
        status = "PASS ✓" if result.ok else "FAIL ✗"
        print(f"[NuSMV] Verification: {status}  PCR={pcr:.2f}")
        for name, held in result.spec_results.items():
            print(f"  {'OK' if held else 'FAIL'} : {name}")
        print(f"[NuSMV] Certificate  → {cert_path}")
        print(f"[NuSMV] Raw output   → {raw_txt_path}")

    return summary
