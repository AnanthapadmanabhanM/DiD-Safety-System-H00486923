from __future__ import annotations

import os
import re
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

@dataclass
class NuSMVResult:
    available: bool
    ok: bool
    spec_results: Dict[str, bool]
    raw_output: str
    nusmv_path: Optional[str] = None
    smv_path: Optional[str] = None
    error: Optional[str] = None


def _which_nusmv() -> Optional[str]:
    return shutil.which("nusmv") or shutil.which("NuSMV") or shutil.which("nusmv.exe")


def run_nusmv(smv_path: str, timeout_s: int = 120) -> NuSMVResult:
    """Run NuSMV on an .smv file and parse LTLSPEC results.

    This is intentionally robust:
      - If NuSMV is not installed, it returns available=False (no crash).
      - It parses common output formats: "-- specification ... is true/false".
    """
    nusmv = _which_nusmv()
    if nusmv is None:
        return NuSMVResult(
            available=False,
            ok=False,
            spec_results={},
            raw_output="",
            nusmv_path=None,
            smv_path=smv_path,
            error="NuSMV binary not found in PATH.",
        )

    try:
        proc = subprocess.run(
            [nusmv, smv_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
        out = proc.stdout or ""
    except Exception as e:
        return NuSMVResult(
            available=True,
            ok=False,
            spec_results={},
            raw_output="",
            nusmv_path=nusmv,
            smv_path=smv_path,
            error=str(e),
        )

    spec_results: Dict[str, bool] = {}
    # Matches: -- specification <spec>  is true
    for m in re.finditer(r"--\s+specification\s+(.*?)\s+is\s+(true|false)", out, flags=re.IGNORECASE):
        spec = m.group(1).strip()
        val = m.group(2).lower() == "true"
        spec_results[spec] = val

    ok = bool(spec_results) and all(spec_results.values())
    return NuSMVResult(
        available=True,
        ok=ok,
        spec_results=spec_results,
        raw_output=out,
        nusmv_path=nusmv,
        smv_path=smv_path,
        error=None if proc.returncode == 0 else f"NuSMV exit code {proc.returncode}",
    )


def save_certificate(result: NuSMVResult, out_path: str) -> str:
    """Save a 'proof certificate' JSON (and embed raw output)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "available": result.available,
        "ok": result.ok,
        "spec_results": result.spec_results,
        "nusmv_path": result.nusmv_path,
        "smv_path": result.smv_path,
        "error": result.error,
        "raw_output": result.raw_output[-20000:],  # keep last 20k chars
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_certificate(cert_path: str) -> Dict[str, Any]:
    with open(cert_path, "r", encoding="utf-8") as f:
        return json.load(f)
