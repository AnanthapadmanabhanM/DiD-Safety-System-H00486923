from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---
# Unified benchmark schema loader
#
# Purpose:
# This module converts heterogeneous benchmark datasets (SafeMind, IS-Bench, AgentSafe)
# into a unified schema compatible with eval_runner.py, enabling reproducible evaluation
# without modifying the core pipeline.
#
# Usage:
# python -m bench.load_unified_bench --dataset path/to/file.csv

# Supported input formats:
#   - CSV (recommended)
#   - JSONL (one JSON object per line)
#
# Minimal required field (any of these will be accepted):
#   - prompt OR instruction OR text
#
# Recommended fields (preserved into the prompt dict):
#   - id / prompt_id
#   - bench (safemind | isbench | agentsafe | custom)
#   - category / group
#   - risk
#   - scene (defaults to caller scene)
#   - is_unsafe (0/1 or true/false). If missing, defaults to 1 for safety datasets.

# ---

_TEXT_KEYS = ("prompt", "instruction", "text", "query")

def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return str(d[k])
    return None

def _to_bool(x: Any, default: bool = True) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return default

def _normalize_row(r: Dict[str, Any], default_bench: str = "external") -> Dict[str, Any]:
    prompt = _first_present(r, _TEXT_KEYS)
    if prompt is None:
        raise ValueError(f"Row missing text field (expected one of {_TEXT_KEYS}). Keys={list(r.keys())[:20]}")

    out: Dict[str, Any] = dict(r)

    # standardize id
    if "prompt_id" not in out:
        out["prompt_id"] = out.get("id") or out.get("uid") or out.get("example_id")

    out["prompt"] = prompt

    # group/category
    if "group" not in out:
        out["group"] = out.get("category") or out.get("topic") or out.get("type")

    # bench
    if "bench" not in out:
        out["bench"] = out.get("source") or out.get("dataset") or default_bench

    # is_unsafe default: True (these safety benchmarks are mostly unsafe prompts)
    out["is_unsafe"] = _to_bool(out.get("is_unsafe", None), default=True)

    return out

def load_unified(path: str, default_bench: str = "external") -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    path_l = path.lower()
    rows: List[Dict[str, Any]] = []

    if path_l.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r is None:
                    continue
                rows.append(_normalize_row(r, default_bench=default_bench))

    elif path_l.endswith(".jsonl") or path_l.endswith(".json"):
        # JSONL preferred. If .json, accept list[dict] too.
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if path_l.endswith(".json"):
            obj = json.loads(txt)
            if isinstance(obj, list):
                for r in obj:
                    rows.append(_normalize_row(dict(r), default_bench=default_bench))
            elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                for r in obj["data"]:
                    rows.append(_normalize_row(dict(r), default_bench=default_bench))
            else:
                rows.append(_normalize_row(dict(obj), default_bench=default_bench))
        else:
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(_normalize_row(dict(r), default_bench=default_bench))
    else:
        raise ValueError("Unsupported file type. Use .csv, .jsonl, or .json")

    return rows
