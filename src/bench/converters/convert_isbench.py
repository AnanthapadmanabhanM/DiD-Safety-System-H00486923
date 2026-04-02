"""Converter template.

These scripts intentionally make minimal assumptions because different releases
of SafeMind / IS-Bench / AgentSafe may store fields differently.

Usage examples (run from repo root):

  python -m bench.converters.convert_safemind --in raw.jsonl --out bench/datasets/safemind_unified.csv --bench safemind
  python -m bench.converters.convert_isbench --in raw.csv --out bench/datasets/isbench_unified.csv --bench isbench
  python -m bench.converters.convert_agentsafe --in raw.json --out bench/datasets/agentsafe_unified.csv --bench agentsafe

After conversion, we can MERGE CSVs:
  python -m bench.converters.merge_unified --outs bench/datasets/unified_bench.csv \
      bench/datasets/safemind_unified.csv bench/datasets/isbench_unified.csv bench/datasets/agentsafe_unified.csv

Then run:
  python -m bench.run_benchmark_colab --dataset bench/datasets/unified_bench.csv

"""
from __future__ import annotations

import logging
log = logging.getLogger(__name__)


import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

_TEXT_KEYS = ("prompt", "instruction", "text", "query")

def _get_text(r: Dict[str, Any]) -> Optional[str]:
    for k in _TEXT_KEYS:
        v = r.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    # common alternates
    for k in ("input", "user", "request", "question"):
        v = r.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return None

def _read_any(path: str) -> List[Dict[str, Any]]:
    pl = path.lower()
    if pl.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    if pl.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    if pl.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        return [obj]
    raise ValueError("Unsupported input type. Use .csv, .jsonl, or .json")

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        raise ValueError("No rows to write.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # stable columns
    cols = ["bench","prompt_id","prompt","group","category","risk","is_unsafe","scene","source_path"]
    # include any extra columns present
    extra = []
    for r in rows:
        for k in r.keys():
            if k not in cols and k not in extra:
                extra.append(k)
    cols = cols + extra
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

def main(default_bench: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Raw benchmark file (.csv/.jsonl/.json)")
    ap.add_argument("--out", dest="out", required=True, help="Unified output CSV")
    ap.add_argument("--bench", dest="bench", default=default_bench, help="bench name stored in output")
    ap.add_argument("--scene", dest="scene", default="", help="optional fixed scene (e.g., FloorPlan1)")
    args = ap.parse_args()

    raw = _read_any(args.inp)
    unified: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        text = _get_text(r)
        if text is None:
            # skip rows we can't interpret
            continue

        u = {
            "bench": args.bench,
            "prompt_id": r.get("prompt_id") or r.get("id") or r.get("uid") or f"{args.bench}_{i}",
            "prompt": text,
            "group": r.get("group") or r.get("category") or r.get("type") or "",
            "category": r.get("category") or r.get("group") or "",
            "risk": r.get("risk") or r.get("harm") or r.get("label") or "",
            "is_unsafe": r.get("is_unsafe", 1),
            "scene": r.get("scene") or args.scene,
            "source_path": args.inp,
        }
        # keep a few raw hints (won't break anything)
        for k in ("subtype","domain","attack_type","difficulty","severity"):
            if k in r and k not in u:
                u[k] = r.get(k)
        unified.append(u)

    _write_csv(args.out, unified)
    log.info("Wrote: %s rows=%s", args.out, len(unified))

if __name__ == "__main__":
    main(default_bench="isbench")