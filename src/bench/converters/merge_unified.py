from __future__ import annotations

import logging
log = logging.getLogger(__name__)


import argparse
import csv
import os
from typing import Dict, Any, List

def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        raise ValueError("No rows.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # union columns
    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output unified CSV")
    ap.add_argument("inputs", nargs="+", help="input unified CSVs")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        rows.extend(read_csv(p))
    write_csv(args.out, rows)
    log.info("Merged rows: %s -> %s", len(rows), args.out)

if __name__ == "__main__":
    main()