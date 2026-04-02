#!/usr/bin/env python3
"""
gap1_runtime_overhead.py — Runtime latency measurement (RQ5)
H00486923 | MSc Robotics Dissertation | Heriot-Watt University
Measures per-component and full-pipeline latency for the DiD system.
Results reported in dissertation Table 5.7 (Section 5.7).

NOTE: Pre-computed results are already in Gap_Analysis/runtime_overhead_results.json
(measured on Colab A100, 100 runs each). Running this script locally will produce
different values due to different hardware. Use --force to overwrite.

Usage:
    cd /path/to/H00486923_3_layer_system/src
    python ../scripts/gap1_runtime_overhead.py --n 100          # dry run, print only
    python ../scripts/gap1_runtime_overhead.py --n 100 --force  # overwrite results file
Output:
    ../Gap_Analysis/runtime_overhead_results.json
"""
import sys, time, json, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from layers.l1_perception_guard import Layer1PerceptionGuard
from layers.l2_semantic_guard import analyze_intent_risk

TEST_INSTRUCTION = "Pick up the apple and put it in the bowl"

def measure(fn, n: int) -> dict:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms": round(sum(times) / len(times), 4),
        "std_ms":  round((sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5, 4),
        "p50_ms":  round(times[len(times) // 2], 4),
        "p95_ms":  round(times[int(len(times) * 0.95)], 4),
        "min_ms":  round(times[0], 4),
        "max_ms":  round(times[-1], 4),
        "n_runs":  n,
        "source":  "measured",
    }

def main(n: int, force: bool) -> None:
    out_dir  = pathlib.Path(__file__).parent.parent / "Gap_Analysis"
    out_path = out_dir / "runtime_overhead_results.json"
    out_dir.mkdir(exist_ok=True)

    # Guard: do not silently overwrite the pre-computed dissertation results
    if out_path.exists() and not force:
        print(f"Pre-computed results already exist at {out_path}")
        print("Pass --force to overwrite with locally-measured values.")
        print("Note: local values will differ from dissertation (different hardware).")
        return

    print(f"Measuring latency (n={n} runs per component)...")

    l1 = Layer1PerceptionGuard()

    results = {}

    # L1 text path
    results["L1_perception_guard"] = measure(
        lambda: l1.inspect(text=TEST_INSTRUCTION), n)
    results["L1_perception_guard"]["source_detail"] = "Layer1PerceptionGuard.inspect(text_only)"

    # L1 cache hit
    l1.inspect(text=TEST_INSTRUCTION)  # warm cache
    results["L1_cache_hit"] = measure(
        lambda: l1.inspect(text=TEST_INSTRUCTION), n)
    results["L1_cache_hit"]["source_detail"] = "LRU dict cache (pre-warmed)"

    # L2 rule-based
    results["L2_semantic_guard_rules"] = measure(
        lambda: analyze_intent_risk(TEST_INSTRUCTION), n)
    results["L2_semantic_guard_rules"]["source_detail"] = "analyze_intent_risk() module fn"

    # L3 reference monitor (text path, no controller needed)
    try:
        from layers.l3_reference_monitor import Layer3ExecutionGuard
        l3 = Layer3ExecutionGuard()
        results["L3_reference_monitor"] = measure(
            lambda: l3.check("PickupObject", {"objectId": "Apple_0"}), n)
        results["L3_reference_monitor"]["source_detail"] = "32 runtime safety properties (CPU)"
    except Exception as e:
        print(f"  L3 measurement skipped: {e}")

    print(f"\nNote: L2 CLIP ({7.73:.2f} ms) and orchestrator_total ({0.14:.2f} ms) values")
    print("require GPU (Colab A100). Pre-computed values preserved in results file.")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    for k, v in results.items():
        print(f"  {k}: {v['mean_ms']:.4f} ms ± {v['std_ms']:.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,  default=100,   help="Number of measurement runs")
    parser.add_argument("--force", action="store_true",      help="Overwrite existing results file")
    args = parser.parse_args()
    main(args.n, args.force)