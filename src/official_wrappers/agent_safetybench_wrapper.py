"""
official_wrappers/agent_safetybench_wrapper.py
-------------------------------------------------------------------
Drop-in wrapper for Agent-SafetyBench (https://github.com/thu-coai/Agent-SafetyBench)

Agent-SafetyBench is text-based: it sends tool-use prompts to an LLM and
scores whether the LLM's response is safe/unsafe using their ShieldAgent scorer.

Integration point:
  Their pipeline:  prompt → LLM API → response → ShieldAgent scorer
  With DiD:        prompt → L2 semantic guard → if REFUSE: canned refusal
                                               → if ALLOW: LLM API → response → ShieldAgent scorer

Usage:
  1. Clone Agent-SafetyBench: git clone https://github.com/thu-coai/Agent-SafetyBench.git
  2. Copy this file into Agent-SafetyBench/evaluation/
  3. In their eval.py, replace the model API call with DiDModelWrapper

  Example:
    from agent_safetybench_wrapper import DiDModelWrapper

    # Their original model:
    original_model = load_model("gpt-4o")

    # Wrapped with DiD:
    wrapped_model = DiDModelWrapper(original_model, l2_guard)
    response = wrapped_model.generate(prompt)  # L2 filters first

This wrapper does NOT modify their ShieldAgent scoring.
It only adds a pre-filter that can refuse before the LLM sees the prompt.
-------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# Standard refusal response when DiD blocks a prompt
DEFAULT_REFUSAL = (
    "I'm sorry, but I cannot assist with this request. "
    "This action appears to involve potential safety risks. "
    "I would recommend consulting appropriate safety guidelines."
)


class DiDModelWrapper:
    """
    Wraps an LLM model API with DiD's L2 semantic guard as a pre-filter.

    For Agent-SafetyBench, the L2 guard checks the instruction text for
    unsafe intent BEFORE passing it to the LLM. If L2 classifies the
    prompt as HIGH/CRITICAL risk, the wrapper returns a refusal response
    without calling the LLM at all.

    This tests the claim: "Adding DiD's semantic guard reduces unsafe
    model outputs in Agent-SafetyBench's evaluation."
    """

    def __init__(
        self,
        original_model,
        l2_guard=None,
        refusal_text: str = DEFAULT_REFUSAL,
        enabled: bool = True,
        log_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        original_model : the LLM model object with a .generate() method
                         (matches Agent-SafetyBench's model API interface)
        l2_guard       : Layer2SemanticGuard instance (from layers/l2_semantic_guard.py)
                         If None, only text-based heuristics are used.
        refusal_text   : response to return when DiD blocks a prompt
        enabled        : if False, acts as passthrough (baseline)
        log_path       : optional JSONL log file path
        """
        self.model = original_model
        self.l2_guard = l2_guard
        self.refusal_text = refusal_text
        self.enabled = enabled
        self.log_path = log_path
        self.decision_log: List[Dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response, optionally filtered by DiD's L2 guard.

        This method matches Agent-SafetyBench's expected model interface.
        """
        if not self.enabled:
            return self.model.generate(prompt, **kwargs)

        # --- L2 semantic classification ---
        t0 = time.time()
        l2_decision, l2_reason, l2_risk = self._classify_prompt(prompt)
        guard_time = (time.time() - t0) * 1000.0

        log_entry = {
            "prompt_preview": prompt[:200],
            "l2_decision": l2_decision,
            "l2_reason": l2_reason,
            "l2_risk_level": l2_risk,
            "guard_latency_ms": guard_time,
        }

        if l2_decision == "REFUSE":
            log_entry["response"] = "REFUSAL (DiD blocked)"
            log_entry["llm_called"] = False
            self.decision_log.append(log_entry)
            log.info("DiD L2 BLOCKED prompt: %s (risk=%s)", l2_reason, l2_risk)
            return self.refusal_text

        # --- L2 ALLOWED — pass to LLM ---
        response = self.model.generate(prompt, **kwargs)
        log_entry["response_preview"] = response[:200] if response else ""
        log_entry["llm_called"] = True
        self.decision_log.append(log_entry)

        return response

    def _classify_prompt(self, prompt: str):
        """
        Classify a prompt using L2's semantic analysis.

        Returns (decision, reason, risk_level)
        """
        import numpy as np

        if self.l2_guard is not None:
            # Use the full L2 guard with CLIP + rules
            # Create a dummy RGB (Agent-SafetyBench is text-only)
            dummy_rgb = np.zeros((300, 300, 3), dtype=np.uint8)
            result = self.l2_guard.decide(
                rgb=dummy_rgb,
                instruction=prompt,
                visible_object_types=[],
                holding=False,
            )
            return (
                result.get("decision", "ALLOW"),
                result.get("reason", ""),
                result.get("risk_level", "LOW"),
            )

        # Fallback: use L2's standalone analyze_intent_risk if available
        try:
            from layers.l2_semantic_guard import analyze_intent_risk
            analysis = analyze_intent_risk(prompt)
            risk = analysis.get("risk_level", "LOW")
            if risk in ("HIGH", "CRITICAL"):
                return ("REFUSE", analysis.get("rationale", "Unsafe pattern"), risk)
            return ("ALLOW", "Low/moderate risk", risk)
        except ImportError:
            log.warning("L2 guard not available; passing all prompts through")
            return ("ALLOW", "No guard available", "UNKNOWN")

    def save_log(self, path: Optional[str] = None):
        """Save decision log as JSONL."""
        path = path or self.log_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for entry in self.decision_log:
                f.write(json.dumps(entry, default=str) + "\n")
        log.info("Saved decision log (%d entries) → %s",
                 len(self.decision_log), path)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.decision_log)
        blocked = sum(1 for e in self.decision_log if e.get("l2_decision") == "REFUSE")
        return {
            "total_prompts": total,
            "blocked": blocked,
            "passed": total - blocked,
            "block_rate": blocked / max(total, 1),
        }


# ---
# Convenience: batch evaluation matching their eval.py interface
# ---
def evaluate_agent_safetybench(
    data_path: str,
    model,
    l2_guard=None,
    output_dir: str = "evaluation/evaluation_results",
    enabled: bool = True,
):
    """
    Run Agent-SafetyBench evaluation with optional DiD guard.

    Parameters
    ----------
    data_path   : path to Agent-SafetyBench data/ directory
    model       : LLM model with .generate() method
    l2_guard    : optional Layer2SemanticGuard
    output_dir  : where to save results
    enabled     : True for DiD mode, False for baseline
    """
    mode = "did" if enabled else "baseline"
    os.makedirs(output_dir, exist_ok=True)

    wrapped = DiDModelWrapper(
        original_model=model,
        l2_guard=l2_guard,
        enabled=enabled,
        log_path=os.path.join(output_dir, f"did_guard_log_{mode}.jsonl"),
    )

    # Load their test data
    results = []
    for data_file in sorted(os.listdir(data_path)):
        if not data_file.endswith(".json"):
            continue

        with open(os.path.join(data_path, data_file)) as f:
            tasks = json.load(f)

        for task in tasks:
            prompt = task.get("prompt", "")
            response = wrapped.generate(prompt)
            results.append({
                "task_id": task.get("id", ""),
                "category": task.get("category", ""),
                "response": response,
            })

    # Save in their expected format for scoring
    results_path = os.path.join(output_dir, f"model_output_{mode}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    wrapped.save_log()
    log.info("Evaluation complete: %d tasks, %s",
             len(results), wrapped.get_summary())

    return results, wrapped.get_summary()
