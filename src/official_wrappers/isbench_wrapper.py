"""
official_wrappers/isbench_wrapper.py
-------------------------------------------------------------------
Drop-in wrapper for IS-Bench (https://github.com/AI45Lab/IS-Bench)

IS-Bench uses OmniGibson (NOT AI2-THOR). The full DiD system (L1+L2+L3)
cannot run directly because L1 and L3 depend on AI2-THOR state. However,
L2's semantic analysis operates on text instructions and can guard actions
in any simulator.

Integration strategy:
  - L2 (semantic guard): fully functional — checks instruction text
  - L1 (perception guard): limited — can check OmniGibson frames if
    provided, but CLIP embeddings may need recalibration
  - L3 (reference monitor): NOT available — depends on AI2-THOR state

Integration point:
  IS-Bench agent loop:  observation → VLM proposes action → execute in OmniGibson
  With DiD L2:          observation → VLM proposes action → L2 checks instruction
                                                          → if REFUSE: skip/safe action
                                                          → if ALLOW: execute in OmniGibson

Usage:
  1. Clone IS-Bench: git clone https://github.com/AI45Lab/IS-Bench.git
  2. Copy this file into IS-Bench/
  3. In their agent code, wrap the action proposal with DiDActionFilter

  Example:
    from isbench_wrapper import DiDActionFilter

    guard = DiDActionFilter(l2_guard=l2_instance)

    # In the agent's step loop:
    proposed_action = vlm.propose_action(observation)
    filtered_action = guard.filter_action(
        instruction=task_instruction,
        proposed_action=proposed_action,
        visible_objects=scene_objects,
    )
    # filtered_action is either the original action or a safe replacement

This wrapper does NOT modify IS-Bench's process-oriented scoring.
It only adds a semantic filter that can block unsafe actions.
-------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# Safe fallback actions for OmniGibson (equivalents of THOR's RotateRight)
OMNIGIBSON_SAFE_ACTIONS = [
    "idle",           # do nothing
    "rotate_right",   # safe navigation action
]


class DiDActionFilter:
    """
    Lightweight action filter using DiD's L2 semantic guard.

    For IS-Bench (OmniGibson), we can only use L2 (text-based semantic
    analysis) since L1 and L3 depend on AI2-THOR state.

    This still provides value because IS-Bench's key safety failures
    are in instruction understanding — exactly what L2 targets.
    """

    def __init__(
        self,
        l2_guard=None,
        safe_action: str = "idle",
        enabled: bool = True,
        log_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        l2_guard    : Layer2SemanticGuard instance
        safe_action : action to substitute when blocking (OmniGibson action)
        enabled     : if False, acts as passthrough
        log_path    : optional JSONL log path
        """
        self.l2_guard = l2_guard
        self.safe_action = safe_action
        self.enabled = enabled
        self.log_path = log_path
        self.decision_log: List[Dict[str, Any]] = []

    def filter_action(
        self,
        instruction: str,
        proposed_action: str,
        visible_objects: Optional[List[str]] = None,
        holding: bool = False,
        rgb=None,
    ) -> Dict[str, Any]:
        """
        Filter a proposed action through L2 semantic analysis.

        Parameters
        ----------
        instruction      : the task instruction
        proposed_action  : the VLM's proposed action string
        visible_objects  : list of visible object types
        holding          : whether the agent is holding something
        rgb              : optional OmniGibson RGB frame

        Returns
        -------
        dict with:
            action          : the action to execute (original or safe replacement)
            blocked         : whether the action was blocked
            guard_decision  : L2 decision
            reason          : explanation
        """
        if not self.enabled:
            return {
                "action": proposed_action,
                "blocked": False,
                "guard_decision": "NO_GUARD",
                "reason": "Guard disabled (baseline mode)",
            }

        t0 = time.time()
        l2_decision, l2_reason, l2_risk = self._check_semantic_safety(
            instruction, proposed_action, visible_objects, holding, rgb
        )
        guard_time = (time.time() - t0) * 1000.0

        blocked = (l2_decision == "REFUSE")
        result_action = self.safe_action if blocked else proposed_action

        log_entry = {
            "instruction_preview": instruction[:200],
            "proposed_action": proposed_action,
            "l2_decision": l2_decision,
            "l2_reason": l2_reason,
            "l2_risk": l2_risk,
            "blocked": blocked,
            "executed_action": result_action,
            "guard_latency_ms": guard_time,
        }
        self.decision_log.append(log_entry)

        if blocked:
            log.info("DiD L2 BLOCKED action '%s' for instruction: %s (risk=%s)",
                     proposed_action, instruction[:80], l2_risk)

        return {
            "action": result_action,
            "blocked": blocked,
            "guard_decision": l2_decision,
            "reason": l2_reason if blocked else "Allowed",
        }

    def _check_semantic_safety(
        self, instruction, proposed_action, visible_objects, holding, rgb
    ):
        """Run L2 semantic classification."""
        import numpy as np

        if self.l2_guard is not None:
            # Combine instruction + proposed action for richer context
            combined_instruction = f"{instruction} [Proposed: {proposed_action}]"
            dummy_rgb = rgb if rgb is not None else np.zeros((300, 300, 3), dtype=np.uint8)

            result = self.l2_guard.decide(
                rgb=dummy_rgb,
                instruction=combined_instruction,
                visible_object_types=visible_objects or [],
                holding=holding,
            )
            return (
                result.get("decision", "ALLOW"),
                result.get("reason", ""),
                result.get("risk_level", "LOW"),
            )

        # Fallback: standalone intent analysis
        try:
            from layers.l2_semantic_guard import analyze_intent_risk
            analysis = analyze_intent_risk(instruction)
            risk = analysis.get("risk_level", "LOW")
            if risk in ("HIGH", "CRITICAL"):
                return ("REFUSE", analysis.get("rationale", ""), risk)
            return ("ALLOW", "Low/moderate risk", risk)
        except ImportError:
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

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.decision_log)
        blocked = sum(1 for e in self.decision_log if e.get("blocked"))
        return {
            "total_actions": total,
            "blocked": blocked,
            "allowed": total - blocked,
            "block_rate": blocked / max(total, 1),
            "layers_used": "L2 only (OmniGibson — no L1/L3)",
        }


# ---
# Integration notes for IS-Bench's evaluation scripts
# ---
INTEGRATION_NOTES = """
IS-Bench Official Integration Guide
====================================

IS-Bench supports evaluation via:
  bash entrypoints/eval_close.sh $MODEL_NAME $DATA_PARALLEL   # API models
  bash entrypoints/eval_open.sh ...                            # open models

To add DiD as a guard layer:

1. Locate the agent's action-proposal function in IS-Bench's code
   (typically in the model-specific evaluation script).

2. After the VLM proposes an action, route it through DiDActionFilter:

   guard = DiDActionFilter(l2_guard=l2_instance)
   result = guard.filter_action(
       instruction=task_instruction,
       proposed_action=vlm_proposed_action,
       visible_objects=scene_state.visible_objects,
   )
   if result["blocked"]:
       # Execute safe action instead
       action_to_execute = result["action"]  # "idle"
   else:
       action_to_execute = result["action"]  # original

3. IS-Bench's process-oriented scorer evaluates the action SEQUENCE.
   When DiD blocks an action, the safe replacement ("idle") enters
   the sequence instead. This may affect:
   - Mitigation ordering (positive: blocks dangerous actions early)
   - Task completion (negative: may prevent necessary actions)
   Both effects are measured by their scorer — no changes needed.

4. Their prompt settings v0-v3 add safety awareness to the VLM prompt.
   DiD is an ADDITIONAL layer on top of these settings, so the
   comparison should be:
   - v0 baseline vs v0+DiD
   - v3 baseline vs v3+DiD
   This shows DiD's added value beyond prompt-based safety.

Limitations:
- Only L2 (semantic guard) is available — L1/L3 require AI2-THOR state
- If this limitation reduces the effect size, acknowledge it honestly
"""
