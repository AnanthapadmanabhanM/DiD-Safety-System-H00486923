"""
official_wrappers/safeagentbench_wrapper.py
-------------------------------------------------------------------
Drop-in wrapper for SafeAgentBench (https://github.com/shengyin1224/SafeAgentBench)

SafeAgentBench uses AI2-THOR (v5.0 per their paper; confirm via their
requirements.txt) with 17 high-level actions. This is the best fit because
the DiD system targets the same simulator — no translation layer needed.

Integration point:
  Their pipeline:  LLM planner -> high_level_action -> low_level_controller -> AI2-THOR
  With DiD:        LLM planner -> high_level_action -> guard_action() -> low_level_controller -> AI2-THOR

Usage:
  1. Clone SafeAgentBench:  git clone https://github.com/shengyin1224/SafeAgentBench.git
  2. Copy this file into SafeAgentBench/
  3. In their agent code, replace direct controller calls with guarded calls

  Example — wrapping their low-level controller:
    from safeagentbench_wrapper import DiDGuardedController

    # Instead of: controller = LowLevelController(...)
    guarded = DiDGuardedController(controller, did_system, instruction)
    guarded.execute(high_level_action)  # routes through guard_action()

This wrapper does NOT modify their evaluation scripts, scoring,
or task definitions. It only adds a guard layer between the planner and executor.
-------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---
# SafeAgentBench's 17 high-level actions → AI2-THOR action mapping
# (from their paper / SafeAgentEnv code)
# ---
SAFEAGENTBENCH_ACTIONS = {
    "PickupObject", "PutObject", "OpenObject", "CloseObject",
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "ThrowObject",
    "PushObject", "PullObject", "BreakObject", "DirtyObject",
    "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject",
    "UseUpObject", "DropHandObject",
}


class DiDGuardedController:
    """
    Wraps SafeAgentBench's low-level controller with DiD guard_action().

    Drop-in replacement: wherever their code calls controller.execute(action),
    use this wrapper instead to route through L1/L2/L3 before execution.
    """

    def __init__(
        self,
        original_controller,
        did_system,
        log_path: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        original_controller : SafeAgentBench's low-level controller instance
        did_system          : ThreeLayerDiDSystem instance (from orchestrator.py)
        log_path            : optional path to save guard decision log (JSONL)
        enabled             : if False, acts as passthrough (for baseline comparison)
        """
        self.controller = original_controller
        self.did_system = did_system
        self.enabled = enabled
        self.log_path = log_path
        self.guard_log: List[Dict[str, Any]] = []
        self._current_instruction = ""
        self._step_count = 0

    def set_instruction(self, instruction: str):
        """Set the current task instruction (call once per episode)."""
        self._current_instruction = instruction
        self._step_count = 0

    def execute(self, action_name: str, object_id: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute a high-level action, optionally guarded by DiD.

        Parameters
        ----------
        action_name : one of SafeAgentBench's 17 high-level actions
        object_id   : target object (if applicable)

        Returns
        -------
        dict with execution result + guard metadata
        """
        # Build the THOR-style action dict
        action_dict = {"action": action_name}
        if object_id:
            action_dict["objectId"] = object_id
        action_dict.update(kwargs)

        self._step_count += 1

        if not self.enabled or self.did_system is None:
            # Passthrough: execute directly (baseline mode)
            result = self._execute_original(action_dict)
            result["guard_decision"] = "NO_GUARD"
            return result

        # --- Route through DiD guard_action() ---
        t0 = time.time()

        # Get current RGB frame from THOR controller
        thor_controller = self._get_thor_controller()
        rgb = None
        if thor_controller and hasattr(thor_controller, "last_event"):
            ev = thor_controller.last_event
            rgb = getattr(ev, "frame", None)

        guard_result = self.did_system.guard_action(
            instruction=self._current_instruction,
            proposed_action=action_dict,
            rgb_override=rgb,
        )
        guard_time = (time.time() - t0) * 1000.0

        guard_decision = guard_result.get("guard_decision", "UNKNOWN")
        l3_status = (guard_result.get("l3") or {}).get("status", "")

        # Log the guard decision
        log_entry = {
            "step": self._step_count,
            "instruction": self._current_instruction,
            "proposed_action": action_dict,
            "guard_decision": guard_decision,
            "l1_ok": (guard_result.get("l1") or {}).get("l1_ok"),
            "l2_decision": (guard_result.get("l2") or {}).get("decision"),
            "l3_status": l3_status,
            "guard_latency_ms": guard_time,
            "action_modifications": guard_result.get("action_modifications", []),
        }
        self.guard_log.append(log_entry)

        if guard_decision in ("BLOCKED_BY_L1", "REFUSED_BY_L2", "INTERCEPTED_BY_L3",
                              "BLOCKED_BY_VLM", "HALTED_BY_VLM"):
            # DiD blocked the action — return a blocked result
            block_reason = (
                (guard_result.get("l2") or {}).get("reason") or
                (guard_result.get("l1") or {}).get("reason") or
                (guard_result.get("l3") or {}).get("violation") or
                "Blocked by DiD safety guard"
            )
            log.info("DiD BLOCKED action '%s' at step %d: %s",
                     action_name, self._step_count, guard_decision)

            return {
                "success": False,
                "blocked_by_did": True,
                "guard_decision": guard_decision,
                "block_reason": block_reason,
                "action": action_dict,
                "guard_result": guard_result,
            }

        # DiD allowed — the action was already executed by L3 via THOR
        # (guard_action() calls L3.execute() which steps the simulator)
        return {
            "success": l3_status in ("EXEC_OK", "EXECUTED_SUCCESS"),
            "blocked_by_did": False,
            "guard_decision": guard_decision,
            "action": action_dict,
            "guard_result": guard_result,
        }

    def _execute_original(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute via the original SafeAgentBench controller."""
        action_name = action_dict.get("action", "")
        object_id = action_dict.get("objectId")

        try:
            if object_id:
                result = self.controller.execute(action_name, object_id)
            else:
                result = self.controller.execute(action_name)
            return {"success": True, "original_result": result, "action": action_dict}
        except Exception as e:
            return {"success": False, "error": str(e), "action": action_dict}

    def _get_thor_controller(self):
        """Get the underlying AI2-THOR controller from SafeAgentBench's wrapper."""
        # SafeAgentBench stores the THOR controller in various places
        # depending on their code version:
        if hasattr(self.controller, "controller"):
            return self.controller.controller
        if hasattr(self.controller, "env"):
            env = self.controller.env
            if hasattr(env, "controller"):
                return env.controller
        return None

    def save_guard_log(self, path: Optional[str] = None):
        """Save the guard decision log as JSONL."""
        path = path or self.log_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for entry in self.guard_log:
                f.write(json.dumps(entry, default=str) + "\n")
        log.info("Saved guard log (%d entries) → %s", len(self.guard_log), path)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of guard decisions."""
        total = len(self.guard_log)
        blocked = sum(1 for e in self.guard_log if "BLOCKED" in e.get("guard_decision", "")
                      or "REFUSED" in e.get("guard_decision", "")
                      or "INTERCEPTED" in e.get("guard_decision", ""))
        allowed = sum(1 for e in self.guard_log if e.get("guard_decision") == "ALLOWED")
        return {
            "total_actions": total,
            "blocked": blocked,
            "allowed": allowed,
            "block_rate": blocked / max(total, 1),
            "avg_guard_latency_ms": (
                sum(e.get("guard_latency_ms", 0) for e in self.guard_log) / max(total, 1)
            ),
        }


# ---
# Convenience: wrap SafeAgentBench's full evaluation loop
# ---
def run_safeagentbench_with_did(
    safeagent_tasks: List[Dict],
    agent_class,
    did_system,
    controller,
    output_dir: str = "outputs/safeagentbench_did",
    enabled: bool = True,
):
    """
    Run SafeAgentBench tasks with DiD guard enabled/disabled.

    Parameters
    ----------
    safeagent_tasks : list of task dicts from SafeAgentBench dataset
    agent_class     : the LLM agent class (e.g., ReActAgent, LoTaAgent)
    did_system      : ThreeLayerDiDSystem instance
    controller      : SafeAgentBench's low-level controller
    output_dir      : where to save results
    enabled         : True for DiD mode, False for baseline

    Returns
    -------
    dict with per-task results and aggregate metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    mode = "did" if enabled else "baseline"

    guarded = DiDGuardedController(
        original_controller=controller,
        did_system=did_system,
        log_path=os.path.join(output_dir, f"guard_log_{mode}.jsonl"),
        enabled=enabled,
    )

    results = []
    for task in safeagent_tasks:
        task_id = task.get("id", "unknown")
        instruction = task.get("instruction", "")
        is_unsafe = task.get("is_unsafe", False)

        guarded.set_instruction(instruction)

        # Run the agent with the guarded controller
        # (Pass `guarded` into the agent's execution loop)
        # This is task-specific; see SafeAgentBench's agent implementations
        log.info("[%s] Task %s: %s", mode, task_id, instruction[:80])

        results.append({
            "task_id": task_id,
            "instruction": instruction,
            "is_unsafe": is_unsafe,
            "mode": mode,
            "guard_summary": guarded.get_summary(),
        })

    # Save results
    guarded.save_guard_log()
    results_path = os.path.join(output_dir, f"results_{mode}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved %s results → %s", mode, results_path)

    return results
