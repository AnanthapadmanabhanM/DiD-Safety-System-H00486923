from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, List
import time

from layers.l1_perception_guard import Layer1PerceptionGuard
from layers.l2_semantic_guard import Layer2SemanticGuard, analyze_intent_risk
from layers.l3_reference_monitor import ThesisGuard_V19_2_FORMAL_RV_SHIELD

from utils.thor import get_event_rgb

log = logging.getLogger(__name__)


def _split_action_dict(action_dict: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    ad = dict(action_dict)
    name = ad.pop("action", None)
    if name is None:
        raise ValueError("action dict missing 'action' key")
    return str(name), ad


# ---
# Plan-level screening constants (used by screen_full_plan)

# ---
DANGEROUS_BIGRAMS = {
    ("ToggleObjectOn", "MoveAhead"),
    ("ToggleObjectOn", "MoveBack"),
    ("ToggleObjectOn", "MoveLeft"),
    ("ToggleObjectOn", "MoveRight"),
    ("PutObject", "ToggleObjectOn"),
}

DANGEROUS_TRIGRAMS = {
    ("PutObject", "CloseObject", "ToggleObjectOn"),
    ("PutObject", "ToggleObjectOn", "MoveAhead"),
    ("PutObject", "ToggleObjectOn", "MoveBack"),
}

class ThreeLayerDiDSystem:
    """
    (RGB, instruction) -> L1 -> L2 -> L3 -> execute

    Parallel architecture support (v2):
      - L2 now runs three checks in parallel (rules + CLIP + Qwen3-VL).
      - In "full" deployment mode, Qwen3-VL runs asynchronously.
      - check_vlm_override() is called before each subsequent action
        to collect the async Qwen3-VL result and HALT if danger detected.
      - Graceful degradation: full → standard → edge.

    Fixes:
      - L1 method compatibility wrapper (check/inspect/run/forward...)
      - Param fill for OpenObject/CloseObject (objectId required by THOR)
      - Guards against Put/Drop when not holding anything
      - MoveAhead retry on physics block
    """

    def __init__(
        self,
        controller,
        l1_cfg: Optional[Dict[str, Any]] = None,
        l2_cfg: Optional[Dict[str, Any]] = None,
        l3_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.controller = controller
        self.l1 = Layer1PerceptionGuard(l1_cfg or {})
        self.l2 = Layer2SemanticGuard(l2_cfg or {})
        self.l3 = ThesisGuard_V19_2_FORMAL_RV_SHIELD(controller, **(l3_cfg or {}))
        self._l1_fitted = False
        self._halted = False          # True if Qwen3-VL async override halted agent
        self._halt_reason = None      # reason for halt

    # -------------------------
    # L1 fitting on scene frames
    # -------------------------
    def fit_l1_on_scene(self, num_frames: int = 80):
        frames = []
        for _ in range(int(num_frames)):
            ev = self.controller.step(action="RotateRight")
            frames.append(get_event_rgb(ev))
        if hasattr(self.l1, "fit"):
            self.l1.fit(frames)
        self._l1_fitted = True

    # -------------------------
    # L1 compatibility wrapper
    # -------------------------
    def _call_l1(self, rgb, instruction: str) -> Dict[str, Any]:
        """
        Compatibility wrapper for Layer-1.
        We prefer stable keyword calls (rgb=..., text=...) and only then fall back
        to other names or positional calls.
        """
        candidates = []
        for name in ["inspect", "check", "run", "forward", "__call__", "detect", "evaluate"]:
            fn = getattr(self.l1, name, None)
            if callable(fn):
                candidates.append((name, fn))

        if not candidates:
            raise AttributeError("Layer1PerceptionGuard has no callable method.")

        # 1) Preferred explicit keyword signatures (tried in order)
        kw_trials = [
            {"rgb": rgb, "text": instruction},          # current API: inspect(rgb, text="")
            {"rgb": rgb, "instruction": instruction},   # legacy / alternative
            {"image": rgb, "text": instruction},        # common alt naming
            {"frame": rgb, "text": instruction},        # common alt naming
            {"rgb": rgb, "prompt": instruction},        # some L1s use prompt=
            {"rgb": rgb},                               # allow text default ""
        ]

        # 2) Positional fallback (least preferred)
        pos_trials = [
            (rgb, instruction),
            (rgb,),
            (instruction,),  # if a model expects only text (rare)
        ]

        last_err = None

        for _fname, fn in candidates:
            # keyword attempts
            for kw in kw_trials:
                try:
                    out = fn(**kw)
                    if isinstance(out, dict) and "l1_ok" in out:
                        return out
                except TypeError as e:
                    last_err = e
                except Exception:
                    continue

            # positional attempts
            for args in pos_trials:
                try:
                    out = fn(*args)
                    if isinstance(out, dict) and "l1_ok" in out:
                        return out
                except TypeError as e:
                    last_err = e
                except Exception:
                    continue

        raise TypeError(
            "Failed to call Layer-1 with any supported signature. "
            "Expected a method returning {'l1_ok': ...}. "
            f"Last TypeError: {last_err}"
        )

    # -------------------------
    # Metadata helpers
    # -------------------------
    def _meta(self) -> Dict[str, Any]:
        ev = getattr(self.controller, "last_event", None)
        meta = getattr(ev, "metadata", None) if ev is not None else None
        return meta if isinstance(meta, dict) else {}

    def _visible_objects(self) -> List[Dict[str, Any]]:
        return list(self._meta().get("objects", []) or [])

    def _visible_types(self) -> List[str]:
        vt = []
        for o in self._visible_objects():
            if o.get("visible", False):
                t = o.get("objectType", None)
                if t is not None:
                    vt.append(str(t))
        return sorted(list(set(vt)))

    def _held(self) -> bool:
        inv = self._meta().get("inventoryObjects", []) or []
        return len(inv) > 0

    def _pick_object_id_for_openclose(self, want_open: bool) -> Optional[str]:
        """
        Pick a visible openable objectId.
        want_open=True  -> prefer objects where isOpen==False
        want_open=False -> prefer objects where isOpen==True
        """
        candidates = []
        for o in self._visible_objects():
            if not o.get("visible", False):
                continue
            if not o.get("openable", False):
                continue
            oid = o.get("objectId", None)
            if not oid:
                continue
            is_open = bool(o.get("isOpen", False))
            score = 0
            if want_open and (not is_open):
                score = 2
            if (not want_open) and is_open:
                score = 2
            candidates.append((score, oid))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _safe_fallback_action(self) -> Dict[str, Any]:
        return {"action": "RotateRight"}

    # -------------------------
    # Main pipeline step
    # -------------------------
    def step(self, instruction: str, rgb_override: Optional[Any] = None) -> Dict[str, Any]:
        t0 = time.time()
        t_l1 = t_l2 = t_l3 = None

        # --- Qwen3-VL async override check (between actions) ---
        # In "full" deployment mode, Qwen3-VL runs asynchronously on the
        # previous frame. Before processing the next instruction, we
        # collect its result. If Qwen3-VL flagged danger, HALT the agent.
        vlm_override = self._check_and_handle_vlm_override()
        if vlm_override is not None:
            return vlm_override

        # --- Agent halted by previous Qwen3-VL override ---
        if self._halted:
            t1 = time.time()
            return {
                "l1": {"l1_ok": False, "reason": "Agent halted by Qwen3-VL override"},
                "l2": {
                    "decision": "REFUSE",
                    "reason": f"Agent HALTED: {self._halt_reason}",
                    "meta": {"halted": True, "halt_reason": self._halt_reason},
                },
                "l3": {"status": "HALTED_BY_VLM", "violation": self._halt_reason, "executed_action": None},
                "latency_ms": (t1 - t0) * 1000.0,
                "latency_l1_ms": 0.0,
                "latency_l2_ms": 0.0,
                "latency_l3_ms": 0.0,
            }

        rgb = rgb_override if rgb_override is not None else get_event_rgb(self.controller.last_event)
        visible_types = self._visible_types()
        held = self._held()

        # ---- L1 ----
        t_l1_0 = time.time()
        l1_out = self._call_l1(rgb=rgb, instruction=instruction)
        t_l1 = (time.time() - t_l1_0) * 1000.0
        if not l1_out.get("l1_ok", False):
            t1 = time.time()
            return {
                "l1": l1_out,
                "l2": {
                    "decision": "REFUSE",
                    "reason": f"Blocked by Layer-1: {l1_out.get('reason')}",
                    "meta": {"backend": None, "blocked_by": "L1", "l1_reason": l1_out.get("reason")},
                },
                "l3": {"status": "BLOCKED_BY_L1", "violation": l1_out.get("reason"), "executed_action": None},
                "latency_ms": (t1 - t0) * 1000.0,
                "latency_l1_ms": t_l1,
                "latency_l2_ms": t_l2,
                "latency_l3_ms": t_l3,
            }

        # ---- L2 (parallel checks: rules + CLIP sync, Qwen3-VL async) ----
        t_l2_0 = time.time()
        l2_out = self.l2.decide(
            rgb=rgb,
            instruction=instruction,
            visible_object_types=visible_types,
            holding=held,
        )
        t_l2 = (time.time() - t_l2_0) * 1000.0

        if l2_out.get("decision") == "REFUSE":
            t1 = time.time()
            return {
                "l1": l1_out,
                "l2": l2_out,
                "l3": {"status": "REFUSED_BY_L2", "violation": l2_out.get("reason"), "executed_action": None},
                "latency_ms": (t1 - t0) * 1000.0,
                "latency_l1_ms": t_l1,
                "latency_l2_ms": t_l2,
                "latency_l3_ms": t_l3,
            }

        action_dict = l2_out.get("action") or {"action": "RotateRight"}

        # ---- Param fill for Open/Close (objectId required) ----
        if action_dict.get("action") == "OpenObject" and "objectId" not in action_dict:
            oid = self._pick_object_id_for_openclose(want_open=True)
            if oid is not None:
                action_dict["objectId"] = oid
                action_dict.setdefault("forceAction", True)
            else:
                action_dict = self._safe_fallback_action()

        if action_dict.get("action") == "CloseObject" and "objectId" not in action_dict:
            oid = self._pick_object_id_for_openclose(want_open=False)
            if oid is not None:
                action_dict["objectId"] = oid
                action_dict.setdefault("forceAction", True)
            else:
                action_dict = self._safe_fallback_action()

        # ---- Prevent Put/Drop if not holding ----
        if action_dict.get("action") in {"PutObject", "DropHandObject"} and not held:
            action_dict = self._safe_fallback_action()

        # ---- L3 ----
        act_name, act_params = _split_action_dict(action_dict)
        l3_res = self.l3.execute(act_name, act_params)

        # Normalize Layer-3 return types
        if isinstance(l3_res, tuple) and len(l3_res) == 2:
            ok, msg = l3_res
            l3_res = {
                'status': 'EXEC_OK' if ok else 'FAILED_IN_SIM',
                'violation': None if ok else str(msg),
                'executed_action': {'action': act_name, **act_params},
                'reason': str(msg),
            }

        # Retry once for MoveAhead sim physics failure
        if isinstance(l3_res, dict):
            status = str(l3_res.get("status", ""))
            vio = str(l3_res.get("violation", "") or "")
            if status in {"FAILED_IN_SIM", "SIM_FAIL", "FAILED"} and act_name == "MoveAhead":
                retry = self.l3.execute("RotateRight", {})
                try:
                    retry = dict(retry)
                    retry["meta_retry_from"] = "MoveAhead"
                    retry["meta_retry_reason"] = vio
                except Exception:
                    pass
                l3_res = retry

        t1 = time.time()
        if not isinstance(l3_res, dict):
            l3_res = {"status": str(l3_res), "violation": None, "executed_action": action_dict}

        return {
            "l1": l1_out,
            "l2": l2_out,
            "l3": l3_res,
            "latency_ms": (t1 - t0) * 1000.0,
            "latency_l1_ms": t_l1,
            "latency_l2_ms": t_l2,
            "latency_l3_ms": t_l3,
        }

    # -------------------------
    # Qwen3-VL async override
    # -------------------------
    def _check_and_handle_vlm_override(self) -> Optional[Dict[str, Any]]:
        """
        Check for pending Qwen3-VL async override.
        
        Called at the START of each step() to collect async Qwen3-VL results
        from the previous action. If Qwen3-VL flagged danger:
          - HALT the agent
          - Block the next action
          - Log semantic danger alert
        
        Returns:
            None if no override (continue normally).
            A full pipeline result dict if Qwen3-VL triggered an override.
        """
        if not hasattr(self.l2, 'check_vlm_override'):
            return None

        override = self.l2.check_vlm_override()
        if not override.get("override", False):
            return None

        reason = override.get("reason", "Qwen3-VL detected semantic scene danger")
        severity = override.get("severity", "HIGH")
        self._halted = True
        self._halt_reason = reason

        log.warning(
            "Qwen3-VL ASYNC OVERRIDE — HALTING AGENT: %s (severity=%s)",
            reason, severity
        )

        return {
            "l1": {"l1_ok": True, "reason": "Passed (override from Qwen3-VL)"},
            "l2": {
                "decision": "REFUSE",
                "reason": f"Qwen3-VL async override: {reason}",
                "meta": {
                    "vlm_override": True,
                    "severity": severity,
                    "vlm_result": override.get("vlm_result"),
                },
            },
            "l3": {
                "status": "HALTED_BY_VLM_OVERRIDE",
                "violation": reason,
                "executed_action": None,
            },
            "latency_ms": 0.0,
            "latency_l1_ms": 0.0,
            "latency_l2_ms": 0.0,
            "latency_l3_ms": 0.0,
        }

    def reset_halt(self) -> None:
        """Reset the agent halt state (e.g., after human review)."""
        self._halted = False
        self._halt_reason = None

    @property
    def is_halted(self) -> bool:
        """Whether the agent is currently halted by a Qwen3-VL override."""
        return self._halted

    # -------------------------
    # Guard-only pathway (for LLM agent comparison)
    # -------------------------
    def guard_action(
        self,
        instruction: str,
        proposed_action: Dict[str, Any],
        rgb_override: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Guard an *externally-proposed* action through L1/L2/L3 **without**
        letting L2 replace the action.

        This is the key method for fair benchmark comparison:
          - Same LLM policy proposes the same action in both modes.
          - In "llm_only" mode the action executes directly.
          - In "did" mode this method checks perception (L1), classifies
            intent/risk (L2), and enforces runtime constraints (L3) on
            **that same action**.

        Pipeline
        --------
        1. L1: perception integrity on (rgb, instruction).
        2. L2: semantic classification only — REFUSE / ALLOW verdict.
              The action L2 would have chosen is *ignored*.
        3. Apply param-fill / safety pre-checks on the LLM's action.
        4. L3: execute the LLM's action under runtime constraints.

        Returns
        -------
        dict matching the same schema as step():
            l1, l2, l3, latency_ms, latency_l{1,2,3}_ms,
            llm_proposed_action, guard_decision
        """
        t0 = time.time()
        t_l1 = t_l2 = t_l3 = None

        # --- Qwen3-VL async override ---
        vlm_override = self._check_and_handle_vlm_override()
        if vlm_override is not None:
            vlm_override["llm_proposed_action"] = proposed_action
            vlm_override["guard_decision"] = "BLOCKED_BY_VLM"
            return vlm_override

        if self._halted:
            t1 = time.time()
            return {
                "l1": {"l1_ok": False, "reason": "Agent halted by Qwen3-VL override"},
                "l2": {"decision": "REFUSE", "reason": f"Agent HALTED: {self._halt_reason}",
                        "meta": {"halted": True}},
                "l3": {"status": "HALTED_BY_VLM", "violation": self._halt_reason,
                        "executed_action": None},
                "latency_ms": (t1 - t0) * 1000.0,
                "latency_l1_ms": 0.0, "latency_l2_ms": 0.0, "latency_l3_ms": 0.0,
                "llm_proposed_action": proposed_action,
                "guard_decision": "BLOCKED_BY_VLM",
            }

        rgb = rgb_override if rgb_override is not None else get_event_rgb(
            self.controller.last_event
        )
        visible_types = self._visible_types()
        held = self._held()

        # --- L1: perception guard ---
        t_l1_0 = time.time()
        l1_out = self._call_l1(rgb=rgb, instruction=instruction)
        t_l1 = (time.time() - t_l1_0) * 1000.0

        if not l1_out.get("l1_ok", False):
            t1 = time.time()
            return {
                "l1": l1_out,
                "l2": {"decision": "REFUSE",
                        "reason": f"Blocked by Layer-1: {l1_out.get('reason')}",
                        "meta": {"blocked_by": "L1"}},
                "l3": {"status": "BLOCKED_BY_L1", "violation": l1_out.get("reason"),
                        "executed_action": None},
                "latency_ms": (t1 - t0) * 1000.0,
                "latency_l1_ms": t_l1, "latency_l2_ms": None, "latency_l3_ms": None,
                "llm_proposed_action": proposed_action,
                "guard_decision": "BLOCKED_BY_L1",
            }

        # --- L2: semantic classification ONLY (verdict, not action) ---
        t_l2_0 = time.time()
        l2_out = self.l2.decide(
            rgb=rgb,
            instruction=instruction,
            visible_object_types=visible_types,
            holding=held,
        )
        t_l2 = (time.time() - t_l2_0) * 1000.0

        if l2_out.get("decision") == "REFUSE":
            t1 = time.time()
            return {
                "l1": l1_out,
                "l2": l2_out,
                "l3": {"status": "REFUSED_BY_L2", "violation": l2_out.get("reason"),
                        "executed_action": None},
                "latency_ms": (t1 - t0) * 1000.0,
                "latency_l1_ms": t_l1, "latency_l2_ms": t_l2, "latency_l3_ms": None,
                "llm_proposed_action": proposed_action,
                "guard_decision": "REFUSED_BY_L2",
            }

        # --- L2 ALLOWED — now gate the LLM's action (NOT L2's action) ---
        action_dict = dict(proposed_action)  # work on a copy
        action_modifications = []            # track any fallbacks applied

        # Param fill for Open/Close
        if action_dict.get("action") == "OpenObject" and "objectId" not in action_dict:
            oid = self._pick_object_id_for_openclose(want_open=True)
            if oid is not None:
                action_dict["objectId"] = oid
                action_dict.setdefault("forceAction", True)
                action_modifications.append({
                    "type": "param_fill", "field": "objectId",
                    "action": "OpenObject", "resolved_to": oid,
                })
            else:
                orig = action_dict.get("action")
                action_dict = self._safe_fallback_action()
                action_modifications.append({
                    "type": "safe_fallback", "original_action": orig,
                    "reason": "No openable object visible",
                    "replaced_with": action_dict.get("action"),
                })

        if action_dict.get("action") == "CloseObject" and "objectId" not in action_dict:
            oid = self._pick_object_id_for_openclose(want_open=False)
            if oid is not None:
                action_dict["objectId"] = oid
                action_dict.setdefault("forceAction", True)
                action_modifications.append({
                    "type": "param_fill", "field": "objectId",
                    "action": "CloseObject", "resolved_to": oid,
                })
            else:
                orig = action_dict.get("action")
                action_dict = self._safe_fallback_action()
                action_modifications.append({
                    "type": "safe_fallback", "original_action": orig,
                    "reason": "No closeable object visible",
                    "replaced_with": action_dict.get("action"),
                })

        # Prevent Put/Drop if not holding
        if action_dict.get("action") in {"PutObject", "DropHandObject"} and not held:
            orig = action_dict.get("action")
            action_dict = self._safe_fallback_action()
            action_modifications.append({
                "type": "safe_fallback", "original_action": orig,
                "reason": "Not holding any object",
                "replaced_with": action_dict.get("action"),
            })

        if action_modifications:
            log.info("guard_action: %d modification(s) applied to LLM action: %s",
                     len(action_modifications),
                     "; ".join(m["type"] for m in action_modifications))

        # --- L3: execute the LLM's action under runtime constraints ---
        act_name, act_params = _split_action_dict(action_dict)

        t_l3_0 = time.time()
        l3_res = self.l3.execute(
            act_name, act_params,
            risk_hint=l2_out.get("risk_level", "LOW"),
            original_instruction=instruction,
            l2_metadata=l2_out.get("meta"),
        )
        t_l3 = (time.time() - t_l3_0) * 1000.0

        # Normalize L3 return
        if isinstance(l3_res, tuple) and len(l3_res) == 2:
            ok, msg = l3_res
            l3_res = {
                "status": "EXEC_OK" if ok else "INTERCEPTED_BY_L3",
                "violation": None if ok else str(msg),
                "executed_action": {"action": act_name, **act_params},
                "reason": str(msg),
            }

        # MoveAhead retry on physics block
        if isinstance(l3_res, dict):
            status = str(l3_res.get("status", ""))
            vio = str(l3_res.get("violation", "") or "")
            if status in {"FAILED_IN_SIM", "SIM_FAIL", "FAILED"} and act_name == "MoveAhead":
                retry = self.l3.execute("RotateRight", {})
                try:
                    retry = dict(retry)
                    retry["meta_retry_from"] = "MoveAhead"
                except Exception:
                    pass
                l3_res = retry

        t1 = time.time()
        if not isinstance(l3_res, dict):
            l3_res = {"status": str(l3_res), "violation": None,
                       "executed_action": action_dict}

        # Determine guard decision
        l3_status = str(l3_res.get("status", ""))
        if "INTERCEPT" in l3_status or "BLOCKED" in l3_status:
            guard_decision = "INTERCEPTED_BY_L3"
        elif l3_status in ("EXEC_OK", "EXECUTED_SUCCESS"):
            guard_decision = "ALLOWED"
        else:
            guard_decision = l3_status

        return {
            "l1": l1_out,
            "l2": l2_out,
            "l3": l3_res,
            "latency_ms": (t1 - t0) * 1000.0,
            "latency_l1_ms": t_l1,
            "latency_l2_ms": t_l2,
            "latency_l3_ms": t_l3,
            "llm_proposed_action": proposed_action,
            "guard_decision": guard_decision,
            "action_modifications": action_modifications,
        }

    def screen_full_plan(self, instruction: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse an entire action sequence *before* execution begins to catch
        sequence-level hazards that may not be flagged by per-step guards.

        Returns: {"decision": "ALLOW"} or {"decision": "BLOCK", "reason": ..., "index": ...}
        """
        action_names = [step.get("action", "") for step in (plan or []) if isinstance(step, dict)]

        # Bigram checks
        for i in range(len(action_names) - 1):
            pair = (action_names[i], action_names[i + 1])
            if pair in DANGEROUS_BIGRAMS:
                return {
                    "decision": "BLOCK",
                    "reason": f"Dangerous action sequence: {pair[0]} → {pair[1]}",
                    "layer": "plan_screen",
                    "index": i,
                }

        # Trigram checks
        for i in range(len(action_names) - 2):
            triple = (action_names[i], action_names[i + 1], action_names[i + 2])
            if triple in DANGEROUS_TRIGRAMS:
                return {
                    "decision": "BLOCK",
                    "reason": f"Dangerous 3-step sequence: {' → '.join(triple)}",
                    "layer": "plan_screen",
                    "index": i,
                }

        # Lightweight L2 scan over full plan text (no vision)
        full_text = (instruction or "") + " " + " ".join(action_names)
        l2_seq = analyze_intent_risk(full_text)
        if l2_seq.get("risk_level") in ("HIGH", "CRITICAL"):
            return {
                "decision": "BLOCK",
                "reason": f"L2 sequence-level risk: {l2_seq.get('rationale')}",
                "layer": "plan_screen_l2",
            }

        return {"decision": "ALLOW"}

    def shutdown(self) -> None:
        """Cleanly shut down background threads/resources."""
        try:
            if hasattr(self.l2, "shutdown"):
                self.l2.shutdown()
        finally:
            if hasattr(self.l3, "shutdown"):
                try:
                    self.l3.shutdown()
                except Exception:
                    pass


