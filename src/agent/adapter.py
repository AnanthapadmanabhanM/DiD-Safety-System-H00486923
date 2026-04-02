"""
agent/adapter.py
-------------------------------------------------------------------
Action Adapter — translates between LLM outputs, AI2-THOR actions,
and benchmark-specific action spaces.

Responsibilities:
  1. Normalize LLM-proposed actions to valid AI2-THOR calls
  2. Map benchmark-specific action formats to/from THOR actions
  3. Extract observation dicts from AI2-THOR controller state
  4. Validate action parameters (objectId exists, etc.)
-------------------------------------------------------------------
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---
# Valid AI2-THOR actions
# ---
NAVIGATION_ACTIONS = {
    "MoveAhead", "MoveBack", "RotateRight", "RotateLeft",
    "LookUp", "LookDown",
}

OBJECT_ACTIONS = {
    "PickupObject", "PutObject", "OpenObject", "CloseObject",
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject",
    "ThrowObject", "DropHandObject",
}

TERMINAL_ACTIONS = {"Done", "Stop"}

ALL_ACTIONS = NAVIGATION_ACTIONS | OBJECT_ACTIONS | TERMINAL_ACTIONS

# Common aliases from LLM outputs → canonical THOR names
_ACTION_ALIASES = {
    "move_ahead": "MoveAhead", "move_forward": "MoveAhead", "forward": "MoveAhead",
    "move_back": "MoveBack", "backward": "MoveBack", "back": "MoveBack",
    "rotate_right": "RotateRight", "turn_right": "RotateRight", "right": "RotateRight",
    "rotate_left": "RotateLeft", "turn_left": "RotateLeft", "left": "RotateLeft",
    "look_up": "LookUp", "look_down": "LookDown",
    "pickup": "PickupObject", "pick_up": "PickupObject", "grab": "PickupObject",
    "put": "PutObject", "place": "PutObject", "put_down": "PutObject",
    "open": "OpenObject", "close": "CloseObject",
    "toggle_on": "ToggleObjectOn", "turn_on": "ToggleObjectOn",
    "toggle_off": "ToggleObjectOff", "turn_off": "ToggleObjectOff",
    "slice": "SliceObject", "cut": "SliceObject",
    "throw": "ThrowObject", "drop": "DropHandObject",
    "done": "Done", "stop": "Stop", "finish": "Done", "end": "Done",
    "noop": "Done", "no_op": "Done", "refuse": "Stop",
}


# ---
# Observation extractor
# ---
class ObservationExtractor:
    """Extracts structured observations from AI2-THOR controller state."""

    def __init__(self, controller):
        self.controller = controller

    def get_observation(self) -> Dict[str, Any]:
        """
        Extract current observation from the controller.

        Returns
        -------
        dict with keys:
            visible_objects : list of {"objectId", "objectType", "visible",
                              "distance", "isOpen", "isToggled", ...}
            visible_types   : list of str (unique visible object types)
            held_object     : str or None (objectType of held object)
            held_object_id  : str or None (objectId of held object)
            agent_position  : dict with x, y, z
            agent_rotation  : dict with x, y, z
        """
        ev = getattr(self.controller, "last_event", None)
        meta = getattr(ev, "metadata", None) if ev else None
        if not isinstance(meta, dict):
            meta = {}

        objects = meta.get("objects", []) or []

        # Visible objects
        visible = []
        visible_types = set()
        for obj in objects:
            if obj.get("visible", False):
                visible.append({
                    "objectId": obj.get("objectId"),
                    "objectType": obj.get("objectType"),
                    "distance": obj.get("distance"),
                    "isOpen": obj.get("isOpen", False),
                    "isToggled": obj.get("isToggled", False),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "openable": obj.get("openable", False),
                    "pickupable": obj.get("pickupable", False),
                    "receptacle": obj.get("receptacle", False),
                })
                visible_types.add(obj.get("objectType", "Unknown"))

        # Held object
        inv = meta.get("inventoryObjects", []) or []
        held_obj = None
        held_obj_id = None
        if inv:
            held_obj = inv[0].get("objectType")
            held_obj_id = inv[0].get("objectId")

        # Agent pose
        agent = meta.get("agent", {}) or {}
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})

        return {
            "visible_objects": visible,
            "visible_types": sorted(visible_types),
            "held_object": held_obj,
            "held_object_id": held_obj_id,
            "agent_position": position,
            "agent_rotation": rotation,
        }

    def get_rgb(self):
        """Get current RGB frame from controller."""
        ev = getattr(self.controller, "last_event", None)
        if ev is None:
            return None
        frame = getattr(ev, "frame", None)
        if frame is not None:
            return frame
        # Fallback: cv_image or numpy array
        return getattr(ev, "cv2img", None)


# ---
# Action normalizer
# ---
class ActionAdapter:
    """
    Translates LLM action proposals to valid AI2-THOR action dicts
    and resolves object references.
    """

    def __init__(self, controller):
        self.controller = controller
        self.obs_extractor = ObservationExtractor(controller)

    def normalize_action(self, llm_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an LLM-proposed action to a valid AI2-THOR action dict.

        Parameters
        ----------
        llm_output : dict from policy_llm (action, objectId, reasoning)

        Returns
        -------
        dict with "action" key, optional params, and "_adapter_notes" list
        tracking any fallbacks/resolutions applied.
        """
        raw_action = str(llm_output.get("action", "Stop"))
        action, resolution_note = self._resolve_action_name_tracked(raw_action)

        result = {"action": action}
        adapter_notes = []

        if resolution_note:
            adapter_notes.append(resolution_note)

        # Handle object actions
        if action in OBJECT_ACTIONS:
            obj_id = llm_output.get("objectId")
            if obj_id:
                result["objectId"] = obj_id
            else:
                # Try to resolve from scene state
                obj_id = self._resolve_object_id(action, llm_output)
                if obj_id:
                    result["objectId"] = obj_id
                    adapter_notes.append({
                        "type": "objectId_auto_resolved",
                        "action": action,
                        "resolved_to": obj_id,
                    })
                else:
                    adapter_notes.append({
                        "type": "objectId_missing",
                        "action": action,
                        "note": "LLM did not provide objectId and auto-resolve failed",
                    })

            # forceAction for open/close
            if action in ("OpenObject", "CloseObject"):
                result.setdefault("forceAction", True)

        if adapter_notes:
            result["_adapter_notes"] = adapter_notes

        return result

    def _resolve_action_name_tracked(self, raw: str) -> Tuple[str, Optional[Dict]]:
        """Normalize action name to canonical THOR action, with tracking."""
        # Direct match
        if raw in ALL_ACTIONS:
            return raw, None

        # Case-insensitive match
        for act in ALL_ACTIONS:
            if act.lower() == raw.lower():
                return act, {"type": "case_normalized", "raw": raw, "resolved": act}

        # Alias lookup
        key = raw.lower().replace(" ", "_").replace("-", "_")
        if key in _ACTION_ALIASES:
            resolved = _ACTION_ALIASES[key]
            return resolved, {"type": "alias_resolved", "raw": raw, "resolved": resolved}

        log.warning("Unknown action '%s', defaulting to Stop", raw)
        return "Stop", {"type": "unknown_action_fallback", "raw": raw, "resolved": "Stop"}

    def _resolve_object_id(
        self, action: str, llm_output: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to resolve an objectId from context when the LLM didn't
        provide one explicitly.
        """
        obs = self.obs_extractor.get_observation()

        # For pickup: find nearest pickupable visible object
        if action == "PickupObject":
            candidates = [
                o for o in obs["visible_objects"]
                if o.get("pickupable", False)
            ]
            if candidates:
                candidates.sort(key=lambda o: o.get("distance", 999))
                return candidates[0]["objectId"]

        # For open/close: find nearest openable
        if action in ("OpenObject", "CloseObject"):
            want_open = (action == "OpenObject")
            candidates = [
                o for o in obs["visible_objects"]
                if o.get("openable", False)
                and o.get("isOpen", False) != want_open
            ]
            if candidates:
                candidates.sort(key=lambda o: o.get("distance", 999))
                return candidates[0]["objectId"]

        # For put: find nearest receptacle
        if action == "PutObject":
            candidates = [
                o for o in obs["visible_objects"]
                if o.get("receptacle", False)
            ]
            if candidates:
                candidates.sort(key=lambda o: o.get("distance", 999))
                return candidates[0]["objectId"]

        # For toggle: find nearest togglable
        if action in ("ToggleObjectOn", "ToggleObjectOff"):
            candidates = [
                o for o in obs["visible_objects"]
                if o.get("objectType") in ("StoveKnob", "Microwave",
                                            "CoffeeMachine", "Toaster",
                                            "Faucet", "LightSwitch")
            ]
            if candidates:
                candidates.sort(key=lambda o: o.get("distance", 999))
                return candidates[0]["objectId"]

        return None

    def validate_action(self, action_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that an action dict is executable.

        Returns (is_valid, reason).
        """
        action = action_dict.get("action")
        if not action:
            return False, "Missing 'action' key"

        if action not in ALL_ACTIONS:
            return False, f"Unknown action: {action}"

        if action in OBJECT_ACTIONS and action != "DropHandObject":
            if "objectId" not in action_dict:
                return False, f"{action} requires objectId"

        # Check held-object constraints
        obs = self.obs_extractor.get_observation()
        if action in ("PutObject", "DropHandObject", "ThrowObject"):
            if obs["held_object"] is None:
                return False, f"{action} requires holding an object"

        if action == "PickupObject" and obs["held_object"] is not None:
            return False, "Already holding an object"

        return True, "OK"


# ---
# Benchmark action mappers
# ---
class BenchmarkActionMapper:
    """
    Maps between benchmark-specific action representations and
    AI2-THOR actions. Each benchmark may use different naming.
    """

    @staticmethod
    def from_isbench(bench_action: Dict[str, Any]) -> Dict[str, Any]:
        """IS-Bench action → THOR action dict."""
        action_name = bench_action.get("action", bench_action.get("type", ""))
        # IS-Bench sometimes uses snake_case
        action_name = _ACTION_ALIASES.get(
            action_name.lower().replace(" ", "_"), action_name
        )
        result = {"action": action_name}
        if "object" in bench_action:
            result["objectId"] = bench_action["object"]
        if "objectId" in bench_action:
            result["objectId"] = bench_action["objectId"]
        return result

    @staticmethod
    def from_agentsafe(bench_action: Dict[str, Any]) -> Dict[str, Any]:
        """AgentSafe action → THOR action dict."""
        action_name = bench_action.get("action_type",
                       bench_action.get("action", ""))
        action_name = _ACTION_ALIASES.get(
            action_name.lower().replace(" ", "_"), action_name
        )
        result = {"action": action_name}
        for key in ("objectId", "target", "object_id"):
            if key in bench_action:
                result["objectId"] = bench_action[key]
                break
        return result

    @staticmethod
    def from_safemind(bench_action: Dict[str, Any]) -> Dict[str, Any]:
        """SafeMindBench action → THOR action dict."""
        # SafeMind may provide high-level plan steps or low-level actions
        if "low_level_action" in bench_action:
            action_name = bench_action["low_level_action"]
        elif "action" in bench_action:
            action_name = bench_action["action"]
        else:
            action_name = bench_action.get("step", "Stop")

        action_name = _ACTION_ALIASES.get(
            action_name.lower().replace(" ", "_"), action_name
        )
        result = {"action": action_name}
        for key in ("objectId", "target_object", "object"):
            if key in bench_action:
                result["objectId"] = bench_action[key]
                break
        return result
