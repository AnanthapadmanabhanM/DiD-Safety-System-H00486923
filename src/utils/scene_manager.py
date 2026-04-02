# utils/scene_manager.py

# ---
# Fix: Multi-scene AI2-THOR iteration for benchmarking.
#
# Maps BAP hazard categories to appropriate AI2-THOR FloorPlans,
# spawns the agent in a reachable position, and optionally injects
# a "human-proxy" object (Bed/Chair/Sofa) near the agent to trigger
# Layer-3 human-proximity invariants.
#
# Usage (in benchmark runner):
#   sm = SceneManager(controller)
#   sm.setup_for_category("physical_harm_high")   # resets + configures
#   sm.setup_for_category("navigation_hazard")

# ---

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


# ---
# Category → FloorPlan mapping
# Each list provides variety; one is chosen per trial.

# ---
CATEGORY_SCENES: Dict[str, List[str]] = {
    # Kitchen manipulation (knives, stoves, counters)
    "physical_harm_high":        ["FloorPlan1",  "FloorPlan2",  "FloorPlan3",  "FloorPlan4"],
    "sharp_neutral":             ["FloorPlan1",  "FloorPlan5",  "FloorPlan6"],
    # Living room / bedroom (human zones: sofa, bed, armchair)
    "contextual_risk":           ["FloorPlan201","FloorPlan202","FloorPlan203","FloorPlan204"],
    "euphemistic_harm":          ["FloorPlan201","FloorPlan205","FloorPlan206"],
    # Office / bathroom
    "theft_privacy":             ["FloorPlan301","FloorPlan302","FloorPlan303"],
    "jailbreak":                 ["FloorPlan1",  "FloorPlan201","FloorPlan301","FloorPlan401"],
    # Navigation
    "navigation_hazard":         ["FloorPlan201","FloorPlan202","FloorPlan401","FloorPlan402"],
    # Safe baseline
    "safe":                      ["FloorPlan1",  "FloorPlan201","FloorPlan301","FloorPlan401"],
    # Default fallback
    "default":                   ["FloorPlan1"],
}

# Human-proxy object types (present in most THOR scenes)
_HUMAN_PROXY_TYPES = ["Bed","Sofa","ArmChair","Chair","DiningChair","Ottoman"]


class SceneManager:
    """
    Wraps an AI2-THOR Controller to provide per-category scene setup
    for benchmarking.
    """

    def __init__(self, controller, seed: int = 42):
        self.controller = controller
        self._rng = random.Random(seed)
        self._current_scene: Optional[str] = None

    # ----------------------------------------------------------
    def scene_for_category(self, category: str, trial: int = 0) -> str:
        """Return a deterministic FloorPlan for a (category, trial) pair."""
        pool = CATEGORY_SCENES.get(
            category.lower(),
            CATEGORY_SCENES.get("default", ["FloorPlan1"])
        )
        return pool[trial % len(pool)]

    # ----------------------------------------------------------
    def setup_for_category(
        self,
        category: str,
        trial: int = 0,
        spawn_human_proxy: bool = True,
        gridSize: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Reset the controller to the appropriate scene for this category/trial,
        teleport the agent to a random reachable position, and optionally
        note a human-proxy object for proximity testing.

        Returns a metadata dict with:
          scene, agent_position, human_proxy_type, human_proxy_distance
        """
        scene = self.scene_for_category(category, trial)
        if scene != self._current_scene:
            self.controller.reset(scene=scene)
            self._current_scene = scene

        # Teleport agent to a random reachable position
        reachable_pos = self._get_reachable_positions()
        if reachable_pos:
            pos = self._rng.choice(reachable_pos)
            self.controller.step(
                action="TeleportFull",
                x=pos["x"], y=pos["y"], z=pos["z"],
                rotation={"x":0,"y":self._rng.choice([0,90,180,270]),"z":0},
                horizon=0,
            )

        # Find nearest human-proxy object
        proxy_type, proxy_dist = self._find_human_proxy()

        meta = self.controller.last_event.metadata if hasattr(self.controller,"last_event") else {}
        agent_pos = (meta.get("agent",{}).get("position",{}) or {}) if isinstance(meta,dict) else {}

        return {
            "scene": scene,
            "trial": trial,
            "category": category,
            "agent_position": agent_pos,
            "human_proxy_type": proxy_type,
            "human_proxy_distance": proxy_dist,
        }

    # ----------------------------------------------------------
    def _get_reachable_positions(self) -> List[Dict[str, float]]:
        try:
            ev = self.controller.step(action="GetReachablePositions")
            return ev.metadata.get("actionReturn", []) or []
        except Exception:
            return []

    # ----------------------------------------------------------
    def _find_human_proxy(self) -> Tuple[Optional[str], Optional[float]]:
        """Return (objectType, distance) of nearest human-proxy object, or (None, None)."""
        try:
            import numpy as np
            meta = self.controller.last_event.metadata
            agent_pos = meta.get("agent",{}).get("position",{}) or {}
            ax, az = float(agent_pos.get("x",0)), float(agent_pos.get("z",0))
            best_t, best_d = None, None
            for obj in (meta.get("objects",[]) or []):
                if obj.get("objectType") in _HUMAN_PROXY_TYPES:
                    opos = obj.get("position",{}) or {}
                    d = float(np.hypot(ax - float(opos.get("x",0)), az - float(opos.get("z",0))))
                    if best_d is None or d < best_d:
                        best_d, best_t = d, obj["objectType"]
            return best_t, best_d
        except Exception:
            return None, None

    # ----------------------------------------------------------
    def all_scenes_for_ablation(self) -> List[Tuple[str, str]]:
        """Return list of (category, scene) pairs for full ablation coverage."""
        pairs = []
        for cat, scenes in CATEGORY_SCENES.items():
            if cat == "default":
                continue
            for sc in scenes:
                pairs.append((cat, sc))
        return pairs
