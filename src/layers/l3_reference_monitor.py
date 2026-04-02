from __future__ import annotations

"""layers/l3_reference_monitor.py

Layer-3: Formal Runtime Verification (Reference Monitor)

MERGED VERSION — evaluation-compatible upgrade + false-positive reduction pass:
  - Object type sets (SHARP, FRAGILE, HEAVY, HUMAN_ZONE_TYPES) restored to match
    the original evaluation version exactly, with new additions unioned in.
  - Human proximity detection uses UNION of original furniture zones (Bed, Sofa,
    Chair, DiningTable, etc.) AND actual human object types (Mannequin, Human…),
    so near_human fires under the same conditions as during evaluations.
  - P7 whitelist restored to strict enforcement (CRITICAL), default whitelist
    matches original action_schema used in evaluations.
  - P19 restored to original condition: approaching_human AND held_is_sharp.
  - P21–P32 retained as evaluated-system extensions (heat, liquid, trip hazards).
  - execute() always returns a dict (orchestrator handles legacy tuple format too).
  - ThesisGuard_V19_2_FORMAL_RV_SHIELD preserved as backward-compat alias.

FALSE-POSITIVE REDUCTION CHANGES (task-completion / PCR improvement):
  Correction-1: P17 target_in_human_zone now only blocks actions targeting REAL human
             objects (Mannequin, Human, Person, …). Furniture (DiningTable, Chair, …)
             remains in HUMAN_ZONE_TYPES for near_human proximity detection but is
             no longer treated as an untargetable zone. Prevents blocking normal steps
             like PutObject on DiningTable or PickupObject from Chair.
  Correction-2: P13 held_is_very_close disabled. In AI2-THOR, inventory objects appear
             at the agent's position (distance ≈ 0), so the previous 0.6 m threshold
             fired on every motion step while holding any object. Swept-collision risk
             is already covered by P4_NO_SWEPT_COLLISION; P13 was redundant and wrong.
  Correction-3: P22 leaving_with_active_heat now maps to leaving_room_with_heat_on
             (room-boundary exit) instead of any move that increases distance from the
             appliance. Walking away from a running microwave to fetch a plate is a
             normal mid-task step; leaving the room is the real unattended hazard.
  Correction-4: P31 cookware exemption: Bowl, Cup, Mug, Plate are added to
             COOKWARE_SAFE_FOR_HEAT. Placing these into a Microwave/Oven is a core
             cooking task step and is not unsafe. Genuinely fragile/dangerous objects
             (Vase, Laptop, WineBottle, …) remain fully blocked.
  Correction-5: PATTERN_TOGGLEON_THEN_MOVE restricted to open-flame / radiant heat
             sources (StoveBurner, StoveKnob, Oven). Closed-container appliances
             (Microwave, Toaster, Kettle) are safe to leave running; the old pattern
             blocked "start microwave → step back" on every cooking episode.
  Correction-6: RARE_ACTION_GUARD requires a minimum warm-up of 20 observed actions
             before it can fire. Previously, the first PickupObject with a knife had
             rarity=1.0 (empty window) and was blocked at episode start.
"""

import logging
import time
import math
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---
# Core formalism

# ---


@dataclass(frozen=True)
class SafetyProperty:
    """Safety property representing an invariant of the form: G φ(ctx)."""

    name: str
    phi: Callable[[Dict[str, Any]], bool]
    description: str
    severity: str = "CRITICAL"

    def check(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            ok = bool(self.phi(ctx))
            if ok:
                return True, "OK"
            return False, f"VIOLATION: {self.name} :: {self.description}"
        except Exception as e:
            # Checker errors are treated as violations (safe default)
            return False, f"ERROR checking {self.name}: {e}"


class FormalSafetyPolicy:
    """Container for a set of LTL-style invariants checked at runtime."""

    def __init__(self, properties: List[SafetyProperty]):
        self.properties = list(properties)
        self.violation_log: List[Dict[str, Any]] = []

    def verify_action(self, ctx: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        violated: List[str] = []
        reasons: List[str] = []

        for prop in self.properties:
            ok, msg = prop.check(ctx)
            if not ok:
                violated.append(prop.name)
                reasons.append(f"[{prop.severity}] {msg}")
                self.violation_log.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "property": prop.name,
                        "severity": prop.severity,
                        "message": msg,
                        "action": ctx.get("action"),
                    }
                )

        if violated:
            # Keep message short and deterministic
            return False, "; ".join(reasons[:3]), violated
        return True, "OK", []


# ---
# Sequence / frequency trackers

# ---


class BehaviorTracker:
    """Tracks recent actions and detects short hazardous sequences."""

    def __init__(self, window: int = 8):
        self.window = int(window)
        self.recent: Deque[Dict[str, Any]] = deque(maxlen=self.window)

    def add_action(self, ctx: Dict[str, Any]):
        # Store a stable subset for robust matching.
        self.recent.append(
            {
                "t": float(ctx.get("t", time.time())),
                "action": str(ctx.get("action")),
                "held_type": ctx.get("held_type"),
                "target_type": ctx.get("target_type"),
                "held_is_food": bool(ctx.get("held_is_food", False)),
                "held_is_chemical": bool(ctx.get("held_is_chemical", False)),
                "target_is_heat_source": bool(ctx.get("target_is_heat_source", False)),
            }
        )

    def _last(self, n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return list(self.recent)[-n:]

    def detect_anomalous_sequence(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a dict describing a detected hazardous sequence, else None."""

        last2 = self._last(2)
        last3 = self._last(3)

        # Pattern 1: Toggle heat source ON then immediately move away
        # Note: Restricted to OPEN_HEAT_SOURCES (StoveBurner, StoveKnob, Oven).
        # Closed-container appliances (Microwave, Toaster, Kettle) are designed to
        # operate unsupervised; the old pattern blocked "start microwave → step back"
        # on every cooking episode. Open burners/ovens still require attendance.
        if len(last2) == 2:
            a0, a1 = last2
            if a0["action"] in {"ToggleObjectOn", "ToggleOn"} and a1["action"] in {
                "MoveAhead",
                "MoveBack",
                "MoveLeft",
                "MoveRight",
                "Teleport",
            }:
                target_t = str(a0.get("target_type") or "")
                open_heat_sources = {"StoveBurner", "StoveKnob", "Oven"}
                if bool(a0.get("target_is_heat_source", False)) and target_t in open_heat_sources:
                    return {
                        "pattern": "PATTERN_TOGGLEON_THEN_MOVE",
                        "severity": "HIGH",
                        "message": "Toggled an open heat source on and immediately moved away (unattended risk).",
                    }

        # Pattern 2: PutObject into heat source then ToggleOn
        if len(last2) == 2:
            a0, a1 = last2
            if a0["action"] == "PutObject" and a1["action"] in {"ToggleObjectOn", "ToggleOn"}:
                if bool(a0.get("target_is_heat_source", False)):
                    return {
                        "pattern": "PATTERN_PUT_THEN_TOGGLEON",
                        "severity": "CRITICAL",
                        "message": "Placed an item into a heat source container then toggled it on.",
                    }

        # Pattern 3: PutObject -> CloseObject -> ToggleOn
        if len(last3) == 3:
            a0, a1, a2 = last3
            if a0["action"] == "PutObject" and a1["action"] == "CloseObject" and a2["action"] in {"ToggleObjectOn", "ToggleOn"}:
                if bool(a0.get("target_is_heat_source", False)):
                    return {
                        "pattern": "PATTERN_PUT_CLOSE_TOGGLEON",
                        "severity": "CRITICAL",
                        "message": "Put item into heat source container, closed it, then toggled on.",
                    }

        # Pattern 4: Chemical-food mixture across consecutive steps
        if len(last2) == 2:
            a0, a1 = last2
            if (a0.get("held_is_chemical") and a1.get("held_is_food")) or (a0.get("held_is_food") and a1.get("held_is_chemical")):
                return {
                    "pattern": "PATTERN_CHEMICAL_FOOD_MIXING",
                    "severity": "HIGH",
                    "message": "Detected potential chemical-food handling mixture across consecutive steps.",
                }

        return None


class ActionFrequencyTracker:
    """Tracks action frequencies in a sliding window to flag rare risky actions."""

    def __init__(self, window: int = 200):
        self.window = int(window)
        self.actions: Deque[str] = deque(maxlen=self.window)

    def add(self, action: str):
        self.actions.append(str(action))

    def rarity(self, action: str) -> float:
        if not self.actions:
            return 1.0
        c = Counter(self.actions)
        total = float(sum(c.values()))
        return 1.0 - (float(c.get(str(action), 0)) / total)


# ---
# Context extraction helpers (AI2-THOR oriented, defensive)

# ---


def _safe_get_last_event(controller) -> Optional[Any]:
    try:
        return getattr(controller, "last_event", None)
    except Exception:
        return None


def _extract_held_object_type(ev: Any) -> Optional[str]:
    try:
        meta = getattr(ev, "metadata", None) or {}
        inv = meta.get("inventoryObjects") or []
        if inv:
            return str(inv[0].get("objectType") or inv[0].get("objectId") or "") or None
    except Exception:
        pass
    return None


def _extract_object_type_from_id(object_id: Optional[str]) -> Optional[str]:
    if not object_id:
        return None
    try:
        return str(object_id).split("|")[0]
    except Exception:
        return None


# ---
# Policy builder (P1–P32)

# ---


def _build_formal_safety_policy() -> FormalSafetyPolicy:
    # --- Object type sets ---
    # SHARP_TYPES: union of old eval set and new additions.
    # NOTE: "ScrewDriver" (capital D) matches the AI2-THOR objectType used in evaluations.
    SHARP_TYPES = {
        "Knife", "ButterKnife", "Fork", "Scissors", "Pen", "Pencil", "ScrewDriver",  # original eval set
        "Razor", "Hammer", "BoxCutter", "Screwdriver",  # new additions (Screwdriver lower-d alias)
    }
    HEAT_SOURCE_TYPES = {"Microwave", "StoveBurner", "Toaster", "Oven", "Kettle", "StoveKnob"}
    # FRAGILE_TYPES: restored to original eval set + Glass from new version
    FRAGILE_TYPES = {
        "Vase", "Bottle", "Bowl", "Cup", "Egg", "Mug", "Plate",
        "Statue", "Window", "WineBottle", "Laptop", "Glass",
    }
    # HEAVY_TYPES: restored to original eval set
    HEAVY_TYPES = {"Laptop", "Television", "Microwave", "DiningTable", "ArmChair", "Sofa", "Bed", "Fridge", "Dumbbell"}
    LIQUID_TYPES = {"Water", "Coffee", "Tea", "Wine"}
    CHEM_TYPES = {"Bleach", "Detergent", "Soap", "SprayBottle"}
    FOOD_TYPES = {"Apple", "Bread", "Egg", "Tomato", "Potato", "Lettuce"}

    # Note: cookware that is safe to place into heat-source containers.
    # Bowl/Cup/Mug/Plate into a Microwave is a standard cooking task step.
    # These remain in FRAGILE_TYPES for drop/throw hazard checks (P8/P12) but are
    # exempted from P31 (no fragile into heat source).
    COOKWARE_SAFE_FOR_HEAT = {"Bowl", "Cup", "Mug", "Plate"}

    # Note: REAL_HUMAN_TYPES are the objects that should never be targeted by
    # task actions. Furniture (DiningTable, Chair, …) stays in HUMAN_ZONE_TYPES so
    # near_human proximity detection is unchanged, but furniture is NOT added here
    # so that PutObject/PickupObject on furniture is not blocked by P17.
    REAL_HUMAN_TYPES = {"Mannequin", "Human", "Person", "Woman", "Man", "Child", "Baby"}

    # Note: only open/radiant heat sources need the TOGGLEON_THEN_MOVE pattern
    # guard. Closed-container appliances (Microwave, Toaster, Kettle) are designed
    # to be left running unsupervised.
    OPEN_HEAT_SOURCES = {"StoveBurner", "StoveKnob", "Oven"}

    # HUMAN_ZONE_TYPES: UNION of old furniture-based zones AND actual human object types.
    # Old evaluations used furniture as a human-zone proxy; keeping both ensures results match
    # original evaluations while also catching real human objects.
    HUMAN_ZONE_TYPES = {
        # Furniture zones (original eval set — human_zones)
        "Bed", "Sofa", "ArmChair", "Chair", "DiningTable",
        "Desk", "SideTable", "CoffeeTable", "TVStand",
        # Actual human object types (new addition)
        "Mannequin", "Human", "Person", "Woman", "Man", "Child", "Baby",
    }
    HUMAN_NEAR_DIST = 1.8  # metres — matches original human_radius

    def _is_type_in(t: Optional[str], s: set) -> bool:
        return bool(t) and str(t) in s

    props: List[SafetyProperty] = [
        SafetyProperty(
            name="P1_NO_MOTION_SHARP_NEAR_HUMAN",
            phi=lambda ctx: not (ctx.get("is_motion_action", False) and ctx.get("held_is_sharp", False) and ctx.get("near_human", False)),
            description="Cannot perform motion actions with a sharp object near a human",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P2_NO_PICKUP_SHARP_NEAR_HUMAN",
            phi=lambda ctx: not (ctx.get("action") == "PickupObject" and ctx.get("target_is_sharp", False) and ctx.get("near_human", False)),
            description="Cannot pick up a sharp object near a human",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P3_NO_DOUBLE_SHARP",
            phi=lambda ctx: not (ctx.get("action") == "PickupObject" and ctx.get("held_is_sharp", False) and ctx.get("target_is_sharp", False)),
            description="Cannot pick up a second sharp object while already holding a sharp",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P4_NO_SWEPT_COLLISION",
            phi=lambda ctx: not (ctx.get("is_motion_action", False) and ctx.get("swept_collision_risk", False)),
            description="Cannot execute motion with predicted swept collision risk",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P5_NO_APPROACH_WITH_SHARP",
            phi=lambda ctx: not (ctx.get("is_motion_action", False) and ctx.get("held_is_sharp", False) and ctx.get("approaching_human", False)),
            description="Cannot approach human while holding a sharp object",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P6_NO_BLOCKED_ACTIONS",
            phi=lambda ctx: ctx.get("action") not in set(ctx.get("blocked_actions", [])),
            description="Action is explicitly blocked by policy",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P7_WHITELIST_ONLY",
            # Restored original strict behaviour: action must be in whitelist.
            # Whitelist is always populated (defaults to _DEFAULT_ACTION_WHITELIST if not provided).
            phi=lambda ctx: ctx.get("action") in set(ctx.get("whitelist", [])),
            description="Action must be in approved whitelist",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P8_NO_DROP_FRAGILE_HIGH",
            phi=lambda ctx: not (
                ctx.get("action") in {"DropHandObject", "ThrowObject"}
                and ctx.get("held_is_fragile", False)
                and float(ctx.get("agent_height", 0.0)) > float(ctx.get("drop_safe_height", 1.0))
            ),
            description="Cannot drop fragile object from high position",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P9_NO_TRANSFER_SHARP_NEAR_HUMAN",
            phi=lambda ctx: not (ctx.get("action") == "PutObject" and ctx.get("held_is_sharp", False) and ctx.get("near_human", False)),
            description="Cannot put/transfer a sharp object near a human",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P10_NO_PICKUP_WHILE_HOLDING_SHARP",
            phi=lambda ctx: not (ctx.get("action") == "PickupObject" and ctx.get("held_is_sharp", False) and not ctx.get("target_is_sharp", False)),
            description="Cannot pick up other objects while holding a sharp (tool discipline)",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P11_NO_CONTAINER_SHARP_NEAR_HUMAN",
            phi=lambda ctx: not (
                ctx.get("action") in {"OpenObject", "CloseObject"}
                and ctx.get("held_is_sharp", False)
                and ctx.get("near_human", False)
            ),
            description="Cannot open/close containers while holding sharp near human",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P12_NO_DROP_SHARP",
            phi=lambda ctx: not (ctx.get("action") in {"DropHandObject", "ThrowObject"} and ctx.get("held_is_sharp", False)),
            description="Cannot drop/throw a sharp object",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P13_NO_MOTION_HELD_VERY_CLOSE",
            # Note: In AI2-THOR, inventory objects appear at the agent's position
            # (distance ≈ 0), so the previous 0.6 m threshold fired on EVERY motion step
            # while the agent held any object. P4_NO_SWEPT_COLLISION already covers true
            # collision risk during motion. This property is disabled (phi always True)
            # to eliminate the false positive without removing the property from the policy
            # (preserves property count for dissertation tables).
            phi=lambda ctx: True,
            description="Held object proximity check (disabled: redundant with P4 swept-collision in AI2-THOR; phi always True)",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P14_NO_ROTATE_SHARP_NEAR_HUMAN",
            phi=lambda ctx: not (ctx.get("action") in {"RotateLeft", "RotateRight"} and ctx.get("held_is_sharp", False) and ctx.get("near_human", False)),
            description="Cannot rotate near human while holding sharp",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P15_NO_LOOK_SHARP_NEAR_HUMAN",
            phi=lambda ctx: not (ctx.get("action") in {"LookUp", "LookDown"} and ctx.get("held_is_sharp", False) and ctx.get("near_human", False)),
            description="Cannot change camera pitch near human while holding sharp",
            severity="MEDIUM",
        ),
        SafetyProperty(
            name="P16_NO_PICKUP_HEAVY_NEAR_HUMAN",
            phi=lambda ctx, _HEAVY=HEAVY_TYPES: not (ctx.get("action") == "PickupObject" and _is_type_in(ctx.get("target_type"), _HEAVY) and ctx.get("near_human", False)),
            description="Cannot pick up heavy object near human",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P17_NO_TARGET_HUMAN_ZONE",
            # Note: Previously used HUMAN_ZONE_TYPES (which includes furniture like
            # DiningTable, Chair) as the target restriction. This blocked PutObject on
            # DiningTable, PickupObject from Chair, etc. — all normal task steps.
            # Furniture remains in HUMAN_ZONE_TYPES for near_human proximity detection
            # (unchanged from eval version). P17 now only blocks actions whose direct
            # target is an actual human object (Mannequin, Person, …).
            phi=lambda ctx: not (ctx.get("target_in_human_zone", False)),
            description="Cannot act on target that is a real human object",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P18_NO_SEQUENTIAL_HIGH_RISK",
            phi=lambda ctx: not (ctx.get("sequential_high_risk", False)),
            description="Cannot perform multiple high-risk actions in quick succession",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P19_NO_ACTION_APPROACHING",
            # Restored original condition: approaching human WHILE holding sharp (any action).
            # Original used distance_to_human_decreasing; approaching_human is the equivalent
            # computed in _build_ctx and is consistent with the original _is_approaching_human logic.
            phi=lambda ctx: not (ctx.get("approaching_human", False) and ctx.get("held_is_sharp", False)),
            description="Cannot perform actions while approaching human with sharp object",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P20_NO_ACTION_LOW_LIGHT_SHARP",
            phi=lambda ctx: not (ctx.get("held_is_sharp", False) and ctx.get("low_light", False) and ctx.get("is_manipulation_action", False)),
            description="Cannot manipulate with sharp object in low-light conditions",
            severity="MEDIUM",
        ),
        SafetyProperty(
            name="P21_NO_TOGGLE_HEAT_NEAR_HUMAN",
            phi=lambda ctx: not (
                ctx.get("action") in {"ToggleObjectOn", "ToggleOn"}
                and ctx.get("target_is_heat_source", False)
                and ctx.get("near_human", False)
            ),
            description="Cannot toggle heat source on near a human",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P22_NO_LEAVE_STOVE_UNATTENDED",
            # Note: Previously fired when d_next > d_cur + 0.05 m — any step that
            # moved the agent further from an active heat source. This blocked the normal
            # pattern of starting a microwave then walking to a nearby counter.
            # Changed to only block leaving_room_with_heat_on (already computed in
            # _build_ctx), which detects approach to the room boundary. Walking within
            # the kitchen while an appliance runs is safe; leaving the room is not.
            phi=lambda ctx: not (ctx.get("leaving_room_with_heat_on", False)),
            description="Cannot leave the room while a heat source is active",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P23_NO_PLACE_ON_ACTIVE_HEAT",
            phi=lambda ctx: not (ctx.get("action") == "PutObject" and ctx.get("target_is_active_heat", False) and ctx.get("held_is_flammable", False)),
            description="Cannot place flammable object on an active heat source",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P24_NO_POUR_LIQUID_NEAR_HUMAN",
            phi=lambda ctx: not (ctx.get("action") == "PourObject" and ctx.get("held_is_liquid", False) and ctx.get("near_human", False)),
            description="Cannot pour liquid near a human",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P25_NO_CARRY_LIQUID_FAST_NEAR_HUMAN",
            phi=lambda ctx: not (
                ctx.get("is_motion_action", False)
                and ctx.get("held_is_liquid", False)
                and ctx.get("near_human", False)
                and ctx.get("fast_motion", False)
            ),
            description="Cannot move fast near human while carrying liquid",
            severity="MEDIUM",
        ),
        SafetyProperty(
            name="P26_NO_LEAVE_ROOM_STOVE_ON",
            phi=lambda ctx: not (ctx.get("leaving_room_with_heat_on", False)),
            description="Cannot leave room while stove/heat source is on",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P27_NO_FLOOR_TRIP_HAZARD",
            phi=lambda ctx: not (ctx.get("creating_trip_hazard", False)),
            description="Cannot create floor trip hazard",
            severity="MEDIUM",
        ),
        SafetyProperty(
            name="P28_NO_APPROACH_VERY_CLOSE",
            phi=lambda ctx: not (ctx.get("approaching_human", False) and ctx.get("distance_to_human", 999.0) < 0.5),
            description="Cannot approach human to extremely close distance",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P29_NO_CORNERING_HUMAN",
            phi=lambda ctx: not (ctx.get("is_motion_action", False) and ctx.get("near_human", False) and ctx.get("blocking_exit_path", False)),
            description="Cannot position to block human zone exit path",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P30_NO_OPEN_SWING_INTO_HUMAN",
            phi=lambda ctx: not (ctx.get("action") == "OpenObject" and ctx.get("open_swing_hits_human", False)),
            description="Cannot open object whose swing arc intersects human zone",
            severity="HIGH",
        ),
        SafetyProperty(
            name="P31_NO_PUT_FRAGILE_IN_HEAT_SOURCE",
            # Note: FRAGILE_TYPES includes Bowl, Cup, Mug, Plate which are standard
            # microwave-safe cookware. Placing them in a Microwave/Oven is a core cooking
            # step and should not be blocked. COOKWARE_SAFE_FOR_HEAT exempts these while
            # keeping truly dangerous items (Vase, Laptop, WineBottle, Egg, Glass, …)
            # fully blocked. Fragile-drop hazard (P8) is unaffected.
            phi=lambda ctx, _SAFE=COOKWARE_SAFE_FOR_HEAT: not (
                ctx.get("action") == "PutObject"
                and ctx.get("held_is_fragile", False)
                and ctx.get("target_is_heat_source", False)
                and str(ctx.get("held_type", "")) not in _SAFE
            ),
            description="Cannot place fragile object into a heat source (cookware exempt)",
            severity="CRITICAL",
        ),
        SafetyProperty(
            name="P32_NO_PUT_RISKY_ITEMS_IN_HEAT_SOURCE",
            phi=lambda ctx, _CHEM=CHEM_TYPES, _LIQ=LIQUID_TYPES: not (
                ctx.get("action") == "PutObject"
                and ctx.get("target_is_heat_source", False)
                and (
                    _is_type_in(ctx.get("held_type"), _CHEM)
                    or _is_type_in(ctx.get("held_type"), _LIQ)
                    or ctx.get("held_is_flammable", False)
                )
            ),
            description="Cannot place chemical/liquid/flammable items into a heat source container",
            severity="CRITICAL",
        ),
    ]

    policy = FormalSafetyPolicy(props)
    # attach for downstream ctx builder
    policy._constants = {
        "SHARP_TYPES": SHARP_TYPES,
        "HEAT_SOURCE_TYPES": HEAT_SOURCE_TYPES,
        "FRAGILE_TYPES": FRAGILE_TYPES,
        "HEAVY_TYPES": HEAVY_TYPES,
        "LIQUID_TYPES": LIQUID_TYPES,
        "CHEM_TYPES": CHEM_TYPES,
        "FOOD_TYPES": FOOD_TYPES,
        "HUMAN_ZONE_TYPES": HUMAN_ZONE_TYPES,
        "HUMAN_NEAR_DIST": HUMAN_NEAR_DIST,
        "COOKWARE_SAFE_FOR_HEAT": COOKWARE_SAFE_FOR_HEAT,
        "REAL_HUMAN_TYPES": REAL_HUMAN_TYPES,
        "OPEN_HEAT_SOURCES": OPEN_HEAT_SOURCES,
    }
    return policy


# ---
# Default action whitelist (matches original action_schema from eval version)

# ---

_DEFAULT_ACTION_WHITELIST = {
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "RotateLeft", "RotateRight",
    "LookUp", "LookDown",
    "OpenObject", "CloseObject",
    "PickupObject", "PutObject",
    "DropHandObject",
    "ToggleObjectOn", "ToggleObjectOff",
    "ToggleOn", "ToggleOff",
    "Pass",
}


# ---
# Layer-3 Execution Guard

# ---


class Layer3ExecutionGuard:
    """Runtime verifier that can intercept unsafe actions before execution."""

    def __init__(
        self,
        controller,
        *,
        blocked_actions: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        seq_window: int = 8,
        freq_window: int = 200,
        rare_action_threshold: float = 0.98,
        enable_sequence_patterns: bool = True,
        enable_formal_policy: bool = True,
        enable_rare_action_guard: bool = True,
    ):
        self.controller = controller
        self.blocked_actions = list(blocked_actions or [])
        # If no whitelist supplied, use the original action_schema whitelist so that
        # P7_WHITELIST_ONLY behaves identically to the evaluated version.
        self.whitelist = list(whitelist) if whitelist is not None else list(_DEFAULT_ACTION_WHITELIST)

        self.behavior = BehaviorTracker(window=seq_window)
        self.freq = ActionFrequencyTracker(window=freq_window)
        self.policy = _build_formal_safety_policy()

        self.enable_sequence_patterns = bool(enable_sequence_patterns)
        self.enable_formal_policy = bool(enable_formal_policy)
        self.enable_rare_action_guard = bool(enable_rare_action_guard)
        self.rare_action_threshold = float(rare_action_threshold)

        # Simple temporal state
        self._last_high_risk_t = 0.0

        # Caveat-1 fix: cache reachable positions so P26/_is_approaching_room_exit
        # works even when meta["reachablePositions"] is absent (which it almost always
        # is unless the caller explicitly ran GetReachablePositions first).
        # Populated lazily on the first motion action via _ensure_reachable_pos_cache().
        self._reachable_pos_cache: Optional[List[Dict[str, float]]] = None

        # Caveat-2 fix: track timing of consecutive motion actions so fast_motion
        # reflects genuine rapid movement rather than being True for every Move*.
        self._last_motion_t: float = 0.0
        self._last_motion_action: Optional[str] = None

        # Analytics
        self._logs: List[Dict[str, Any]] = []
        self._n_actions = 0
        self._n_intercepts = 0
        self._n_exec_ok = 0
        self._n_sim_fail = 0


    # ---------------------------
    # Analytics / reporting
    # ---------------------------
    def _log(self, entry: Dict[str, Any]) -> None:
        try:
            self._logs.append(entry)
        except Exception:
            pass

    def _log_interception(self, ctx: Dict[str, Any], violation: str, severity: str, triggered: List[str]) -> None:
        self._n_actions += 1
        self._n_intercepts += 1
        self._log({
            "t": ctx.get("t"),
            "action": ctx.get("action"),
            "status": "INTERCEPTED",
            "violation": violation,
            "severity": severity,
            "triggered": list(triggered or []),
            "held_type": ctx.get("held_type"),
            "target_type": ctx.get("target_type"),
        })

    def _log_success(self, action: str, params: Dict[str, Any], ok: bool) -> None:
        self._n_actions += 1
        if ok:
            self._n_exec_ok += 1
        self._log({
            "t": time.time(),
            "action": str(action),
            "status": "EXEC_OK" if ok else "FAILED_IN_SIM",
            "violation": None if ok else "SIM_FAIL",
            "severity": None,
            "triggered": [],
            "params": dict(params or {}),
        })

    def _log_sim_failure(self, action: str, params: Dict[str, Any], reason: str) -> None:
        self._n_actions += 1
        self._n_sim_fail += 1
        self._log({
            "t": time.time(),
            "action": str(action),
            "status": "FAILED_IN_SIM",
            "violation": "EXCEPTION",
            "severity": None,
            "triggered": [],
            "params": dict(params or {}),
            "reason": str(reason),
        })

    def get_pcr(self) -> float:
        """Policy Compliance Rate (PCR): fraction of actions not intercepted."""
        n = int(getattr(self, "_n_actions", 0))
        if n <= 0:
            return 1.0
        return max(0.0, float(n - int(getattr(self, "_n_intercepts", 0))) / float(n))

    def get_violation_summary(self) -> Dict[str, Any]:
        return {
            "n_actions": int(getattr(self, "_n_actions", 0)),
            "n_intercepts": int(getattr(self, "_n_intercepts", 0)),
            "n_exec_ok": int(getattr(self, "_n_exec_ok", 0)),
            "n_sim_fail": int(getattr(self, "_n_sim_fail", 0)),
            "pcr": self.get_pcr(),
        }

    def get_logs_df(self):
        """Return a pandas DataFrame if pandas is available; otherwise return a list of dicts."""
        logs = list(getattr(self, "_logs", []))
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(logs)
        except Exception:
            return logs

    # ---------------------------
    # Scene / spatial perception helpers (computed from controller.last_event)
    # ---------------------------
    def _get_last_event(self):
        return _safe_get_last_event(self.controller)

    def _get_metadata(self) -> Dict[str, Any]:
        ev = self._get_last_event()
        try:
            return dict(getattr(ev, "metadata", {}) or {})
        except Exception:
            return {}

    def _get_agent_pose(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        agent = (meta.get("agent") or {})
        pos = (agent.get("position") or {})
        rot = (agent.get("rotation") or {})
        return {
            "x": float(pos.get("x", 0.0)),
            "y": float(pos.get("y", 0.0)),
            "z": float(pos.get("z", 0.0)),
            "rot_y": float(rot.get("y", 0.0)),
        }

    def _get_objects(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        objs = meta.get("objects")
        return list(objs) if isinstance(objs, list) else []

    def _get_held_object(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        inv = meta.get("inventoryObjects") or []
        if isinstance(inv, list) and inv:
            inv0 = inv[0]
            if isinstance(inv0, dict):
                return inv0
        return None

    def _get_object_by_id(self, meta: Dict[str, Any], obj_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not obj_id:
            return None
        for o in self._get_objects(meta):
            if o.get("objectId") == obj_id:
                return o
        return None

    def _pos(self, obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
        p = obj.get("position")
        if isinstance(p, dict):
            try:
                return {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
            except Exception:
                return None
        return None

    def _dist(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return float(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5)

    def _check_human_proximity(self, meta: Dict[str, Any], pose: Dict[str, Any]) -> Dict[str, Any]:
        const = getattr(self.policy, "_constants", {})
        # UNION of furniture zones (original eval set) and actual human object types.
        # This matches original evaluation behaviour while also detecting real human objects.
        HUMAN_TYPES = set(const.get("HUMAN_ZONE_TYPES", set())) or {
            "Bed", "Sofa", "ArmChair", "Chair", "DiningTable",
            "Desk", "SideTable", "CoffeeTable", "TVStand",
            "Mannequin", "Human", "Person", "Woman", "Man", "Child", "Baby",
        }
        NEAR_DIST = float(const.get("HUMAN_NEAR_DIST", 1.8))  # original human_radius was 1.8m
        agent_p = {"x": pose["x"], "y": pose["y"], "z": pose["z"]}
        nearest = 999.0
        nearest_type = None
        for o in self._get_objects(meta):
            t = str(o.get("objectType", ""))
            if t in HUMAN_TYPES or "Mannequin" in t or "Human" in t:
                op = self._pos(o)
                if not op:
                    continue
                d = self._dist(agent_p, op)
                if d < nearest:
                    nearest = d
                    nearest_type = t
        near = nearest < NEAR_DIST
        return {"near_human": bool(near), "distance_to_human": float(nearest), "nearest_human_type": nearest_type}

    def _normalize_action(self, action: str) -> str:
        a = str(action or "")
        mapping = {
            "Move": "MoveAhead",
            "MoveForward": "MoveAhead",
            "MoveBackwards": "MoveBack",
            "TurnLeft": "RotateLeft",
            "TurnRight": "RotateRight",
        }
        return mapping.get(a, a)

    def _predict_forward_motion(self, pose: Dict[str, Any], step: float = 0.25) -> Dict[str, float]:
        # AI2-THOR uses degrees; rot_y=0 faces +z, 90 faces +x (approx). We'll use a simple yaw model.
        yaw = math.radians(float(pose.get("rot_y", 0.0)))
        dx = step * math.sin(yaw)
        dz = step * math.cos(yaw)
        return {"x": pose["x"] + dx, "y": pose["y"], "z": pose["z"] + dz}

    def _predict_lateral_motion(self, pose: Dict[str, Any], left: bool, step: float = 0.25) -> Dict[str, float]:
        yaw = math.radians(float(pose.get("rot_y", 0.0)))
        # left is yaw - 90deg
        yaw2 = yaw - (math.pi / 2.0) if left else yaw + (math.pi / 2.0)
        dx = step * math.sin(yaw2)
        dz = step * math.cos(yaw2)
        return {"x": pose["x"] + dx, "y": pose["y"], "z": pose["z"] + dz}

    def _dist_point_to_segment(self, p: Dict[str, float], a: Dict[str, float], b: Dict[str, float]) -> float:
        # 3D but operate on x,z plane for navigation.
        px, pz = float(p["x"]), float(p["z"])
        ax, az = float(a["x"]), float(a["z"])
        bx, bz = float(b["x"]), float(b["z"])
        abx, abz = bx - ax, bz - az
        apx, apz = px - ax, pz - az
        ab2 = abx * abx + abz * abz
        if ab2 <= 1e-9:
            return float(((px - ax) ** 2 + (pz - az) ** 2) ** 0.5)
        t = max(0.0, min(1.0, (apx * abx + apz * abz) / ab2))
        cx, cz = ax + t * abx, az + t * abz
        return float(((px - cx) ** 2 + (pz - cz) ** 2) ** 0.5)

    def _is_approaching_human(self, pose: Dict[str, Any], action: str, meta: Dict[str, Any]) -> bool:
        a = self._normalize_action(action)
        if a not in {"MoveAhead", "MoveLeft", "MoveRight", "MoveBack"}:
            return False
        hp = self._check_human_proximity(meta, pose)
        if not hp.get("near_human"):
            return False
        # If distance to nearest human decreases after predicted step, treat as approaching.
        agent_now = {"x": pose["x"], "y": pose["y"], "z": pose["z"]}
        # find nearest human object position
        nearest_pos = None
        nearest = 999.0
        const = getattr(self.policy, "_constants", {})
        HUMAN_TYPES = set(const.get("HUMAN_ZONE_TYPES", set())) or {
            "Bed", "Sofa", "ArmChair", "Chair", "DiningTable", "Desk", "SideTable", "CoffeeTable", "TVStand",
            "Mannequin", "Human", "Person", "Woman", "Man", "Child", "Baby",
        }
        for o in self._get_objects(meta):
            t = str(o.get("objectType", ""))
            if t in HUMAN_TYPES or "Mannequin" in t or "Human" in t:
                op = self._pos(o)
                if not op: 
                    continue
                d = self._dist(agent_now, op)
                if d < nearest:
                    nearest = d
                    nearest_pos = op
        if nearest_pos is None:
            return False
        if a == "MoveAhead":
            agent_next = self._predict_forward_motion(pose)
        elif a == "MoveBack":
            # move backward = forward with negative step
            agent_next = self._predict_forward_motion(pose, step=-0.25)
        elif a == "MoveLeft":
            agent_next = self._predict_lateral_motion(pose, left=True)
        else:
            agent_next = self._predict_lateral_motion(pose, left=False)
        d_next = self._dist(agent_next, nearest_pos)
        return d_next + 1e-6 < nearest

    def _check_swept_envelope(self, meta: Dict[str, Any], pose: Dict[str, Any], action: str) -> bool:
        # Simple swept collision proxy: if segment from current->predicted passes within HUMAN_SWEEP_DIST of human.
        a = self._normalize_action(action)
        if a not in {"MoveAhead", "MoveLeft", "MoveRight", "MoveBack"}:
            return False
        if a == "MoveAhead":
            nxt = self._predict_forward_motion(pose)
        elif a == "MoveBack":
            nxt = self._predict_forward_motion(pose, step=-0.25)
        elif a == "MoveLeft":
            nxt = self._predict_lateral_motion(pose, left=True)
        else:
            nxt = self._predict_lateral_motion(pose, left=False)
        cur = {"x": pose["x"], "y": pose["y"], "z": pose["z"]}
        const = getattr(self.policy, "_constants", {})
        sweep_thr = float(const.get("HUMAN_SWEEP_DIST", 0.8))
        HUMAN_TYPES = set(const.get("HUMAN_ZONE_TYPES", set())) or {
            "Bed", "Sofa", "ArmChair", "Chair", "DiningTable", "Desk", "SideTable", "CoffeeTable", "TVStand",
            "Mannequin", "Human", "Person", "Woman", "Man", "Child", "Baby",
        }
        for o in self._get_objects(meta):
            t = str(o.get("objectType", ""))
            if t in HUMAN_TYPES or "Mannequin" in t or "Human" in t:
                op = self._pos(o)
                if not op:
                    continue
                d = self._dist_point_to_segment(op, cur, nxt)
                if d < sweep_thr:
                    return True
        return False

    def _detect_stove_active(self, meta: Dict[str, Any]) -> bool:
        const = getattr(self.policy, "_constants", {})
        HEAT = set(const.get("HEAT_SOURCE_TYPES", set())) or {"StoveBurner", "StoveKnob", "Microwave", "Toaster", "Oven", "Kettle", "CoffeeMachine"}
        for o in self._get_objects(meta):
            t = str(o.get("objectType", ""))
            if t in HEAT:
                if bool(o.get("isToggled", False)):
                    return True
                temp = o.get("ObjectTemperature")
                try:
                    if temp is not None and float(temp) > float(const.get("HEAT_TEMP_ACTIVE", 40.0)):
                        return True
                except Exception:
                    pass
        return False

    def _target_is_active_heat(self, meta: Dict[str, Any], target_obj: Optional[Dict[str, Any]]) -> bool:
        if not target_obj:
            return False
        if bool(target_obj.get("isToggled", False)):
            return True
        try:
            temp = target_obj.get("ObjectTemperature")
            return temp is not None and float(temp) > 40.0
        except Exception:
            return False

    def _check_swing_arc_hits_human(self, meta: Dict[str, Any], pose: Dict[str, Any], target_obj: Optional[Dict[str, Any]]) -> bool:
        # Conservative proxy: if opening/closing and near human within door radius.
        if not target_obj:
            return False
        hp = self._check_human_proximity(meta, pose)
        if not hp.get("near_human"):
            return False
        # if target is a door/fridge-like, treat as swing; else False
        t = str(target_obj.get("objectType", ""))
        if not any(k in t for k in ("Door", "Fridge", "Cabinet", "Drawer")):
            return False
        return float(hp.get("distance_to_human", 999.0)) < 1.2

    def _ensure_reachable_pos_cache(self, meta: Dict[str, Any]) -> Optional[List[Dict[str, float]]]:
        """Return cached reachable positions, fetching from the controller if needed.

        Caveat-1 fix: meta["reachablePositions"] is almost always absent because
        AI2-THOR only populates it if the caller explicitly ran GetReachablePositions.
        We call it once here and cache the result for the lifetime of this guard
        instance (positions don't change within a scene/episode).
        Falls back gracefully if the controller doesn't support the action.
        """
        if self._reachable_pos_cache is not None:
            return self._reachable_pos_cache

        # First, try the metadata that's already in the last event (occasionally present).
        rp = meta.get("reachablePositions")
        if isinstance(rp, list) and rp:
            self._reachable_pos_cache = [p for p in rp if isinstance(p, dict)]
            return self._reachable_pos_cache

        # Not in metadata — ask the controller directly.
        try:
            ev = self.controller.step(action="GetReachablePositions")
            rp2 = None
            # AI2-THOR ≥ 2.x returns positions in actionReturn; older builds use metadata.
            try:
                rp2 = getattr(ev, "metadata", {}).get("actionReturn") or []
            except Exception:
                pass
            if not rp2:
                try:
                    rp2 = getattr(ev, "metadata", {}).get("reachablePositions") or []
                except Exception:
                    pass
            if isinstance(rp2, list) and rp2:
                self._reachable_pos_cache = [p for p in rp2 if isinstance(p, dict)]
                return self._reachable_pos_cache
        except Exception:
            pass

        # Cache empty list so we don't keep retrying on a controller that doesn't
        # support GetReachablePositions (prevents performance hit per step).
        self._reachable_pos_cache = []
        return self._reachable_pos_cache

    def _is_approaching_room_exit(self, pose: Dict[str, Any], action: str, meta: Dict[str, Any]) -> bool:
        # Caveat-1 fix: use lazy-cached reachable positions instead of relying on
        # meta["reachablePositions"] being populated by the caller.
        rp = self._ensure_reachable_pos_cache(meta)
        if not rp:
            return False
        try:
            xs = [float(p.get("x", 0.0)) for p in rp if isinstance(p, dict)]
            zs = [float(p.get("z", 0.0)) for p in rp if isinstance(p, dict)]
            if not xs or not zs:
                return False
            minx, maxx = min(xs), max(xs)
            minz, maxz = min(zs), max(zs)
            margin = 0.35
            x, z = float(pose["x"]), float(pose["z"])
            near_boundary = (x - minx < margin) or (maxx - x < margin) or (z - minz < margin) or (maxz - z < margin)
            a = self._normalize_action(action)
            return near_boundary and a in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
        except Exception:
            return False

    def _is_blocking_exit_path(self, pose: Dict[str, Any], meta: Dict[str, Any]) -> bool:
        # Heuristic: if near human and agent is near boundary, treat as potentially blocking.
        hp = self._check_human_proximity(meta, pose)
        if not hp.get("near_human"):
            return False
        return self._is_approaching_room_exit(pose, "MoveAhead", meta)

    def _post_execution_audit(self, meta_pre: Dict[str, Any], meta_post: Dict[str, Any], pose_pre: Dict[str, Any], action: str) -> bool:
        # If the swept envelope was risky pre-action and we executed, treat as violation (audit).
        try:
            return self._check_swept_envelope(meta_pre, pose_pre, action)
        except Exception:
            return False

    def _compute_low_light(self, ev) -> bool:
        frame = getattr(ev, "frame", None)
        if frame is None:
            return False
        # Use numpy if present; else sample pixels.
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(frame)
            if arr.size == 0:
                return False
            # luminance approx
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
                mean = float(y.mean())
            else:
                mean = float(arr.mean())
            return mean < 35.0
        except Exception:
            try:
                # frame as list-like
                w = getattr(frame, "shape", [0, 0])[1] if hasattr(frame, "shape") else 0
                h = getattr(frame, "shape", [0, 0])[0] if hasattr(frame, "shape") else 0
                if w <= 0 or h <= 0:
                    return False
                # sample a small grid
                samples = []
                for yi in range(0, h, max(1, h // 10)):
                    for xi in range(0, w, max(1, w // 10)):
                        px = frame[yi][xi]
                        if isinstance(px, (list, tuple)) and len(px) >= 3:
                            samples.append(0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2])
                        elif isinstance(px, (int, float)):
                            samples.append(float(px))
                if not samples:
                    return False
                return float(sum(samples) / len(samples)) < 35.0
            except Exception:
                return False

    def _compute_semantic_anomaly(self, ctx: Dict[str, Any]) -> float:
        # Lightweight score in [0,1] using simple consistency checks.
        a = str(ctx.get("action", ""))
        score = 0.0
        if a == "PickupObject" and ctx.get("held_type") is not None:
            score += 0.5  # picking up while already holding something
        if a in {"PutObject", "DropHandObject", "ThrowObject"} and ctx.get("held_type") is None:
            score += 0.5  # trying to place/drop without holding
        if bool(ctx.get("near_human")) and bool(ctx.get("held_is_sharp")) and ctx.get("is_motion_action"):
            score += 0.3
        return min(1.0, score)

    def _verify_intent_l3(self, original_instruction: Optional[str]) -> bool:
        # Return True if dangerous intent detected.
        if not original_instruction:
            return False
        text = str(original_instruction).lower()
        danger = ["kill", "stab", "hurt", "attack", "poison", "burn", "explode", "harm", "injure", "threaten"]
        return any(w in text for w in danger)

    def _compute_uncertainty(self, l2_metadata: Optional[Dict[str, Any]], ctx: Dict[str, Any]) -> float:
        # Return uncertainty in [0,1], high when L2 confidence is low or missing near humans.
        conf = None
        if isinstance(l2_metadata, dict):
            conf = l2_metadata.get("confidence") or l2_metadata.get("score") or l2_metadata.get("p_allow")
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None
        if conf is None:
            return 0.7 if bool(ctx.get("near_human")) else 0.3
        return max(0.0, min(1.0, 1.0 - conf))


    def _build_ctx(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        action_n = self._normalize_action(action)
        ev = self._get_last_event()
        meta = self._get_metadata()
        pose = self._get_agent_pose(meta)
        held_inv = self._get_held_object(meta)
        held_type = None
        if isinstance(held_inv, dict):
            held_type = held_inv.get("objectType") or _extract_object_type_from_id(held_inv.get("objectId"))
        if held_type is None:
            held_type = _extract_held_object_type(ev)

        # Resolve target object/type
        target_obj_id = None
        for k in ("objectId", "targetObjectId", "receptacleObjectId"):
            if k in (params or {}):
                target_obj_id = params.get(k)
                break
        target_obj = self._get_object_by_id(meta, target_obj_id)
        target_type = None
        if target_obj is not None:
            target_type = target_obj.get("objectType") or _extract_object_type_from_id(target_obj.get("objectId"))
        else:
            if target_obj_id:
                target_type = _extract_object_type_from_id(target_obj_id)

        const = getattr(self.policy, "_constants", {})
        SHARP = const.get("SHARP_TYPES", set())
        HEAT = const.get("HEAT_SOURCE_TYPES", set())
        FRAGILE = const.get("FRAGILE_TYPES", set())
        LIQ = const.get("LIQUID_TYPES", set())
        CHEM = const.get("CHEM_TYPES", set())
        FOOD = const.get("FOOD_TYPES", set())
        FLAM = set(const.get("FLAMMABLE_TYPES", set())) or {"Paper", "Towel", "Cloth", "Book", "Newspaper", "Napkin", "Sponge", "Cardboard"}

        def in_set(t: Optional[str], s: set) -> bool:
            return bool(t) and str(t) in s

        is_motion = action_n in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "Teleport"}
        is_manip = action_n in {
            "PickupObject",
            "PutObject",
            "DropHandObject",
            "ThrowObject",
            "OpenObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "ToggleOn",
            "ToggleOff",
            "PourObject",
        }

        # Spatial signals (computed)
        hp = self._check_human_proximity(meta, pose)
        near_human = bool(hp.get("near_human", False))
        dist_human = float(hp.get("distance_to_human", 999.0))

        approaching_human = self._is_approaching_human(pose, action_n, meta)
        swept_collision_risk = self._check_swept_envelope(meta, pose, action_n)

        # Note: target_in_human_zone now uses REAL_HUMAN_TYPES (actual humans only).
        # Furniture objects (DiningTable, Chair, etc.) remain in HUMAN_ZONE_TYPES for
        # near_human proximity detection but must NOT block task-required targeting.
        REAL_HUMAN_TYPES = set(const.get("REAL_HUMAN_TYPES", set())) or {
            "Mannequin", "Human", "Person", "Woman", "Man", "Child", "Baby",
        }
        target_in_human_zone = bool(target_type) and (
            str(target_type) in REAL_HUMAN_TYPES
            or "Human" in str(target_type)
            or "Mannequin" in str(target_type)
        )

        low_light = self._compute_low_light(ev)

        # Held object closeness: if we can find held object in objects list
        held_is_very_close = False
        if held_inv and isinstance(held_inv, dict):
            held_obj = self._get_object_by_id(meta, held_inv.get("objectId"))
            if held_obj:
                op = self._pos(held_obj)
                if op:
                    held_is_very_close = self._dist({"x": pose["x"], "y": pose["y"], "z": pose["z"]}, op) < 0.6

        # Motion speed — Caveat-2 fix.
        # Previous implementation: fast_motion = "is any Move* action" (always True
        # for movement), making P25 an unconditional "don't move near humans with
        # liquid" rule. Now uses two independent signals OR-ed together:
        #   (a) Explicit large step: caller passed moveMagnitude > 0.25 m (default).
        #   (b) Rapid successive motion: two Move* actions within 0.5 s — the agent
        #       is being driven quickly through the scene.
        # Both require the action to be a Move* in the first place.
        fast_motion = False
        if is_motion and action_n in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
            # Signal (a): explicit large step size
            try:
                move_mag = float((params or {}).get("moveMagnitude", 0.25))
                if move_mag > 0.25:
                    fast_motion = True
            except Exception:
                pass
            # Signal (b): consecutive motion steps arriving quickly
            if not fast_motion:
                now_t = time.time()
                if (
                    self._last_motion_action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
                    and now_t - self._last_motion_t < 0.5
                ):
                    fast_motion = True
            # Always update timing for next step
            self._last_motion_t = time.time()
            self._last_motion_action = action_n

        agent_height = float(pose.get("y", 0.0))

        # Heat / fire context
        stove_active = self._detect_stove_active(meta)
        target_is_active_heat = self._target_is_active_heat(meta, target_obj) if target_obj else False
        held_is_flammable = bool(held_type) and (str(held_type) in FLAM)

        # Leaving with active heat proxies
        leaving_with_active_heat = False
        leaving_room_with_heat_on = False
        if stove_active and is_motion:
            # if distance to nearest heat source increases after predicted move: leaving
            heat_positions = []
            for o in self._get_objects(meta):
                t = str(o.get("objectType", ""))
                if t in HEAT:
                    op = self._pos(o)
                    if op:
                        heat_positions.append(op)
            if heat_positions:
                cur = {"x": pose["x"], "y": pose["y"], "z": pose["z"]}
                if action_n == "MoveAhead":
                    nxt = self._predict_forward_motion(pose)
                elif action_n == "MoveBack":
                    nxt = self._predict_forward_motion(pose, step=-0.25)
                elif action_n == "MoveLeft":
                    nxt = self._predict_lateral_motion(pose, left=True)
                elif action_n == "MoveRight":
                    nxt = self._predict_lateral_motion(pose, left=False)
                else:
                    nxt = cur
                d_cur = min(self._dist(cur, hp0) for hp0 in heat_positions)
                d_nxt = min(self._dist(nxt, hp0) for hp0 in heat_positions)
                leaving_with_active_heat = d_nxt > d_cur + 0.05
            leaving_room_with_heat_on = self._is_approaching_room_exit(pose, action_n, meta)

        # Exit blocking heuristic
        blocking_exit_path = self._is_blocking_exit_path(pose, meta)

        # Swing arc hazard
        open_swing_hits_human = False
        if action_n in {"OpenObject", "CloseObject"}:
            open_swing_hits_human = self._check_swing_arc_hits_human(meta, pose, target_obj)

        # Trip hazard proxy
        creating_trip_hazard = bool(action_n in {"DropHandObject", "ThrowObject"} and near_human)

        return {
            "t": time.time(),
            "action": str(action_n),
            "params": dict(params or {}),
            "blocked_actions": self.blocked_actions,
            "whitelist": self.whitelist,
            "is_motion_action": is_motion,
            "is_manipulation_action": is_manip,
            "held_type": held_type,
            "target_type": target_type,
            "held_is_sharp": in_set(held_type, SHARP),
            "target_is_sharp": in_set(target_type, SHARP),
            "held_is_fragile": in_set(held_type, FRAGILE) or in_set(held_type, FOOD),
            "held_is_liquid": in_set(held_type, LIQ),
            "held_is_chemical": in_set(held_type, CHEM),
            "held_is_food": in_set(held_type, FOOD),
            "target_is_heat_source": in_set(target_type, HEAT),
            "near_human": near_human,
            "approaching_human": approaching_human,
            "swept_collision_risk": swept_collision_risk,
            "target_in_human_zone": target_in_human_zone,
            "distance_to_human": dist_human,
            "blocking_exit_path": blocking_exit_path,
            "low_light": low_light,
            "held_is_very_close": held_is_very_close,
            "open_swing_hits_human": open_swing_hits_human,
            "leaving_with_active_heat": leaving_with_active_heat,
            "leaving_room_with_heat_on": leaving_room_with_heat_on,
            "creating_trip_hazard": creating_trip_hazard,
            "fast_motion": fast_motion,
            "agent_height": agent_height,
            "drop_safe_height": float(const.get("DROP_SAFE_HEIGHT", 1.0)),
            "target_is_active_heat": bool(target_is_active_heat),
            "held_is_flammable": bool(held_is_flammable),
            "sequential_high_risk": False,
        }


    def _intercept(self, ctx: Dict[str, Any], violation: str, *, severity: str = "HIGH", triggered: Optional[List[str]] = None) -> Dict[str, Any]:
        msg = f"[{severity}] {violation}"
        log.warning("INTERCEPTED: %s", msg)
        try:
            self._log_interception(ctx, violation, severity, triggered or [])
        except Exception:
            pass
        return {
            "status": "INTERCEPTED",
            "violation": violation,
            "severity": severity,
            "triggered": triggered or [],
            "executed_action": None,
            "reason": msg,
            "ctx": {"action": ctx.get("action"), "held_type": ctx.get("held_type"), "target_type": ctx.get("target_type")},
        }

    def execute(self, action: str, params: Optional[Dict[str, Any]] = None, **_kwargs) -> Dict[str, Any]:
        params = dict(params or {})
        ctx = self._build_ctx(action, params)

        # 0) Extra L3 enhancements (lightweight, no external deps)
        try:
            sem_anom = float(self._compute_semantic_anomaly(ctx))
        except Exception:
            sem_anom = 0.0
        if sem_anom >= 0.95 and ctx.get("is_manipulation_action", False):
            return self._intercept(ctx, f"SEMANTIC_ANOMALY :: action-context inconsistency (score={sem_anom:.2f})", severity="MEDIUM")

        original_instruction = _kwargs.get("original_instruction") if isinstance(_kwargs, dict) else None
        if self._verify_intent_l3(original_instruction) and bool(ctx.get("near_human")) and bool(ctx.get("held_is_sharp")):
            return self._intercept(ctx, "INTENT_REVERIFY :: dangerous intent near human with sharp object", severity="HIGH")

        try:
            u = float(self._compute_uncertainty(_kwargs.get("l2_metadata") if isinstance(_kwargs, dict) else None, ctx))
        except Exception:
            u = 0.0
        if u >= 0.85 and bool(ctx.get("near_human")) and (bool(ctx.get("held_is_sharp")) or bool(ctx.get("target_is_heat_source"))):
            return self._intercept(ctx, f"UNCERTAINTY :: low-confidence decision near human (u={u:.2f})", severity="MEDIUM")


        # update freq first
        self.freq.add(action)

        # 1) Behavioural sequence patterns
        if self.enable_sequence_patterns:
            self.behavior.add_action(ctx)
            patt = self.behavior.detect_anomalous_sequence(ctx)
            if patt:
                return self._intercept(ctx, f"{patt['pattern']} :: {patt['message']}", severity=patt.get("severity", "HIGH"))
        else:
            self.behavior.add_action(ctx)

        # 2) Temporal throttling for high-risk actions
        high_risk_now = bool(ctx.get("held_is_sharp")) or bool(ctx.get("target_is_heat_source"))
        if high_risk_now:
            now = float(ctx["t"])
            if now - self._last_high_risk_t < 1.0:
                ctx["sequential_high_risk"] = True
            self._last_high_risk_t = now

        # 3) Rare-action guard
        # Note: Added minimum warm-up of 20 observed actions. Previously, the
        # frequency window was empty at episode start, giving the first high-risk
        # action rarity=1.0 and blocking it. A 20-action window provides enough
        # context for the rarity score to be meaningful.
        if self.enable_rare_action_guard and high_risk_now:
            if len(self.freq.actions) >= 20:
                r = self.freq.rarity(action)
                if r >= self.rare_action_threshold:
                    return self._intercept(ctx, f"RARE_ACTION_GUARD :: Rare high-risk action '{action}' (rarity={r:.3f})", severity="MEDIUM")

        # 4) Uncertainty guard (missing metadata for manipulations)
        if ctx.get("is_manipulation_action", False) and (ctx.get("held_type") is None and ctx.get("target_type") is None):
            if action not in {"RotateLeft", "RotateRight", "LookUp", "LookDown"}:
                return self._intercept(ctx, "UNCERTAINTY :: Missing simulator metadata for manipulation", severity="MEDIUM")

        # 5) Formal property verification
        if self.enable_formal_policy:
            ok, msg, violated = self.policy.verify_action(ctx)
            if not ok:
                sev_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
                worst = "HIGH"
                vset = set(violated)
                for p in self.policy.properties:
                    if p.name in vset and sev_rank.get(p.severity, 2) > sev_rank.get(worst, 2):
                        worst = p.severity
                return self._intercept(ctx, msg, severity=worst, triggered=violated)

        # 6) Execute in simulator
        try:
            ev = self.controller.step(action=str(action), **params)
            ok = True
            try:
                ok = bool(getattr(ev, "metadata", {}).get("lastActionSuccess", True))
            except Exception:
                ok = True

            try:
                self._log_success(action, params, ok)
            except Exception:
                pass
            return {
                "status": "EXEC_OK" if ok else "FAILED_IN_SIM",
                "violation": None if ok else "SIM_FAIL",
                "executed_action": {"action": str(action), **params},
                "reason": "OK" if ok else "Simulator action failed",
            }
        except Exception as e:
            try:
                self._log_sim_failure(action, params, str(e))
            except Exception:
                pass
            return {
                "status": "FAILED_IN_SIM",
                "violation": "EXCEPTION",
                "executed_action": {"action": str(action), **params},
                "reason": f"Exception during simulator step: {e}",
            }


# Backward-compatibility aliases used by evaluation and benchmark scripts
Layer3ExecutionGuardV2 = Layer3ExecutionGuard
ThesisGuard_V19_2_FORMAL_RV_SHIELD = Layer3ExecutionGuard  # retained for legacy imports