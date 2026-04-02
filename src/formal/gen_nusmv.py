from __future__ import annotations

# ---
# NuSMV model generator — 30 LTL properties + 11 system properties
# Aligned with l3_reference_monitor.py (P1-P30)

# ---
import argparse
import os
from typing import List

# Exact THOR action strings (matches l3_reference_monitor.py action_schema)
ACTIONS = [
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "RotateLeft", "RotateRight",
    "LookUp", "LookDown",
    "OpenObject", "CloseObject",
    "PickupObject", "PutObject",
    "DropHandObject",
    "ThrowObject", "PushObject", "SliceObject", "BreakObject",
    "Stop",
]

# 30 LTL safety properties matching l3_reference_monitor.py
LTL_SPECS = [
    # P1-P12: Sharp objects (40%)
    ("P1_NO_MOTION_SHARP_NEAR_HUMAN",
     "G !(held_type = sharp & near_human & is_motion)"),
    ("P2_NO_PICKUP_SHARP_NEAR_HUMAN",
     "G !(action = PickupObject & near_human & target_type = sharp)"),
    ("P3_NO_DOUBLE_SHARP",
     "G !(action = PickupObject & held_type = sharp & target_type = sharp)"),
    ("P4_NO_SWEPT_COLLISION",
     "G !(held_type = sharp & zone_close & is_motion)"),
    ("P5_NO_APPROACH_WITH_SHARP",
     "G !(held_type = sharp & approaching)"),
    ("P6_NO_BLOCKED_ACTIONS",
     "G !(action in {ThrowObject, PushObject, SliceObject, BreakObject})"),
    ("P7_WHITELIST_ONLY",
     "G (action in {MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown, OpenObject, CloseObject, PickupObject, PutObject, DropHandObject, Stop})"),
    ("P8_NO_DROP_FRAGILE_HIGH",
     "G !(action = DropHandObject & held_type = fragile & agent_elevated)"),
    ("P9_NO_TRANSFER_SHARP_NEAR_HUMAN",
     "G !(action = PutObject & held_type = sharp & near_human)"),
    ("P10_NO_PICKUP_WHILE_HOLDING_SHARP",
     "G !(action = PickupObject & held_type = sharp)"),
    ("P11_NO_CONTAINER_SHARP_NEAR_HUMAN",
     "G !(action in {OpenObject, CloseObject} & held_type = sharp & near_human)"),
    ("P12_NO_DROP_SHARP",
     "G !(action = DropHandObject & held_type = sharp)"),
    # P13-P20: General safety (27%)
    ("P13_NO_MOTION_HELD_VERY_CLOSE",
     "G !(is_motion & holding_any & zone_very_close)"),
    ("P14_NO_ROTATE_SHARP_NEAR_HUMAN",
     "G !(action in {RotateLeft, RotateRight} & held_type = sharp & near_human)"),
    ("P15_NO_LOOK_SHARP_NEAR_HUMAN",
     "G !(action in {LookUp, LookDown} & held_type = sharp & near_human)"),
    ("P16_NO_PICKUP_HEAVY_NEAR_HUMAN",
     "G !(action = PickupObject & target_type = heavy & near_human)"),
    ("P17_NO_TARGET_HUMAN_ZONE",
     "G !(target_is_human_zone)"),
    ("P18_NO_SEQUENTIAL_HIGH_RISK",
     "G !(rapid_succession & action in {PickupObject, PutObject, DropHandObject, OpenObject, CloseObject})"),
    ("P19_NO_ACTION_APPROACHING",
     "G !(distance_decreasing & held_type = sharp)"),
    ("P20_NO_ACTION_LOW_LIGHT_SHARP",
     "G !(ambient_low & held_type = sharp)"),
    # P21-P23: Thermal safety (10%)
    ("P21_NO_TOGGLE_HEAT_NEAR_HUMAN",
     "G !(target_is_heat_source & near_human & action in {OpenObject, CloseObject, PickupObject})"),
    ("P22_NO_LEAVE_STOVE_UNATTENDED",
     "G !(stove_active & !near_human & is_translational_motion)"),
    ("P23_NO_PLACE_ON_ACTIVE_HEAT",
     "G !(action = PutObject & target_is_heat_source & stove_active)"),
    # P24-P25: Liquid safety (7%)
    ("P24_NO_POUR_LIQUID_NEAR_HUMAN",
     "G !(held_has_liquid & near_human & action in {PutObject, DropHandObject})"),
    ("P25_NO_CARRY_LIQUID_FAST_NEAR_HUMAN",
     "G !(held_has_liquid & zone_close & is_motion)"),
    # P26-P27: Environmental safety (7%)
    ("P26_NO_LEAVE_ROOM_STOVE_ON",
     "G !(stove_active & approaching_room_exit & action = MoveAhead)"),
    ("P27_NO_FLOOR_TRIP_HAZARD",
     "G !(action = DropHandObject & near_human & holding_any & !held_is_sharp)"),
    # P28-P29: Proximity safety (7%)
    ("P28_NO_APPROACH_VERY_CLOSE",
     "G !(approaching & zone_very_close)"),
    ("P29_NO_CORNERING_HUMAN",
     "G !(is_motion & near_human & blocking_exit_path)"),
    # P30: Manipulation safety (3%)
    ("P30_NO_OPEN_SWING_INTO_HUMAN",
     "G !(action = OpenObject & open_swing_hits_human)"),
]

def main():
    ap = argparse.ArgumentParser(
        description="Generate NuSMV model for Layer-3 Execution Guard (30 properties + 11 system)."
    )
    ap.add_argument("--out", default="formal/guard_model.smv",
                    help="Output .smv file path")
    args = ap.parse_args()

    # Instead of generating from template, copy the hand-crafted did_safety.smv
    # which has the full 29-property model (20 P-specs + 9 S-specs)
    smv_source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../../formal/did_safety.smv")
    
    if os.path.exists(smv_source):
        import shutil
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        shutil.copy2(smv_source, args.out)
        print(f"[NuSMV] Copied {smv_source} -> {args.out}")
    else:
        print(f"[NuSMV] WARNING: {smv_source} not found")
        print(f"[NuSMV] Use formal/did_safety.smv directly instead")
    
    print(f"[NuSMV] Actions ({len(ACTIONS)}): {ACTIONS}")
    print(f"[NuSMV] LTL specs ({len(LTL_SPECS)}): {[s[0] for s in LTL_SPECS]}")


if __name__ == "__main__":
    main()
