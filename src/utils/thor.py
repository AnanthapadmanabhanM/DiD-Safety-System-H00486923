from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def get_event_rgb(event) -> np.ndarray:
    "Return HxWx3 uint8 RGB frame from AI2-THOR event."
    frame = event.frame
    if frame is None:
        raise ValueError("event.frame is None")
    return frame

def visible_objects(event) -> List[Dict[str, Any]]:
    objs = event.metadata.get("objects", [])
    return [o for o in objs if o.get("visible", False)]

def inventory_objects(event) -> List[Dict[str, Any]]:
    inv = event.metadata.get("inventoryObjects", [])
    return inv if isinstance(inv, list) else []

def agent_position(event) -> Tuple[float, float, float]:
    pos = event.metadata.get("agent", {}).get("position", {})
    return float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))

def obj_center(o: Dict[str, Any]) -> Tuple[float, float, float]:
    c = o.get("axisAlignedBoundingBox", {}).get("center", {})
    return float(c.get("x", 0.0)), float(c.get("y", 0.0)), float(c.get("z", 0.0))

def l2_norm_xy(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    return float(np.linalg.norm([a[0]-b[0], a[2]-b[2]]))

def find_nearest_visible_object(event, object_type: str) -> Optional[Dict[str, Any]]:
    cand = [o for o in visible_objects(event) if o.get("objectType")==object_type]
    if not cand:
        return None
    ap = agent_position(event)
    cand.sort(key=lambda o: l2_norm_xy(ap, obj_center(o)))
    return cand[0]

def resolve_object_id(event, preferred_type: Optional[str], preferred_id: Optional[str]) -> Optional[str]:
    "Resolve an object id using either explicit id or nearest visible of a type."
    if preferred_id:
        return preferred_id
    if preferred_type:
        o = find_nearest_visible_object(event, preferred_type)
        if o:
            return o.get("objectId")
    return None

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
