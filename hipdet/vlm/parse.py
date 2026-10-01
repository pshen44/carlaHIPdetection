"""Turn a model reply into a normalised prediction dict.

Even with structured output enabled we parse defensively: some endpoints
ignore the schema, and replies that cannot be parsed are counted as failures
in the metrics instead of crashing the run (the old regex + ``json.loads``
crashed on the first malformed reply).
"""

from __future__ import annotations

import json
import re

_TYPE_ALIASES = {
    "police car": "police", "police_car": "police", "fire truck": "firetruck",
    "fire_truck": "firetruck", "hazard": "hazard_vehicle", "hazard lights": "hazard_vehicle",
    "hazard_car": "hazard_vehicle",
}


def _extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_prediction(text: str) -> dict:
    obj = _extract_json(text or "")
    if not isinstance(obj, dict):
        return {"parse_ok": False, "hip_present": None, "hips": [], "car_response": None,
                "ego_lane": None, "n_lanes": None, "confidence": None}
    hips = []
    for h in obj.get("hips") or []:
        if isinstance(h, dict):
            t = str(h.get("type", "other")).strip().lower()
            hips.append({"type": _TYPE_ALIASES.get(t, t), "relation": str(h.get("relation", "other"))})
        elif isinstance(h, str):
            t = h.strip().lower()
            hips.append({"type": _TYPE_ALIASES.get(t, t), "relation": "other"})
    present = _as_bool(obj.get("hip_present"))
    if present is None:
        present = bool(hips)
    conf = obj.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 1.0 if present else 0.0
    return {
        "parse_ok": True,
        "hip_present": present,
        "hips": hips,
        "car_response": _as_bool(obj.get("car_response")),
        "ego_lane": _as_int(obj.get("ego_lane")),
        "n_lanes": _as_int(obj.get("n_lanes")),
        "confidence": conf,
    }
