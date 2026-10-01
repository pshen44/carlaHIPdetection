"""Ground-truth labels for a captured sample, computed from simulator state.

This replaces the hand-typed "Enter score:" step in the old pipeline. CARLA
knows where every actor is, what its lights are doing and which pixels it
covers (via the instance-segmentation camera), so every sample can be labelled
automatically and models can be scored with ordinary metrics.

Labelling rules (the prompt states the same rules to the model):

* A HIP is an emergency vehicle (ambulance, police car, fire truck) with its
  emergency lights on, or any vehicle with its hazard lights on.
* A HIP counts as present only if it is visible: it covers at least
  ``MIN_VISIBLE_PIXELS`` pixels in the camera image.
* The ego car must respond (slow down / stop / change lanes) if a visible HIP
  is on its side of the road (same lane or an adjacent same-direction lane),
  or if a visible *emergency* HIP is in oncoming traffic, and the HIP is
  within ``RESPONSE_RANGE_M`` metres.
"""

from __future__ import annotations

import numpy as np

from .scenarios import EMERGENCY_TYPES

MIN_VISIBLE_PIXELS = 40
RESPONSE_RANGE_M = 80.0


def is_hip(hip_type: str, lights_on: bool) -> bool:
    return hip_type != "none" and lights_on


def response_required(hip_type: str, lights_on: bool, placement: str,
                      visible: bool, distance: float) -> bool:
    if not (visible and is_hip(hip_type, lights_on)) or distance > RESPONSE_RANGE_M:
        return False
    if placement in ("same_lane", "adjacent_lane"):
        return True
    if placement == "oncoming":
        return hip_type in EMERGENCY_TYPES
    return False


def instance_stats(instance_ids: np.ndarray, actor_id_candidates):
    """Pixel count and bounding box for the first candidate id present in the mask.

    CARLA documents the G/B channels as a "unique object id"; for vehicles we
    found it equals the actor id, but the caller passes candidates so this keeps
    working if that mapping changes.
    """
    for cid in actor_id_candidates:
        mask = instance_ids == cid
        n = int(mask.sum())
        if n:
            ys, xs = np.nonzero(mask)
            return n, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return 0, None


def label_sample(scenario, ego_lane: int, n_lanes: int, hip_pixels: int, hip_bbox,
                 true_distance: float | None):
    visible = hip_pixels >= MIN_VISIBLE_PIXELS
    hip_present = visible and is_hip(scenario.hip_type, scenario.lights_on)
    dist = true_distance if true_distance is not None else scenario.distance
    return {
        "hip_present": bool(hip_present),
        "hip_types": [scenario.hip_type] if hip_present else [],
        "target_visible": bool(visible),
        "target_pixels": int(hip_pixels),
        "target_bbox": hip_bbox,
        "target_distance_m": None if true_distance is None else round(float(true_distance), 2),
        "response_required": response_required(scenario.hip_type, scenario.lights_on,
                                               scenario.placement, visible, dist),
        "ego_lane": int(ego_lane),
        "n_lanes": int(n_lanes),
    }
