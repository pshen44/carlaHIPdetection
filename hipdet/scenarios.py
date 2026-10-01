"""Scenario specification and placement of actors relative to the ego car.

The old scripts spawned ambulances at random map spawn points, so in a 10 s
run they were almost never in view and most logged frames were empty. Here
the HIP actor is placed at a controlled distance and lane relative to the ego
car, which makes distance, lane relation, weather and light state proper
experimental variables.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass, field

HIP_TYPES = ("ambulance", "police", "firetruck", "hazard_car")
EMERGENCY_TYPES = ("ambulance", "police", "firetruck")
PLACEMENTS = ("same_lane", "adjacent_lane", "oncoming")


@dataclass
class Scenario:
    sample_id: str
    town: str = "Town10HD"
    weather: str = "ClearNoon"
    hip_type: str = "ambulance"      # one of HIP_TYPES or "none"
    lights_on: bool = True
    placement: str = "same_lane"     # one of PLACEMENTS or "none"
    distance: float = 25.0           # metres ahead of the ego car (along the lane)
    n_distractors: int = 2
    seed: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class GridSpec:
    """Full-factorial grid over the scenario factors, repeated ``repeats`` times.

    Negative samples (no HIP at all) are added at ``negative_fraction`` of the
    positive grid size.
    """
    towns: list = field(default_factory=lambda: ["Town10HD"])
    weathers: list = field(default_factory=lambda: ["ClearNoon", "ClearNight"])
    hip_types: list = field(default_factory=lambda: list(HIP_TYPES))
    lights: list = field(default_factory=lambda: [True, False])
    placements: list = field(default_factory=lambda: list(PLACEMENTS))
    distances: list = field(default_factory=lambda: [15.0, 30.0, 60.0])
    repeats: int = 1
    negative_fraction: float = 0.15
    n_distractors: int = 2
    seed: int = 0

    def expand(self) -> list:
        rng = random.Random(self.seed)
        out = []
        combos = list(itertools.product(self.towns, self.weathers, self.hip_types, self.lights,
                                        self.placements, self.distances, range(self.repeats)))
        for town, weather, hip, lit, place, dist, rep in combos:
            out.append(Scenario(sample_id="", town=town, weather=weather, hip_type=hip,
                                lights_on=lit, placement=place, distance=float(dist),
                                n_distractors=self.n_distractors, seed=rng.randrange(1 << 30)))
        n_neg = int(round(len(out) * self.negative_fraction))
        for i in range(n_neg):
            out.append(Scenario(sample_id="", town=self.towns[i % len(self.towns)],
                                weather=self.weathers[i % len(self.weathers)], hip_type="none",
                                lights_on=False, placement="none", distance=0.0,
                                n_distractors=self.n_distractors + 1,
                                seed=rng.randrange(1 << 30)))
        rng.shuffle(out)
        # Group by town so the map is loaded as few times as possible.
        out.sort(key=lambda s: s.town)
        for i, s in enumerate(out):
            s.sample_id = f"s{i:05d}"
        return out


# --------------------------------------------------------------------------
# Lane geometry helpers (need carla at call time only)
# --------------------------------------------------------------------------

def _same_direction(a, b) -> bool:
    return a.lane_id * b.lane_id > 0


def lane_layout(wp):
    """Return (ego_lane_number, n_lanes) counting same-direction driving lanes from the left.

    "Left" is relative to the direction of travel. Opposing lanes and shoulders
    are not counted, which is the rule the prompt also states.
    """
    import carla

    left = 0
    w = wp.get_left_lane()
    while w is not None and w.lane_type == carla.LaneType.Driving and _same_direction(w, wp):
        left += 1
        w = w.get_left_lane()
    right = 0
    w = wp.get_right_lane()
    while w is not None and w.lane_type == carla.LaneType.Driving and _same_direction(w, wp):
        right += 1
        w = w.get_right_lane()
    return left + 1, left + right + 1


def _walk(wp, distance: float, step: float = 2.0):
    """Follow the lane ``distance`` metres forward. Returns None if it crosses a junction."""
    travelled = 0.0
    cur = wp
    while travelled < distance:
        nxt = cur.next(min(step, distance - travelled))
        if not nxt:
            return None
        cur = nxt[0]
        if cur.is_junction:
            return None
        travelled += step
    return cur


def _oncoming_lane(wp):
    import carla

    w = wp.get_left_lane()
    hops = 0
    while w is not None and _same_direction(w, wp) and hops < 6:
        w = w.get_left_lane()
        hops += 1
    if w is None or w.lane_type != carla.LaneType.Driving or _same_direction(w, wp):
        return None
    return w


def _adjacent_lanes(wp):
    import carla

    out = []
    for w in (wp.get_left_lane(), wp.get_right_lane()):
        if w is not None and w.lane_type == carla.LaneType.Driving and _same_direction(w, wp):
            out.append(w)
    return out


def find_layout(world_map, scenario: Scenario, rng: random.Random, max_tries: int = 400):
    """Pick an ego start waypoint and the HIP target waypoint for ``scenario``.

    Returns (ego_wp, target_wp_or_None) or None if no valid location was found.
    """
    import carla

    spawns = world_map.get_spawn_points()
    rng.shuffle(spawns)
    for sp in spawns[:max_tries]:
        ego_wp = world_map.get_waypoint(sp.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if ego_wp is None or ego_wp.is_junction:
            continue
        # Need clear road in front: the target distance plus some margin.
        reach = max(scenario.distance, 20.0) + 10.0
        if _walk(ego_wp, reach) is None:
            continue
        if scenario.placement == "none":
            return ego_wp, None
        ahead = _walk(ego_wp, scenario.distance)
        if ahead is None:
            continue
        if scenario.placement == "same_lane":
            return ego_wp, ahead
        if scenario.placement == "adjacent_lane":
            lanes = _adjacent_lanes(ahead)
            if lanes:
                return ego_wp, rng.choice(lanes)
            continue
        if scenario.placement == "oncoming":
            w = _oncoming_lane(ahead)
            if w is not None:
                return ego_wp, w
            continue
        raise ValueError(f"unknown placement {scenario.placement}")
    return None


def distractor_waypoints(ego_wp, target_wp, n: int, rng: random.Random):
    """Waypoints for ordinary parked/stopped traffic ahead of the ego car."""
    import carla

    out = []
    tries = 0
    while len(out) < n and tries < 40:
        tries += 1
        d = rng.uniform(8.0, 70.0)
        ahead = _walk(ego_wp, d)
        if ahead is None:
            continue
        lanes = [ahead] + _adjacent_lanes(ahead)
        onc = _oncoming_lane(ahead)
        if onc is not None:
            lanes.append(onc)
        w = rng.choice(lanes)
        if w.lane_type != carla.LaneType.Driving:
            continue
        # Keep the HIP's own lane clear so distractors do not simply hide it.
        if target_wp is not None and (w.road_id, w.lane_id) == (target_wp.road_id, target_wp.lane_id):
            continue
        loc = w.transform.location
        taken = [ego_wp] + out + ([target_wp] if target_wp is not None else [])
        if any(loc.distance(t.transform.location) < 9.0 for t in taken):
            continue
        out.append(w)
    return out
