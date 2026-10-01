"""Tests for the parts of the pipeline that do not need a running simulator."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import baselines, ground_truth as gt  # noqa: E402
from hipdet.evaluate import auc, binary_metrics, wilson  # noqa: E402
from hipdet.scenarios import GridSpec, Scenario  # noqa: E402
from hipdet.vlm.parse import parse_prediction  # noqa: E402


# ---------------------------------------------------------------- colour order

def test_bgra_is_converted_to_rgb():
    carla_utils = pytest.importorskip("hipdet.carla_utils")
    # One pure-red pixel as CARLA delivers it: B, G, R, A.
    img = SimpleNamespace(raw_data=bytes([0, 0, 255, 255]), height=1, width=1)
    rgb = carla_utils.to_rgb_array(img)
    assert rgb.tolist() == [[[255, 0, 0]]]


def test_instance_ids_decode_green_and_blue():
    carla_utils = pytest.importorskip("hipdet.carla_utils")
    # tag 14 (car), id = G + 256 * B = 25 + 256 * 1 = 281
    img = SimpleNamespace(raw_data=bytes([1, 25, 14, 255]), height=1, width=1)
    tag, ids = carla_utils.to_instance_arrays(img)
    assert tag[0, 0] == 14 and ids[0, 0] == 281


# ---------------------------------------------------------------- parsing

def test_parse_strict_json():
    p = parse_prediction('{"hip_present": true, "hips": [{"type": "Police Car", "relation": "oncoming"}],'
                         ' "car_response": true, "ego_lane": 2, "n_lanes": 3, "confidence": 0.9}')
    assert p["parse_ok"] and p["hip_present"] and p["hips"][0]["type"] == "police"
    assert p["ego_lane"] == 2 and p["confidence"] == 0.9


def test_parse_fenced_and_sloppy():
    p = parse_prediction('Sure!\n```json\n{"hip_present": "false", "hips": [], "car_response": "no"}\n```')
    assert p["parse_ok"] and p["hip_present"] is False and p["car_response"] is False


def test_parse_garbage_is_a_failure_not_a_crash():
    p = parse_prediction("I cannot see anything <True>")
    assert not p["parse_ok"] and p["hip_present"] is None


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("hip,lit,place,expected", [
    ("ambulance", True, "same_lane", True),
    ("ambulance", True, "oncoming", True),
    ("hazard_car", True, "oncoming", False),   # hazards across the median: no action needed
    ("hazard_car", True, "adjacent_lane", True),
    ("police", False, "same_lane", False),     # lights off: not a HIP
])
def test_response_rule(hip, lit, place, expected):
    assert gt.response_required(hip, lit, place, visible=True, distance=30) is expected


def test_invisible_or_far_hip_needs_no_response():
    assert not gt.response_required("ambulance", True, "same_lane", visible=False, distance=30)
    assert not gt.response_required("ambulance", True, "same_lane", visible=True, distance=120)


def test_label_sample_visibility_threshold():
    s = Scenario("x", hip_type="police", lights_on=True, placement="same_lane", distance=30)
    assert not gt.label_sample(s, 1, 2, gt.MIN_VISIBLE_PIXELS - 1, None, 30.0)["hip_present"]
    assert gt.label_sample(s, 1, 2, gt.MIN_VISIBLE_PIXELS, [0, 0, 1, 1], 30.0)["hip_present"]


# ---------------------------------------------------------------- grid

def test_grid_expansion_is_deterministic_and_complete():
    g = GridSpec(weathers=["ClearNoon"], hip_types=["police"], lights=[True, False],
                 placements=["same_lane"], distances=[15, 30], negative_fraction=0.5, seed=3)
    a, b = g.expand(), g.expand()
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    assert len(a) == 4 + 2
    assert sum(s.hip_type == "none" for s in a) == 2
    assert len({s.sample_id for s in a}) == len(a)


# ---------------------------------------------------------------- metrics

def test_binary_metrics_counts_unparsed_as_wrong():
    m = binary_metrics([(True, True), (True, None), (False, False), (False, None)])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1)


def test_auc_and_wilson():
    assert auc([(0.9, True), (0.1, False)]) == 1.0
    assert auc([(0.5, True), (0.5, False)]) == 0.5
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


# ---------------------------------------------------------------- baselines

def _scene(light_on: bool):
    f = np.full((60, 80, 3), 90, dtype=np.uint8)
    if light_on:
        f[20:26, 30:36] = (255, 20, 20)  # bright red strobe
    return f


def test_flicker_baseline_prefers_flashing_over_static_light():
    flashing = [_scene(i % 2 == 0) for i in range(4)]
    static = [_scene(True) for _ in range(4)]
    dark = [_scene(False) for _ in range(4)]
    assert baselines.flicker_score(flashing) > baselines.flicker_score(static)
    assert baselines.flicker_score(static) == baselines.flicker_score(dark) == 0.0
    # The single-frame colour score cannot tell a static red light from a strobe.
    assert baselines.color_score(static) == baselines.color_score(flashing) > 0
