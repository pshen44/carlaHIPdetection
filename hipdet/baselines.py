"""Hand-written colour/flicker detectors used as reference points for the VLMs.

Neither uses any learned model. They answer "is there a HIP?" with a score;
a threshold is picked on a calibration split.

* ``color_score``   single frame: amount of bright, saturated red/blue light.
* ``flicker_score`` burst: amount of bright, saturated red/blue/amber light
                    whose brightness changes between frames. Static lights
                    (tail lights, traffic lights, street lamps) are ignored,
                    flashing ones (emergency strobes, hazard blinkers) are not.
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV hue is 0..179.
_RED = ((0, 10), (170, 180))
_AMBER = ((11, 28),)
_BLUE = ((95, 135),)


def _hue_mask(h, ranges):
    m = np.zeros(h.shape, dtype=bool)
    for lo, hi in ranges:
        m |= (h >= lo) & (h < hi)
    return m


def _light_mask(frame: np.ndarray, ranges, min_v=170, min_s=110):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return _hue_mask(h, ranges) & (s >= min_s) & (v >= min_v)


# Parameters below were chosen on the calibration half of the first 100 samples.
# The blur matters: CARLA's low-quality software rendering adds per-pixel temporal
# noise (mean |frame difference| ~40 grey levels on a static scene), which a
# per-pixel flicker test mistakes for flashing lights.
BLUR = 5
MIN_S, MIN_V = 160, 150


def _smooth(frame):
    return cv2.GaussianBlur(frame, (BLUR, BLUR), 0)


def color_score(frames) -> float:
    """Single-frame score: log pixel count of bright red/blue light in the first frame."""
    m = _light_mask(_smooth(frames[0]), _RED + _BLUE, min_v=MIN_V, min_s=MIN_S)
    return float(np.log1p(m.sum()))


def flicker_score(frames, min_delta=80) -> float:
    """Burst score: log pixel count of coloured light whose brightness changes across the burst."""
    if len(frames) < 2:
        return color_score(frames)
    frames = [_smooth(f) for f in frames]
    vals = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 2].astype(np.int16) for f in frames])
    delta = vals.max(0) - vals.min(0)
    colored = np.zeros(vals.shape[1:], dtype=bool)
    for f in frames:
        colored |= _light_mask(f, _RED + _BLUE + _AMBER, min_v=MIN_V, min_s=MIN_S)
    flick = colored & (delta >= min_delta)
    return float(np.log1p(flick.sum()))


BASELINES = {"color_single": color_score, "flicker_burst": flicker_score}
