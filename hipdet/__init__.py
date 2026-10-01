"""HIP (High Illumination Priority) detection benchmark built on CARLA.

Package layout:
    carla_utils   connection, synchronous mode, actor bookkeeping, image conversion
    scenarios     scenario specs and placement of HIP actors relative to the ego car
    ground_truth  per-sample labels computed from simulator state
    collect       runs a grid of scenarios and writes a dataset to disk
    dataset       reading datasets back
    vlm           prompt, output schema, parsing and model backends
    baselines     non-learned detectors used as reference points
    evaluate      metrics and breakdowns
"""

__version__ = "0.2.0"
