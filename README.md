# HIP detection benchmark in CARLA

Can a vision model tell when an autonomous car is looking at a **High Illumination Priority (HIP)**
object, such as an emergency vehicle with its lights on or a car with its hazards on, and whether
the car has to react?

This repository builds that question into a reproducible benchmark using the
[CARLA simulator](https://carla.org/). CARLA places a HIP at a controlled distance and lane relative to
the ego car, under controlled lighting and weather. It records a short burst of camera frames and labels
every sample automatically from simulator state. Any detector (an API vision-language model, a
human, or a classical baseline) is then scored against those labels.

<!-- RESULTS -->

## How it works

```
configs/*.yaml ──► scripts/collect.py ──► data/<run>/            (frames + samples.jsonl with ground truth)
                                            │
          scripts/run_model.py  (API VLMs) ─┤
          scripts/blind_batches.py (agents / humans, blinded)
          scripts/run_baselines.py (colour / flicker)
                                            ▼
                                  results/predictions/*.jsonl ──► scripts/evaluate.py ──► results/summary.md
                                                                   scripts/make_figures.py ──► results/figures/
```

**Scenarios** (`hipdet/scenarios.py`). The grid is a full factorial over weather/time of day, HIP type
(ambulance, police, fire truck, car with hazards), lights on/off, placement (same lane, adjacent
same-direction lane, oncoming lane) and distance, plus scenes with no HIP. Vehicles with their lights off
are hard negatives: an emergency vehicle with its lights off is *not* a HIP. Each scenario has its own
seed, so the dataset is reproducible.

**Capture** (`hipdet/collect.py`). The world runs in synchronous mode with a fixed time step. For each
sample the collector spawns the ego car, the HIP and distractor traffic, then saves a burst of 4 RGB frames
0.15 s apart. The burst matters because flashing is the defining property of a HIP, and a single still
cannot show it.

**Ground truth** (`hipdet/ground_truth.py`). A pixel-aligned instance-segmentation camera measures how
many pixels the HIP actually covers, so labels reflect what is *visible* rather than what was spawned. The
rules, which the prompt also states:

* A HIP is an emergency vehicle with its emergency lights on, or any vehicle with its hazard lights on,
  covering at least 40 pixels.
* The car must respond if the HIP is in its own lane or a same-direction lane, or if it is an
  emergency vehicle in oncoming traffic, within 80 m.
* Lanes are numbered from the left, counting only same-direction driving lanes.

**Models** (`hipdet/vlm/`). There is one prompt and one strict JSON output schema for all backends:
OpenAI (`openai:gpt-4.1`), Anthropic (`anthropic:claude-opus-5-5`), and a `swap_rb:` wrapper that reproduces
the old colour bug on purpose. `scripts/blind_batches.py` exports randomly renamed frames with the same
instructions for evaluation through any interface without API access, e.g. Claude Code sub-agents or human
labellers.

**Baselines** (`hipdet/baselines.py`). There are two non-learned detectors. The first scores bright,
saturated red/blue light in one frame. The second scores coloured light whose brightness *changes*
across the burst. Their thresholds are fit on half the data and evaluated on the other half.

## Running it

Requirements: a CARLA **0.9.16** server and Python 3.10–3.12.

```bash
pip install -r requirements.txt
./CarlaUE4.sh -RenderOffScreen            # in the CARLA folder; a GPU makes this ~50x faster

python scripts/collect.py --config configs/pilot.yaml --out data/pilot          # a few minutes
python scripts/collect.py --config configs/benchmark.yaml --out data/benchmark  # resumable

python scripts/run_baselines.py --data data/benchmark
python scripts/run_model.py --data data/benchmark --backend openai:gpt-4.1 --mode single
python scripts/run_model.py --data data/benchmark --backend openai:gpt-4.1 --mode burst
python scripts/evaluate.py --data data/benchmark
python scripts/make_figures.py --compare openai_gpt-4.1__single openai_gpt-4.1__burst baseline_flicker_burst

python scripts/live.py --detector flicker --seconds 60   # ego car drives; detector runs on a worker thread
python -m pytest tests                                    # no simulator needed
```

`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` must be set for the API backends.

## Repository layout

| path | what it is |
|---|---|
| `hipdet/` | the library: CARLA helpers, scenarios, ground truth, collection, VLM backends, baselines, metrics |
| `scripts/` | command-line entry points (collect, run models, evaluate, figures, live demo, map plot) |
| `configs/` | scenario grids (`pilot.yaml` small, `benchmark.yaml` main) |
| `results/` | predictions, metrics, summary tables and figures from the runs reported above |
| `tests/` | unit tests for everything that does not need the simulator |
| `legacy/` | the original 2025 scripts and `hip_log.csv`, kept for reference |

## What changed from the 2025 version

The original scripts (now in `legacy/`) spawned emergency vehicles at random map points and sent a few
seconds of frames to GPT-4o/4.1. A human then typed a score into `hip_log.csv`. Issues fixed along the way:

* **Red and blue were swapped in every image sent to the model.** CARLA delivers BGRA. The old code
  dropped alpha and handed BGR to PIL as RGB, so red ambulance lights reached the model as blue and blue
  police lights as red. The PNGs saved to disk were correct, so it was invisible when checking by eye.
* `filter('dodge')[0]` could return a civilian Dodge Charger instead of the police car. Blueprints are now
  exact ids.
* HIPs spawned at random map points were almost never in view, so `hip_log.csv` is ~95% empty frames.
* Asynchronous mode with "every 60th frame" made capture timing machine-dependent. Runs are now synchronous
  and seeded.
* The prompt contradicted itself on when to respond, and its examples were not valid JSON. Replies were
  regex-scraped. There is now one rule set, a strict schema, and tolerant parsing that counts failures
  instead of crashing.
* The hand-entered score is replaced by automatic ground truth and standard metrics with confidence
  intervals.
