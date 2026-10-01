"""Run a list of scenarios in CARLA and write a labelled dataset.

Dataset layout::

    <out>/
      manifest.json        collection settings, CARLA version, camera spec
      samples.jsonl        one JSON object per sample: scenario + ground truth
      frames/<id>_<k>.png  burst of K consecutive RGB frames per sample

Each sample is a short burst of frames rather than one still, because flashing
lights (the defining property of a HIP) are a temporal signal.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

import carla

from . import carla_utils as cu
from . import ground_truth as gt
from . import scenarios as sc

NIGHT_WEATHERS = {"ClearNight", "CloudyNight", "WetNight", "WetCloudyNight",
                  "SoftRainNight", "MidRainyNight", "HardRainNight"}


@dataclass
class CollectSettings:
    fixed_delta: float = 0.1     # simulated seconds per tick
    warmup_ticks: int = 3        # let vehicles settle and lights switch on
    burst_frames: int = 4        # frames saved per sample
    burst_stride: int = 2        # ticks between saved frames
    camera: cu.CameraSpec = field(default_factory=cu.CameraSpec)


def _light_state(hip_type: str, lit: bool, night: bool):
    L = carla.VehicleLightState
    state = L.Position | L.LowBeam if night else L.NONE
    if lit and hip_type in sc.EMERGENCY_TYPES:
        state |= L.Special1
    if lit and hip_type == "hazard_car":
        state |= L.LeftBlinker | L.RightBlinker
    return carla.VehicleLightState(state)


def _spawn_tf(wp):
    tf = wp.transform
    tf.location.z += 0.3
    return tf


def _hold(vehicle, light_state):
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    vehicle.set_light_state(light_state)


def run_sample(client, world, scenario: sc.Scenario, settings: CollectSettings, frames_dir: Path):
    """Spawn, capture and label one scenario. Returns the record dict, or None if it could not be placed."""
    rng = random.Random(scenario.seed)
    world_map = world.get_map()
    night = scenario.weather in NIGHT_WEATHERS
    pool = cu.ActorPool(client, world)
    try:
        ego = target = None
        for _attempt in range(6):
            layout = sc.find_layout(world_map, scenario, rng)
            if layout is None:
                return None
            ego_wp, target_wp = layout
            ego = pool.spawn_vehicle(cu.BLUEPRINTS["ego"], _spawn_tf(ego_wp), role_name="hero")
            if ego is None:
                continue
            if target_wp is None:
                break
            target = pool.spawn_vehicle(cu.BLUEPRINTS[scenario.hip_type], _spawn_tf(target_wp))
            if target is not None:
                break
            pool.destroy()
            ego = None
        if ego is None or (scenario.placement != "none" and target is None):
            return None

        _hold(ego, _light_state("none", False, night))
        if target is not None:
            _hold(target, _light_state(scenario.hip_type, scenario.lights_on, night))
        distractors = []
        for wp in sc.distractor_waypoints(ego_wp, target_wp, scenario.n_distractors, rng):
            v = pool.spawn_vehicle(rng.choice(cu.DISTRACTOR_BLUEPRINTS), _spawn_tf(wp))
            if v is not None:
                _hold(v, _light_state("none", False, night))
                distractors.append(v.type_id)

        for _ in range(settings.warmup_ticks):
            world.tick()
        rig = cu.CameraRig(pool, ego, settings.camera)
        world.tick()  # first frame after spawning sensors is sometimes black
        rig.drain()

        frame_paths, label = [], None
        for k in range(settings.burst_frames):
            for _ in range(settings.burst_stride - 1):
                f = world.tick()
                rig.get(f)
            f = world.tick()
            rgb, inst = rig.get(f)
            path = frames_dir / f"{scenario.sample_id}_{k}.png"
            Image.fromarray(cu.to_rgb_array(rgb)).save(path)
            frame_paths.append(path.name)
            if k == 0:
                _, ids = cu.to_instance_arrays(inst)
                pixels, bbox = (0, None)
                true_dist = None
                if target is not None:
                    pixels, bbox = gt.instance_stats(ids, [target.id])
                    true_dist = ego.get_location().distance(target.get_location())
                lane, n_lanes = sc.lane_layout(world_map.get_waypoint(ego.get_location()))
                label = gt.label_sample(scenario, lane, n_lanes, pixels, bbox, true_dist)
        return {
            "sample_id": scenario.sample_id,
            "scenario": scenario.to_dict(),
            "night": night,
            "distractors": distractors,
            "frames": frame_paths,
            "gt": label,
        }
    finally:
        pool.destroy()
        world.tick()


def collect(scenarios, out_dir, settings: CollectSettings | None = None,
            host: str = "localhost", port: int = 2000, log=print):
    settings = settings or CollectSettings()
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    client = cu.connect(host, port)
    manifest = {
        "carla_server": client.get_server_version(),
        "carla_client": client.get_client_version(),
        "settings": asdict(settings),
        "n_scenarios": len(scenarios),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    samples_path = out / "samples.jsonl"
    done = set()
    if samples_path.exists():  # resume an interrupted collection
        done = {json.loads(l)["sample_id"] for l in samples_path.read_text().splitlines() if l.strip()}
    skipped_path = out / "skipped.txt"

    current_town, current_weather = None, None
    world = client.get_world()
    t0 = time.time()
    n_new = 0
    for i, s in enumerate(scenarios):
        if s.sample_id in done:
            continue
        if s.town != current_town:
            world = cu.load_town(client, s.town)
            current_town, current_weather = s.town, None
        if s.weather != current_weather:
            world.set_weather(getattr(carla.WeatherParameters, s.weather))
            current_weather = s.weather
        with cu.synchronous_mode(client, world, settings.fixed_delta):
            rec = run_sample(client, world, s, settings, out / "frames")
        if rec is None:
            with skipped_path.open("a") as f:
                f.write(s.sample_id + "\n")
            log(f"[{i + 1}/{len(scenarios)}] {s.sample_id} skipped (no valid placement)")
            continue
        with samples_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        n_new += 1
        rate = (time.time() - t0) / n_new
        g = rec["gt"]
        log(f"[{i + 1}/{len(scenarios)}] {s.sample_id} {s.weather} {s.hip_type} lit={s.lights_on} "
            f"{s.placement} {s.distance:.0f}m -> visible={g['target_visible']} px={g['target_pixels']} "
            f"hip={g['hip_present']} resp={g['response_required']} ({rate:.1f}s/sample)")
    return out
