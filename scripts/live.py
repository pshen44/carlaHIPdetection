#!/usr/bin/env python3
"""Closed-loop-style live demo: the ego car drives on autopilot while a detector
watches a rolling window of camera frames.

Replaces legacy/rgb_sensor_collect_analyze_looping.py. Differences:
* model calls run on a worker thread, so the simulator never blocks on the API
  (the old script called the API inside the sensor callback);
* emergency vehicles are spawned on the ego car's route instead of at random
  map points, so they are actually encountered;
* every decision is logged to JSONL together with the simulator ground truth
  for that moment.

    python scripts/live.py --detector flicker --seconds 60
    python scripts/live.py --detector anthropic:claude-opus-5-5 --window 4 --every 6
"""

import argparse
import json
import queue
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import carla  # noqa: E402

from hipdet import carla_utils as cu  # noqa: E402
from hipdet import ground_truth as gt  # noqa: E402
from hipdet import scenarios as sc  # noqa: E402
from hipdet.baselines import flicker_score  # noqa: E402


class FlickerDetector:
    name = "baseline:flicker"

    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def predict(self, frames, dt):
        s = flicker_score(frames)
        return {"hip_present": s >= self.threshold, "score": s, "car_response": None}


def make_detector(spec):
    if spec == "flicker":
        return FlickerDetector()
    from hipdet.vlm.backends import make_backend
    return make_backend(spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="flicker")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--window", type=int, default=4, help="frames per detector call")
    ap.add_argument("--every", type=int, default=2, help="ticks between captured frames")
    ap.add_argument("--n-emergency", type=int, default=3)
    ap.add_argument("--weather", default="ClearNight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="results/live_log.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    detector = make_detector(args.detector)
    client = cu.connect()
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    dt = 0.1
    night = "Night" in args.weather
    jobs: queue.Queue = queue.Queue(maxsize=1)
    results: queue.Queue = queue.Queue()

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            frame_id, frames, truth = job
            t = time.time()
            try:
                pred = detector.predict(frames, dt * args.every)
            except Exception as e:
                pred = {"error": repr(e)}
            results.put({"frame": frame_id, "latency_s": round(time.time() - t, 3), "truth": truth,
                         **{k: v for k, v in pred.items() if k != "raw"}})

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with cu.synchronous_mode(client, world, dt) as tm:
        tm.set_random_device_seed(args.seed)
        pool = cu.ActorPool(client, world)
        try:
            spawns = world.get_map().get_spawn_points()
            rng.shuffle(spawns)
            ego = None
            for sp in spawns:
                ego = pool.spawn_vehicle(cu.BLUEPRINTS["ego"], sp, role_name="hero")
                if ego:
                    break
            ego.set_autopilot(True, tm.get_port())
            L = carla.VehicleLightState
            base = L.Position | L.LowBeam if night else L.NONE
            ego.set_light_state(carla.VehicleLightState(base))
            # Emergency vehicles on the ego car's road, ahead of it, driving on autopilot.
            # Use the spawn transform: in synchronous mode a new actor reports (0, 0, 0) until the first tick.
            ego_wp = world.get_map().get_waypoint(sp.location)
            hips = []
            for k in range(args.n_emergency):
                for d in (25 + 30 * k, 35 + 30 * k, 45 + 30 * k):
                    nxt = ego_wp.next(d)
                    if not nxt:
                        continue
                    wp = rng.choice([nxt[0]] + sc._adjacent_lanes(nxt[0]) + [w for w in [sc._oncoming_lane(nxt[0])] if w])
                    tf = wp.transform
                    tf.location.z += 0.3
                    kind = rng.choice(list(sc.EMERGENCY_TYPES))
                    v = pool.spawn_vehicle(cu.BLUEPRINTS[kind], tf)
                    if v:
                        v.set_light_state(carla.VehicleLightState(base | L.Special1))
                        v.set_autopilot(True, tm.get_port())
                        tm.vehicle_percentage_speed_difference(v, 30.0)  # slower than ego, so it is caught up with
                        hips.append(v)
                        break
            print(f"spawned {len(hips)} emergency vehicles ahead of the ego car", flush=True)
            rig = cu.CameraRig(pool, ego, cu.CameraSpec(width=640, height=480))
            window = deque(maxlen=args.window)
            n_ticks = int(args.seconds / dt)
            with log_path.open("a") as log:
                for tick in range(n_ticks):
                    f = world.tick()
                    rgb, inst = rig.get(f)
                    if tick % args.every:
                        continue
                    window.append(cu.to_rgb_array(rgb))
                    if len(window) == args.window and jobs.empty():
                        _, ids = cu.to_instance_arrays(inst)
                        visible = [h for h in hips if gt.instance_stats(ids, [h.id])[0] >= gt.MIN_VISIBLE_PIXELS]
                        truth = {"hip_visible": bool(visible),
                                 "nearest_m": min((ego.get_location().distance(h.get_location()) for h in visible),
                                                  default=None)}
                        jobs.put((f, list(window), truth))
                    while not results.empty():
                        r = results.get()
                        log.write(json.dumps(r) + "\n")
                        print(f"t={tick * dt:5.1f}s frame {r['frame']}: pred={r.get('hip_present')} "
                              f"truth={r['truth']['hip_visible']} ({r['latency_s']}s)", flush=True)
        finally:
            jobs.put(None)
            for v in pool.vehicles:
                v.set_autopilot(False)
            pool.destroy()


if __name__ == "__main__":
    main()
