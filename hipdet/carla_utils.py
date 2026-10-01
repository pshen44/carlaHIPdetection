"""Thin helpers around the CARLA Python API.

Everything that touches the simulator lives here or in ``scenarios`` /
``collect`` so the analysis side of the package can be used without CARLA.
"""

from __future__ import annotations

import contextlib
import queue
from dataclasses import dataclass

import numpy as np

import carla

# Exact blueprint ids. ``blueprint_library.filter('dodge')[0]`` (used by the old
# scripts) can return the civilian Charger, so we always ask for an exact id.
BLUEPRINTS = {
    "ego": "vehicle.mercedes.coupe_2020",
    "ambulance": "vehicle.ford.ambulance",
    "police": "vehicle.dodge.charger_police_2020",
    "firetruck": "vehicle.carlamotors.firetruck",
    "hazard_car": "vehicle.lincoln.mkz_2020",
}

# Civilian cars used as distractor traffic.
DISTRACTOR_BLUEPRINTS = [
    "vehicle.audi.a2",
    "vehicle.tesla.model3",
    "vehicle.toyota.prius",
    "vehicle.nissan.patrol_2021",
    "vehicle.mini.cooper_s_2021",
    "vehicle.ford.mustang",
]


def connect(host: str = "localhost", port: int = 2000, timeout: float = 60.0):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def load_town(client, town: str):
    """Load ``town`` unless it is already the current map."""
    world = client.get_world()
    name = world.get_map().name.split("/")[-1]
    if name not in (town, f"{town}_Opt"):
        world = client.load_world(town)
    return world


@contextlib.contextmanager
def synchronous_mode(client, world, fixed_delta: float = 0.05, tm_port: int = 8000):
    """Run the world in synchronous mode with a fixed time step.

    In asynchronous mode (what the old scripts used) the number of frames per
    wall-clock second depends on the machine, so "every 60th frame" meant a
    different thing on every run. Synchronous mode makes runs reproducible.
    """
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    # Physics substepping requires fixed_delta <= max_substep_delta_time * max_substeps.
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = min(16, max(10, int(-(-fixed_delta // 0.01))))
    world.apply_settings(settings)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    try:
        yield tm
    finally:
        tm.set_synchronous_mode(False)
        world.apply_settings(original)


class ActorPool:
    """Tracks spawned actors and destroys them (sensors first) on cleanup."""

    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.sensors: list = []
        self.vehicles: list = []

    def spawn_vehicle(self, blueprint_id: str, transform, role_name: str = ""):
        bp = self.world.get_blueprint_library().find(blueprint_id)
        if role_name and bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role_name)
        if bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is not None:
            self.vehicles.append(actor)
        return actor

    def spawn_sensor(self, blueprint, transform, parent):
        actor = self.world.spawn_actor(blueprint, transform, attach_to=parent)
        self.sensors.append(actor)
        return actor

    def destroy(self):
        for s in self.sensors:
            with contextlib.suppress(RuntimeError):
                s.stop()
        ids = [a.id for a in self.sensors + self.vehicles]
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        self.sensors.clear()
        self.vehicles.clear()


@dataclass
class CameraSpec:
    width: int = 800
    height: int = 600
    fov: float = 90.0
    # Roughly the windshield / roof-front position of a sedan.
    x: float = 1.0
    z: float = 1.6


class CameraRig:
    """Forward RGB camera plus a pixel-aligned instance-segmentation camera.

    The instance camera is only used to compute ground truth (which actors are
    actually visible and where); models never see it.
    """

    def __init__(self, pool: ActorPool, parent, spec: CameraSpec):
        self.spec = spec
        lib = pool.world.get_blueprint_library()
        tf = carla.Transform(carla.Location(x=spec.x, z=spec.z))
        self.queues = {}
        self.sensors = {}
        for name, bp_id in (("rgb", "sensor.camera.rgb"),
                            ("inst", "sensor.camera.instance_segmentation")):
            bp = lib.find(bp_id)
            bp.set_attribute("image_size_x", str(spec.width))
            bp.set_attribute("image_size_y", str(spec.height))
            bp.set_attribute("fov", str(spec.fov))
            sensor = pool.spawn_sensor(bp, tf, parent)
            q: queue.Queue = queue.Queue()
            sensor.listen(q.put)
            self.queues[name] = q
            self.sensors[name] = sensor

    def remove(self, name: str, pool: "ActorPool"):
        """Stop and destroy one camera (e.g. the instance camera once ground truth is taken)."""
        sensor = self.sensors.pop(name)
        self.queues.pop(name)
        sensor.stop()
        pool.sensors.remove(sensor)
        sensor.destroy()

    def get(self, frame: int, timeout: float = 120.0):
        """Return the (rgb, inst) images for simulator ``frame`` (inst is None once removed)."""
        out = {"inst": None}
        for name, q in self.queues.items():
            while True:
                img = q.get(timeout=timeout)
                if img.frame == frame:
                    out[name] = img
                    break
                if img.frame > frame:
                    raise RuntimeError(f"{name} camera skipped frame {frame} (got {img.frame})")
        return out["rgb"], out["inst"]

    def drain(self):
        for q in self.queues.values():
            while not q.empty():
                q.get_nowait()


def to_rgb_array(image) -> np.ndarray:
    """Convert a ``carla.Image`` to an HxWx3 uint8 array in **RGB** order.

    CARLA's ``raw_data`` is BGRA. The old scripts sliced ``[:, :, :3]`` and
    handed the result to PIL as RGB, which swapped red and blue in every
    image sent to the model.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, 2::-1].copy()


def to_instance_arrays(image):
    """Decode an instance-segmentation image into (semantic_tag, instance_id) arrays."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b = arr[:, :, 0].astype(np.int32)
    g = arr[:, :, 1].astype(np.int32)
    r = arr[:, :, 2]
    return r, g + (b << 8)
