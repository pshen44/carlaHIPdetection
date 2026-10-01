#!/usr/bin/env python3
"""Collect a labelled HIP dataset from a running CARLA server.

    python scripts/collect.py --config configs/benchmark.yaml --out data/benchmark

Re-running the same command resumes an interrupted collection.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet.carla_utils import CameraSpec  # noqa: E402
from hipdet.collect import CollectSettings, collect  # noqa: E402
from hipdet.scenarios import GridSpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N scenarios")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    grid = GridSpec(**cfg.get("grid", {}))
    coll = dict(cfg.get("collect", {}))
    camera = CameraSpec(**coll.pop("camera", {}))
    settings = CollectSettings(camera=camera, **coll)

    scenarios = grid.expand()
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"{len(scenarios)} scenarios -> {args.out}", flush=True)
    collect(scenarios, args.out, settings, host=args.host, port=args.port,
            log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
