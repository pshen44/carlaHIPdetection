#!/usr/bin/env python3
"""Run a VLM backend over a dataset and write predictions (resumable).

    python scripts/run_model.py --data data/benchmark --backend openai:gpt-4.1 --mode single
    python scripts/run_model.py --data data/benchmark --backend anthropic:claude-opus-5-5 --mode burst

--mode single sends only the first frame of each burst; --mode burst sends all of them.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import dataset  # noqa: E402
from hipdet.vlm.backends import make_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--mode", choices=["single", "burst"], default="single")
    ap.add_argument("--out", default=None, help="predictions .jsonl (default: results/predictions/...)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    samples = dataset.load(args.data)[: args.limit]
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    dt = manifest["settings"]["fixed_delta"] * manifest["settings"]["burst_stride"]
    backend = make_backend(args.backend)
    out = Path(args.out or f"results/predictions/{backend.name.replace(':', '_')}__{args.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in out.read_text().splitlines() if l.strip()}

    for i, s in enumerate(samples):
        if s.sample_id in done:
            continue
        frames = s.frames()
        if args.mode == "single":
            frames = frames[:1]
        try:
            pred = backend.predict(frames, dt=dt)
        except Exception as e:  # keep going; the failure is recorded and scored as wrong
            pred = {"parse_ok": False, "error": repr(e), "hip_present": None, "car_response": None}
        pred.update({"sample_id": s.sample_id, "backend": backend.name, "mode": args.mode})
        with out.open("a") as f:
            f.write(json.dumps(pred) + "\n")
        print(f"[{i + 1}/{len(samples)}] {s.sample_id} gt={s.gt['hip_present']} "
              f"pred={pred.get('hip_present')} ({pred.get('latency_s')}s)", flush=True)


if __name__ == "__main__":
    main()
