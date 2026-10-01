#!/usr/bin/env python3
"""Export blinded labelling batches and import the answers as predictions.

Used when a model is evaluated through an interface other than an API
(e.g. Claude Code sub-agents, or human labellers). Frames are copied under
random names so the labeller cannot see sample ids or scenario metadata, and
the batch instructions contain the same prompt the API backends use.

    # 1. write batches of 30 items
    python scripts/blind_batches.py export --data data/benchmark --mode burst \\
        --out blind/burst --batch-size 30 [--swap-rb]
    # 2. have each labeller follow blind/burst/batch_XX/INSTRUCTIONS.md and write answers.jsonl
    # 3. collect the answers as a normal prediction file
    python scripts/blind_batches.py import --blind blind/burst --name claude-opus__burst

The key that maps blinded names back to samples is written to <out>/../<name>.key.json,
outside the batch folders.
"""

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import dataset  # noqa: E402
from hipdet.vlm.parse import parse_prediction  # noqa: E402
from hipdet.vlm.prompt import SYSTEM_PROMPT, user_text  # noqa: E402

INSTRUCTIONS = """# Labelling batch

{system_prompt}

## Task

This folder contains {n} items. {per_item}

For each item ID below, look at its image file(s) and judge that item on its own.
Do not open any file outside this folder.

Write your answers to `{answers}`, one JSON object per line, in this form:

{{"item": "<item id>", "hip_present": ..., "hips": [...], "car_response": ..., "ego_lane": ..., "n_lanes": ..., "confidence": ...}}

Items:
{items}
"""


def cmd_export(args):
    samples = dataset.load(args.data)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    dt = manifest["settings"]["fixed_delta"] * manifest["settings"]["burst_stride"]
    if args.ids:
        wanted = set(Path(args.ids).read_text().split())
        samples = [s for s in samples if s.sample_id in wanted]
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    n_frames = 1 if args.mode == "single" else None
    for b in range(0, len(samples), args.batch_size):
        bdir = out / f"batch_{b // args.batch_size:02d}"
        bdir.mkdir(exist_ok=True)
        items = []
        for s in samples[b:b + args.batch_size]:
            item = secrets.token_hex(4)
            frames = s.frame_paths()[: n_frames or None]
            names = []
            for k, p in enumerate(frames):
                arr = np.asarray(Image.open(p).convert("RGB"))
                if args.swap_rb:
                    arr = arr[:, :, ::-1]
                name = f"{item}.png" if len(frames) == 1 else f"{item}_t{k}.png"
                Image.fromarray(arr).save(bdir / name)
                names.append(name)
            key[item] = s.sample_id
            items.append(f"- `{item}`: " + ", ".join(f"`{bdir / n}`" for n in names))
        per_item = (user_text(1, dt) + " Each item is one image.") if args.mode == "single" else \
            (user_text(len(frames), dt) + " Each item is a set of files `<item>_t0.png` ... in time order.")
        (bdir / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(
            system_prompt=SYSTEM_PROMPT, n=len(items), per_item=per_item,
            answers=bdir / "answers.jsonl", items="\n".join(items)))
    (out.parent / f"{out.name}.key.json").write_text(json.dumps(
        {"mode": args.mode, "swap_rb": args.swap_rb, "key": key}, indent=0))
    print(f"wrote {len(key)} items in {(len(key) + args.batch_size - 1) // args.batch_size} batches to {out}")


def cmd_import(args):
    out = Path(args.outdir) / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    key, seen, answer_files, mode = {}, {}, [], None
    for b in args.blind:
        blind = Path(b)
        meta = json.loads((blind.parent / f"{blind.name}.key.json").read_text())
        key.update(meta["key"])
        mode = meta["mode"]
        answer_files += sorted(blind.glob("batch_*/answers.jsonl"))
    for ans in answer_files:
        for line in ans.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line).get("item")
            except json.JSONDecodeError:
                continue
            if item in key:
                pred = parse_prediction(line)
                pred.update({"sample_id": key[item], "backend": args.name, "mode": mode})
                seen[item] = pred
    missing = [i for i in key if i not in seen]
    with out.open("w") as f:
        for pred in seen.values():
            f.write(json.dumps(pred) + "\n")
    print(f"imported {len(seen)} predictions to {out}; {len(missing)} items unanswered")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    e = sub.add_parser("export")
    e.add_argument("--data", required=True)
    e.add_argument("--mode", choices=["single", "burst"], default="single")
    e.add_argument("--out", required=True)
    e.add_argument("--batch-size", type=int, default=30)
    e.add_argument("--ids", help="file with sample ids to include (default: all)")
    e.add_argument("--swap-rb", action="store_true", help="reproduce the old red/blue swap bug")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser("import")
    i.add_argument("--blind", required=True, nargs="+", help="one or more exported batch folders")
    i.add_argument("--name", required=True)
    i.add_argument("--outdir", default="results/predictions")
    i.set_defaults(fn=cmd_import)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
