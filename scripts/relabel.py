#!/usr/bin/env python3
"""Recompute ground-truth labels of an existing dataset with the current rules.

The raw measurements (visible pixels, bbox, distance, lane layout) are kept;
only the derived labels are recomputed. The old file is kept as samples.jsonl.bak.

    python scripts/relabel.py --data data/benchmark
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import ground_truth as gt  # noqa: E402
from hipdet.scenarios import Scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    path = Path(args.data) / "samples.jsonl"
    shutil.copy(path, path.with_suffix(".jsonl.bak"))
    out, changed = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        old = rec["gt"]
        new = gt.label_sample(Scenario(**rec["scenario"]), old["ego_lane"], old["n_lanes"],
                              old["target_pixels"], old["target_bbox"], old["target_distance_m"])
        changed += new != old
        rec["gt"] = new
        out.append(json.dumps(rec))
    path.write_text("\n".join(out) + "\n")
    print(f"relabelled {len(out)} samples, {changed} changed")


if __name__ == "__main__":
    main()
