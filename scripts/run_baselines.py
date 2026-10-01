#!/usr/bin/env python3
"""Score every sample with the classical baselines and write predictions.

The decision threshold for each baseline is chosen to maximise balanced accuracy
(Youden's J = TPR - FPR) on the
calibration half of the data (even sample index) and applied to all samples;
evaluate.py reports metrics on the test half (odd index) for baselines.

    python scripts/run_baselines.py --data data/benchmark
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import dataset  # noqa: E402
from hipdet.baselines import BASELINES  # noqa: E402
from hipdet.evaluate import binary_metrics  # noqa: E402


def is_calibration(sample_id: str) -> bool:
    return int(sample_id[1:]) % 2 == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results/predictions")
    args = ap.parse_args()

    samples = dataset.load(args.data)
    scores = {name: {} for name in BASELINES}
    for i, s in enumerate(samples):
        frames = s.frames()
        for name, fn in BASELINES.items():
            scores[name][s.sample_id] = fn(frames)
        if (i + 1) % 50 == 0:
            print(f"scored {i + 1}/{len(samples)}", flush=True)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for name, sc in scores.items():
        cal = [(sc[s.sample_id], s.gt["hip_present"]) for s in samples if is_calibration(s.sample_id)]
        # Maximising F1 picks "everything is a HIP" when the score is weak, so use Youden's J.
        best_t, best_j = 0.0, -2.0
        for t in sorted({v for v, _ in cal}):
            m = binary_metrics([(truth, v >= t) for v, truth in cal])
            j = m["recall"] - m["fpr"]
            if j > best_j:
                best_t, best_j = t, j
        top = max(sc.values()) or 1.0
        with open(Path(args.outdir) / f"baseline_{name}.jsonl", "w") as f:
            for s in samples:
                v = sc[s.sample_id]
                f.write(json.dumps({
                    "sample_id": s.sample_id, "backend": f"baseline:{name}", "mode": name.split("_")[-1],
                    "parse_ok": True, "hip_present": v >= best_t, "car_response": None,
                    "confidence": v / top, "score": v, "threshold": best_t,
                    "split": "cal" if is_calibration(s.sample_id) else "test",
                }) + "\n")
        print(f"{name}: threshold={best_t:.3f} (calibration Youden J={best_j:.3f})")


if __name__ == "__main__":
    main()
