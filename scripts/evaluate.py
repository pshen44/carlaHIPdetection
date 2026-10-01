#!/usr/bin/env python3
"""Evaluate every prediction file against ground truth and write a report.

    python scripts/evaluate.py --data data/benchmark --preds results/predictions --out results

Writes results/metrics.json, results/summary.md and figures in results/figures/.
Baseline prediction files are scored on their test split only (see run_baselines.py).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import dataset  # noqa: E402
from hipdet.evaluate import evaluate, load_predictions  # noqa: E402


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", default="results/predictions")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    samples = dataset.load(args.data)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    results = {}
    for path in sorted(Path(args.preds).glob("*.jsonl")):
        preds = load_predictions(path)
        subset = samples
        if path.stem.startswith("baseline_"):
            test_ids = {k for k, p in preds.items() if p.get("split") == "test"}
            subset = [s for s in samples if s.sample_id in test_ids]
        else:
            subset = [s for s in samples if s.sample_id in preds]
        if not subset:
            continue
        results[path.stem] = evaluate(subset, preds)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=str))

    n_pos = sum(s.gt["hip_present"] for s in samples)
    lines = [
        "# HIP benchmark results", "",
        f"Dataset: `{args.data}`: {len(samples)} samples, {n_pos} with a visible HIP, "
        f"{len(samples) - n_pos} without.", "",
        "## HIP detection", "",
        "| predictor | n | accuracy | precision | recall | F1 | FPR | AUC | response acc | response F1 "
        "| ego-lane acc | parse fail |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        h, resp = r["hip"], r["response"]
        lines.append(
            f"| {name} | {r['n']} | {fmt(h['accuracy'])} | {fmt(h['precision'])} | {fmt(h['recall'])} | "
            f"{fmt(h['f1'])} | {fmt(h['fpr'])} | {fmt(h.get('auc'), pct=False)} | "
            f"{fmt(resp['accuracy']) if 'baseline' not in name else '–'} | "
            f"{fmt(resp['f1']) if 'baseline' not in name else '–'} | "
            f"{fmt(r['lane']['ego_lane_acc']) if 'baseline' not in name else '–'} | {r['parse_failures']} |")
    for factor in ["weather", "distance", "hip_type", "placement"]:
        groups = sorted({g for r in results.values() for g in r["recall_by"][factor]})
        lines += ["", f"## Recall on visible HIPs by {factor}", "",
                  "| predictor | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for name, r in results.items():
            cells = []
            for g in groups:
                v = r["recall_by"][factor].get(g)
                cells.append(f"{fmt(v['recall'])} (n={v['n']})" if v else "–")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    groups = sorted({g for r in results.values() for g in r["fpr_by_negative"]})
    lines += ["", "## False-positive rate on negatives", "",
              "| predictor | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for name, r in results.items():
        cells = []
        for g in groups:
            v = r["fpr_by_negative"].get(g)
            cells.append(f"{fmt(v['fpr'])} (n={v['n']})" if v else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
