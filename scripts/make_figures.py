#!/usr/bin/env python3
"""Figures for the results README, from results/metrics.json.

    python scripts/make_figures.py --metrics results/metrics.json --out results/figures \\
        --compare claude-opus__single claude-opus__burst baseline_flicker_burst
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated reference palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

DIST_ORDER = ["<=20m", "20-40m", "40-65m", ">65m"]
WEATHER_ORDER = ["ClearNoon", "ClearSunset", "ClearNight", "HardRainNight"]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def pretty(name):
    return name.replace("__", " · ").replace("baseline_", "baseline: ").replace("_", " ")


def line_by_factor(metrics, names, factor, order, out, title, xlabel):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for i, n in enumerate(names):
        rb = metrics[n]["recall_by"][factor]
        xs = [g for g in order if g in rb]
        ys = [100 * rb[g]["recall"] for g in xs]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=pretty(n))
        ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    style(ax, title, "recall on visible HIPs (%)")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def grouped_bars(metrics, names, factor, order, out, title):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    groups = [g for g in order if any(g in metrics[n]["recall_by"][factor] for n in names)]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        rb = metrics[n]["recall_by"][factor]
        ys = [100 * rb[g]["recall"] if g in rb else 0 for g in groups]
        xs = [j + (i - (len(names) - 1) / 2) * w for j in range(len(groups))]
        ax.bar(xs, ys, width=w - 0.02, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=pretty(n))
    ax.set_xticks(range(len(groups)), groups)
    style(ax, title, "recall on visible HIPs (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, ncol=min(4, len(names)), loc="upper center",
              bbox_to_anchor=(0.5, -0.1), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def overview(metrics, out):
    names = sorted(metrics, key=lambda n: metrics[n]["hip"]["f1"])
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(names) + 1.2), facecolor=SURFACE)
    f1 = [100 * metrics[n]["hip"]["f1"] for n in names]
    ax.barh(range(len(names)), f1, color=SERIES[0], height=0.6)
    for i, v in enumerate(f1):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)), [pretty(n) for n in names], color=INK)
    style(ax, "HIP detection F1 by predictor", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 (%)", color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--compare", nargs="+", required=True, help="up to 4 predictors for the line/bar charts")
    args = ap.parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.compare if n in metrics][:4]
    overview(metrics, out / "f1_overview.png")
    line_by_factor(metrics, names, "distance", DIST_ORDER, out / "recall_by_distance.png",
                   "Recall falls with distance", "distance to HIP")
    grouped_bars(metrics, names, "weather", WEATHER_ORDER, out / "recall_by_weather.png",
                 "Recall by lighting / weather")
    grouped_bars(metrics, names, "hip_type", ["ambulance", "police", "firetruck", "hazard_car"],
                 out / "recall_by_type.png", "Recall by HIP type")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
