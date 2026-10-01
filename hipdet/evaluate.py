"""Metrics for HIP predictions against simulator ground truth."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def load_predictions(path) -> dict:
    preds = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            preds[p["sample_id"]] = p
    return preds


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binary_metrics(pairs):
    """pairs: iterable of (truth: bool, pred: bool|None). None counts as a wrong answer."""
    tp = fp = tn = fn = 0
    for t, p in pairs:
        if p is None:
            p = not t
        if t and p:
            tp += 1
        elif t and not p:
            fn += 1
        elif not t and p:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and (prec + rec) else 0.0 if n else float("nan")
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall_ci": wilson(tp, tp + fn), "fpr_ci": wilson(fp, fp + tn),
    }


def auc(scores_truth):
    """ROC AUC via the rank-sum formulation (ties count half)."""
    pos = [s for s, t in scores_truth if t]
    neg = [s for s, t in scores_truth if not t]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def distance_bin(d):
    if d is None:
        return "n/a"
    return "<=20m" if d <= 20 else "20-40m" if d <= 40 else "40-65m" if d <= 65 else ">65m"


GROUPERS = {
    "weather": lambda s: s.record["scenario"]["weather"],
    "hip_type": lambda s: s.record["scenario"]["hip_type"],
    "lights_on": lambda s: str(s.record["scenario"]["lights_on"]),
    "placement": lambda s: s.record["scenario"]["placement"],
    "distance": lambda s: distance_bin(s.gt["target_distance_m"]),
    "visible": lambda s: str(s.gt["target_visible"]),
}


def evaluate(samples, preds: dict) -> dict:
    """Overall and per-factor metrics for one prediction set."""
    rows = [(s, preds.get(s.sample_id)) for s in samples]
    missing = sum(1 for _, p in rows if p is None)
    rows = [(s, p) for s, p in rows if p is not None]

    def hip_pairs(rs):
        return [(s.gt["hip_present"], p.get("hip_present")) for s, p in rs]

    out = {
        "n": len(rows),
        "missing": missing,
        "parse_failures": sum(1 for _, p in rows if not p.get("parse_ok", True)),
        "hip": binary_metrics(hip_pairs(rows)),
        "response": binary_metrics([(s.gt["response_required"], p.get("car_response")) for s, p in rows]),
    }
    scored = [(p["confidence"], s.gt["hip_present"]) for s, p in rows if p.get("confidence") is not None]
    out["hip"]["auc"] = auc(scored) if len(scored) == len(rows) else float("nan")

    lane_rows = [(s, p) for s, p in rows if p.get("ego_lane") is not None]
    out["lane"] = {
        "ego_lane_acc": (sum(1 for s, p in lane_rows if p["ego_lane"] == s.gt["ego_lane"]) / len(rows))
        if rows else float("nan"),
        "n_lanes_acc": (sum(1 for s, p in lane_rows if p.get("n_lanes") == s.gt["n_lanes"]) / len(rows))
        if rows else float("nan"),
    }
    lat = [p["latency_s"] for _, p in rows if p.get("latency_s") is not None]
    out["latency_s_mean"] = sum(lat) / len(lat) if lat else None

    by = {}
    for name, key in GROUPERS.items():
        groups = defaultdict(list)
        for s, p in rows:
            groups[key(s)].append((s, p))
        by[name] = {g: {"hip": binary_metrics(hip_pairs(rs)),
                        "response": binary_metrics([(s.gt["response_required"], p.get("car_response"))
                                                    for s, p in rs])}
                    for g, rs in sorted(groups.items())}
    out["by"] = by

    # Recall restricted to HIPs, split by factor: the most informative view.
    pos = [(s, p) for s, p in rows if s.gt["hip_present"]]
    out["recall_by"] = {}
    for name, key in GROUPERS.items():
        groups = defaultdict(list)
        for s, p in pos:
            groups[key(s)].append(bool(p.get("hip_present")))
        out["recall_by"][name] = {g: {"n": len(v), "recall": sum(v) / len(v), "ci": wilson(sum(v), len(v))}
                                  for g, v in sorted(groups.items())}
    # False-positive rate on hard negatives (lit-off emergency vehicles / plain cars).
    neg = [(s, p) for s, p in rows if not s.gt["hip_present"]]
    groups = defaultdict(list)
    for s, p in neg:
        sc = s.record["scenario"]
        kind = "no_target" if sc["hip_type"] == "none" else \
            "not_visible" if not s.gt["target_visible"] else f"{sc['hip_type']}_lights_off"
        groups[kind].append(bool(p.get("hip_present")))
    out["fpr_by_negative"] = {g: {"n": len(v), "fpr": sum(v) / len(v), "ci": wilson(sum(v), len(v))}
                              for g, v in sorted(groups.items())}
    return out
