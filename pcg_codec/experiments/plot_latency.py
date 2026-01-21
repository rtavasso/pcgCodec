#!/usr/bin/env python3
"""Plot latency curves (TTFA vs depth) from summary.csv."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Any, Dict, List, Tuple


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str) and value.strip() != "":
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _make_label(row: Dict[str, Any], group_by: List[str]) -> str:
    parts = []
    for key in group_by:
        val = row.get(key)
        if val is None or val == "":
            continue
        parts.append(f"{key}={val}")
    return ", ".join(parts) if parts else "run"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latency curves from summary CSV.")
    parser.add_argument("--summary", default="pcg_codec/results/summary.csv")
    parser.add_argument("--metric", default="metrics.latency.ttfa_ms")
    parser.add_argument("--only-matched", action="store_true")
    parser.add_argument("--group-by", default="codec,quantizer,sens_eq")
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--output", default="pcg_codec/results/plots/latency.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    rows = _load_rows(args.summary)
    if not rows:
        raise SystemExit(f"No rows found in {args.summary}")

    group_by = [x.strip() for x in args.group_by.split(",") if x.strip()]

    grouped: Dict[str, Dict[float, List[float]]] = {}
    for row in rows:
        if args.only_matched and row.get("bitrate_within_tolerance") not in ("True", "true", True):
            continue
        depth = _as_float(row.get("depth"))
        metric = _as_float(row.get(args.metric))
        if depth is None or metric is None:
            continue
        label = _make_label(row, group_by)
        grouped.setdefault(label, {}).setdefault(depth, []).append(metric)

    if not grouped:
        raise SystemExit("No rows to plot after filtering.")

    plt.figure(figsize=(7, 5))

    for label, depth_map in grouped.items():
        depths = sorted(depth_map.keys())
        values = []
        for d in depths:
            vals = depth_map[d]
            if args.aggregate == "mean":
                values.append(sum(vals) / len(vals))
            else:
                values.append(statistics.median(vals))
        plt.plot(depths, values, marker="o", label=label)

    plt.xlabel("Depth (D)")
    plt.ylabel(args.metric)
    plt.title("Latency vs Depth")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
