#!/usr/bin/env python3
"""Plot rate-distortion curves from summary.csv."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List


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
    parser = argparse.ArgumentParser(description="Plot RD curves from summary CSV.")
    parser.add_argument("--summary", default="pcg_codec/results/summary.csv")
    parser.add_argument("--metric", default="metrics.quality.mrstft")
    parser.add_argument("--only-matched", action="store_true")
    parser.add_argument("--group-by", default="codec,quantizer,depth,sens_eq")
    parser.add_argument("--output", default="pcg_codec/results/plots/rd_curves.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    rows = _load_rows(args.summary)
    if not rows:
        raise SystemExit(f"No rows found in {args.summary}")

    group_by = [x.strip() for x in args.group_by.split(",") if x.strip()]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if args.only_matched and row.get("bitrate_within_tolerance") not in ("True", "true", True):
            continue
        label = _make_label(row, group_by)
        grouped.setdefault(label, []).append(row)

    if not grouped:
        raise SystemExit("No rows to plot after filtering.")

    plt.figure(figsize=(8, 5))

    for label, items in grouped.items():
        pairs = []
        for row in items:
            bitrate = _as_float(row.get("bitrate_kbps"))
            metric = _as_float(row.get(args.metric))
            if bitrate is None or metric is None:
                continue
            pairs.append((bitrate, metric))
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Bitrate (kbps)")
    plt.ylabel(args.metric)
    plt.title("Rate-Distortion Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
