"""Metric utilities."""

from __future__ import annotations

import numpy as np


def bitrate_from_bytes(total_bytes: int, num_frames: int, hop: int, sample_rate: int) -> float:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if hop <= 0 or sample_rate <= 0:
        raise ValueError("hop and sample_rate must be positive")
    seconds = num_frames * hop / sample_rate
    return 8.0 * total_bytes / seconds


def summarize_distribution(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
