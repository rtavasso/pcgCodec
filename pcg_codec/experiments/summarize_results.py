#!/usr/bin/env python3
"""Summarize evaluation JSONs into a flat table + JSON.

Expected eval.json schema (minimal):
{
  "run_id": "...",
  "codec": "pcg" | "encodec",
  "dataset": {"name": "librispeech_dev_clean"},
  "streaming": {"sample_rate_hz": 24000, "hop_samples": 320, "lookahead_samples": 0, "depth": 2},
  "metrics": {
    "bitrate": {"bps": 2400.0, "p5_bps": 2300.0, "p50_bps": 2400.0, "p95_bps": 2500.0},
    "quality": {"mrstft": 0.123},
    "latency": {"ttfa_ms": 12.3, "rtf_cpu_1": 0.8, "rtf_cpu_4": 0.3}
  },
  "bytes_total": 123456,
  "frames_total": 1000,
  "target_bitrate_kbps": 3.0
}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else key, value, out)
    else:
        out[prefix] = obj


def _find_eval_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == "eval.json":
                yield os.path.join(dirpath, name)


def _get_nested(data: Dict[str, Any], path: List[str]) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_present(data: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _compute_bitrate_bps(bytes_total: float, frames_total: float, hop_samples: int, sample_rate_hz: int) -> float:
    seconds = (frames_total * hop_samples) / float(sample_rate_hz)
    if seconds <= 0:
        return 0.0
    return 8.0 * float(bytes_total) / seconds


def _extract_config(run_dir: str, eval_data: Dict[str, Any]) -> Dict[str, Any]:
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        return _load_yaml(cfg_path)
    cfg = eval_data.get("config")
    return cfg if isinstance(cfg, dict) else {}


def _row_from_eval(eval_path: str, tolerance_pct: float | None) -> Dict[str, Any]:
    eval_data = _load_json(eval_path)
    run_dir = os.path.dirname(eval_path)
    cfg = _extract_config(run_dir, eval_data)

    row: Dict[str, Any] = {}
    row["eval_path"] = eval_path
    row["run_id"] = eval_data.get("run_id") or os.path.basename(run_dir)
    row["codec"] = eval_data.get("codec") or cfg.get("codec")

    dataset = eval_data.get("dataset", {})
    row["dataset"] = dataset.get("name") or dataset.get("id") or cfg.get("dataset")

    streaming = eval_data.get("streaming", {})
    if not streaming and "streaming" in cfg:
        streaming = cfg.get("streaming", {})

    row["sample_rate_hz"] = streaming.get("sample_rate_hz")
    row["hop_samples"] = streaming.get("hop_samples")
    row["lookahead_samples"] = streaming.get("lookahead_samples")
    row["depth"] = streaming.get("depth") or _get_nested(cfg, ["model", "dag", "depth"])

    row["quantizer"] = _get_nested(cfg, ["model", "quantizer", "type"])
    row["sens_eq"] = _get_nested(cfg, ["loss", "sens_eq", "enabled"])
    row["lambda_rd"] = _get_nested(cfg, ["training", "lambda_rd"])

    metrics = eval_data.get("metrics", {})
    bitrate = metrics.get("bitrate", {})

    bitrate_bps = bitrate.get("bps")
    if bitrate_bps is None:
        bytes_total = eval_data.get("bytes_total")
        frames_total = _first_present(eval_data, ["frames_total", "frame_count", "num_frames"])
        hop = row.get("hop_samples")
        sr = row.get("sample_rate_hz")
        if bytes_total is not None and frames_total is not None and hop and sr:
            bitrate_bps = _compute_bitrate_bps(bytes_total, frames_total, int(hop), int(sr))
    row["bitrate_bps"] = bitrate_bps
    row["bitrate_kbps"] = (bitrate_bps / 1000.0) if bitrate_bps is not None else None

    target_kbps = eval_data.get("target_bitrate_kbps")
    if target_kbps is None:
        target_kbps = _get_nested(cfg, ["targets", "bitrate_kbps"])
        if isinstance(target_kbps, list):
            target_kbps = None
    row["target_bitrate_kbps"] = target_kbps

    tol = tolerance_pct
    if tol is None:
        tol = _get_nested(cfg, ["metrics", "bitrate_tolerance_pct"]) or _get_nested(cfg, ["targets", "tolerance_pct"])
    row["bitrate_tolerance_pct"] = tol

    if bitrate_bps is not None and target_kbps:
        target_bps = float(target_kbps) * 1000.0
        row["bitrate_error_pct"] = 100.0 * (bitrate_bps - target_bps) / target_bps
        if tol is not None:
            row["bitrate_within_tolerance"] = abs(row["bitrate_error_pct"]) <= float(tol)
    else:
        row["bitrate_error_pct"] = None
        row["bitrate_within_tolerance"] = None

    flat_metrics: Dict[str, Any] = {}
    _flatten("metrics", metrics, flat_metrics)
    row.update(flat_metrics)

    return row


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval.json files into a table.")
    parser.add_argument("--results-dir", default="pcg_codec/results")
    parser.add_argument("--out-json", default="pcg_codec/results/summary.json")
    parser.add_argument("--out-csv", default="pcg_codec/results/summary.csv")
    parser.add_argument("--tolerance-pct", type=float, default=None)
    parser.add_argument("--only-matched", action="store_true")
    args = parser.parse_args()

    eval_files = list(_find_eval_files(args.results_dir))
    if not eval_files:
        raise SystemExit(f"No eval.json files found under {args.results_dir}")

    rows: List[Dict[str, Any]] = []
    for eval_path in eval_files:
        row = _row_from_eval(eval_path, args.tolerance_pct)
        if args.only_matched and not row.get("bitrate_within_tolerance"):
            continue
        rows.append(row)

    summary = {
        "count": len(rows),
        "results_dir": args.results_dir,
        "rows": rows,
    }
    _write_json(args.out_json, summary)
    _write_csv(args.out_csv, rows)

    print(f"Wrote {len(rows)} rows to {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
