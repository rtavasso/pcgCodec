#!/usr/bin/env python3
"""Generate run configs for the PCG-Codec ablation grid.

This script builds per-run config files by merging a base config with
factor overrides and an RD lambda sweep. It does not train; it prepares
run directories and optional runner commands.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _dump_yaml(path: str, data: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def _format_lambda(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def _iter_grid(grid: Dict[str, List[Dict[str, Any]]]) -> Iterable[List[Tuple[str, Dict[str, Any]]]]:
    keys = list(grid.keys())

    def rec(idx: int, acc: List[Tuple[str, Dict[str, Any]]]):
        if idx == len(keys):
            yield acc
            return
        key = keys[idx]
        for entry in grid[key]:
            name = entry.get("name", key)
            overrides = entry.get("overrides", {})
            yield from rec(idx + 1, acc + [(name, overrides)])

    return rec(0, [])


def _load_lambda_grid(grid_cfg: Dict[str, Any], base_cfg: Dict[str, Any]) -> List[float]:
    lambdas = grid_cfg.get("lambda_grid")
    if lambdas:
        return [float(x) for x in lambdas]
    base_lambda = base_cfg.get("training", {}).get("lambda_rd")
    if base_lambda is None:
        raise SystemExit("No lambda_grid in ablation config and no training.lambda_rd in base config.")
    return [float(base_lambda)]


def _build_run_id(prefix: str, names: List[str], lambda_rd: float) -> str:
    parts = [prefix] + [ _slug(n) for n in names ] + [f"l{_format_lambda(lambda_rd)}"]
    return "_".join([p for p in parts if p])


def _prepare_run_config(
    base_cfg: Dict[str, Any],
    overrides_list: List[Dict[str, Any]],
    lambda_rd: float,
    grid_names: List[str],
    targets: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for overrides in overrides_list:
        _deep_update(cfg, overrides)
    cfg.setdefault("training", {})["lambda_rd"] = float(lambda_rd)
    cfg.setdefault("grid", {})["names"] = grid_names
    cfg.setdefault("targets", {})["bitrate_kbps"] = targets.get("bitrate_kbps", [])
    cfg.setdefault("targets", {})["tolerance_pct"] = targets.get("tolerance_pct", None)
    return cfg


def _write_run(run_dir: str, run_id: str, cfg: Dict[str, Any], meta: Dict[str, Any]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    _dump_yaml(os.path.join(run_dir, "config.yaml"), cfg)
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ablation grid run configs.")
    parser.add_argument("--base-config", default="pcg_codec/configs/pcg_base.yaml")
    parser.add_argument("--grid-config", default="pcg_codec/configs/ablations/ablation_grid.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print run IDs without writing files.")
    parser.add_argument("--runner", default=None, help="Command template to execute per run.")
    parser.add_argument(
        "--skip-tests-check",
        action="store_true",
        help="Skip the pre-run `pytest` gate (not recommended; the spec requires tests to pass).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of runs.")
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)
    grid_cfg = _load_yaml(args.grid_config)

    output_dir = args.output_dir or base_cfg.get("output", {}).get("results_dir", "pcg_codec/results")
    runs_root = os.path.join(output_dir, "runs")

    grid = grid_cfg.get("grid", {})
    if not grid:
        raise SystemExit("No grid entries found in ablation config.")

    lambda_grid = _load_lambda_grid(grid_cfg, base_cfg)
    targets = grid_cfg.get("targets", {})

    prefix = base_cfg.get("output", {}).get("run_name_prefix", "pcg")

    if args.runner and not args.skip_tests_check and not args.dry_run:
        try:
            subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit("Refusing to run experiments: `pytest` failed.") from exc

    run_count = 0
    for combo in _iter_grid(grid):
        names = [name for name, _ in combo]
        overrides_list = [overrides for _, overrides in combo]
        for lambda_rd in lambda_grid:
            run_id = _build_run_id(prefix, names, lambda_rd)
            cfg = _prepare_run_config(base_cfg, overrides_list, lambda_rd, names, targets)
            run_dir = os.path.join(runs_root, run_id)

            meta = {
                "run_id": run_id,
                "created_at": dt.datetime.utcnow().isoformat() + "Z",
                "grid_names": names,
                "lambda_rd": lambda_rd,
                "targets": targets,
            }

            if args.dry_run:
                print(run_id)
            else:
                _write_run(run_dir, run_id, cfg, meta)

            if args.runner:
                cmd = args.runner.format(config=os.path.join(run_dir, "config.yaml"), run_dir=run_dir, run_id=run_id)
                print(f"[runner] {cmd}")
                if not args.dry_run:
                    subprocess.run(cmd, shell=True, check=False)

            run_count += 1
            if args.limit is not None and run_count >= args.limit:
                return


if __name__ == "__main__":
    main()
