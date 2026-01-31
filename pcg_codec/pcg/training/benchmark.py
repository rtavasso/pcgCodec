"""Benchmark PCG-Codec training throughput.

This is intended to answer "where is my training time going?" by separating:
  - input pipeline (batch fetch)
  - host->device transfer
  - forward/backward/optimizer step
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

from .losses import MultiResolutionSTFTLoss
from .train import build_components, build_train_dataloader, train_step


def _load_yaml(path: str) -> dict:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pad_to_hop(n: int, hop: int) -> int:
    r = int(n) % int(hop)
    return int(n) if r == 0 else int(n) + int(hop) - r


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark input and training step throughput.")
    parser.add_argument("--config", required=True, help="Run config YAML.")
    parser.add_argument("--datasets", default="pcg_codec/configs/datasets.yaml", help="Dataset registry YAML.")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--sync-cuda", action="store_true", help="Synchronize CUDA for accurate timings.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random synthetic batches (bypasses dataset IO; useful for compute-only benchmarking).",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    ds_cfg = _load_yaml(args.datasets)
    device = torch.device(str(cfg["training"]["device"]))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    seed = int(cfg["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    components = build_components(cfg)
    components.encoder.to(device)
    components.decoder.to(device)
    components.transform.to(device)
    components.quantizer.to(device)
    if components.prior is not None:
        components.prior.to(device)

    stft_cfg = cfg["loss"]["mrstft"]
    stft_loss = None
    if bool(stft_cfg.get("enabled", True)):
        stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg["fft_sizes"],
            hop_sizes=stft_cfg["hop_sizes"],
            win_lengths=stft_cfg["win_lengths"],
        ).to(device)

    params = (
        list(components.encoder.parameters())
        + list(components.decoder.parameters())
        + list(components.transform.parameters())
        + list(components.quantizer.parameters())
        + (list(components.prior.parameters()) if components.prior is not None else [])
    )
    optimizer = torch.optim.Adam(params, lr=float(cfg["training"]["lr"]))

    lambda_rd = float(cfg["training"]["lambda_rd"])
    sens_cfg = cfg["loss"]["sens_eq"]
    beta_sens = float(sens_cfg["weight"]) if bool(sens_cfg.get("enabled", False)) else 0.0

    amp_cfg = cfg.get("training", {}).get("amp", {}) or {}
    autocast_enabled = bool(amp_cfg.get("enabled", device.type == "cuda"))
    dtype_name = str(amp_cfg.get("dtype", "bfloat16")).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16"}:
        autocast_dtype = torch.float16
    elif dtype_name in {"fp32", "float32"}:
        autocast_enabled = False
        autocast_dtype = torch.float32
    else:
        raise ValueError("training.amp.dtype must be one of bf16|fp16|fp32")
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
    if (
        device.type == "cuda"
        and autocast_enabled
        and autocast_dtype == torch.float16
        and bool(amp_cfg.get("grad_scaler", True))
    ):
        grad_scaler = torch.cuda.amp.GradScaler()

    if args.synthetic:
        hop = int(cfg["streaming"]["hop_samples"])
        sr = int(cfg["streaming"]["sample_rate_hz"])
        dataset_name = str(cfg["dataset"])
        d = ds_cfg["datasets"][dataset_name]
        seg_s = float(d.get("segment_seconds", ds_cfg.get("defaults", {}).get("segment_seconds", 4.0)))
        seg_samples = int(round(seg_s * sr))
        seg_samples = _pad_to_hop(seg_samples, hop)
        batch_size = int(cfg["training"]["batch_size"])
        pin_memory = False

        def next_batch():
            return torch.randn(batch_size, seg_samples, device=device)

        data_iter = None
    else:
        loader, pin_memory = build_train_dataloader(cfg, ds_cfg, device=device)
        data_iter = iter(loader)

        def next_batch():
            assert data_iter is not None
            return next(data_iter)

    def maybe_sync():
        if args.sync_cuda and device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(int(args.warmup)):
        batch = next_batch()
        if torch.is_tensor(batch) and batch.device != device:
            batch = batch.to(device, non_blocking=bool(pin_memory and device.type == "cuda"))
        _ = train_step(
            components,
            batch,
            optimizer,
            device,
            stft_loss=stft_loss,
            lambda_rate=lambda_rd,
            beta_sens=beta_sens,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            grad_scaler=grad_scaler,
            sync_metrics=False,
        )
    maybe_sync()

    fetch_s = 0.0
    xfer_s = 0.0
    step_s = 0.0
    t0 = time.perf_counter()
    for _ in range(int(args.iters)):
        t_fetch0 = time.perf_counter()
        batch = next_batch()
        t_fetch1 = time.perf_counter()
        fetch_s += t_fetch1 - t_fetch0

        if torch.is_tensor(batch) and batch.device != device:
            if args.sync_cuda and device.type == "cuda":
                torch.cuda.synchronize()
            t_xfer0 = time.perf_counter()
            batch = batch.to(device, non_blocking=bool(pin_memory and device.type == "cuda"))
            if args.sync_cuda and device.type == "cuda":
                torch.cuda.synchronize()
            t_xfer1 = time.perf_counter()
            xfer_s += t_xfer1 - t_xfer0

        if args.sync_cuda and device.type == "cuda":
            torch.cuda.synchronize()
        t_step0 = time.perf_counter()
        _ = train_step(
            components,
            batch,
            optimizer,
            device,
            stft_loss=stft_loss,
            lambda_rate=lambda_rd,
            beta_sens=beta_sens,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            grad_scaler=grad_scaler,
            sync_metrics=False,
        )
        if args.sync_cuda and device.type == "cuda":
            torch.cuda.synchronize()
        t_step1 = time.perf_counter()
        step_s += t_step1 - t_step0
    maybe_sync()
    t1 = time.perf_counter()

    iters = max(int(args.iters), 1)
    dt = max(t1 - t0, 1e-9)
    steps_per_s = iters / dt
    batch_size = int(cfg["training"]["batch_size"])
    samples_per_s = steps_per_s * batch_size

    mode = "synthetic" if args.synthetic else "dataset"
    fetch_ms = 1e3 * fetch_s / iters
    xfer_ms = 1e3 * xfer_s / iters
    step_ms = 1e3 * step_s / iters
    print(
        f"[bench:{mode}] device={device} steps/s={steps_per_s:.2f} samples/s={samples_per_s:.2f} "
        f"fetch_ms={fetch_ms:.2f} xfer_ms={xfer_ms:.2f} step_ms={step_ms:.2f}"
    )


if __name__ == "__main__":
    main()
