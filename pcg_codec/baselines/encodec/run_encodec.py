#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch

from ...pcg.entropy.streams import ContainerHeader, FramePacket, stable_params_hash, write_container
from ...pcg.training.losses import MultiResolutionSTFTLoss
from ...pcg.training.metrics import bitrate_from_bytes, summarize_distribution
from .wrappers import EncodecConfig, EncodecNotInstalledError, load_encodec_model


def _load_yaml(path: str) -> dict:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if int(sr) != int(target_sr):
        raise RuntimeError("Baseline runner expects audio already at target sample rate")
    return audio.astype(np.float32, copy=False)


def _extract_codes(encoded: Any) -> torch.Tensor:
    """
    Best-effort extraction of EnCodec code indices from the encodec package.

    The encodec API varies across versions; this handles common cases.
    """
    if isinstance(encoded, (list, tuple)) and encoded:
        first = encoded[0]
        if hasattr(first, "codes"):
            return first.codes
    if hasattr(encoded, "codes"):
        return encoded.codes
    if isinstance(encoded, tuple) and len(encoded) >= 1 and torch.is_tensor(encoded[0]):
        return encoded[0]
    raise RuntimeError("Unable to extract codes from encodec encode output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EnCodec raw baseline and write eval.json + container.")
    parser.add_argument("--config", default="pcg_codec/configs/encodec_baseline.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-wav", required=True, help="Path to a 24kHz mono wav file for baseline eval.")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    streaming = cfg["streaming"]
    sr = int(streaming["sample_rate_hz"])
    bitrate_kbps = float(cfg["baseline"]["bitrate_kbps"])

    os.makedirs(args.run_dir, exist_ok=True)

    try:
        model = load_encodec_model(EncodecConfig(sample_rate_hz=sr, bitrate_kbps=bitrate_kbps))
    except EncodecNotInstalledError as exc:
        raise SystemExit(str(exc)) from exc

    audio = _load_audio(args.input_wav, target_sr=sr)
    wav = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        encoded = model.encode(wav)
        decoded = model.decode(encoded).squeeze(0).squeeze(0).cpu()
        codes = _extract_codes(encoded)

    codes = codes.detach().cpu().to(torch.int64)
    if codes.dim() == 2:
        # (n_q, frames) -> (n_q, 1, frames)
        codes = codes.unsqueeze(1)
    if codes.dim() != 3:
        raise RuntimeError("Unexpected codes shape")

    n_q, batch, frames = codes.shape
    if batch != 1:
        raise RuntimeError("Expected batch=1")

    # Derive hop per code frame from audio length.
    hop_samples = max(1, int(round(len(audio) / frames)))

    packets: list[FramePacket] = []
    bytes_total = 0
    bps_per_frame: list[float] = []
    for t in range(frames):
        codes_t = codes[:, 0, t].numpy().astype(np.uint16, copy=False)
        payload = codes_t.tobytes()
        packets.append(FramePacket(frame_index=int(t), layer_bytes=[payload]))
        bytes_total += len(payload)
        bps_per_frame.append(8.0 * len(payload) / (hop_samples / sr))

    header = ContainerHeader(
        sample_rate_hz=sr,
        hop_samples=hop_samples,
        encoder_lookahead_samples=int(streaming.get("lookahead_samples", 0)),
        codec_name="encodec_raw",
        params_hash=stable_params_hash(cfg),
    )
    with open(os.path.join(args.run_dir, "stream.pcg"), "wb") as f:
        write_container(f, header=header, frames=packets)

    # Metrics
    x = torch.from_numpy(audio).unsqueeze(0)
    x_hat = decoded.unsqueeze(0)
    stft = MultiResolutionSTFTLoss().to(x.device)
    with torch.no_grad():
        mrstft = float(stft(x, x_hat).cpu())

    bitrate_bps = bitrate_from_bytes(bytes_total, frames, hop=hop_samples, sample_rate=sr)
    dist = summarize_distribution(bps_per_frame)

    t0 = time.perf_counter()
    _ = model.decode(encoded)
    t1 = time.perf_counter()
    ttfa_ms = 1000.0 * (t1 - t0)

    eval_json = {
        "run_id": os.path.basename(args.run_dir),
        "codec": "encodec",
        "dataset": {"name": os.path.basename(args.input_wav)},
        "streaming": {"sample_rate_hz": sr, "hop_samples": hop_samples, "lookahead_samples": 0, "depth": 1},
        "metrics": {
            "bitrate": {"bps": bitrate_bps, "p5_bps": dist["p5"], "p50_bps": dist["p50"], "p95_bps": dist["p95"]},
            "quality": {"mrstft": mrstft},
            "latency": {"ttfa_ms": ttfa_ms, "rtf_cpu_1": None, "rtf_cpu_4": None},
        },
        "bytes_total": int(bytes_total),
        "frames_total": int(frames),
        "config": cfg,
    }
    with open(os.path.join(args.run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2, sort_keys=False)


if __name__ == "__main__":
    main()
