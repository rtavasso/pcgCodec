#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly  # type: ignore


def _iter_wavs(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                yield os.path.join(dirpath, name)


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32, copy=False), int(sr)


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    g = int(np.gcd(int(sr), int(target_sr)))
    up = int(target_sr) // g
    down = int(sr) // g
    return resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local_wav_dir dataset (mono 24kHz) under pcg_codec/data/…")
    parser.add_argument("--src", required=True, help="Source directory containing wav files.")
    parser.add_argument("--dst", required=True, help="Destination directory to write normalized wavs.")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(_iter_wavs(args.src))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"No wavs found under {args.src}")

    hasher = hashlib.sha256()
    for src_path in files:
        rel = os.path.relpath(src_path, args.src)
        dst_path = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        audio, sr = _load_mono(src_path)
        audio = _resample(audio, sr=sr, target_sr=int(args.target_sr))
        sf.write(dst_path, audio, int(args.target_sr))

        hasher.update(rel.encode("utf-8"))
        hasher.update(str(len(audio)).encode("utf-8"))

    split_hash = hasher.hexdigest()[:16]
    print(f"Prepared {len(files)} files at {args.dst}")
    print(f"Suggested datasets.yaml split_hash: {split_hash}")


if __name__ == "__main__":
    main()

