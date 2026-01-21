from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncodecConfig:
    sample_rate_hz: int = 24000
    hop_samples: int = 320
    encoder_lookahead_samples: int = 0
    bitrate_kbps: float = 6.0


class EncodecNotInstalledError(RuntimeError):
    pass


def load_encodec_model(cfg: EncodecConfig) -> Any:
    try:
        from encodec import EncodecModel
    except Exception as exc:  # noqa: BLE001
        raise EncodecNotInstalledError(
            "Optional dependency `encodec` not installed; install with `pip install -e .[encodec]`."
        ) from exc

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(cfg.bitrate_kbps)
    model.eval()
    return model


def encode_streaming_frames(model: Any, audio: np.ndarray, hop_samples: int) -> list[np.ndarray]:
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    frames = []
    for i in range(0, len(audio), hop_samples):
        frame = audio[i : i + hop_samples]
        if len(frame) < hop_samples:
            frame = np.pad(frame, (0, hop_samples - len(frame)))
        frames.append(frame.astype(np.float32, copy=False))
    return frames

