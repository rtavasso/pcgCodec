"""Training utilities for PCG-Codec.

This package includes some optional heavy dependencies (e.g. torch/scipy/soundfile).
To keep lightweight utilities (like metrics) usable in minimal environments, we lazily
import most symbols on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .eval import evaluate as evaluate
    from .losses import (
        MultiResolutionSTFTLoss,
        sensitivity_equalization,
        waveform_l1,
        waveform_l2,
    )
    from .metrics import bitrate_from_bytes as bitrate_from_bytes
    from .schedulers import WarmupCosineScheduler as WarmupCosineScheduler
    from .train import train_one_epoch as train_one_epoch, train_step as train_step

__all__ = [
    "MultiResolutionSTFTLoss",
    "waveform_l1",
    "waveform_l2",
    "sensitivity_equalization",
    "train_one_epoch",
    "train_step",
    "evaluate",
    "bitrate_from_bytes",
    "WarmupCosineScheduler",
]


def __getattr__(name: str):
    if name in {"MultiResolutionSTFTLoss", "waveform_l1", "waveform_l2", "sensitivity_equalization"}:
        from . import losses as _losses

        return getattr(_losses, name)
    if name in {"train_one_epoch", "train_step"}:
        from . import train as _train

        return getattr(_train, name)
    if name == "evaluate":
        from . import eval as _eval

        return getattr(_eval, name)
    if name == "bitrate_from_bytes":
        from . import metrics as _metrics

        return getattr(_metrics, name)
    if name == "WarmupCosineScheduler":
        from . import schedulers as _schedulers

        return getattr(_schedulers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
