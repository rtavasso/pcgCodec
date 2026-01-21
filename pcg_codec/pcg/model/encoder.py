"""Streaming encoder for PCG-Codec."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class StreamingEncoder(nn.Module):
    """Frame-wise encoder with optional lookahead padding."""

    def __init__(
        self,
        hop: int = 320,
        lookahead: int = 0,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if hop <= 0:
            raise ValueError("hop must be positive")
        if lookahead < 0:
            raise ValueError("lookahead must be non-negative")
        self.hop = int(hop)
        self.lookahead = int(lookahead)
        input_dim = self.hop + self.lookahead
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._cached_device: Optional[torch.device] = None

    def reset(self) -> None:
        """Reset any internal streaming state (none for the minimal encoder)."""
        return None

    def _as_tensor(self, x_frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x_frame):
            x = x_frame
        else:
            x = torch.as_tensor(x_frame)
        if x.dtype != torch.float32:
            x = x.float()
        return x

    def encode_frame(self, x_frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a single frame into a continuous latent vector."""
        x = self._as_tensor(x_frame)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(-1) == self.hop:
            if self.lookahead > 0:
                pad = torch.zeros(x.size(0), self.lookahead, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
        elif x.size(-1) != self.hop + self.lookahead:
            raise ValueError(
                f"Expected frame length {self.hop} (or {self.hop + self.lookahead} with lookahead), "
                f"got {x.size(-1)}"
            )
        z = self.net(x)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames stacked along time."""
        if x.dim() != 2:
            raise ValueError("Expected input shape (batch, samples)")
        if x.size(-1) % self.hop != 0:
            raise ValueError("Input length must be a multiple of hop")
        frames = x.unfold(dimension=-1, size=self.hop, step=self.hop)
        batch, num_frames, _ = frames.shape
        frames = frames.reshape(batch * num_frames, self.hop)
        z = self.encode_frame(frames)
        return z.view(batch, num_frames, -1)
