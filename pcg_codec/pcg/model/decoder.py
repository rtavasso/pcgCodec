"""Streaming decoder for PCG-Codec."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class StreamingDecoder(nn.Module):
    """Frame-wise decoder that maps latents back to waveform frames."""

    def __init__(self, hop: int = 320, latent_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        if hop <= 0:
            raise ValueError("hop must be positive")
        self.hop = int(hop)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.hop),
        )

    def reset(self) -> None:
        """Reset any internal streaming state (none for the minimal decoder)."""
        return None

    def _extract_latent(self, q_frame: Any) -> torch.Tensor:
        if torch.is_tensor(q_frame):
            return q_frame
        if isinstance(q_frame, np.ndarray):
            return torch.as_tensor(q_frame).float()
        if hasattr(q_frame, "z_hat"):
            return q_frame.z_hat
        if isinstance(q_frame, dict) and "z_hat" in q_frame:
            return q_frame["z_hat"]
        raise TypeError("q_frame must be a tensor, numpy array, or have a z_hat attribute")

    def decode_frame(self, q_frame: Any) -> torch.Tensor:
        """Decode a single frame from quantized latents."""
        z_hat = self._extract_latent(q_frame)
        if z_hat.dim() == 1:
            z_hat = z_hat.unsqueeze(0)
        x_hat = self.net(z_hat)
        return x_hat

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent frames stacked along time."""
        if z.dim() != 3:
            raise ValueError("Expected input shape (batch, frames, latent_dim)")
        batch, frames, _ = z.shape
        z_flat = z.reshape(batch * frames, -1)
        x_hat = self.decode_frame(z_flat)
        return x_hat.view(batch, frames * self.hop)
