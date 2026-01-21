"""Losses for PCG-Codec training."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def waveform_l1(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - x_hat))


def waveform_l2(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - x_hat) ** 2)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT magnitude loss."""

    def __init__(
        self,
        fft_sizes: Iterable[int] = (256, 512, 1024),
        hop_sizes: Iterable[int] = (64, 128, 256),
        win_lengths: Iterable[int] = (256, 512, 1024),
    ) -> None:
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("FFT sizes, hop sizes, and win lengths must match in length")

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x_hat.dim() == 1:
            x_hat = x_hat.unsqueeze(0)
        loss = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win, device=x.device, dtype=x.dtype)
            x_stft = torch.stft(x, n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            x_hat_stft = torch.stft(
                x_hat, n_fft=fft, hop_length=hop, win_length=win, window=window, return_complex=True
            )
            mag = torch.abs(x_stft)
            mag_hat = torch.abs(x_hat_stft)
            loss = loss + torch.mean(torch.abs(mag - mag_hat))
        return loss / len(self.fft_sizes)


def sensitivity_equalization(
    z: torch.Tensor,
    perceptual_loss: torch.Tensor,
    num_blocks: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute sensitivity equalization loss from perceptual gradients."""
    if not z.requires_grad:
        return torch.tensor(0.0, device=z.device)
    grads = torch.autograd.grad(perceptual_loss, z, retain_graph=True, create_graph=True)[0]
    if grads is None:
        return torch.tensor(0.0, device=z.device)
    if grads.dim() != 2:
        grads = grads.view(grads.size(0), -1)
    if grads.size(-1) % num_blocks != 0:
        raise ValueError("latent dim must be divisible by num_blocks")
    block_dim = grads.size(-1) // num_blocks
    grads = grads.view(grads.size(0), num_blocks, block_dim)
    norms = torch.sqrt((grads**2).sum(dim=-1) + eps)
    mean_norms = norms.mean(dim=0)
    return torch.var(torch.log(mean_norms + eps))
