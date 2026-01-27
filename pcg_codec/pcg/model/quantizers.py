"""Quantizers for PCG-Codec.

Quantizers must output one discrete symbol per block, consistent with the README spec:

- Symbols shape: (batch, num_blocks)
- Block b symbols are integers in [0, K_b-1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class QuantizerOutput:
    symbols: torch.Tensor  # (batch, num_blocks) int64
    z_hat: torch.Tensor  # (batch, latent_dim) float32


class FSQQuantizer(nn.Module):
    """Finite scalar quantization (FSQ-style) with per-dimension levels, packed per block."""

    def __init__(self, levels: int | Iterable[int] = 8, data_range: float = 1.0) -> None:
        super().__init__()
        if isinstance(levels, int):
            if levels < 2:
                raise ValueError("levels must be >= 2")
        else:
            levels = list(levels)
            if any(l < 2 for l in levels):
                raise ValueError("all levels must be >= 2")
        self.levels = levels
        self.data_range = float(data_range)
        self.num_blocks: int | None = None
        self.block_dim: int | None = None

    def _levels_tensor(self, dim: int, device: torch.device) -> torch.Tensor:
        if isinstance(self.levels, int):
            return torch.full((dim,), self.levels, device=device, dtype=torch.float32)
        levels = torch.tensor(self.levels, device=device, dtype=torch.float32)
        if levels.numel() != dim:
            raise ValueError(f"Expected {dim} level entries, got {levels.numel()}")
        return levels

    def configure_blocks(self, num_blocks: int, block_dim: int) -> None:
        if num_blocks <= 0 or block_dim <= 0:
            raise ValueError("num_blocks and block_dim must be positive")
        self.num_blocks = int(num_blocks)
        self.block_dim = int(block_dim)

    def quantize(self, z: torch.Tensor) -> QuantizerOutput:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        dim = z.size(-1)
        if self.num_blocks is None or self.block_dim is None:
            raise RuntimeError("FSQQuantizer.configure_blocks(num_blocks, block_dim) must be called")
        expected_dim = self.num_blocks * self.block_dim
        if dim != expected_dim:
            raise ValueError(f"Expected latent dim {expected_dim}, got {dim}")
        if not isinstance(self.levels, int):
            raise ValueError("Packed FSQQuantizer currently requires `levels` to be an int")
        levels = self._levels_tensor(dim, z.device)
        step = 2.0 * self.data_range / (levels - 1.0)
        z_clamped = z.clamp(-self.data_range, self.data_range)
        idx = torch.round((z_clamped + self.data_range) / step).long()
        # Avoid torch.clamp(min=int, max=tensor) which is not supported on some torch builds.
        idx = idx.clamp_min(0)
        idx = torch.minimum(idx, levels.to(dtype=idx.dtype) - 1)
        z_q = idx.float() * step - self.data_range
        z_hat = z + (z_q - z).detach()
        # Pack per-dimension indices into one symbol per block (mixed radix).
        idx_blocks = idx.view(z.size(0), self.num_blocks, self.block_dim)
        radix = torch.tensor(int(self.levels), device=idx.device, dtype=torch.int64)
        weights = radix ** torch.arange(self.block_dim, device=idx.device, dtype=torch.int64)
        symbols = (idx_blocks * weights).sum(dim=-1).to(torch.int64)
        return QuantizerOutput(symbols=symbols, z_hat=z_hat)

    def dequantize(self, symbols: torch.Tensor) -> torch.Tensor:
        if self.num_blocks is None or self.block_dim is None:
            raise RuntimeError("FSQQuantizer.configure_blocks(num_blocks, block_dim) must be called")
        if symbols.dim() != 2:
            raise ValueError("symbols must have shape (batch, num_blocks)")
        if symbols.size(-1) != self.num_blocks:
            raise ValueError(f"Expected num_blocks={self.num_blocks}, got {symbols.size(-1)}")
        if not isinstance(self.levels, int):
            raise ValueError("Packed FSQQuantizer currently requires `levels` to be an int")

        batch = symbols.size(0)
        radix = int(self.levels)
        symbols = symbols.to(torch.int64)
        idx = torch.empty((batch, self.num_blocks, self.block_dim), device=symbols.device, dtype=torch.int64)
        s = symbols.clone()
        for i in range(self.block_dim):
            idx[:, :, i] = s % radix
            s = s // radix
        idx_flat = idx.reshape(batch, -1).to(torch.float32)
        step = 2.0 * self.data_range / (radix - 1.0)
        z_q = idx_flat * step - self.data_range
        return z_q

    def usage_fraction(self, symbols: torch.Tensor) -> torch.Tensor:
        if self.num_blocks is None:
            raise RuntimeError("FSQQuantizer.configure_blocks(num_blocks, block_dim) must be called")
        if symbols.dim() != 2 or symbols.size(-1) != self.num_blocks:
            raise ValueError("symbols must have shape (batch, num_blocks)")
        frac = []
        for b in range(self.num_blocks):
            uniq = torch.unique(symbols[:, b]).numel()
            frac.append(float(uniq) / float(symbols.size(0)))
        return torch.tensor(frac, device=symbols.device, dtype=torch.float32)

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        return self.quantize(z)


class BlockCodebookQuantizer(nn.Module):
    """Blockwise vector quantizer with per-block codebooks."""

    def __init__(
        self,
        num_blocks: int,
        block_dim: int,
        codebook_size: int = 256,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_blocks <= 0 or block_dim <= 0:
            raise ValueError("num_blocks and block_dim must be positive")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        self.num_blocks = int(num_blocks)
        self.block_dim = int(block_dim)
        self.codebook_size = int(codebook_size)
        codebook = init_scale * torch.randn(self.num_blocks, self.codebook_size, self.block_dim)
        self.codebook = nn.Parameter(codebook)

    def quantize(self, z: torch.Tensor) -> QuantizerOutput:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        expected_dim = self.num_blocks * self.block_dim
        if z.size(-1) != expected_dim:
            raise ValueError(f"Expected latent dim {expected_dim}, got {z.size(-1)}")
        batch = z.size(0)
        z_blocks = z.view(batch, self.num_blocks, self.block_dim)
        codebook = self.codebook
        dists = (z_blocks[:, :, None, :] - codebook[None, :, :, :]).pow(2).sum(-1)
        idx = dists.argmin(dim=-1)
        codebook_exp = codebook.unsqueeze(0).expand(batch, -1, -1, -1)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(batch, self.num_blocks, 1, self.block_dim)
        z_q = torch.gather(codebook_exp, 2, idx_exp).squeeze(2)
        z_hat = z_blocks + (z_q - z_blocks).detach()
        z_hat = z_hat.reshape(batch, -1)
        return QuantizerOutput(symbols=idx, z_hat=z_hat)

    def codebook_usage(self, symbols: torch.Tensor) -> torch.Tensor:
        """Return per-block usage counts for monitoring."""
        if symbols.dim() != 2:
            raise ValueError("symbols should have shape (batch, num_blocks)")
        counts = []
        for block in range(self.num_blocks):
            counts.append(torch.bincount(symbols[:, block], minlength=self.codebook_size))
        return torch.stack(counts, dim=0)

    def dequantize(self, symbols: torch.Tensor) -> torch.Tensor:
        if symbols.dim() != 2:
            raise ValueError("symbols should have shape (batch, num_blocks)")
        if symbols.size(-1) != self.num_blocks:
            raise ValueError(f"Expected num_blocks={self.num_blocks}, got {symbols.size(-1)}")
        batch = symbols.size(0)
        codebook = self.codebook
        idx = symbols.long().clamp(0, self.codebook_size - 1)
        codebook_exp = codebook.unsqueeze(0).expand(batch, -1, -1, -1)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(batch, self.num_blocks, 1, self.block_dim)
        z_q = torch.gather(codebook_exp, 2, idx_exp).squeeze(2)
        return z_q.reshape(batch, -1)

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        return self.quantize(z)
