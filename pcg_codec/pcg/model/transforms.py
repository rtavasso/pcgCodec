"""Latent transforms for PCG-Codec."""

from __future__ import annotations

import torch
from torch import nn


class IdentityTransform(nn.Module):
    """Identity mapping used for ablations."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def regularizer(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters(), torch.zeros(1)).device)

    def offdiag_covariance(self, z: torch.Tensor, num_blocks: int) -> torch.Tensor:
        return torch.tensor(0.0, device=z.device)


class MixingTransform(nn.Module):
    """Learned linear mixing with orthogonality regularization."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight
        inv_weight = torch.inverse(weight)
        return torch.matmul(z, inv_weight.t())

    def regularizer(self) -> torch.Tensor:
        weight = self.linear.weight
        ident = torch.eye(weight.size(0), device=weight.device, dtype=weight.dtype)
        gram = weight @ weight.t()
        return (gram - ident).pow(2).mean()

    def offdiag_covariance(self, z: torch.Tensor, num_blocks: int) -> torch.Tensor:
        if z.dim() > 2:
            z = z.reshape(-1, z.size(-1))
        if z.size(-1) % num_blocks != 0:
            raise ValueError("latent dim must be divisible by num_blocks")
        block_dim = z.size(-1) // num_blocks
        z_blocks = z.view(-1, num_blocks, block_dim)
        z_blocks = z_blocks - z_blocks.mean(dim=0, keepdim=True)
        block_means = z_blocks.mean(dim=2)
        samples = block_means.size(0)
        if samples < 2:
            return torch.tensor(0.0, device=z.device)
        cov = (block_means.t() @ block_means) / (samples - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).mean()
