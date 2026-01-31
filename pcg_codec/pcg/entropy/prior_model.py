"""Layered causal prior for entropy modeling."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class LayeredCausalPrior(nn.Module):
    """Causal prior that predicts symbols layer-wise given past frames and parents."""

    def __init__(
        self,
        num_blocks: int,
        codebook_size: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        self.num_blocks = int(num_blocks)
        self.codebook_size = int(codebook_size)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.token_embed = nn.Embedding(self.codebook_size, self.embed_dim)
        self.block_embed = nn.Embedding(self.num_blocks, self.hidden_dim)
        self.frame_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.parent_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.codebook_size)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def update_state(self, state: torch.Tensor, q_frame: torch.Tensor) -> torch.Tensor:
        """Update the recurrent state with a full frame of tokens."""
        frame_embed = self.token_embed(q_frame).mean(dim=1)
        frame_feat = self.frame_proj(frame_embed)
        return self.rnn(frame_feat, state)

    def predict_layer(
        self,
        state: torch.Tensor,
        block_ids: Iterable[int] | torch.Tensor,
        parent_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict logits for a layer given prior state and parent tokens."""
        if not torch.is_tensor(block_ids):
            block_ids = torch.tensor(list(block_ids), device=state.device, dtype=torch.long)
        if parent_tokens is None:
            parent_tokens_by_block = None
        else:
            parent_tokens_by_block = parent_tokens

        logits = []
        for i, bid in enumerate(block_ids.tolist()):
            pt = None
            if parent_tokens_by_block is not None:
                if parent_tokens_by_block.dim() == 2:
                    pt = parent_tokens_by_block
                elif parent_tokens_by_block.dim() == 3:
                    pt = parent_tokens_by_block[:, i, :]
                else:
                    raise ValueError("parent_tokens must be shape (batch, parents) or (batch, blocks, parents)")
            logits.append(self.predict_block(state, int(bid), pt))
        return torch.stack(logits, dim=1)

    def predict_block(
        self, state: torch.Tensor, block_id: int, parent_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict logits for a single block (batch, K) given past state and its parent tokens."""
        block_id = int(block_id)
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError("block_id out of range")
        parent_context = torch.zeros_like(state)
        if parent_tokens is not None and parent_tokens.numel() > 0:
            parent_embed = self.token_embed(parent_tokens.long().clamp(0, self.codebook_size - 1)).mean(dim=1)
            parent_context = self.parent_proj(parent_embed)
        context = torch.tanh(state + parent_context + self.block_embed(torch.tensor(block_id, device=state.device)))
        return self.out(context)

    def predict_all_blocks(
        self,
        state: torch.Tensor,
        parent_tokens_by_block: Optional[torch.Tensor] = None,
        parent_mask: Optional[torch.Tensor] = None,
        block_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict logits for all blocks in parallel.

        Args:
            state: (batch, hidden_dim)
            parent_tokens_by_block: Optional (batch, num_blocks, parents_k) teacher-forced parent tokens.
            parent_mask: Optional (num_blocks, parents_k) boolean mask for valid parent positions.
        Returns:
            logits: (batch, num_blocks, codebook_size)
        """
        if state.dim() != 2:
            raise ValueError("state must have shape (batch, hidden_dim)")
        if block_context is None:
            block_ids = torch.arange(self.num_blocks, device=state.device, dtype=torch.long)
            block_context = self.block_embed(block_ids).unsqueeze(0)  # (1, num_blocks, hidden_dim)
        else:
            if block_context.dim() != 3 or block_context.size(1) != self.num_blocks or block_context.size(2) != self.hidden_dim:
                raise ValueError("block_context must have shape (1, num_blocks, hidden_dim) or (batch, num_blocks, hidden_dim)")
        parent_context = torch.zeros(
            state.size(0), self.num_blocks, self.hidden_dim, device=state.device, dtype=state.dtype
        )

        if parent_tokens_by_block is not None and parent_tokens_by_block.numel() > 0:
            if parent_tokens_by_block.dim() != 3 or parent_tokens_by_block.size(1) != self.num_blocks:
                raise ValueError("parent_tokens_by_block must have shape (batch, num_blocks, parents_k)")
            embed = self.token_embed(parent_tokens_by_block.long().clamp(0, self.codebook_size - 1))
            if parent_mask is not None:
                if parent_mask.dim() != 2 or parent_mask.size(0) != self.num_blocks:
                    raise ValueError("parent_mask must have shape (num_blocks, parents_k)")
                mask = parent_mask.to(device=embed.device).unsqueeze(0).unsqueeze(-1)  # (1, num_blocks, parents_k, 1)
                embed = embed * mask
                denom = mask.sum(dim=2).clamp_min(1.0)
                parent_embed = embed.sum(dim=2) / denom
            else:
                parent_embed = embed.mean(dim=2)
            parent_context = self.parent_proj(parent_embed)

        context = torch.tanh(state.unsqueeze(1) + parent_context + block_context)
        return self.out(context)

    def forward(
        self,
        q_frames: torch.Tensor,
        layer_blocks: Iterable[int],
        parent_blocks: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Predict logits for each timestep for the given layer blocks."""
        if q_frames.dim() != 3:
            raise ValueError("Expected q_frames with shape (batch, time, num_blocks)")
        batch, time, _ = q_frames.shape
        state = self.init_state(batch, q_frames.device)
        layer_blocks = list(layer_blocks)
        parent_blocks = list(parent_blocks) if parent_blocks is not None else []
        logits_out = []
        for t in range(time):
            parent_tokens = q_frames[:, t, parent_blocks] if parent_blocks else None
            logits_t = self.predict_layer(state, layer_blocks, parent_tokens)
            logits_out.append(logits_t)
            state = self.update_state(state, q_frames[:, t, :])
        return torch.stack(logits_out, dim=1)
