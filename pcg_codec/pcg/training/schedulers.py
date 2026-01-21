"""Learning rate schedulers."""

from __future__ import annotations

import math


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> None:
        if warmup_steps < 0 or total_steps <= 0:
            raise ValueError("warmup_steps must be >= 0 and total_steps must be > 0")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1
        for idx, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx]
            if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * self.step_num / self.warmup_steps
            else:
                progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            group["lr"] = lr
