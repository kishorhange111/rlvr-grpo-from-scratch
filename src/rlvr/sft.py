"""Supervised fine-tuning (teacher forcing) on (prompt, answer) pairs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT
from .task import PAD_ID, AdditionTask


def sft_loss(model: GPT, prompts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on answer tokens only (prompt tokens and PAD are not scored)."""
    seq = torch.cat([prompts, targets], dim=1)
    logits = model(seq[:, :-1])[:, prompts.shape[1] - 1 :, :]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD_ID
    )


def train_sft(
    model: GPT,
    task: AdditionTask,
    steps: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    rng: np.random.Generator | None = None,
    opt: torch.optim.Optimizer | None = None,
) -> list[float]:
    rng = rng or np.random.default_rng(0)
    opt = opt or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for _ in range(steps):
        a, b = task.sample(batch_size, rng)
        loss = sft_loss(model, task.prompt_ids(a, b), task.target_ids(a, b))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses
