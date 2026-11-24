"""Group-relative policy optimisation and its variants.

One loss function covers the family, selected by two orthogonal switches:

``advantage``
    * ``"grpo"`` — ``(r - mean_g) / (std_g + eps)`` (Shao et al., 2024, DeepSeekMath)
    * ``"dr_grpo"`` — ``r - mean_g``; drops the std term that over-weights
      near-solved / near-impossible prompts (Liu et al., 2025, "Understanding R1-Zero-Like
      Training")
    * ``"rloo"`` — leave-one-out baseline ``r_i - mean_{j!=i} r_j`` (Ahmadian et al., 2024)

``aggregation``
    * ``"seq_mean"`` — mean over each sequence's tokens, then over sequences (original GRPO;
      introduces a length bias)
    * ``"token_mean"`` — mean over all valid tokens in the batch (DAPO, Yu et al., 2025)
    * ``"constant"`` — sum over tokens / (B * L_max) (Dr. GRPO; length-unbiased)

The KL penalty to a frozen reference uses the k3 estimator
``exp(ref - logp) - (ref - logp) - 1`` (unbiased, always >= 0).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import GPT, completion_logprobs, generate
from .task import AdditionTask, completion_mask


@dataclass
class GRPOConfig:
    steps: int = 200
    prompts_per_step: int = 32
    group_size: int = 8
    temperature: float = 1.0
    lr: float = 3e-4
    clip_eps: float = 0.2
    beta_kl: float = 0.0
    inner_epochs: int = 1
    advantage: str = "grpo"  # grpo | dr_grpo | rloo
    aggregation: str = "seq_mean"  # seq_mean | token_mean | constant
    max_grad_norm: float = 1.0
    seed: int = 0


@dataclass
class GRPOLog:
    reward: list[float] = field(default_factory=list)
    zero_var_frac: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    clip_frac: list[float] = field(default_factory=list)
    resp_len: list[float] = field(default_factory=list)


def group_advantages(rewards: torch.Tensor, group_size: int, kind: str) -> torch.Tensor:
    """Rewards ``[B*G]`` ordered group-contiguously -> per-sample advantages ``[B*G]``."""
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    if kind == "grpo":
        adv = (r - mean) / (r.std(dim=1, keepdim=True) + 1e-4)
    elif kind == "dr_grpo":
        adv = r - mean
    elif kind == "rloo":
        adv = (r - mean) * group_size / (group_size - 1)
    else:
        raise ValueError(f"unknown advantage kind: {kind}")
    return adv.reshape(-1)


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    if how == "seq_mean":
        return ((per_token * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    if how == "token_mean":
        return (per_token * mask).sum() / mask.sum().clamp_min(1)
    if how == "constant":
        return (per_token * mask).sum() / mask.numel()
    raise ValueError(f"unknown aggregation: {how}")


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor | None,
    adv: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    beta_kl: float,
    aggregation: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate per token with sequence-level advantages broadcast over tokens."""
    ratio = (logp - old_logp).exp()
    a = adv[:, None]
    surr = torch.min(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)
    per_token = -surr
    kl_val = 0.0
    if ref_logp is not None and beta_kl > 0:
        d = ref_logp - logp
        kl = d.exp() - d - 1.0
        per_token = per_token + beta_kl * kl
        kl_val = ((kl * mask).sum() / mask.sum()).item()
    loss = aggregate(per_token, mask, aggregation)
    with torch.no_grad():
        clipped = ((ratio - 1).abs() > clip_eps).float()
        cf = ((clipped * mask).sum() / mask.sum()).item()
    return loss, {"kl": kl_val, "clip_frac": cf}


def train_grpo(
    model: GPT, task: AdditionTask, cfg: GRPOConfig, rng: np.random.Generator | None = None
) -> GRPOLog:
    rng = rng or np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ref = copy.deepcopy(model).eval().requires_grad_(False) if cfg.beta_kl > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    log = GRPOLog()
    G = cfg.group_size
    for _ in range(cfg.steps):
        a, b = task.sample(cfg.prompts_per_step, rng)
        a, b = np.repeat(a, G), np.repeat(b, G)  # group-contiguous
        prompts = task.prompt_ids(a, b)
        comps = generate(model, prompts, task.max_answer_len, cfg.temperature, generator=gen)
        mask = completion_mask(comps)
        rewards = task.rewards(comps, a, b)
        adv = group_advantages(rewards, G, cfg.advantage)
        with torch.no_grad():
            old_logp = completion_logprobs(model, prompts, comps)
            ref_logp = completion_logprobs(ref, prompts, comps) if ref is not None else None
        model.train()
        for _ in range(cfg.inner_epochs):
            logp = completion_logprobs(model, prompts, comps)
            loss, stats = grpo_loss(
                logp, old_logp, ref_logp, adv, mask, cfg.clip_eps, cfg.beta_kl, cfg.aggregation
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
        r = rewards.view(-1, G)
        log.reward.append(rewards.mean().item())
        log.zero_var_frac.append((r.std(dim=1) == 0).float().mean().item())
        log.kl.append(stats["kl"])
        log.clip_frac.append(stats["clip_frac"])
        log.resp_len.append(mask.sum(1).mean().item())
    return log
