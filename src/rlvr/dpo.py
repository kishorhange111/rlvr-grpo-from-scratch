"""Direct Preference Optimisation (Rafailov et al., 2023), run *online*.

Each step samples a group of completions per prompt from the current policy and
forms (chosen, rejected) = (a correct sample, an incorrect sample) whenever the
group contains both — i.e. preferences come from the verifier, not humans.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, completion_logprobs, generate
from .task import AdditionTask, completion_mask


@dataclass
class DPOConfig:
    steps: int = 200
    prompts_per_step: int = 32
    group_size: int = 8
    temperature: float = 1.0
    beta: float = 0.1
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    seed: int = 0


@dataclass
class DPOLog:
    reward: list[float] = field(default_factory=list)
    pairs: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    margin_acc: list[float] = field(default_factory=list)


def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequence log-probs in, (loss, implicit-reward accuracy) out."""
    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean()


def build_pairs(rewards: torch.Tensor, group_size: int) -> list[tuple[int, int]]:
    """Pick the first correct and first incorrect sample from each mixed group."""
    pairs = []
    r = rewards.view(-1, group_size)
    for g in range(r.shape[0]):
        pos = torch.nonzero(r[g] > 0.5).flatten()
        neg = torch.nonzero(r[g] <= 0.5).flatten()
        if len(pos) and len(neg):
            pairs.append((g * group_size + int(pos[0]), g * group_size + int(neg[0])))
    return pairs


def seq_logprob(model, prompts, comps, mask) -> torch.Tensor:
    return (completion_logprobs(model, prompts, comps) * mask).sum(1)


def train_dpo(
    model: GPT, task: AdditionTask, cfg: DPOConfig, rng: np.random.Generator | None = None
) -> DPOLog:
    rng = rng or np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ref = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    log = DPOLog()
    G = cfg.group_size
    for _ in range(cfg.steps):
        a, b = task.sample(cfg.prompts_per_step, rng)
        a, b = np.repeat(a, G), np.repeat(b, G)
        prompts = task.prompt_ids(a, b)
        comps = generate(model, prompts, task.max_answer_len, cfg.temperature, generator=gen)
        rewards = task.rewards(comps, a, b)
        log.reward.append(rewards.mean().item())
        pairs = build_pairs(rewards, G)
        log.pairs.append(len(pairs))
        if not pairs:
            log.loss.append(float("nan"))
            log.margin_acc.append(float("nan"))
            continue
        ci = torch.tensor([p[0] for p in pairs])
        ri = torch.tensor([p[1] for p in pairs])
        mask = completion_mask(comps)
        with torch.no_grad():
            ref_c = seq_logprob(ref, prompts[ci], comps[ci], mask[ci])
            ref_r = seq_logprob(ref, prompts[ri], comps[ri], mask[ri])
        model.train()
        pi_c = seq_logprob(model, prompts[ci], comps[ci], mask[ci])
        pi_r = seq_logprob(model, prompts[ri], comps[ri], mask[ri])
        loss, acc = dpo_loss(pi_c, pi_r, ref_c, ref_r, cfg.beta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        log.loss.append(loss.item())
        log.margin_acc.append(acc.item())
    return log
