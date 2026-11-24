"""Evaluation: greedy accuracy and unbiased pass@k on a fixed held-out problem set."""

from __future__ import annotations

from math import comb

import numpy as np
import torch

from .model import GPT, generate
from .task import AdditionTask

EVAL_SEED = 12345  # disjoint generator stream from all training seeds


def eval_problems(task: AdditionTask, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    return task.sample(n, np.random.default_rng(EVAL_SEED))


@torch.no_grad()
def greedy_accuracy(model: GPT, task: AdditionTask, n: int = 1000) -> float:
    a, b = eval_problems(task, n)
    comps = generate(model, task.prompt_ids(a, b), task.max_answer_len, greedy=True)
    return task.rewards(comps, a, b).mean().item()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021, Codex)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


@torch.no_grad()
def sampled_pass_at_k(
    model: GPT,
    task: AdditionTask,
    ks: tuple[int, ...] = (1, 4, 16),
    n_samples: int = 16,
    n_problems: int = 300,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[int, float]:
    a, b = eval_problems(task, n_problems)
    a, b = np.repeat(a, n_samples), np.repeat(b, n_samples)
    gen = torch.Generator().manual_seed(seed)
    comps = generate(model, task.prompt_ids(a, b), task.max_answer_len, temperature, generator=gen)
    correct = task.rewards(comps, a, b).view(n_problems, n_samples).sum(1).long().tolist()
    return {k: float(np.mean([pass_at_k(n_samples, c, k) for c in correct])) for k in ks}
