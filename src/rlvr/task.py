"""Verifiable task: n-digit addition with a character-level tokenizer.

Prompts are fixed-length (operands zero-padded), e.g. ``"047+512="``, so a batch
of prompts needs no padding. The completion is the answer digits followed by
``<eos>``; the reward is computed by parsing the completion — the model never
sees the reward function, which is what makes it *verifiable*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

CHARS = list("0123456789+=")
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
VOCAB = [PAD, BOS, EOS, *CHARS]
STOI = {s: i for i, s in enumerate(VOCAB)}
PAD_ID, BOS_ID, EOS_ID = STOI[PAD], STOI[BOS], STOI[EOS]


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i in (PAD_ID, BOS_ID):
            continue
        out.append(VOCAB[i])
    return "".join(out)


@dataclass(frozen=True)
class AdditionTask:
    n_digits: int = 3

    @property
    def prompt_len(self) -> int:
        return 1 + 2 * self.n_digits + 2  # <bos> a + b =

    @property
    def max_answer_len(self) -> int:
        return self.n_digits + 2  # up to n+1 digits + <eos>

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        hi = 10**self.n_digits
        return rng.integers(0, hi, size=n), rng.integers(0, hi, size=n)

    def prompt(self, a: int, b: int) -> str:
        return f"{a:0{self.n_digits}d}+{b:0{self.n_digits}d}="

    def prompt_ids(self, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        rows = [[BOS_ID, *encode(self.prompt(int(x), int(y)))] for x, y in zip(a, b, strict=True)]
        return torch.tensor(rows, dtype=torch.long)

    def target_ids(self, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        """Answer tokens + EOS, right-padded with PAD to ``max_answer_len``."""
        rows = []
        for x, y in zip(a, b, strict=True):
            ids = [*encode(str(int(x) + int(y))), EOS_ID]
            rows.append(ids + [PAD_ID] * (self.max_answer_len - len(ids)))
        return torch.tensor(rows, dtype=torch.long)

    def reward(self, completion_ids, a: int, b: int) -> float:
        """1.0 iff the completion is exactly the decimal sum terminated by <eos>."""
        ids = [int(i) for i in completion_ids]
        if EOS_ID not in ids:
            return 0.0
        return 1.0 if decode(ids) == str(int(a) + int(b)) else 0.0

    def rewards(self, completions: torch.Tensor, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            [self.reward(c, x, y) for c, x, y in zip(completions, a, b, strict=True)],
            dtype=torch.float32,
        )


def completion_mask(completions: torch.Tensor) -> torch.Tensor:
    """1 for tokens up to and including the first <eos>, 0 afterwards."""
    is_eos = (completions == EOS_ID).long()
    after_eos = (is_eos.cumsum(dim=1) - is_eos) > 0
    return (~after_eos).float()
