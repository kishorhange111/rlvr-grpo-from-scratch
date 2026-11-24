"""Controlled experiments: one shared SFT checkpoint per seed, then each post-training method.

Every method starts from the *same* SFT weights for a given seed and gets the
same number of optimizer steps, so differences are attributable to the method.

Example::

    rlvr --methods grpo dr_grpo rloo dpo sft_more --seeds 0 1 2 --out results
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dpo import DPOConfig, train_dpo
from .evaluate import greedy_accuracy, sampled_pass_at_k
from .grpo import GRPOConfig, train_grpo
from .model import GPT, GPTConfig
from .sft import train_sft
from .task import VOCAB, AdditionTask

METHODS: dict[str, dict] = {
    "grpo": {"kind": "grpo", "advantage": "grpo", "aggregation": "token_mean"},
    "grpo_seqmean": {"kind": "grpo", "advantage": "grpo", "aggregation": "seq_mean"},
    "grpo_kl": {"kind": "grpo", "advantage": "grpo", "aggregation": "token_mean", "beta_kl": 0.04},
    "dr_grpo": {"kind": "grpo", "advantage": "dr_grpo", "aggregation": "constant"},
    "dr_grpo_lr3x": {"kind": "grpo", "advantage": "dr_grpo", "aggregation": "constant", "lr": 9e-4},
    "rloo": {"kind": "grpo", "advantage": "rloo", "aggregation": "token_mean"},
    "dpo": {"kind": "dpo"},
    "sft_more": {"kind": "sft"},  # supervised control: uses gold labels, same optimizer steps
}


def build_model(seed: int, n_layer: int, d_model: int) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=len(VOCAB), block_size=16, n_layer=n_layer, d_model=d_model))


def evaluate_all(model: GPT, task: AdditionTask) -> dict:
    p = sampled_pass_at_k(model, task, ks=(1, 4, 16), n_samples=16, n_problems=300)
    return {
        "greedy_acc": greedy_accuracy(model, task, 1000),
        **{f"pass@{k}": v for k, v in p.items()},
    }


def run_method(name: str, base: GPT, task: AdditionTask, seed: int, steps: int, lr: float) -> dict:
    spec = dict(METHODS[name])
    kind = spec.pop("kind")
    model = copy.deepcopy(base)
    t0 = time.perf_counter()
    curve: dict[str, list[float]] = {}
    if kind == "grpo":
        cfg = GRPOConfig(steps=steps, prompts_per_step=32, group_size=8, seed=seed, lr=lr)
        for k, v in spec.items():
            setattr(cfg, k, v)
        log = train_grpo(model, task, cfg, rng=np.random.default_rng(1000 + seed))
        curve = {"reward": log.reward, "zero_var_frac": log.zero_var_frac, "kl": log.kl}
    elif kind == "dpo":
        cfg = DPOConfig(steps=steps, prompts_per_step=32, group_size=8, seed=seed, lr=lr)
        log = train_dpo(model, task, cfg, rng=np.random.default_rng(1000 + seed))
        curve = {"reward": log.reward, "pairs": [float(p) for p in log.pairs]}
    else:
        losses = train_sft(
            model, task, steps, batch_size=256, lr=lr, rng=np.random.default_rng(1000 + seed)
        )
        curve = {"sft_loss": losses}
    return {
        "method": name,
        "seed": seed,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        **evaluate_all(model, task),
        "curve": curve,
    }


def summarize(runs: list[dict], metrics=("greedy_acc", "pass@1", "pass@16")) -> list[dict]:
    rows = []
    for m in dict.fromkeys(r["method"] for r in runs):
        rs = [r for r in runs if r["method"] == m]
        row = {"method": m, "n_seeds": len(rs)}
        for k in metrics:
            vals = np.array([r[k] for r in rs])
            row[k] = float(vals.mean())
            row[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    out = ["| method | seeds | greedy acc | pass@1 | pass@16 |", "|---|---|---|---|---|"]
    for r in rows:
        cells = [
            f"{100 * r[k]:.1f} ± {100 * r[k + '_std']:.1f}"
            for k in ("greedy_acc", "pass@1", "pass@16")
        ]
        out.append(f"| {r['method']} | {r['n_seeds']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rlvr")
    p.add_argument("--methods", nargs="+", default=["grpo", "dr_grpo", "rloo", "dpo", "sft_more"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--sft-steps", type=int, default=150)
    p.add_argument("--rl-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    for m in args.methods:
        if m not in METHODS:
            p.error(f"unknown method {m!r}; choose from {sorted(METHODS)}")

    torch.set_num_threads(args.threads)
    task = AdditionTask(args.digits)
    runs = []
    for seed in args.seeds:
        base = build_model(seed, args.n_layer, args.d_model)
        train_sft(base, task, args.sft_steps, batch_size=128, rng=np.random.default_rng(seed))
        runs.append({"method": "sft_base", "seed": seed, **evaluate_all(base, task), "curve": {}})
        print(f"seed={seed} sft_base greedy={runs[-1]['greedy_acc']:.3f}", flush=True)
        for m in args.methods:
            r = run_method(m, base, task, seed, args.rl_steps, args.lr)
            print(
                f"seed={seed} {m} greedy={r['greedy_acc']:.3f} ({r['wall_seconds']}s)", flush=True
            )
            runs.append(r)

    rows = summarize(runs)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta["torch"] = torch.__version__
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"addition{args.digits}{('_' + args.tag) if args.tag else ''}"
    (args.out / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": rows, "runs": runs}, indent=1)
    )
    (args.out / f"{stem}.md").write_text(to_markdown(rows) + "\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
