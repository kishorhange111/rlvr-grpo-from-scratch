"""Aggregate per-seed result files into a cross-seed table (mean ± std, and paired gain vs base).

python scripts/aggregate.py results/addition2_sft250_lr1e-4_seed*.json
"""

import json
import sys

import numpy as np

METRICS = ("greedy_acc", "pass@1", "pass@16")


def main(paths: list[str]) -> None:
    per: dict[str, dict[str, list[float]]] = {}
    for p in sorted(paths):
        for row in json.load(open(p))["summary"]:
            d = per.setdefault(row["method"], {m: [] for m in METRICS})
            for m in METRICS:
                d[m].append(row[m])
    base = per["sft_base"]
    print("| method | seeds | greedy acc | Δ greedy vs base (per seed) | pass@1 | pass@16 |")
    print("|---|---|---|---|---|---|")
    for meth, d in per.items():
        g = np.array(d["greedy_acc"]) * 100
        delta = g - np.array(base["greedy_acc"]) * 100
        cells = [
            meth,
            str(len(g)),
            f"{g.mean():.1f} ± {g.std():.1f}",
            "—" if meth == "sft_base" else " / ".join(f"{x:+.1f}" for x in delta),
            f"{100 * np.mean(d['pass@1']):.1f}",
            f"{100 * np.mean(d['pass@16']):.1f}",
        ]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
