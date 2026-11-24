# rlvr-grpo-from-scratch

Reinforcement learning from verifiable rewards (RLVR), implemented from scratch in PyTorch. The pipeline is SFT, then GRPO / Dr. GRPO / RLOO / online DPO, on a tiny character-level GPT learning 2-digit addition. Every comparison is controlled and multi-seed, and runs on a CPU.

The question is the one behind recent debate about RLVR: **does RL teach the model new solutions, or only make it more reliable at ones it can already sample?**

## What's implemented

| Piece | Where | Notes |
|---|---|---|
| Verifiable task and reward | `task.py` | n-digit addition; reward is 1 for an exact answer, 0 otherwise |
| Tiny GPT and sampling | `model.py` | 3 layers, d = 128 |
| SFT | `sft.py` | Teacher-forced cross-entropy on gold answers |
| GRPO | `grpo.py` | Group-normalised advantages (G = 8), PPO-style clipped ratio; `seq_mean` / `token_mean` / `constant` loss aggregation; optional k3 KL to the SFT reference |
| Dr. GRPO | `grpo.py` | No std normalisation; constant-length aggregation (Liu et al., 2025) |
| RLOO | `grpo.py` | Leave-one-out baseline |
| Online DPO | `dpo.py` | Pairs built on-policy from verifier-labelled samples (correct vs incorrect) |
| Evaluation | `evaluate.py` | Greedy accuracy plus the **unbiased pass@k** estimator (Chen et al., 2021) from 16 samples per problem |
| **Supervised control** | `cli.py` (`sft_more`) | Continued SFT with gold labels for the **same number of optimizer steps** as RL |

## Results

Each RL method starts from the same SFT checkpoint per seed and runs 400 RL steps; there are 3 seeds per regime. To aggregate:

```
python scripts/aggregate.py results/addition2_sft250_lr1e-4_seed*.json
```

### Regime A: moderate base (250 SFT steps), RL learning rate 1e-4

| method | greedy acc (mean ± sd) | Δ greedy vs base, per seed | pass@1 | pass@16 |
|---|---|---|---|---|
| SFT base | 29.2 ± 18.0 | — | 18.0 | 77.9 |
| GRPO | 50.3 ± 31.1 | +36.2 / +25.7 / +1.4 | 35.6 | 76.4 |
| GRPO + KL | 52.5 ± 31.3 | +37.9 / +29.5 / +2.4 | 36.4 | 82.2 |
| Dr. GRPO | 48.5 ± 33.9 | +39.4 / +19.7 / −1.3 | 36.3 | 72.1 |
| RLOO | 51.0 ± 34.1 | +39.5 / +28.7 / −2.8 | 37.5 | 69.7 |
| online DPO | 50.5 ± 29.7 | +34.6 / +25.9 / +3.3 | 34.5 | 83.6 |
| **continued SFT (control)** | **76.3 ± 23.5** | +45.0 / +65.4 / +30.7 | **57.7** | **97.9** |

### Regime B: weak base (150 SFT steps), RL learning rate 3e-4

| method | greedy acc | Δ greedy vs base, per seed | pass@1 | pass@16 |
|---|---|---|---|---|
| SFT base | 8.1 ± 1.5 | — | 5.2 | 50.0 |
| GRPO | 7.0 ± 5.1 | +7.8 / −5.1 / −6.2 | 5.1 | 31.0 |
| GRPO (seq-mean) | 5.5 ± 1.1 | +0.0 / −4.1 / −3.8 | 4.4 | 32.0 |
| GRPO + KL | 19.6 ± 13.7 | +32.6 / +3.6 / −1.9 | 13.6 | 65.4 |
| Dr. GRPO | 2.8 ± 1.4 | −1.7 / −6.8 / −7.6 | 2.7 | 12.1 |
| Dr. GRPO, 3× LR | 0.2 ± 0.3 | −6.3 / −7.5 / −10.0 | 0.4 | 0.8 |
| RLOO | 1.8 ± 0.4 | −5.1 / −6.1 / −7.8 | 1.6 | 8.6 |
| online DPO | 4.0 ± 1.0 | −1.9 / −3.1 / −7.3 | 2.9 | 24.0 |
| **continued SFT (control)** | **58.3 ± 30.0** | +92.2 / +42.0 / +16.4 | **43.7** | **90.7** |

## Findings

1. **RL sharpens; it doesn't expand.**
   - In regime A, every RL method roughly doubles pass@1 (18 → 34–38%).
   - pass@16 barely moves (77.9% → 69.7–83.6%): GRPO + KL and DPO edge up, while RLOO and Dr. GRPO fall.
   - RL converts "sometimes right" into "usually right", but it does not make new problems solvable.
2. **RL's gain tracks the base model's coverage.**
   - In regime A, the per-seed gains line up with the SFT base's pass@16:
     - seed 0 (pass@16 = 93.7%): +35 to +39 points;
     - seed 1 (72.3%): +20 to +30 points;
     - seed 2 (67.7%): about zero.
   - Outcome-reward RL needs correct samples in each group to learn from. When they are rare, the group advantages are mostly zero.
3. **With a weak base and a high learning rate, RL collapses diversity.**
   - In regime B, most RL variants *lose* accuracy, and pass@16 falls from 50% to 1–32%.
   - Tripling the learning rate drives Dr. GRPO to about 0.
   - The KL anchor to the SFT reference is the only variant that helped on average (pass@16 65.4%), mostly on one seed. It acts as the trust region that a single inner epoch does not provide.
4. **The supervised control wins every time.**
   - Continued SFT with gold labels for the same number of steps beats every RL method in both regimes, on all three metrics.
   - RLVR earns its place when labels are unavailable or the verifier is cheaper than demonstrations. When gold answers exist and the base is weak, imitation is the stronger use of compute.
5. **Seed variance is huge.** Base accuracy ranges from 13% to 54% across seeds of the *same* recipe. A single-seed comparison here could support almost any conclusion, which is why every number above is a 3-seed result with per-seed deltas.

All runs are on CPU with a tiny model. These are mechanism-level observations about the algorithms, not claims about LLM-scale RLVR. They do agree with recent reports that RLVR mostly improves pass@1 within the base model's pass@k.

## Reproduce

```
pip install -e ".[dev]"
rlvr --methods grpo grpo_kl dr_grpo rloo dpo sft_more --seeds 0 --sft-steps 250 --lr 1e-4 --tag sft250_lr1e-4_seed0
pytest -q
```
