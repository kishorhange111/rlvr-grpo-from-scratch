import numpy as np
import pytest
import torch

from rlvr.dpo import DPOConfig, build_pairs, dpo_loss, train_dpo
from rlvr.evaluate import greedy_accuracy, pass_at_k
from rlvr.grpo import GRPOConfig, aggregate, group_advantages, grpo_loss, train_grpo
from rlvr.model import GPT, GPTConfig
from rlvr.sft import train_sft
from rlvr.task import VOCAB, AdditionTask


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=len(VOCAB), block_size=16, n_layer=2, d_model=64))


# ---------------------------------------------------------------- advantages
def test_grpo_advantage_is_standardised_within_group():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = group_advantages(r, 4, "grpo").view(2, 4)
    assert adv[0].mean().abs() < 1e-6
    assert adv[0, 0] > 0 > adv[0, 1]
    assert torch.all(adv[1] == 0)  # zero-variance group -> no signal


def test_dr_grpo_advantage_has_no_std_scaling():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(group_advantages(r, 4, "dr_grpo"), r - 0.25)


def test_rloo_matches_leave_one_out_definition():
    r = torch.tensor([3.0, 1.0, 2.0, 6.0])
    adv = group_advantages(r, 4, "rloo")
    loo = torch.stack([r[i] - torch.cat([r[:i], r[i + 1 :]]).mean() for i in range(4)])
    torch.testing.assert_close(adv, loo)


def test_unknown_advantage_kind_raises():
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(4), 2, "ppo")


# --------------------------------------------------------------- aggregation
def test_aggregations_differ_on_unequal_lengths():
    x = torch.ones(2, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    assert aggregate(x, mask, "seq_mean").item() == pytest.approx(1.0)
    assert aggregate(x, mask, "token_mean").item() == pytest.approx(1.0)
    assert aggregate(x, mask, "constant").item() == pytest.approx(5 / 8)
    # seq_mean weights the short sequence's single token 4x more than token_mean does
    y = torch.tensor([[0.0, 0, 0, 0], [8.0, 0, 0, 0]])
    assert aggregate(y, mask, "seq_mean").item() == pytest.approx(4.0)
    assert aggregate(y, mask, "token_mean").item() == pytest.approx(8 / 5)


# ---------------------------------------------------------------------- loss
def test_grpo_loss_on_policy_equals_negative_mean_advantage():
    logp = torch.randn(4, 3)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(4, 3)
    loss, s = grpo_loss(logp, logp, None, adv, mask, 0.2, 0.0, "token_mean")
    assert loss.item() == pytest.approx(-adv.mean().item(), abs=1e-6)
    assert s["clip_frac"] == 0.0


def test_k3_kl_is_zero_at_reference_and_positive_elsewhere():
    logp = torch.zeros(2, 3, requires_grad=True)
    adv, mask = torch.zeros(2), torch.ones(2, 3)
    _, s = grpo_loss(logp, logp.detach(), logp.detach(), adv, mask, 0.2, 0.1, "token_mean")
    assert s["kl"] == pytest.approx(0.0, abs=1e-7)
    _, s = grpo_loss(logp, logp.detach(), logp.detach() - 1.0, adv, mask, 0.2, 0.1, "token_mean")
    assert s["kl"] > 0


def test_dpo_loss_at_reference_is_log2():
    z = torch.zeros(5)
    loss, _ = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(np.log(2))


def test_build_pairs_skips_uniform_groups():
    r = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert build_pairs(r, 2) == [(0, 1)]


def test_pass_at_k_estimator():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)


# ----------------------------------------------------------- learning checks
def test_sft_then_grpo_improves_reward_on_easy_task():
    task = AdditionTask(n_digits=1)
    model = tiny_model()
    train_sft(model, task, steps=60, batch_size=128, rng=np.random.default_rng(0))
    log = train_grpo(
        model, task, GRPOConfig(steps=150, prompts_per_step=16, group_size=8, lr=3e-4, seed=0)
    )
    assert len(log.reward) == 150
    # On-policy sampled reward trends upward (greedy accuracy is too noisy at this budget).
    assert np.mean(log.reward[-30:]) > np.mean(log.reward[:30]) + 0.02
    assert 0.0 <= greedy_accuracy(model, task, n=100) <= 1.0


def test_dpo_runs_and_logs_pairs():
    task = AdditionTask(n_digits=1)
    model = tiny_model()
    train_sft(model, task, steps=40, batch_size=128, rng=np.random.default_rng(0))
    log = train_dpo(model, task, DPOConfig(steps=5, prompts_per_step=8, group_size=4))
    assert len(log.pairs) == 5 and sum(log.pairs) > 0
