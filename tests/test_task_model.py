import numpy as np
import torch

from rlvr.model import GPT, GPTConfig, completion_logprobs, generate
from rlvr.task import EOS_ID, PAD_ID, VOCAB, AdditionTask, completion_mask, decode, encode


def test_encode_decode_roundtrip():
    assert decode(encode("123+456=")) == "123+456="
    assert decode([*encode("42"), EOS_ID, *encode("99")]) == "42"


def test_prompt_and_target_shapes():
    task = AdditionTask(n_digits=3)
    a, b = np.array([7, 999]), np.array([5, 999])
    p = task.prompt_ids(a, b)
    t = task.target_ids(a, b)
    assert p.shape == (2, task.prompt_len)
    assert t.shape == (2, task.max_answer_len)
    assert decode(p[0]) == "007+005="
    assert decode(t[0]) == "12" and t[0, 2] == EOS_ID and t[0, 3] == PAD_ID
    assert decode(t[1]) == "1998"


def test_reward_requires_exact_answer_and_eos():
    task = AdditionTask(2)
    assert task.reward([*encode("68"), EOS_ID], 23, 45) == 1.0
    assert task.reward([*encode("68")], 23, 45) == 0.0  # no EOS
    assert task.reward([*encode("068"), EOS_ID], 23, 45) == 0.0
    assert task.reward([*encode("67"), EOS_ID], 23, 45) == 0.0


def test_completion_mask_includes_first_eos_only():
    c = torch.tensor([[3, 4, EOS_ID, 5, EOS_ID], [3, 3, 3, 3, 3]])
    m = completion_mask(c)
    assert m.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_completion_logprobs_match_manual_forward():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(VOCAB), block_size=16, n_layer=1, d_model=32))
    prompts = torch.randint(0, len(VOCAB), (2, 5))
    comps = generate(model, prompts, 3)
    lp = completion_logprobs(model, prompts, comps)
    assert lp.shape == (2, 3)
    # manual: log-prob of the 2nd completion token
    seq = torch.cat([prompts, comps], 1)
    logits = model(seq)[:, 5, :]
    manual = torch.log_softmax(logits, -1).gather(-1, comps[:, 1:2]).squeeze(-1)
    torch.testing.assert_close(lp[:, 1], manual)


def test_greedy_generation_is_deterministic():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(VOCAB), block_size=16, n_layer=1, d_model=32))
    p = torch.randint(0, len(VOCAB), (3, 4))
    assert torch.equal(generate(model, p, 4, greedy=True), generate(model, p, 4, greedy=True))
