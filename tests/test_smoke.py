"""Smoke tests for SOAR components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config():
    from soar_llm.config import SOARConfig
    cfg = SOARConfig()
    assert cfg.num_layers == 12
    assert cfg.early_exit_layers == [4, 8, 12]


def test_plastic_adapter():
    import torch
    from soar_llm.ttt.adapter import PlasticAdapter
    a = PlasticAdapter(768, 32)
    x = torch.randn(2, 10, 768)
    y = a(x)
    assert y.shape == x.shape


def test_ttt_router():
    from soar_llm.config import SOARConfig
    from soar_llm.ttt.adapter import TTTRouter
    cfg = SOARConfig()
    r = TTTRouter(cfg)
    assert len(r.adapters_attn) == 12


def test_shampoo_lite():
    import torch
    from soar_llm.ttt.shampoo_lite import ShampooLite
    p = torch.nn.Parameter(torch.randn(10, 10))
    opt = ShampooLite([p], lr=0.01)
    p.grad = torch.randn_like(p)
    opt.step()


def test_tool_parse():
    from soar_llm.agent.tool_tokens import parse_tool_calls
    text = 'Use <SEARCH>{"query": "weather"} to find.'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0][0] == "SEARCH"
    assert calls[0][1]["query"] == "weather"


def test_rlvr_penalty():
    import torch
    from soar_llm.continual.rlvr import RLVR

    model = torch.nn.Linear(4, 2, bias=False)
    rlvr = RLVR(lambda_=10.0)
    rlvr.snapshot_reference(model)
    with torch.no_grad():
        model.weight.add_(0.1)
    p0 = rlvr.penalty(model, reward=0.0)
    p1 = rlvr.penalty(model, reward=1.0)
    assert p0.item() > 0
    assert p1.item() == 0


def test_grpo_penalty_and_reward_normalization():
    import torch
    from soar_llm.continual.grpo import GRPO

    model = torch.nn.Linear(4, 2, bias=False)
    grpo = GRPO(lambda_=10.0, group_size=2)
    grpo.snapshot_reference(model)
    with torch.no_grad():
        model.weight.add_(0.1)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = grpo.normalize_rewards(rewards)
    penalty = grpo.penalty(model, rewards=rewards)
    assert normalized.shape == rewards.shape
    assert torch.isfinite(normalized).all()
    assert penalty.item() > 0


def test_tokenizer():
    from soar_llm.tokenizer import get_soar_tokenizer, SPECIAL_TOKENS
    tok, n = get_soar_tokenizer("gpt2")
    assert n == len(SPECIAL_TOKENS)
    enc = tok.encode("hello world")
    assert len(enc) >= 2


def test_model_forward():
    import torch
    from soar_llm.config import SOARConfig
    from soar_llm.model import SOARModel
    cfg = SOARConfig()
    model = SOARModel(cfg)
    x = torch.randint(0, 50257, (1, 8))
    out = model.forward(x)
    assert "logits" in out
    assert out["logits"].shape[-1] == 50257
