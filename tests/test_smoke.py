"""Smoke tests for SOAR components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config():
    from soar_llm.config import SOARConfig
    cfg = SOARConfig()
    assert cfg.num_layers == 24
    assert cfg.early_exit_layers == [8, 16, 24]
    assert cfg.hidden_size == 1024
    assert cfg.model_name == "Qwen/Qwen3.5-0.8B"


def test_plastic_adapter():
    import torch
    from soar_llm.ttt.adapter import PlasticAdapter
    a = PlasticAdapter(1024, 42)
    x = torch.randn(2, 10, 1024)
    y = a(x)
    assert y.shape == x.shape


def test_ttt_router():
    from soar_llm.config import SOARConfig
    from soar_llm.ttt.adapter import TTTRouter
    cfg = SOARConfig()
    r = TTTRouter(cfg)
    assert len(r.adapters_attn) == 24


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


def test_dynamic_rl():
    import torch
    from soar_llm.continual.dynamic_rl import DynamicRL

    drl = DynamicRL()

    # Confidence reward: logits with a clear winner → high confidence
    logits = torch.zeros(1, 5, 10)
    logits[0, :, 0] = 10.0  # token 0 always wins
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    reward = drl.compute_confidence_reward(logits, input_ids)
    assert 0.0 < reward <= 1.0

    # Perplexity reward: when logits match labels perfectly, reward is high
    reward_ppl = drl.compute_perplexity_reward(logits[:, :4], input_ids[:, :4])
    assert 0.0 < reward_ppl <= 1.0

    # adapt_lr: reward=0.0 → LR = base * min_scale
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([p], lr=1e-3)
    lr_low = drl.adapt_lr(opt, reward=0.0, base_lr=1e-3, min_scale=0.1, max_scale=2.0)
    assert abs(lr_low - 1e-3 * 0.1) < 1e-9

    # reward=1.0 → LR = base * max_scale
    lr_high = drl.adapt_lr(opt, reward=1.0, base_lr=1e-3, min_scale=0.1, max_scale=2.0)
    assert abs(lr_high - 1e-3 * 2.0) < 1e-9

    # reward=0.5 → LR in (min, max)
    lr_mid = drl.adapt_lr(opt, reward=0.5, base_lr=1e-3, min_scale=0.1, max_scale=2.0)
    assert 1e-3 * 0.1 < lr_mid < 1e-3 * 2.0


def test_replay_buffer_dynamic_reward_priority():
    import torch
    from soar_llm.continual.replay_buffer import ReplayBuffer

    rb = ReplayBuffer(max_tokens=4, sample_ratio=1.0)
    rb.add(torch.tensor([[1, 2]]), loss=1.0, reward=1.0)  # low priority
    rb.add(torch.tensor([[3, 4]]), loss=1.0, reward=0.0)  # high priority
    rb.add(torch.tensor([[5, 6]]), loss=1.0, reward=0.0)  # forces eviction

    kept_first_tokens = [int(s.input_ids[0, 0].item()) for s in rb._samples]
    assert 1 not in kept_first_tokens
    assert 3 in kept_first_tokens

    # Priority combines loss and reward multiplicatively.
    rb2 = ReplayBuffer(max_tokens=2, sample_ratio=1.0)
    rb2.add(torch.tensor([[7, 8]]), loss=2.0, reward=1.0)
    rb2.add(torch.tensor([[9, 10]]), loss=0.9, reward=0.0)
    kept_first = int(rb2._samples[0].input_ids[0, 0].item())
    assert kept_first == 9


def test_ttt_continual_trainer():
    import torch
    import torch.nn as nn
    from soar_llm.config import SOARConfig
    from soar_llm.ttt.trainer import TTTContinualTrainer

    # Tiny linear model as stand-in (hooks won't attach, graceful fallback).
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 1024)
            self.proj = nn.Linear(1024, 100)

        def forward(self, input_ids):
            import types
            h = self.embed(input_ids)
            logits = self.proj(h)
            out = types.SimpleNamespace()
            out.logits = logits
            return out

    cfg = SOARConfig()
    tiny = TinyModel()
    trainer = TTTContinualTrainer(tiny, cfg, torch.device("cpu"), ttt_lr=1e-4)

    input_ids = torch.randint(0, 100, (1, 8))
    loss = trainer.train_step(input_ids, reward=0.6)
    assert isinstance(loss, float)
    assert trainer.step_count == 1

    # Reward omitted: trainer computes confidence reward and applies dynamic LR.
    loss2 = trainer.train_step(input_ids)
    assert isinstance(loss2, float)
    assert trainer.step_count == 2
    current_lr = trainer.optimizer.param_groups[0]["lr"]
    assert 1e-4 * 0.1 <= current_lr <= 1e-4 * 2.0
    assert trainer.replay_buffer._samples[-1].reward is not None

    stats = trainer.stats
    assert "step_count" in stats
    assert "running_loss" in stats
    assert "replay_tokens" in stats


def test_tokenizer():
    from soar_llm.tokenizer import get_soar_tokenizer
    tok, n = get_soar_tokenizer("Qwen/Qwen3.5-0.8B")
    assert n >= 0
    enc = tok.encode("hello world")
    assert len(enc) >= 2


def test_package_top_level_exports():
    import soar_llm

    assert hasattr(soar_llm, "SOARConfig")
    assert hasattr(soar_llm, "SOARModel")
    assert hasattr(soar_llm, "get_soar_tokenizer")
    assert hasattr(soar_llm, "SPECIAL_TOKENS")


def test_model_forward():
    import torch
    from soar_llm.config import SOARConfig
    from soar_llm.model import SOARModel
    cfg = SOARConfig()
    model = SOARModel(cfg)
    x = torch.randint(0, 50257, (1, 8))
    out = model.forward(x)
    assert "logits" in out
    assert out["logits"].shape[-1] == cfg.vocab_size
