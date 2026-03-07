"""MAML-style pretrain for TTT adapter meta-learned prior."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from soar_llm.config import SOARConfig
from soar_llm.ttt.adapter import TTTRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--out", default="./checkpoints/meta_prior")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = torch.device("cuda")
    config = SOARConfig(model_name=args.model)
    base = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    adapter = TTTRouter(config).to(device)
    handles = adapter.register_hooks(base)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    def dummy_batch():
        return torch.randint(0, config.vocab_size, (args.batch, args.seq_len), device=device)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        total_loss = 0.0
        for _ in range(100):
            ids = dummy_batch()
            labels = ids.clone()
            labels[:, :-1] = ids[:, 1:]
            labels[:, -1] = -100
            out = base(input_ids=ids)
            logits = out.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        for h in handles:
            h.remove()
        handles = adapter.register_hooks(base)
        torch.save(adapter.state_dict(), f"{args.out}/adapter_ep{ep}.pt")
        print(f"Epoch {ep} loss={total_loss / 100:.4f}")


if __name__ == "__main__":
    main()
