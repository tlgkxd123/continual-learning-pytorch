"""Continual fine-tune with EWC++/RLVR/GRPO, replay, LoRA snapshots."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from soar_llm.config import SOARConfig
from soar_llm.continual.ewc_plus import EWCPlus
from soar_llm.continual.grpo import GRPO
from soar_llm.continual.replay_buffer import ReplayBuffer
from soar_llm.continual.lora_archive import LoRAArchive
from soar_llm.continual.rlvr import RLVR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--regularizer", choices=["ewc", "rlvr", "grpo"], default="ewc")
    parser.add_argument(
        "--regularizer_lambda", "--ewc_lambda", dest="regularizer_lambda", type=float, default=1000.0
    )
    parser.add_argument("--out", default="./checkpoints/continual")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model = model.to(device)
    config = SOARConfig()
    if args.regularizer == "ewc":
        regularizer = EWCPlus(lambda_=args.regularizer_lambda)
    elif args.regularizer == "rlvr":
        regularizer = RLVR(lambda_=args.regularizer_lambda)
    else:
        regularizer = GRPO(lambda_=args.regularizer_lambda)
    replay = ReplayBuffer(max_tokens=100_000, sample_ratio=0.05)
    archive = LoRAArchive(f"{args.out}/lora")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def dummy_loader():
        for _ in range(10):
            yield {"input_ids": torch.randint(0, config.vocab_size, (2, 64), device=device)}

    for task in args.tasks:
        for ep in range(args.epochs):
            for batch in dummy_loader():
                ids = batch["input_ids"]
                labels = ids.clone()
                labels[:, :-1] = ids[:, 1:]
                labels[:, -1] = -100
                out = model(input_ids=ids, labels=labels)
                loss = out.loss
                if task != args.tasks[0]:
                    loss = loss + regularizer.penalty(model)
                replay_sample = replay.sample(ids.size(0), device)
                if replay_sample is not None:
                    rid = replay_sample["input_ids"]
                    rlabels = rid.clone()
                    rlabels[:, :-1] = rid[:, 1:]
                    rlabels[:, -1] = -100
                    r_out = model(**replay_sample, labels=rlabels)
                    loss = 0.95 * loss + 0.05 * r_out.loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                replay.add(ids, loss.item())
        if args.regularizer == "ewc":
            regularizer.compute_fisher(model, iter(dummy_loader()), device, max_batches=5)
        else:
            regularizer.snapshot_reference(model)
        lora_state = {n: p.detach().clone() for n, p in model.named_parameters()}
        archive.save(lora_state, task)

    model.save_pretrained(args.out)
    print("Done")


if __name__ == "__main__":
    main()
