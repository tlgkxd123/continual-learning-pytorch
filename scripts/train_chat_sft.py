"""Chat SFT: train on instruction/chat data, ChatGPT-style."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel

from soar_llm.tokenizer import get_soar_tokenizer, format_chat_for_sft, resize_model_embeddings


class ChatSFTDataset(Dataset):
    """Dataset of chat conversations for SFT."""

    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        messages = self.samples[i]["messages"]
        input_ids, labels = format_chat_for_sft(
            messages, self.tokenizer, max_length=self.max_length
        )
        if input_ids is None:
            return self.__getitem__((i + 1) % len(self))
        return {"input_ids": input_ids, "labels": labels}


def collate_pad(batch, pad_id=0):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = []
    labels = []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def get_dummy_chat(n: int = 50):
    """Return n dummy chat samples for quick testing."""
    dummies = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Explain Python in one sentence."},
                {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity."},
            ]
        },
    ]
    return (dummies * ((n // len(dummies)) + 1))[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--data", default=None, help="JSONL file: each line {messages:[{role,content}]}")
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset: tatsu-lab/alpaca, OpenAssistant/oasst1, etc.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2, help="Reduce if OOM (e.g. 8GB GPU)")
    parser.add_argument("--seq_len", type=int, default=128, help="Reduce if OOM")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--out", default="./checkpoints/chat_sft")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = torch.device("cuda")

    tokenizer, _ = get_soar_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model)
    resize_model_embeddings(model, tokenizer)
    model = model.to(device)

    if args.data:
        samples = load_jsonl(args.data)
        if args.max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:args.max_samples]
        print(f"Loaded {len(samples)} samples from {args.data}")
    elif args.dataset:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
        samples = []
        for ex in ds:
            msgs = ex.get("messages") or ex.get("conversations")
            if not msgs:
                inst = ex.get("instruction") or ex.get("prompt", "")
                inp = ex.get("input", "") or ""
                out = ex.get("output") or ex.get("response", "")
                if inst and out:
                    user_content = inst if not inp else f"{inst}\n\n{inp}".strip()
                    msgs = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": str(out)},
                    ]
            if msgs:
                samples.append({"messages": msgs})
        if args.max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:args.max_samples]
        print(f"Loaded {len(samples)} samples from {args.dataset}")
    else:
        print("No --data or --dataset; using Alpaca (tatsu-lab/alpaca)")
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        samples = []
        for ex in ds:
            inst = ex.get("instruction", "")
            inp = ex.get("input", "") or ""
            out = ex.get("output", "")
            if inst and out:
                user_content = inst if not inp else f"{inst}\n\n{inp}".strip()
                samples.append({
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": str(out)},
                    ]
                })
        if args.max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:args.max_samples]
        print(f"Loaded {len(samples)} Alpaca samples")

    dataset = ChatSFTDataset(samples, tokenizer, max_length=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id=tokenizer.pad_token_id or 0),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    step = 0
    batch_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        step += 1
        if step % 20 == 0:
            print(f"Step {step} loss={out.loss.item():.4f}")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
