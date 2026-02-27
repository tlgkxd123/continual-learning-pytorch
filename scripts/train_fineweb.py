"""Train SOAR base on FineWeb (small run)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from soar_llm.config import SOARConfig

# Enable TF32 for matmuls (faster on Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def iter_fineweb(tokenizer, seq_len, batch_size):
    """Stream FineWeb (normal) samples, yield tokenized batches."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    buffer = []
    for ex in ds:
        text = ex.get("text", ex.get("content", ""))
        if not text or len(text) < 30:
            continue
        enc = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
        )
        buffer.append(enc["input_ids"][0])
        if len(buffer) >= batch_size:
            yield torch.stack(buffer[:batch_size])
            buffer = buffer[batch_size:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--steps", type=int, default=600, help="Max steps (train until loss < target_loss or this)")
    parser.add_argument("--target_loss", type=float, default=2.0, help="Stop when loss drops below this")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--out", default="./checkpoints/fineweb")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    config = SOARConfig()

    use_amp = torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=use_amp)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    try:
        gen = iter_fineweb(tokenizer, args.seq_len, args.batch)
    except Exception as e:
        print(f"FineWeb load failed ({e}), using wikitext")
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        def collate(examples):
            texts = [ex["text"] for ex in examples if ex.get("text") and len(ex["text"]) > 20]
            if not texts:
                return torch.randint(0, config.vocab_size, (args.batch, args.seq_len))
            enc = tokenizer(texts[:args.batch], truncation=True, max_length=args.seq_len,
                           padding="max_length", return_tensors="pt")
            return enc["input_ids"]
        loader = DataLoader(
            ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
            num_workers=2, pin_memory=True, prefetch_factor=2,
        )
        step = 0
        for _ in range((args.steps * args.batch) // len(ds) + 1):
            for batch in loader:
                if step >= args.steps:
                    break
                ids = batch.to(device, non_blocking=True)
                if ids.size(0) < 2:
                    continue
                labels = ids.clone()
                labels[:, :-1] = ids[:, 1:]
                labels[:, -1] = -100
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(input_ids=ids, labels=labels)
                if scaler is not None:
                    scaler.scale(out.loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    out.loss.backward()
                    opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"Step {step} loss={out.loss.item():.4f}")
            if step >= args.steps:
                break
        save_model = getattr(model, "_orig_mod", model)
        save_model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        print(f"Saved to {args.out}")
        return

    step = 0
    accum_loss = 0.0
    for ids in gen:
        if step >= args.steps:
            break
        ids = ids.to(device, non_blocking=True)
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids=ids, labels=labels)
            loss = out.loss / args.grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += out.loss.item()
        if (step + 1) % args.grad_accum == 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            avg = accum_loss / 100
            print(f"Step {step} loss={avg:.4f}")
            if avg < args.target_loss:
                print(f"Target loss {args.target_loss} reached.")
                break
            accum_loss = 0.0
    save_model = getattr(model, "_orig_mod", model)
    save_model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Done. Saved to {args.out}")


if __name__ == "__main__":
    main()
