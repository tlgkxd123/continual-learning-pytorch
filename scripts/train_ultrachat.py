"""Fine-tune Qwen3.5-0.8B on UltraChat with LoRA."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from soar_llm.config import SOARConfig


class UltraChatDataset(Dataset):
    """Streams UltraChat conversations formatted for SFT.

    Uses only the last user-assistant pair from each conversation to keep
    prompts short enough to leave room for response tokens.
    """

    def __init__(self, tokenizer, max_length=512, max_samples=5000, split="train_sft"):
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        skipped = 0
        seen = 0

        for row in ds:
            if len(self.samples) >= max_samples:
                break
            seen += 1
            messages = row.get("messages", [])
            pair = self._extract_last_pair(messages)
            if pair is None:
                skipped += 1
                continue

            prompt_text = tokenizer.apply_chat_template(
                pair[:-1], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False, truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(prompt_ids) >= max_length - 10:
                skipped += 1
                continue
            self.samples.append(pair)

        print(f"Loaded {len(self.samples)} conversations from UltraChat "
              f"(scanned {seen}, skipped {skipped})", flush=True)

    @staticmethod
    def _extract_last_pair(messages):
        """Get last user->assistant exchange, optionally preceded by system msg."""
        if not messages or len(messages) < 2:
            return None
        if messages[-1].get("role") != "assistant":
            messages = messages[:-1]
        if not messages or messages[-1].get("role") != "assistant":
            return None
        last_asst_idx = len(messages) - 1
        user_idx = last_asst_idx - 1
        if user_idx < 0 or messages[user_idx].get("role") != "user":
            return None
        pair = []
        if messages[0].get("role") == "system":
            pair.append(messages[0])
        pair.append(messages[user_idx])
        pair.append(messages[last_asst_idx])
        return pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        prompt_messages = messages[:-1]
        assistant_content = messages[-1]["content"]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + assistant_content + self.tokenizer.eos_token

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = min(prompt_enc["input_ids"].shape[1], self.max_length)
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def find_lora_target_modules(model):
    """Discover linear layers suitable for LoRA in any architecture."""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_part = name.split(".")[-1]
            if last_part in ("lm_head", "embed_tokens"):
                continue
            targets.add(last_part)
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = targets & candidates
    if found:
        return sorted(found)
    return sorted(targets)[:6]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on UltraChat with LoRA")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Base model")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max training samples to load")
    parser.add_argument("--out", default="./checkpoints/ultrachat_sft", help="Output directory")
    parser.add_argument("--log_every", type=int, default=5, help="Log every N steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = find_lora_target_modules(model)
    print(f"LoRA target modules: {target_modules}", flush=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)", flush=True)

    model = model.to(device)
    model.train()

    print("Loading UltraChat dataset...", flush=True)
    dataset = UltraChatDataset(
        tokenizer,
        max_length=args.seq_len,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(20, args.steps // 10),
        num_training_steps=args.steps,
    )

    print(f"\nTraining for {args.steps} steps (batch={args.batch}, grad_accum={args.grad_accum})...", flush=True)
    step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    while step < args.steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % args.grad_accum == 0 or step == args.steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % args.log_every == 0:
                avg = running_loss / args.log_every * args.grad_accum
                lr = scheduler.get_last_lr()[0]
                print(f"  step {step}/{args.steps}  loss={avg:.4f}  lr={lr:.2e}", flush=True)
                running_loss = 0.0

            if step >= args.steps:
                break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMerging LoRA weights and saving to {out_dir}...", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Done! Checkpoint saved.", flush=True)


if __name__ == "__main__":
    main()
