"""Inference with TTT, TTC (refinement, early exit), tool dispatch."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from transformers import AutoTokenizer

from soar_llm.config import SOARConfig
from soar_llm.model import SOARModel
from soar_llm.tokenizer import get_soar_tokenizer, resize_model_embeddings, format_chat

_MIN_TEMPERATURE = 0.05
_MAX_TEMPERATURE = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--use_ttt", action="store_true", help="Enable test-time training adapters")
    parser.add_argument("--use_early_exit", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (default: chat_sft_alpaca if exists)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Reduce repetition in output")
    parser.add_argument("--temperature", type=float, default=None, help=">0 sampling, 0 greedy; default 0 for chat, 0.7 for completion")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling; ignored when temperature=0")
    parser.add_argument("--no_repeat_ngram", type=int, default=0, help="Block repeating n-grams; 0=disabled (recommended for chat)")
    parser.add_argument("--chat", action="store_true", help="Use chat format (ChatGPT-style); auto if checkpoint has chat")
    parser.add_argument("--system", default=None, help="System message when using --chat")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = torch.device("cuda")
    config = SOARConfig()

    # Prefer chat checkpoint for better quality when available
    root = Path(__file__).resolve().parent.parent
    if args.checkpoint is None:
        for ckpt in ("checkpoints/chat_sft_alpaca", "checkpoints/chat_sft", "checkpoints/fineweb"):
            if (root / ckpt / "config.json").exists():
                args.checkpoint = str(root / ckpt)
                if "chat" in ckpt and not args.chat:
                    args.chat = True
                break
    if args.temperature is None:
        args.temperature = 0.0 if args.chat else 0.7

    model = SOARModel(config)
    if args.checkpoint:
        from transformers import GPT2LMHeadModel
        model._base = GPT2LMHeadModel.from_pretrained(args.checkpoint)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        except Exception:
            tokenizer, _ = get_soar_tokenizer(config.model_name)
    else:
        tokenizer, _ = get_soar_tokenizer(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resize_model_embeddings(model._base, tokenizer)
    model = model.to(device)
    model.eval()

    if args.use_ttt:
        model._ensure_ttt_hooks()

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        prompt = format_chat(messages, tokenizer)
    else:
        prompt = args.prompt

    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram
    if math.isfinite(args.temperature) and args.temperature > 0:
        temperature = max(_MIN_TEMPERATURE, min(args.temperature, _MAX_TEMPERATURE))
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = args.top_p
        gen_kwargs["remove_invalid_values"] = True
        gen_kwargs["renormalize_logits"] = True
    if args.chat:
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id != tokenizer.unk_token_id:
                gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id, im_end_id]
        except Exception:
            pass
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        out = model.generate(ids, **gen_kwargs)
    new_ids = out[0][ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if args.chat:
        text = text.split("<|im_end|>")[0].strip()
    else:
        text = text.strip()
    print(text)


if __name__ == "__main__":
    main()
