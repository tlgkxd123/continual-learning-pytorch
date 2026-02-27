# SOAR LLM

PyTorch implementation of SOAR: Test-Time Training, Continual Learning, Test-Time Compute scaling, and Agentic Tools. GPT-2 (124M) base with optional chat fine-tuning.

## Components

- **TTT**: Plastic adapters (~0.1% params), Shampoo-lite optimizer, meta-learned prior
- **Continual Learning**: EWC++, replay buffer, LoRA archive, expert reservation
- **TTC**: Early exit, confidence-gated refinement, MCTS, scratchpad
- **Agent**: Tool tokens, hierarchy, memory (working/episodic/procedural)
- **Matryoshka** (design): Register-level dynamic precision, SM-aware scaling — see [docs/DESIGN_MATRYOSHKA_SLICING.md](docs/DESIGN_MATRYOSHKA_SLICING.md)

## Requirements

- Python 3.10+
- CUDA-capable GPU
- PyTorch 2.0+, Transformers 4.36+

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
from soar_llm.model import SOARModel
from soar_llm.config import SOARConfig
from soar_llm.tokenizer import get_soar_tokenizer

config = SOARConfig()
tokenizer, _ = get_soar_tokenizer("gpt2")
model = SOARModel(config)
```

### Inference

Text completion (auto-loads best available checkpoint):

```bash
python scripts/run_inference.py --prompt "The quick brown fox"
```

Chat mode (instruction-tuned models):

```bash
python scripts/run_inference.py --chat --prompt "Hello"
python scripts/run_inference.py --chat --system "You are helpful." --prompt "What is 2+2?"
```

Options: `--checkpoint`, `--temperature`, `--max_tokens`, `--use_ttt`, `--use_early_exit`

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_inference.py` | Inference with optional TTT, early exit; chat or completion |
| `scripts/train_chat_sft.py` | Chat SFT on Alpaca or custom JSONL |
| `scripts/train_fineweb.py` | Pretrain on FineWeb (or WikiText fallback) |
| `scripts/train_continual.py` | Multi-task continual fine-tuning |
| `scripts/pretrain_meta_prior.py` | Pretrain TTT adapter meta prior |

## Training Examples

**Chat SFT** (recommended for instruction-following):

```bash
# Alpaca (default, ~52k samples)
python scripts/train_chat_sft.py --steps 500 --batch 4 --seq_len 256 --out ./checkpoints/chat_sft

# Custom JSONL: each line {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
python scripts/train_chat_sft.py --data data.jsonl --steps 300 --out ./checkpoints/chat_sft
```

**Pretrain on web text**:

```bash
python scripts/train_fineweb.py --steps 600 --batch 16 --out ./checkpoints/fineweb
```

## Checkpoints

- `checkpoints/chat_sft` — Chat-tuned (Alpaca)
- `checkpoints/chat_sft_alpaca` — Same, explicit Alpaca run
- `checkpoints/fineweb` — FineWeb-pretrained base
- `checkpoints/continual` — Multi-task continual

Inference auto-selects `chat_sft_alpaca` → `chat_sft` → `fineweb` when `--checkpoint` is omitted.
