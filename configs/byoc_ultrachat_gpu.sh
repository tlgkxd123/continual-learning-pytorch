#!/usr/bin/env bash
set -euo pipefail

# BYOC reference run for a CUDA box with >=16GB VRAM.
# For 8-12GB VRAM, reduce --seq_len to 256 and keep --batch 1.
python3 scripts/train_ultrachat.py \
  --device cuda \
  --bf16 \
  --steps 2000 \
  --batch 1 \
  --grad_accum 8 \
  --seq_len 512 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --max_samples 50000 \
  --lr 2e-4 \
  --out ./checkpoints/ultrachat_sft_byoc \
  --log_every 10
