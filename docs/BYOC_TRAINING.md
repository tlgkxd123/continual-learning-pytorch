# BYOC GPU training

Use this when you want to bring your own CUDA machine for longer SOAR training runs.
The local Devin VM may only have CPU, which is useful for smoke tests but not for
meaningful 0.8B-parameter fine-tuning.

## 1. Provision a CUDA host

Recommended minimum:

- NVIDIA GPU with 16GB+ VRAM
- 80GB+ free disk
- Python 3.10+
- CUDA-compatible PyTorch

For better runs, use an A100/H100/L40S or multi-GPU box and raise `--steps`,
`--max_samples`, and `--seq_len`.

## 2. Install

```bash
git clone https://github.com/tlgkxd123/continual-learning-pytorch.git
cd continual-learning-pytorch
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional, for higher Hugging Face limits:

```bash
export HF_TOKEN=...
```

## 3. Run the reference job

```bash
bash configs/byoc_ultrachat_gpu.sh
```

The script runs UltraChat LoRA SFT with bf16 autocast and writes a checkpoint to
`checkpoints/ultrachat_sft_byoc/`.

## 4. Smaller GPUs

If you hit OOM:

```bash
python3 scripts/train_ultrachat.py \
  --device cuda \
  --fp16 \
  --steps 1000 \
  --batch 1 \
  --grad_accum 16 \
  --seq_len 256 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --max_samples 20000 \
  --out ./checkpoints/ultrachat_sft_byoc_small
```

## 5. CPU smoke test

This verifies the pipeline only; it is not a meaningful quality run.

```bash
python3 scripts/train_ultrachat.py \
  --device cpu \
  --steps 1 \
  --batch 1 \
  --grad_accum 1 \
  --seq_len 64 \
  --lora_rank 2 \
  --lora_alpha 4 \
  --max_samples 4 \
  --out ./checkpoints/ultrachat_sft_cpu_smoke \
  --log_every 1
```

## 6. Promote a checkpoint

After training, run smoke tests and a small prompt set before using a checkpoint:

```bash
ruff check .
python3 -m pytest tests/test_smoke.py -v
python3 webui.py
```

The web UI automatically prefers `checkpoints/ultrachat_sft/` when present, so
copy or symlink your selected BYOC checkpoint there for local chat testing.
