# AGENTS.md

## Cursor Cloud specific instructions

This is a single-process Python ML research project (SOAR LLM) with no external services, databases, or Docker dependencies. See `README.md` for full usage docs.

### Quick reference

- **Lint:** `ruff check .` (48 pre-existing warnings as of initial setup)
- **Tests:** `python3 -m pytest tests/test_smoke.py -v` (6/7 pass; `test_tokenizer` has a pre-existing assertion mismatch: tokenizer returns 11 added tokens but `SPECIAL_TOKENS` list has 9)
- **Python API demo:** see `README.md` → "Python API" section

### Caveats

- **No CUDA in cloud VM.** Training scripts (`scripts/train_*.py`) and `scripts/run_inference.py` require CUDA and will raise `RuntimeError("CUDA is required but not available")`. All component-level tests and the Python API work on CPU.
- **HuggingFace Hub downloads.** The first run of tests or model creation downloads GPT-2 weights (~500 MB). These are cached in `~/.cache/huggingface/` after initial download.
- Ensure `$HOME/.local/bin` is on `PATH` for `pytest` and `ruff` commands.
