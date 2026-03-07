"""SOAR LLM Web UI — FastAPI chat interface with streaming and continuous TTT."""

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from soar_llm.config import SOARConfig

app = FastAPI(title="SOAR LLM")

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ttt_trainer = None
_ttt_lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 256
    system: str = "You are a helpful assistant."
    enable_ttt: bool = True


class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    enable_ttt: bool = True


class RememberRequest(BaseModel):
    text: str
    save: bool = True


def _sanitize_max_tokens(max_tokens: int) -> int:
    return max(1, min(int(max_tokens), 1024))


def _build_gen_kwargs(temperature: float, max_tokens: int, *, chat: bool) -> dict:
    gen_kwargs = {
        "max_new_tokens": _sanitize_max_tokens(max_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = 0.9
    if chat:
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id >= 0:
                eos_ids = [tokenizer.eos_token_id, im_end_id]
                gen_kwargs["eos_token_id"] = [i for i in eos_ids if i is not None]
        except Exception:
            pass
    return gen_kwargs


def load_model():
    global model, tokenizer, ttt_trainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = SOARConfig()
    root = Path(__file__).resolve().parent

    checkpoint = None
    for ckpt in ("checkpoints/ultrachat_sft", "checkpoints/chat_sft_alpaca",
                  "checkpoints/chat_sft", "checkpoints/fineweb"):
        if (root / ckpt / "config.json").exists():
            checkpoint = str(root / ckpt)
            break

    model_path = checkpoint or config.model_name
    print(f"Loading model from: {model_path}")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    print(f"Model loaded: {type(model).__name__}, vocab={len(tokenizer)}, device={device}, dtype={dtype}")

    # Initialise the continuous TTT trainer.
    try:
        from soar_llm.ttt.trainer import TTTContinualTrainer
        ttt_trainer = TTTContinualTrainer(model, config, device)
        hooks_msg = "hooks active" if ttt_trainer.hooks_active else "hooks inactive (unsupported arch)"
        print(f"TTT continual trainer ready — {hooks_msg}")
    except Exception as exc:  # pragma: no cover
        print(f"TTT trainer could not be initialised: {exc}")
        ttt_trainer = None


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)


def _generate(input_ids: torch.Tensor, temperature: float, max_tokens: int, chat: bool = False) -> str:
    gen_kwargs = _build_gen_kwargs(temperature, max_tokens, chat=chat)
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _format_chat(messages_dicts, system_msg):
    """Format using tokenizer's native chat template."""
    chat = []
    if system_msg:
        chat.append({"role": "system", "content": system_msg})
    chat.extend(messages_dicts)

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    parts = []
    for m in chat:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def _do_ttt_step(prompt_text: str, response_text: str) -> None:
    """Run one TTT gradient step on (prompt + response) context."""
    if ttt_trainer is None:
        return
    full_text = prompt_text + response_text
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return

    # Compute self-supervised confidence reward for RLVR / dynamic GRPO.
    reward: Optional[float] = None
    rewards_tensor = None
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
        reward = ttt_trainer.compute_reward(logits, input_ids)
        # Build a synthetic group-rewards tensor for dynamic GRPO.
        rewards_tensor = torch.tensor([reward], device=device)
    except Exception as exc:
        print(f"[TTT] reward computation failed: {exc}")

    try:
        ttt_trainer.train_step(
            input_ids,
            reward=reward,
            rewards_tensor=rewards_tensor,
        )
    except Exception as exc:
        print(f"[TTT] train_step failed: {exc}")


def _save_ttt_state() -> str:
    if ttt_trainer is None:
        raise RuntimeError("TTT trainer is not enabled")
    out_dir = Path(__file__).resolve().parent / "checkpoints" / "ttt_memory"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / "ttt_router.pt"
    stats_path = out_dir / "stats.json"
    with _ttt_lock:
        torch.save(ttt_trainer.ttt_router.state_dict(), adapter_path)
        stats_path.write_text(json.dumps(ttt_trainer.stats, indent=2))
    return str(adapter_path)


def _stream_chunks(input_ids: torch.Tensor, temperature: float, max_tokens: int, *, chat: bool):
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = _build_gen_kwargs(temperature, max_tokens, chat=chat)
    gen_kwargs["streamer"] = streamer
    worker = threading.Thread(
        target=model.generate,
        kwargs={"input_ids": input_ids, **gen_kwargs},
        daemon=True,
    )
    worker.start()

    for piece in streamer:
        text_piece = piece or ""
        if not text_piece:
            continue
        if chat and "<|im_end|>" in text_piece:
            text_piece = text_piece.split("<|im_end|>")[0]
            if text_piece:
                yield text_piece
            break
        yield text_piece
    worker.join(timeout=1.0)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = _format_chat(messages, req.system)
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    ids = enc["input_ids"].to(device)

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _generate, ids, req.temperature, req.max_tokens, True)
    text = text.split("<|im_end|>")[0].strip()

    # Continuous TTT: train in background after returning the response.
    if req.enable_ttt and ttt_trainer is not None:
        async def _ttt_bg():
            try:
                await loop.run_in_executor(None, _do_ttt_step, prompt, text)
            except Exception as exc:
                print(f"[TTT] background step error: {exc}")
        asyncio.create_task(_ttt_bg())

    return {"response": text, "ttt_step": ttt_trainer.step_count if ttt_trainer else 0}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = _format_chat(messages, req.system)
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    ids = enc["input_ids"].to(device)

    async def _sse():
        loop = asyncio.get_event_loop()
        try:
            full_text = ""
            for piece in _stream_chunks(ids, req.temperature, req.max_tokens, chat=True):
                full_text += piece
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"

            text = full_text.strip()
            if req.enable_ttt and ttt_trainer is not None:
                await loop.run_in_executor(None, _do_ttt_step, prompt, text)
            yield f"data: {json.dumps({'type': 'done', 'response': text, 'ttt_step': ttt_trainer.step_count if ttt_trainer else 0})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/api/complete")
async def complete(req: CompletionRequest):
    enc = tokenizer(req.prompt, return_tensors="pt", padding=True, truncation=True)
    ids = enc["input_ids"].to(device)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _generate, ids, req.temperature, req.max_tokens, False)

    # Continuous TTT: train in background after returning the response.
    if req.enable_ttt and ttt_trainer is not None:
        async def _ttt_bg_complete():
            try:
                await loop.run_in_executor(None, _do_ttt_step, req.prompt, text)
            except Exception as exc:
                print(f"[TTT] background step error: {exc}")
        asyncio.create_task(_ttt_bg_complete())

    return {
        "response": text,
        "ttt_step": ttt_trainer.step_count if ttt_trainer else 0,
    }


@app.post("/api/complete/stream")
async def complete_stream(req: CompletionRequest):
    enc = tokenizer(req.prompt, return_tensors="pt", padding=True, truncation=True)
    ids = enc["input_ids"].to(device)
    loop = asyncio.get_event_loop()

    async def _sse():
        try:
            full_text = ""
            for piece in _stream_chunks(ids, req.temperature, req.max_tokens, chat=False):
                full_text += piece
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"

            text = full_text.strip()
            if req.enable_ttt and ttt_trainer is not None:
                await loop.run_in_executor(None, _do_ttt_step, req.prompt, text)
            yield f"data: {json.dumps({'type': 'done', 'response': text, 'ttt_step': ttt_trainer.step_count if ttt_trainer else 0})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "supports_streaming": True,
    }


@app.get("/api/ttt_status")
async def ttt_status():
    """Return current TTT / continual-learning statistics."""
    if ttt_trainer is None:
        return {"enabled": False}
    return {"enabled": True, **ttt_trainer.stats}


@app.post("/api/ttt/remember")
async def ttt_remember(req: RememberRequest):
    if ttt_trainer is None:
        raise HTTPException(status_code=400, detail="TTT trainer is not enabled")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_ttt_step, text, "")
    saved_path = ""
    if req.save:
        saved_path = await loop.run_in_executor(None, _save_ttt_state)
    return {
        "ok": True,
        "message": "Added to continual memory/weights via TTT",
        "saved_path": saved_path,
        "ttt_step": ttt_trainer.step_count,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webui:app", host="0.0.0.0", port=8000, reload=False)
