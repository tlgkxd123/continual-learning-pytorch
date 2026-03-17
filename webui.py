"""SOAR LLM Omnimodal Web UI — text + image chat powered by Qwen3.5."""

import asyncio
import base64
import io
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="SOAR LLM")

model = None
processor = None
device = torch.device("cpu")


class ContentPart(BaseModel):
    type: str  # "text" or "image"
    text: Optional[str] = None
    image: Optional[str] = None  # base64-encoded image data


class Message(BaseModel):
    role: str
    content: str | List[ContentPart]


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 256
    system: str = "You are a helpful assistant."


class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256


def load_model():
    global model, processor
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from soar_llm.config import SOARConfig

    config = SOARConfig()
    root = Path(__file__).resolve().parent

    checkpoint = None
    for ckpt in ("checkpoints/ultrachat_sft",):
        if (root / ckpt / "config.json").exists():
            checkpoint = str(root / ckpt)
            break

    model_path = config.model_name
    print(f"Loading multimodal model from: {model_path}", flush=True)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
    )

    if checkpoint:
        print(f"Loading fine-tuned text weights from: {checkpoint}", flush=True)
        from safetensors.torch import load_file
        ckpt_path = Path(checkpoint) / "model.safetensors"
        if ckpt_path.exists():
            state_dict = load_file(str(ckpt_path))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded checkpoint: {len(state_dict)} tensors "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    model.to(device)
    model.eval()
    print(f"Model loaded: {type(model).__name__}", flush=True)


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)


def _build_messages(req_messages: List[Message], system_msg: str):
    """Convert API messages to Qwen3.5 multimodal format."""
    from PIL import Image

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_msg}]})

    images_for_processing = []

    for m in req_messages:
        if isinstance(m.content, str):
            messages.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
        else:
            parts = []
            for p in m.content:
                if p.type == "text" and p.text:
                    parts.append({"type": "text", "text": p.text})
                elif p.type == "image" and p.image:
                    img_data = base64.b64decode(p.image)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    images_for_processing.append(img)
                    parts.append({"type": "image", "image": img})
            if parts:
                messages.append({"role": m.role, "content": parts})

    return messages, images_for_processing


def _generate(messages, images, temperature, max_tokens):
    """Run multimodal generation."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vis_images, vis_videos = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=vis_images if vis_images else None,
        videos=vis_videos if vis_videos else None,
        return_tensors="pt",
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages, images = _build_messages(req.messages, req.system)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, _generate, messages, images, req.temperature, req.max_tokens
    )
    text = text.split("<|im_end|>")[0].strip()
    return {"response": text}


@app.post("/api/complete")
async def complete(req: CompletionRequest):
    messages = [{"role": "user", "content": [{"type": "text", "text": req.prompt}]}]
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, _generate, messages, [], req.temperature, req.max_tokens
    )
    return {"response": text}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "omnimodal": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webui:app", host="0.0.0.0", port=8000, reload=False)
