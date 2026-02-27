"""Extended tokenizer with SOAR special tokens and chat template."""

from typing import Dict, List, Optional, Union

SPECIAL_TOKENS: List[str] = [
    "<think>",
    "</think>",
    "<SEARCH>",
    "<CODE_EXEC>",
    "<FILE_READ>",
    "<API_CALL>",
    "<MEMORY_WRITE>",
    "<SPAWN_AGENT>",
    "<TOOL_RESULT>",
]

# ChatML-style (ChatGPT-like) tokens
CHAT_SPECIAL_TOKENS: List[str] = ["<|im_start|>", "<|im_end|>"]
CHAT_ROLES: List[str] = ["system", "user", "assistant"]


def get_soar_tokenizer(base_name: str = "gpt2", add_chat_tokens: bool = True):
    """Load base tokenizer and add SOAR special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    to_add = list(SPECIAL_TOKENS)
    if add_chat_tokens:
        to_add.extend(CHAT_SPECIAL_TOKENS)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": to_add}
    )
    return tokenizer, num_added


def format_chat(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: Optional[int] = None,
) -> str:
    """Format messages into ChatML string. Each msg: {role, content}."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    text = "\n".join(parts) + "\n<|im_start|>assistant\n"
    return text


def format_chat_for_sft(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = 512,
) -> tuple:
    """Format chat for SFT. Returns (input_ids, labels) with -100 on non-assistant tokens."""
    if not messages or messages[-1].get("role") != "assistant":
        return None, None
    prompt_parts = []
    for m in messages[:-1]:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")
    response = messages[-1].get("content", "").strip()
    prompt_text = "".join(prompt_parts)
    full_text = prompt_text + response + "<|im_end|>"
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return input_ids, labels


def get_special_token_ids(tokenizer) -> Dict[str, int]:
    """Return mapping of special token names to IDs."""
    return {name: tokenizer.convert_tokens_to_ids(name) for name in SPECIAL_TOKENS}


def resize_model_embeddings(model, tokenizer):
    """Resize embedding and lm_head for new vocab size."""
    old_vocab = model.config.vocab_size
    new_vocab = len(tokenizer)
    if new_vocab != old_vocab:
        model.resize_token_embeddings(new_vocab)
        model.config.vocab_size = new_vocab
    return model
