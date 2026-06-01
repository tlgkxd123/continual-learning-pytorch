import webui


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        if token == "<|im_end|>":
            return 7
        return -1


def test_sanitize_max_tokens_clamps_range():
    webui.tokenizer = None
    assert webui._sanitize_max_tokens(0) == 1
    assert webui._sanitize_max_tokens(10) == 10
    assert webui._sanitize_max_tokens(9999) == 4096


def test_build_gen_kwargs_has_chat_eos_ids(monkeypatch):
    monkeypatch.setattr(webui, "tokenizer", _DummyTokenizer())
    kwargs = webui._build_gen_kwargs(temperature=0.6, max_tokens=2048, chat=True)
    assert kwargs["max_new_tokens"] == 2048
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.6
    assert kwargs["eos_token_id"] == [2, 7]


def test_device_defaults_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert webui._default_device().type == "cpu"
