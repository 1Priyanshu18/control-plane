from typing import Callable

_FAKE_RESPONSES = [
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "The Eiffel Tower was completed in 1887 and was privately funded by Gustave Eiffel alone.",
    "The Eiffel Tower was fully funded by the French government from the start, "
    "though Gustave Eiffel privately paid for the entire project out of his own "
    "pocket due to a lack of government funding.",
]


def _fake_generate(prompt: str, attempt: int = 0) -> str:
    return _FAKE_RESPONSES[attempt % len(_FAKE_RESPONSES)]


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_real_model = None
_real_tokenizer = None


def _load_real_model():
    global _real_model, _real_tokenizer
    if _real_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        _real_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if _real_tokenizer.pad_token is None:
            _real_tokenizer.pad_token = _real_tokenizer.eos_token
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        _real_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quant_config, device_map="auto"
        )
    return _real_model, _real_tokenizer


def _real_generate(prompt: str, attempt: int = 0, max_new_tokens: int = 80) -> str:
    """Real Qwen generation, Kaggle-only (needs GPU + bitsandbytes). attempt>0
    switches on sampling so a regeneration can actually differ from the first
    try -- unlike the fake generator's fixed canned alternatives."""
    import torch

    model, tokenizer = _load_real_model()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
    if attempt > 0:
        gen_kwargs.update(do_sample=True, temperature=0.8)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response.strip()


def load_generator(cfg: dict) -> Callable[..., str]:
    name = cfg["generator"]["name"]
    if name == "fake":
        return _fake_generate
    if name == "real":
        return _real_generate
    raise ValueError(f"unknown generator: {name}")
