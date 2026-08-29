from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def build_prompt(knowledge: str, context: str, response: str) -> tuple[str, int, int]:
    prefix = f"Knowledge: {knowledge}\nContext: {context}\nResponse: "
    text = prefix + response
    return text, len(prefix), len(text)


def _char_range_to_token_range(offsets, char_start: int, char_end: int) -> tuple[int | None, int | None]:
    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if token_start is None and end > char_start:
            token_start = i
        if start < char_end:
            token_end = i + 1
    return token_start, token_end


def align_response_span(record: dict) -> dict:
    """Locates which token indices in the tokenized prompt+response text
    correspond to the response substring, via char offsets."""
    tokenizer = get_tokenizer()
    text, char_start, char_end = build_prompt(record["knowledge"], record["context"], record["response"])
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    token_start, token_end = _char_range_to_token_range(offsets, char_start, char_end)

    return {
        "text": text,
        "input_ids": encoding["input_ids"],
        "tokens": tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
        "offsets": offsets,
        "char_span": (char_start, char_end),
        "token_span": (token_start, token_end),
    }


def align_sub_span(record: dict, sub_text: str) -> dict | None:
    """Locates a substring WITHIN the response (e.g. a HaluEval
    hallucination_span) as token indices in the full prompt+response text.
    Returns None if sub_text isn't found verbatim in the response."""
    tokenizer = get_tokenizer()
    text, resp_char_start, _resp_char_end = build_prompt(record["knowledge"], record["context"], record["response"])
    local_idx = record["response"].find(sub_text)
    if local_idx == -1:
        return None
    char_start = resp_char_start + local_idx
    char_end = char_start + len(sub_text)

    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    token_start, token_end = _char_range_to_token_range(offsets, char_start, char_end)

    return {
        "input_ids": encoding["input_ids"],
        "tokens": tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
        "char_span": (char_start, char_end),
        "token_span": (token_start, token_end),
    }
