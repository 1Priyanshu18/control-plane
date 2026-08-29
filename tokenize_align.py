from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def build_prompt(knowledge: str, question: str, response: str) -> tuple[str, int, int]:
    prefix = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer: "
    text = prefix + response
    return text, len(prefix), len(text)


def align_response_span(record: dict) -> dict:
    """Locates which token indices in the tokenized prompt+response text
    correspond to the response substring, via char offsets."""
    tokenizer = get_tokenizer()
    text, char_start, char_end = build_prompt(record["knowledge"], record["question"], record["response"])
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if token_start is None and end > char_start:
            token_start = i
        if start < char_end:
            token_end = i + 1

    return {
        "text": text,
        "input_ids": encoding["input_ids"],
        "tokens": tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
        "offsets": offsets,
        "char_span": (char_start, char_end),
        "token_span": (token_start, token_end),
    }
