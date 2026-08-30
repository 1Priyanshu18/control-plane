import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

_nli_tokenizer = None
_nli_model = None
_contradiction_idx = None


def get_nli_model():
    global _nli_tokenizer, _nli_model, _contradiction_idx
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _nli_model.eval()
        _contradiction_idx = next(
            idx for idx, label in _nli_model.config.id2label.items() if label.lower() == "contradiction"
        )
    return _nli_tokenizer, _nli_model


_CLAUSE_BOUNDARY = re.compile(
    r"(?<=[.!?,])\s+(?=(?:though|but|yet|however|while|whereas|although)\b)|(?<=[.!?])\s+",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Splits into clauses, not just sentences -- a contrast joined by a comma
    within one grammatical sentence ("funded X, though Y") is exactly the
    pattern the Marie Curie fixture needs pairwise NLI to catch."""
    parts = _CLAUSE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def contradiction_prob(tokenizer, model, premise: str, hypothesis: str) -> float:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=0)
    return probs[_contradiction_idx].item()


def nli_contradiction_score(record: dict) -> float:
    """Max contradiction probability over: (a) all ordered pairs of sentences
    within the response (self-contradiction), (b) knowledge vs response, if
    knowledge is present (input-contradiction)."""
    tokenizer, model = get_nli_model()
    sentences = split_sentences(record["response"])
    scores = []

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            scores.append(contradiction_prob(tokenizer, model, sentences[i], sentences[j]))

    if record.get("knowledge"):
        scores.append(contradiction_prob(tokenizer, model, record["knowledge"], record["response"]))

    return max(scores) if scores else 0.0
