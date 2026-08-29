import hashlib
import json
import urllib.request
from pathlib import Path

QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
CACHE_PATH = Path("data/qa_data.json")


def _download_qa() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(QA_URL, CACHE_PATH)


def load_qa() -> list[dict]:
    if not CACHE_PATH.exists():
        _download_qa()
    records = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _response_id(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}||{answer}".encode("utf-8")).hexdigest()[:16]


def _group_id(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def flatten_responses(records: list[dict]) -> list[dict]:
    flattened = []
    for r in records:
        for answer_key, label in [("right_answer", 0), ("hallucinated_answer", 1)]:
            flattened.append(
                {
                    "response_id": _response_id(r["question"], r[answer_key]),
                    "group_id": _group_id(r["question"]),
                    "knowledge": r["knowledge"],
                    "question": r["question"],
                    "response": r[answer_key],
                    "label": label,
                }
            )
    return flattened
