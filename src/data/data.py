import hashlib
import json
import urllib.request
from pathlib import Path

DIALOGUE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json"
GENERAL_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json"
DIALOGUE_CACHE = Path("data/dialogue_data.json")
GENERAL_CACHE = Path("data/general_data.json")


def _download(url: str, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)


def _load_jsonl(cache_path: Path, url: str) -> list[dict]:
    if not cache_path.exists():
        _download(url, cache_path)
    records = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dialogue() -> list[dict]:
    return _load_jsonl(DIALOGUE_CACHE, DIALOGUE_URL)


def load_general() -> list[dict]:
    """HaluEval's general_data.json: 130 single natural ChatGPT responses with
    human hallucination labels AND real sub-response hallucination_spans.
    Held out entirely from training/split -- used only as a qualitative check
    that a trained probe fires on genuine fabrication spans."""
    return _load_jsonl(GENERAL_CACHE, GENERAL_URL)


def _response_id(context: str, answer: str) -> str:
    return hashlib.sha256(f"{context}||{answer}".encode("utf-8")).hexdigest()[:16]


def _group_id(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]


def flatten_dialogue(records: list[dict]) -> list[dict]:
    flattened = []
    for r in records:
        for answer_key, label in [("right_response", 0), ("hallucinated_response", 1)]:
            flattened.append(
                {
                    "response_id": _response_id(r["dialogue_history"], r[answer_key]),
                    "group_id": _group_id(r["dialogue_history"]),
                    "knowledge": r["knowledge"],
                    "context": r["dialogue_history"],
                    "response": r[answer_key],
                    "label": label,
                }
            )
    return flattened


def flatten_general(records: list[dict]) -> list[dict]:
    flattened = []
    for r in records:
        flattened.append(
            {
                "response_id": _response_id(r["user_query"], r["chatgpt_response"]),
                "group_id": _group_id(r["user_query"]),
                "knowledge": "",
                "context": r["user_query"],
                "response": r["chatgpt_response"],
                "label": 1 if r["hallucination"] == "yes" else 0,
                "hallucination_spans": r.get("hallucination_spans", []),
            }
        )
    return flattened
