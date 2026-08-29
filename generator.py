from typing import Callable

_FAKE_RESPONSES = [
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "The Eiffel Tower was completed in 1887 and was privately funded by Gustave Eiffel alone.",
]


def _fake_generate(prompt: str, attempt: int = 0) -> str:
    return _FAKE_RESPONSES[attempt % len(_FAKE_RESPONSES)]


def load_generator(cfg: dict) -> Callable[..., str]:
    name = cfg["generator"]["name"]
    if name == "fake":
        return _fake_generate
    raise ValueError(f"unknown generator: {name}")
