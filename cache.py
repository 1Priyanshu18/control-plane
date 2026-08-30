import hashlib

_CACHE: dict[str, str] = {}


def cache_key(user_input: str, attempt: int) -> str:
    return hashlib.sha256(f"{user_input}||{attempt}".encode("utf-8")).hexdigest()


def cache_get(key: str) -> str | None:
    return _CACHE.get(key)


def cache_set(key: str, value: str) -> None:
    _CACHE[key] = value
