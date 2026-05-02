from __future__ import annotations

import json
import re
import unicodedata


STOPWORDS = {
    "a",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "for",
    "of",
    "o",
    "os",
    "para",
    "the",
    "with",
}

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency is declared for normal installs.
    import difflib

    class _FallbackFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FallbackFuzz()


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = strip_accents(text.lower())
    text = text.replace("&", " e ")
    text = re.sub(r"['`´]", "", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_brand(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def normalize_category(value: object) -> tuple[str, str]:
    raw = "" if value is None else str(value)
    if not raw.strip():
        return "", ""
    parts = re.split(r"[>›/|]+", raw)
    cleaned = [normalize_text(part) for part in parts if normalize_text(part)]
    if not cleaned:
        return "", ""
    return ">".join(cleaned), cleaned[-1]


def parse_jsonish(value: str) -> tuple[object | None, bool]:
    if not value or not str(value).strip():
        return None, False
    try:
        return json.loads(value), False
    except json.JSONDecodeError:
        return None, True


def flatten_jsonish(value: object | None) -> str:
    pieces: list[str] = []

    def walk(item: object) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                pieces.append(str(key))
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        else:
            pieces.append(str(item))

    walk(value)
    return normalize_text(" ".join(pieces))


def informative_tokens(text: str, limit: int = 5) -> tuple[str, ...]:
    tokens = [tok for tok in normalize_text(text).split() if len(tok) > 2 and tok not in STOPWORDS]
    return tuple(tokens[:limit])
