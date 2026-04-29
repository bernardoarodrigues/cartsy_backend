from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable


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

BRAND_ALIASES = {
    "l oreal": "loreal",
    "loreal paris": "loreal",
    "l oreal paris": "loreal",
    "loreal professionnel": "loreal professionnel",
    "l oreal professionnel": "loreal professionnel",
    "la roche posay": "la roche posay",
    "wella professionals": "wella professionals",
    "wella professional": "wella professionals",
    "cerave": "cerave",
    "eucerin": "eucerin",
}


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
    brand = normalize_text(value)
    return BRAND_ALIASES.get(brand, brand)


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


def stable_join(values: Iterable[str]) -> str:
    return "|".join(sorted(value for value in values if value))
