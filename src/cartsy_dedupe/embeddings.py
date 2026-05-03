from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embedding_provider_name() -> str:
    provider = os.getenv("CARTSY_EMBEDDING_PROVIDER", "openai").strip().lower()
    aliases = {
        "sentence_transformers": "sentence-transformers",
        "sentence_transformer": "sentence-transformers",
        "local": "sentence-transformers",
        "st": "sentence-transformers",
    }
    return aliases.get(provider, provider)


def default_embedding_model(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
    if provider == "sentence-transformers":
        return DEFAULT_SENTENCE_TRANSFORMERS_MODEL
    raise ValueError("CARTSY_EMBEDDING_PROVIDER must be one of: openai, sentence-transformers")


def configured_embedding_model(provider: str | None = None, model: str | None = None) -> str:
    resolved_provider = provider or embedding_provider_name()
    return model or os.getenv("CARTSY_EMBEDDING_MODEL") or default_embedding_model(resolved_provider)


def configured_embedding_dimensions(provider: str | None = None, model: str | None = None) -> int:
    resolved_provider = provider or embedding_provider_name()
    resolved_model = configured_embedding_model(resolved_provider, model)
    if resolved_provider == "openai":
        if resolved_model == "text-embedding-3-large":
            return 3072
        return 1536
    if os.getenv("CARTSY_EMBEDDING_DIMENSIONS"):
        return int(os.environ["CARTSY_EMBEDDING_DIMENSIONS"])
    if resolved_provider == "sentence-transformers":
        if resolved_model.endswith("all-MiniLM-L6-v2"):
            return 384
        return sentence_transformer_dimensions(resolved_model)
    raise ValueError("CARTSY_EMBEDDING_PROVIDER must be one of: openai, sentence-transformers")


def ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env or the environment before using OpenAI embeddings.")


@dataclass(slots=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    usage: Any = None


class EmbeddingProvider:
    def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or embedding_provider_name()
        self.model = configured_embedding_model(self.provider, model)
        self._sentence_transformer_model: Any | None = None

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult([])
        if self.provider == "openai":
            return self._embed_openai(texts)
        if self.provider == "sentence-transformers":
            return self._embed_sentence_transformers(texts)
        raise ValueError("CARTSY_EMBEDDING_PROVIDER must be one of: openai, sentence-transformers")

    def _embed_openai(self, texts: list[str]) -> EmbeddingResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment setup.
            raise RuntimeError("Install openai before using OpenAI embeddings.") from exc
        ensure_openai_api_key()
        response = OpenAI().embeddings.create(model=self.model, input=texts)
        return EmbeddingResult([list(item.embedding) for item in response.data], getattr(response, "usage", None))

    def _embed_sentence_transformers(self, texts: list[str]) -> EmbeddingResult:
        model = self._get_sentence_transformer_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return EmbeddingResult([[float(value) for value in vector] for vector in vectors])

    def _get_sentence_transformer_model(self) -> Any:
        if self._sentence_transformer_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError(
                    "Install sentence-transformers before using CARTSY_EMBEDDING_PROVIDER=sentence-transformers."
                ) from exc
            self._sentence_transformer_model = SentenceTransformer(self.model)
        return self._sentence_transformer_model


def sentence_transformer_dimensions(model_name: str) -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "Set CARTSY_EMBEDDING_DIMENSIONS or install sentence-transformers so the model dimension can be detected."
        ) from exc
    return int(SentenceTransformer(model_name).get_sentence_embedding_dimension())
