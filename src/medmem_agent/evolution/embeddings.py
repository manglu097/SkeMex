from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

from .utils import atomic_write_json, cosine_similarity, read_json

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingManager:
    cache_path: str
    model: str = "text-embedding-3-small"
    api_base: str = ""
    api_key: str = ""
    global_query_cache_path: str = ""

    def __post_init__(self) -> None:
        self._cache: Dict[str, dict] = read_json(self.cache_path, default={})
        self._global_query_cache: Dict[str, dict] = read_json(self.global_query_cache_path, default={}) if self.global_query_cache_path else {}
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            kwargs = {}
            api_key = os.getenv("EMBEDDING_API_KEY") or self.api_key or os.getenv("OPENAI_API_KEY") or None
            if api_key:
                kwargs["api_key"] = api_key
            base = os.getenv("EMBEDDING_BASE_URL") or self.api_base
            if base:
                kwargs["base_url"] = base
            self._client = OpenAI(**kwargs)
        return self._client

    def get(self, text: str, *, is_query: bool = False) -> Optional[List[float]]:
        if is_query and self.global_query_cache_path:
            item = self._global_query_cache.get(text)
            if item and item.get("model") == self.model:
                return item.get("embedding")
        item = self._cache.get(text)
        if item and item.get("model") == self.model:
            return item.get("embedding")
        return None

    def embed_text(self, text: str, *, is_query: bool = False) -> List[float]:
        cached = self.get(text, is_query=is_query)
        if cached is not None:
            return cached
        client = self._get_client()
        response = client.embeddings.create(input=text, model=self.model)
        vector = list(response.data[0].embedding)
        self._cache[text] = {"model": self.model, "embedding": vector}
        if is_query and self.global_query_cache_path:
            self._global_query_cache[text] = {"model": self.model, "embedding": vector}
        self.flush()
        return vector

    def embed_batch(self, texts: Iterable[str], *, is_query: bool = False) -> Dict[str, List[float]]:
        text_list = list(texts)
        pending = [text for text in text_list if self.get(text, is_query=is_query) is None]
        if pending:
            client = self._get_client()
            response = client.embeddings.create(input=pending, model=self.model)
            for text, item in zip(pending, response.data):
                vector = list(item.embedding)
                self._cache[text] = {"model": self.model, "embedding": vector}
                if is_query and self.global_query_cache_path:
                    self._global_query_cache[text] = {"model": self.model, "embedding": vector}
            self.flush()
        return {text: self.get(text, is_query=is_query) for text in text_list if self.get(text, is_query=is_query) is not None}

    def similarity(self, query_text: str, candidate_text: str) -> float:
        query_vec = self.embed_text(query_text, is_query=True)
        candidate_vec = self.embed_text(candidate_text, is_query=False)
        return cosine_similarity(query_vec, candidate_vec)

    def flush(self) -> None:
        atomic_write_json(self.cache_path, self._cache)
        if self.global_query_cache_path:
            atomic_write_json(self.global_query_cache_path, self._global_query_cache)
