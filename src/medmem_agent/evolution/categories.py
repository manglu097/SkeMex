from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import List

from medmem_agent.config import LLMConfig
from medmem_agent.llm import build_llm

from .config import EvoConfig
from .prompts import build_task_classifier_prompt
from .storage import SkillStore

logger = logging.getLogger(__name__)


def _query_hash(query: str) -> str:
    """Return the SHA-256 hex digest of a query string (used as cache key)."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


@dataclass
class DynamicTaskCategoryRegistry:
    store: SkillStore
    evo_config: EvoConfig
    provider: str = "openai_compatible"
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None

    def classify(self, query: str) -> List[str]:
        """Classify *query* into task categories.

        Behaviour is controlled by ``evo_config.categories.classifier_mode``:

        ``"always"`` (default)
            Call the LLM classifier on every invocation — original behaviour.
        ``"cache"``
            Return the cached result for previously seen queries (keyed by
            SHA-256 hash of the query string).  On a cache miss, call the LLM
            and persist the result before returning.
        ``"precomputed"``
            This method should not be called in precomputed mode; ``online.py``
            reads ``task_category`` directly from the sample dict and bypasses
            this registry entirely.  If called anyway, falls back to
            ``"always"`` behaviour with a warning.
        """
        mode = getattr(self.evo_config.categories, "classifier_mode", "always")

        if mode == "precomputed":
            logger.warning(
                "DynamicTaskCategoryRegistry.classify() called in 'precomputed' mode. "
                "This should not happen — task_category should be read from the sample dict. "
                "Falling back to 'always' mode for this call."
            )
            return self._classify_via_llm(query)

        if mode == "cache":
            return self._classify_with_cache(query)

        # mode == "always" (default) — call LLM every time
        return self._classify_via_llm(query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_with_cache(self, query: str) -> List[str]:
        """Look up the cache; call LLM on miss and persist the result."""
        key = _query_hash(query)
        cache = self.store.load_category_cache()
        if key in cache:
            logger.debug("Category cache hit for query hash %s.", key[:8])
            return list(cache[key])

        logger.debug("Category cache miss for query hash %s; calling LLM.", key[:8])
        result = self._classify_via_llm(query)
        cache[key] = result
        self.store.save_category_cache(cache)
        return result

    def _classify_via_llm(self, query: str) -> List[str]:
        """Call the LLM classifier and return the selected category labels."""
        data = self.store.load_categories()
        active = data.get("active", [])
        if not active:
            return []

        llm = build_llm(
            LLMConfig(
                provider=self.provider,
                model=self.evo_config.categories.classifier_model,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=256,
                timeout_seconds=30,
            )
        )
        raw = llm.generate(
            system_prompt="You output strict JSON only.",
            user_prompt=build_task_classifier_prompt(active, query),
        ).content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Category classifier returned non-JSON. Falling back to empty selection.")
            return []

        selected = [item for item in parsed.get("selected", []) if isinstance(item, str)]
        selected = selected[: self.evo_config.categories.max_selected_categories]
        if parsed.get("is_new") and parsed.get("new_label"):
            label = str(parsed["new_label"]).strip()
            if label:
                self.store.update_pending_category(label)
                selected.append(label)
        return selected[: self.evo_config.categories.max_selected_categories]
