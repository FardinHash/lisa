import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        self.enabled = settings.cache_enabled
        self.ttl = settings.cache_ttl
        self.max_size = settings.cache_max_size
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _generate_key(self, prefix: str, **kwargs) -> str:
        data = json.dumps(kwargs, sort_keys=True)
        hash_obj = hashlib.sha256(data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry["expires_at"]:
                logger.debug(f"Cache hit for key: {key[:50]}...")
                return entry["value"]
            else:
                del self._cache[key]
                logger.debug(f"Cache expired for key: {key[:50]}...")

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self.enabled:
            return

        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.items(), key=lambda x: x[1]["created_at"])[0]
            del self._cache[oldest_key]
            logger.debug("Cache size limit reached, evicted oldest entry")

        ttl = ttl or self.ttl
        self._cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }
        logger.debug(f"Cached value for key: {key[:50]}...")

    def invalidate(self, prefix: str) -> None:
        keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_delete:
            del self._cache[key]
        logger.info(
            f"Invalidated {len(keys_to_delete)} cache entries with prefix: {prefix}"
        )

    def clear(self) -> None:
        self._cache.clear()
        logger.info("Cache cleared")

    def get_rag_result(self, query: str, k: int) -> Optional[Any]:
        key = self._generate_key("rag", query=query, k=k)
        return self.get(key)

    def set_rag_result(self, query: str, k: int, result: Any) -> None:
        key = self._generate_key("rag", query=query, k=k)
        self.set(key, result)

    def get_llm_result(self, messages: list, temperature: float) -> Optional[str]:
        key = self._generate_key("llm", messages=str(messages), temperature=temperature)
        return self.get(key)

    def set_llm_result(self, messages: list, temperature: float, result: str) -> None:
        key = self._generate_key("llm", messages=str(messages), temperature=temperature)
        self.set(key, result)


cache_service = CacheService()
