"""Cache factory class."""

from ols import constants
from ols.app.models.config import ConversationCacheConfig
from ols.src.cache.cache import Cache
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.src.cache.redis_cache import RedisCache


class CacheFactory:
    """Cache factory class."""

    @staticmethod
    def conversation_cache(config: ConversationCacheConfig) -> Cache:
        """Create an instance of Cache based on loaded configuration.

        Returns:
            An instance of `Cache` (either `RedisCache` or `InMemoryCache`).
        """
        match config.type:
            case constants.REDIS_CACHE:
                return RedisCache(config.redis)
            case constants.IN_MEMORY_CACHE:
                return InMemoryCache(config.memory)
            case _:
                raise ValueError(
                    f"Invalid cache type: {config.type}. "
                    f"Use '{constants.REDIS_CACHE}' or '{constants.IN_MEMORY_CACHE}' options."
                )
