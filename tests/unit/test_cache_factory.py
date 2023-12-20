import pytest

from src import constants
from src.cache.cache_factory import CacheFactory, InMemoryCache
from app.models.config import ConversationCacheConfig


@pytest.fixture(scope="module")
def in_memory_cache_config():
    # os.environ["OLS_CONVERSATION_CACHE"] = constants.IN_MEMORY_CACHE
    c = ConversationCacheConfig(
        {
            "type": constants.IN_MEMORY_CACHE,
            constants.IN_MEMORY_CACHE: {"max_entries": 10},
        }
    )
    return c


@pytest.fixture(scope="module")
def invalid_cache_type_config():
    # os.environ["OLS_CONVERSATION_CACHE"] = "foo bar baz"
    c = ConversationCacheConfig()
    c.type = "foo bar baz"
    return c


def test_conversation_cache(in_memory_cache_config):
    """Check if InMemoryCache is returned by factory with proper env.var setup."""
    cache = CacheFactory.conversation_cache(in_memory_cache_config)
    assert cache is not None
    # check if the object has the right type
    assert isinstance(cache, InMemoryCache)


def test_conversation_cache_wrong_cache(invalid_cache_type_config):
    """Check if wrong cache env.variable is detected properly."""
    with pytest.raises(ValueError):
        CacheFactory.conversation_cache(invalid_cache_type_config)
