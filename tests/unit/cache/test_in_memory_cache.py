"""Unit tests for InMemoryCache class."""

import pytest

from ols.app.models.config import MemoryConfig
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.utils import suid

conversation_id = suid.get_suid()


@pytest.fixture
def cache():
    """Fixture with constucted and initialized in memory cache object."""
    mc = MemoryConfig({"max_entries": "10"})
    c = InMemoryCache(mc)
    c.initialize_cache(mc)
    return c


def test_insert_or_append(cache):
    """Test the behavior of insert_or_append method."""
    cache.insert_or_append("user1", conversation_id, "value1")
    assert cache.get("user1", conversation_id) == "value1"


def test_insert_or_append_existing_key(cache):
    """Test the behavior of insert_or_append method for existing item."""
    cache.insert_or_append("user1", conversation_id, "value1")
    cache.insert_or_append("user1", conversation_id, "value2")
    assert cache.get("user1", conversation_id) == "value1\nvalue2"


def test_insert_or_append_overflow(cache):
    """Test if items in cache with defined capacity is handled correctly."""
    capacity = 5
    cache.capacity = capacity
    for i in range(capacity + 1):
        user = f"user{i}"
        value = f"value{i}"
        cache.insert_or_append(user, conversation_id, value)

    # Ensure the oldest entry is evicted
    assert cache.get("user0", conversation_id) is None
    # Ensure the newest entry is still present
    assert cache.get(f"user{capacity}", conversation_id) == f"value{capacity}"


def test_get_nonexistent_user(cache):
    """Test how non-existent items are handled by the cache."""
    assert cache.get("nonexistent_user", conversation_id) is None


def test_get_improper_user_id(cache):
    """Test how improper user ID is handled."""
    with pytest.raises(ValueError, match="Incorrect user ID"):
        cache.get(":", conversation_id)
    with pytest.raises(ValueError, match="Incorrect user ID"):
        cache.get("foo:bar", conversation_id)


def test_get_improper_conversation_id(cache):
    """Test how improper conversation ID is handled."""
    with pytest.raises(ValueError, match="Incorrect conversation ID"):
        cache.get("user1", "this-is-not-valid-uuid")


def test_singleton_pattern():
    """Test if in memory cache exists as one instance in memory."""
    mc = MemoryConfig({"max_entries": "10"})
    cache1 = InMemoryCache(mc)
    cache2 = InMemoryCache(mc)
    assert cache1 is cache2
