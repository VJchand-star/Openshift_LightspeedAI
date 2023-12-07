from collections import deque
from typing import Union
import threading
from lightspeed_service import constants


class InMemoryCache:
    """An in-memory LRU cache implementation in O(1) time."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Implement Singleton pattern with thread safety."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(InMemoryCache, cls).__new__(cls)
                cls._instance.initialize_cache()
        return cls._instance

    def initialize_cache(self):
        """
        Initialize the InMemoryCache.
        """
        self.capacity = constants.IN_MEMORY_CACHE_MAX_ENTRIES
        self.deque = deque()
        self.cache = {}

    def get(self, key: str) -> Union[str, None]:
        """
        Get the value associated with the given key.

        Args:
        - key (str): The key to look up in the cache.

        Returns:
        - Union[str, None]: The value associated with the key, or None if the key is not present.
        """
        if key not in self.cache:
            return None

        self.deque.remove(key)
        self.deque.appendleft(key)
        value = self.cache[key]
        return value

    def insert_or_append(self, key: str, value: str) -> None:
        """
        sets the value if a key is not present or else simply appends.

        Args:
        - key (str): The key to set in the cache.
        - value (str): The value to associate with the key.

        Returns:
        - None
        """
        with self._lock:
            if key not in self.cache:
                if len(self.deque) == self.capacity:
                    oldest = self.deque.pop()
                    del self.cache[oldest]
                self.cache[key] = value
            else:
                self.deque.remove(key)
                oldValue = self.cache[key]
                self.cache[key] = oldValue + "\n" + value
            self.deque.appendleft(key)
