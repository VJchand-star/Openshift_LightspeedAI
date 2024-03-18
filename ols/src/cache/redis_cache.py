"""Cache that uses Redis to store cached values."""

import logging
import pickle
import threading
from typing import Any, Optional

import redis
from langchain_core.messages.base import BaseMessage
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    BusyLoadingError,
    ConnectionError,
    RedisError,
    WatchError,
)
from redis.retry import Retry

from ols.app.models.config import RedisConfig
from ols.src.cache.cache import Cache

logger = logging.getLogger(__name__)


# TODO
# Good for on-premise hosting for now
# Extend it to distributed setting using cloud offerings
class RedisCache(Cache):
    """Cache that uses Redis to store cached values."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls: type["RedisCache"], config: RedisConfig) -> "RedisCache":
        """Create a new instance of the `RedisCache` class."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(RedisCache, cls).__new__(cls)
                cls._instance.initialize_redis(config)
        return cls._instance

    def initialize_redis(self, config: RedisConfig) -> None:
        """Initialize the Redis client and logger.

        This method sets up the Redis client with custom configuration parameters.
        """
        kwargs: dict[str, Any] = {}
        if config.password is not None:
            kwargs["password"] = config.password
        if config.ca_cert_path is not None:
            kwargs["ssl"] = True
            kwargs["ssl_cert_reqs"] = "required"
            kwargs["ssl_ca_certs"] = config.ca_cert_path

        # setup Redis retry logic
        retry: Optional[Retry] = None
        if config.number_of_retries is not None and config.number_of_retries > 0:
            retry = Retry(ExponentialBackoff(), config.number_of_retries)

        retry_on_error: Optional[list[type[RedisError]]] = None
        if config.retry_on_error:
            retry_on_error = [BusyLoadingError, ConnectionError]

        # initialize Redis client
        self.redis_client = redis.StrictRedis(
            host=str(config.host),
            port=int(config.port),
            decode_responses=False,  # we store serialized messages as bytes, not strings
            retry=retry,
            retry_on_timeout=bool(config.retry_on_timeout),
            retry_on_error=retry_on_error,
            **kwargs,
        )
        # Set custom configuration parameters
        self.redis_client.config_set("maxmemory", config.max_memory)
        self.redis_client.config_set("maxmemory-policy", config.max_memory_policy)

    def get(self, user_id: str, conversation_id: str) -> Optional[list[BaseMessage]]:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """
        key = super().construct_key(user_id, conversation_id)

        value = self.redis_client.get(key)
        if value is None:
            return None
        return pickle.loads(value, errors="strict")  # noqa S301

    def insert_or_append(
        self, user_id: str, conversation_id: str, value: list[BaseMessage]
    ) -> None:
        """Set the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            value: The value to set.

        Raises:
            OutOfMemoryError: If item is evicted when Redis allocated
                memory is higher than maxmemory.
        """
        # lock in order to not allow client from the same process to make
        # changes
        with self._lock:
            self.insert_or_append_in_transaction(user_id, conversation_id, value)

    def insert_or_append_in_transaction(
        self, user_id: str, conversation_id: str, value: list[BaseMessage]
    ) -> None:
        """Set the value associated with given key in transaction."""
        key = super().construct_key(user_id, conversation_id)
        while True:
            # try to execute GET operation followed by transaction with
            # watching changes (made by other clients) in updated value
            with self.redis_client.pipeline(transaction=True) as pipeline:
                try:
                    logger.debug("Transaction started")
                    pipeline.watch(key)
                    old_value = self.get(user_id, conversation_id)
                    pipeline.multi()
                    if old_value:
                        old_value.extend(value)
                        pipeline.set(
                            key,
                            pickle.dumps(old_value, protocol=pickle.HIGHEST_PROTOCOL),
                        )
                    else:
                        pipeline.set(
                            key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                        )
                    pipeline.execute()
                    pipeline.unwatch()
                    logger.debug("Transaction finished")
                    # transaction finished with success, we can leave the "try" loop
                    break
                except WatchError as err:
                    logger.info(
                        "Watch error, need to repeat insert_or_append operation", err
                    )
