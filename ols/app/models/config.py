"""Config classes for the configuration structure."""

import logging
from typing import Dict, Optional

from pydantic import BaseModel

from ols.src import constants


class InvalidConfigurationError(Exception):
    """Exception raised when configuration is invalid."""

    pass


class ModelConfig(BaseModel):
    """Model configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credential_path: Optional[str] = None
    credentials: Optional[str] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credential_path = data.get("credential_path", None)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: Optional[str] = None
    url: Optional[str] = None
    credential_path: Optional[str] = None
    credentials: Optional[str] = None
    models: Dict[str, ModelConfig] = {}

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.name = data.get("name", None)
        self.url = data.get("url", None)
        self.credential_path = data.get("credential_path", None)
        if "models" not in data or len(data["models"]) == 0:
            raise InvalidConfigurationError(
                f"no models configured for provider {data['name']}"
            )
        for m in data["models"]:
            if "name" not in m:
                raise InvalidConfigurationError("model name is missing")
            model = ModelConfig(m)
            self.models[m["name"]] = model


class LLMConfig(BaseModel):
    """LLM configuration."""

    providers: Dict[str, ProviderConfig] = {}

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        for p in data:
            if "name" not in p:
                raise InvalidConfigurationError("provider name is missing")
            provider = ProviderConfig(p)
            self.providers[p["name"]] = provider


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: Optional[str] = None
    port: Optional[int] = None
    max_memory: Optional[str] = None
    max_memory_policy: Optional[str] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.host = data.get("host", None)
        self.port = data.get("port", None)
        self.max_memory = data.get("max_memory", constants.REDIS_CACHE_MAX_MEMORY)
        self.max_memory_policy = data.get(
            "max_memory_policy", constants.REDIS_CACHE_MAX_MEMORY_POLICY
        )


class MemoryConfig(BaseModel):
    """In-memory cache configuration."""

    max_entries: Optional[int] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.max_entries = data.get(
            "max_entries", constants.IN_MEMORY_CACHE_MAX_ENTRIES
        )


class ConversationCacheConfig(BaseModel):
    """Conversation cache configuration."""

    type: Optional[str] = None
    redis: Optional[RedisConfig] = None
    memory: Optional[MemoryConfig] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.type = data.get("type", None)
        if self.type == "redis":
            if "redis" not in data:
                raise InvalidConfigurationError("redis config is missing")
            self.redis = RedisConfig(data["redis"])
        elif self.type == "in-memory":
            if "in-memory" not in data:
                raise InvalidConfigurationError("in-memory config is missing")
            self.memory = MemoryConfig(data.get("memory", None))


class LoggerConfig(BaseModel):
    """Logger configuration."""

    default_level: Optional[int | str] = None
    default_filename: Optional[str] = None
    default_size: Optional[int] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        level = logging.getLevelName(data.get("default_level", "INFO"))
        if level is None:
            raise InvalidConfigurationError(
                f"invalid log level for default log: {data.get('default_level',None)}"
            )
        self.default_level = level
        self.default_filename = data.get("default_filename", None)
        self.default_size = data.get("default_size", (1048576 * 100))


class OLSConfig(BaseModel):
    """OLS configuration."""

    conversation_cache: Optional[ConversationCacheConfig] = None
    logger_config: Optional[LoggerConfig] = None

    enable_debug_ui: Optional[bool] = False
    default_model: Optional[str] = None
    default_provider: Optional[str] = None

    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None
    summarizer_provider: Optional[str] = None
    summarizer_model: Optional[str] = None
    validator_provider: Optional[str] = None
    validator_model: Optional[str] = None
    yaml_provider: Optional[str] = None
    yaml_model: Optional[str] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.default_provider = data.get("default_provider", None)
        self.default_model = data.get("default_model", None)
        self.classifier_provider = data.get(
            "classifier_provider", self.default_provider
        )
        self.classifier_model = data.get("classifier_model", self.default_model)
        self.summarizer_provider = data.get(
            "summarizer_provider", self.default_provider
        )
        self.summarizer_model = data.get("summarizer_model", self.default_model)
        self.validator_provider = data.get("validator_provider", self.default_provider)
        self.validator_model = data.get("validator_model", self.default_model)
        self.yaml_provider = data.get("yaml_provider", self.default_provider)
        self.yaml_model = data.get("yaml_model", self.default_model)

        self.enable_debug_ui = data.get("enable_debug_ui", False)
        self.conversation_cache = ConversationCacheConfig(
            data.get("conversation_cache", None)
        )
        self.logger_config = LoggerConfig(data.get("logger_config", None))


class Config:
    """Global service configuration."""

    llm_config: Optional[LLMConfig] = None
    ols_config: Optional[OLSConfig] = None

    def __init__(self, data: Optional[dict] = None):
        """Initialize configuration and perform basic validation."""
        super().__init__()
        if data is None:
            return
        self.llm_config = LLMConfig(data.get("llm_providers", None))
        self.ols_config = OLSConfig(data.get("ols_config", None))

    def validate(self) -> None:
        """Validate all configurations."""
        if self.llm_config is None:
            raise InvalidConfigurationError("no llm config found")
        if self.llm_config.providers is None or len(self.llm_config.providers) == 0:
            raise InvalidConfigurationError("no llm providers found")
        if self.ols_config is None:
            raise InvalidConfigurationError("no ols config found")
        if self.ols_config.default_model is None:
            raise InvalidConfigurationError("default model is not set")
        if self.ols_config.classifier_model is None:
            raise InvalidConfigurationError("classifier model is not set")
        if self.ols_config.conversation_cache is None:
            raise InvalidConfigurationError("conversation cache is not set")
        if self.ols_config.logger_config is None:
            raise InvalidConfigurationError("logger config is not set")
