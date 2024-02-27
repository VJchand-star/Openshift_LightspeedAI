"""LLM providers registry."""

import logging
from typing import Callable, ClassVar

from ols.src.llms.providers.provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMProvidersRegistry:
    """Registry for LLM providers."""

    llm_providers: ClassVar = {}

    @classmethod
    def register(cls, provider_type: str, llm_provider: Callable) -> None:
        """Register LLM provider."""
        if not issubclass(llm_provider, LLMProvider):
            raise ValueError(
                f"LLMProvider subclass required, got '{type(llm_provider)}'"
            )
        cls.llm_providers[provider_type] = llm_provider
        logger.debug(f"LLM provider '{provider_type}' registered")


def register_llm_provider_as(provider_type: str) -> Callable:
    """Register LLM provider in the `LLMProvidersRegistry`.

    Example:
    ```python
    @register_llm_provider_as("openai")
    class OpenAI(LLMProvider):
       pass
    ```
    """

    def decorator(cls: LLMProvider) -> Callable:
        LLMProvidersRegistry.register(provider_type, cls)
        return cls

    return decorator
