"""OpenAI provider implementation."""

import logging
from typing import Any

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_OPENAI)
class OpenAI(LLMProvider):
    """OpenAI provider."""

    url: str = "https://api.openai.com/v1"

    @property
    def default_params(self) -> dict[str, Any]:
        """Default LLM params."""
        return {
            "base_url": str(self.provider_config.url) or self.url,
            "openai_api_key": self.provider_config.credentials,
            "model": self.model,
            "model_kwargs": {
                "top_p": 0.95,
                "frequency_penalty": 1.03,
            },
            "organization": None,
            "cache": None,
            "streaming": True,
            "temperature": 0.01,
            "max_tokens": 512,
            "verbose": False,
        }

    def load(self) -> LLM:
        """Load LLM."""
        return ChatOpenAI(**self.params)  # type: ignore [return-value]
