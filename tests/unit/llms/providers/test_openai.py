"""Unit tests for OpenAI provider."""

import pytest
from langchain_openai.chat_models.base import ChatOpenAI

from ols.app.models.config import LLMProviderConfig
from ols.src.llms.providers.openai import OpenAI
from ols.utils import config


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for OpenAI."""
    return LLMProviderConfig(
        **{
            "name": "some_provider",
            "type": "openai",
            "url": "http://test_url.com",
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_url.com",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    openai = OpenAI(model="uber-model", params={}, provider_config=provider_config)
    llm = openai.load()
    assert isinstance(llm, ChatOpenAI)
    assert openai.default_params
    assert "base_url" in openai.default_params
    assert "model" in openai.default_params
    assert "max_tokens" in openai.default_params


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
        "verbose": True,
    }

    openai = OpenAI(model="uber-model", params=params, provider_config=provider_config)
    llm = openai.load()
    assert isinstance(llm, ChatOpenAI)
    assert openai.default_params
    assert openai.params

    # known parameters should be there
    assert "temperature" in openai.params
    assert "verbose" in openai.params
    assert openai.params["temperature"] == 0.3
    assert openai.params["verbose"] is True

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in openai.params
    assert "max_new_tokens" not in openai.params
    assert "unknown_parameter" not in openai.params


def test_none_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
        "organization": None,
        "cache": None,
    }

    openai = OpenAI(model="uber-model", params=params, provider_config=provider_config)
    llm = openai.load()
    assert isinstance(llm, ChatOpenAI)
    assert openai.default_params
    assert openai.params

    # known parameters should be there, with None values
    assert "cache" in openai.params
    assert "organization" in openai.params
    assert openai.params["cache"] is None
    assert openai.params["organization"] is None

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in openai.params
    assert "max_new_tokens" not in openai.params
    assert "unknown_parameter" not in openai.params
