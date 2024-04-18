"""Unit tests for OpenAI provider."""

from unittest.mock import patch

from ols.app.models.config import LLMProviderConfig
from ols.src.llms.providers.watsonx import WatsonX
from ols.utils import config
from tests.mock_classes.mock_watsonxllm import WatsonxLLM


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_basic_interface():
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params
    provider_cfg = LLMProviderConfig(
        **{
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret.txt",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_url.com",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )

    watsonx = WatsonX(model="uber-model", params={}, provider_config=provider_cfg)
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
