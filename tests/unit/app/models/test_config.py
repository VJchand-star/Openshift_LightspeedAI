"""Unit tests for data models."""

import logging

import pytest
from pydantic import ValidationError

from ols import constants
from ols.app.models.config import (
    AuthenticationConfig,
    Config,
    ConversationCacheConfig,
    InMemoryCacheConfig,
    InvalidConfigurationError,
    LLMProviders,
    LoggingConfig,
    ModelConfig,
    OLSConfig,
    PostgresConfig,
    ProviderConfig,
    QueryFilter,
    RedisConfig,
    ReferenceContent,
    UserDataCollection,
)


def test_model_config():
    """Test the ModelConfig model."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "options": {
                "foo": 1,
                "bar": 2,
            },
        }
    )

    assert model_config.name == "test_name"
    assert model_config.url == "test_url"
    assert model_config.credentials == "secret_key"
    assert model_config.options == {
        "foo": 1,
        "bar": 2,
    }

    model_config = ModelConfig()
    assert model_config.name is None
    assert model_config.url is None
    assert model_config.credentials is None
    assert model_config.options is None


def test_model_config_path_to_secret_directory():
    """Test the ModelConfig model."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "test_url",
            "credentials_path": "tests/config/secret",
            "options": {
                "foo": 1,
                "bar": 2,
            },
        }
    )

    assert model_config.credentials == "secret_key"


def test_model_config_equality():
    """Test the ModelConfig equality check."""
    model_config_1 = ModelConfig()
    model_config_2 = ModelConfig()

    # compare the same model configs
    assert model_config_1 == model_config_2

    # compare different model configs
    model_config_2.name = "some non-default name"
    assert model_config_1 != model_config_2

    # compare with value of different type
    other_value = "foo"
    assert model_config_1 != other_value


def test_model_config_validation_proper_config():
    """Test the ModelConfig model validation."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "options": {
                "foo": 1,
                "bar": 2,
            },
        }
    )
    # validation should not fail
    model_config.validate_yaml()


def test_model_config_no_options():
    """Test the ModelConfig model validation."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
        }
    )
    # validation should not fail because model options are fully optional
    model_config.validate_yaml()


def test_model_config_validation_no_credentials_path():
    """Test the ModelConfig model validation when path to credentials is not provided."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": None,
        }
    )
    # validation should not fail
    model_config.validate_yaml()
    assert model_config.credentials is None


def test_model_config_validation_empty_model():
    """Test the ModelConfig model validation when model is empty."""
    model_config = ModelConfig()

    # validation should fail
    with pytest.raises(InvalidConfigurationError, match="model name is missing"):
        model_config.validate_yaml()


def test_model_config_wrong_options():
    """Test the ModelConfig model validation."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "options": "not-dictionary",
        }
    )

    # validation should fail
    with pytest.raises(
        InvalidConfigurationError, match="model options must be dictionary"
    ):
        model_config.validate_yaml()


def test_model_config_wrong_option_key():
    """Test the ModelConfig model validation."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "options": {
                42: "answer",
            },
        }
    )

    # validation should fail
    with pytest.raises(
        InvalidConfigurationError, match="key for model option must be string"
    ):
        model_config.validate_yaml()


def test_model_config_validation_missing_name():
    """Test the ModelConfig model validation when model name is missing."""
    model_config = ModelConfig(
        {
            "name": None,
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
        }
    )

    # validation should fail
    with pytest.raises(InvalidConfigurationError, match="model name is missing"):
        model_config.validate_yaml()


def test_model_config_validation_improper_url():
    """Test the ModelConfig model validation when URL is incorrect."""
    model_config = ModelConfig(
        {
            "name": "test_name",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret/apitoken",
        }
    )

    # validation should fail
    with pytest.raises(InvalidConfigurationError, match="model URL is invalid"):
        model_config.validate_yaml()


def test_model_config_invalid_response_token():
    """Test the model config with invalid response token limit."""
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid response_token_limit = 0, positive value expected",
    ):
        ModelConfig({"name": "test_model_name", "response_token_limit": 0})


def test_model_config_higher_response_token():
    """Test the model config with response token >= context window."""
    with pytest.raises(
        InvalidConfigurationError,
        match="Context window size 2, should be greater than response token limit 2",
    ):
        ModelConfig(
            {
                "name": "test_model_name",
                "context_window_size": 2,
                "response_token_limit": 2,
            }
        )


def test_provider_config():
    """Test the ProviderConfig model."""
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    assert provider_config.name == "test_name"
    assert provider_config.type == "bam"
    assert provider_config.url == "test_url"
    assert provider_config.credentials == "secret_key"
    assert provider_config.project_id == "test_project_id"
    assert len(provider_config.models) == 1
    assert provider_config.models["test_model_name"].name == "test_model_name"
    assert provider_config.models["test_model_name"].url == "test_model_url"
    assert provider_config.models["test_model_name"].credentials == "secret_key"
    assert (
        provider_config.models["test_model_name"].context_window_size
        == constants.DEFAULT_CONTEXT_WINDOW_SIZE
    )
    assert (
        provider_config.models["test_model_name"].response_token_limit
        == constants.DEFAULT_RESPONSE_TOKEN_LIMIT
    )
    assert provider_config.openai_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None

    provider_config = ProviderConfig()
    assert provider_config.name is None
    assert provider_config.url is None
    assert provider_config.credentials is None
    assert provider_config.project_id is None
    assert len(provider_config.models) == 0

    assert provider_config.openai_config is None
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [],
            }
        )
    assert "no models configured for provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [
                    {
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )
    assert "model name is missing" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ProviderConfig(
            {
                "name": "azure_openai",
                "type": "azure_openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [
                    {
                        "name": "test_model",
                    }
                ],
            }
        )
    assert "deployment_name is required" in str(excinfo.value)


def test_that_url_is_required_provider_parameter():
    """Test that provider-specific URL is required attribute."""
    # provider type is set to "azure_openai"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "azure_openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "azure_openai_config": {
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "openai"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "credentials_path": "tests/config/secret/apitoken",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "bam"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "bam_config": {
                    "credentials_path": "tests/config/secret/apitoken",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "watsonx"
    with pytest.raises(ValidationError, match="url"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "watsonx",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "watsonx_config": {
                    "credentials_path": "tests/config/secret/apitoken",
                    "project_id": "*project id*",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_that_credentials_is_required_provider_parameter():
    """Test that provider-specific credentials is required attribute for any provider but Azure."""
    # provider type is set to "openai"
    with pytest.raises(ValidationError, match="credentials"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "bam"
    with pytest.raises(ValidationError, match="credentials"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "bam_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )

    # provider type is set to "watsonx"
    with pytest.raises(ValidationError, match="credentials"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "watsonx",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "watsonx_config": {
                    "project_id": "*project id*",
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_azure_openai_specific():
    """Test if Azure OpenAI-specific config is loaded and validated."""
    # provider type is set to "azure_openai" and Azure OpenAI-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "deployment_name": "deploment-name",
            "azure_openai_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret_azure_tenant_id_client_id_client_secret",
                "deployment_name": "deployment-name",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # Azure OpenAI-specific configuration must be present
    assert provider_config.azure_config is not None
    assert str(provider_config.azure_config.url) == "http://localhost/"
    assert (
        provider_config.azure_config.tenant_id == "00000000-0000-0000-0000-000000000001"
    )
    assert (
        provider_config.azure_config.client_id == "00000000-0000-0000-0000-000000000002"
    )
    assert provider_config.azure_config.deployment_name == "deployment-name"
    assert provider_config.azure_config.client_secret == "client secret"  # noqa: S105
    assert provider_config.azure_config.api_key is None

    # configuration for other providers must not be set
    assert provider_config.openai_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_azure_openai_unknown_parameters():
    """Test if unknown Azure OpenAI parameters are detected."""
    # provider type is set to "azure_openai" and Azure OpenAI-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "azure_openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "azure_openai_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_openai_specific():
    """Test if OpenAI-specific config is loaded and validated."""
    # provider type is set to "openai" and OpenAI-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "openai_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # OpenAI-specific configuration must be present
    assert provider_config.openai_config is not None
    assert str(provider_config.openai_config.url) == "http://localhost/"
    assert provider_config.openai_config.api_key == "secret_key"

    # configuration for other providers must not be set
    assert provider_config.azure_config is None
    assert provider_config.watsonx_config is None
    assert provider_config.bam_config is None


def test_provider_config_openai_unknown_parameters():
    """Test if unknown OpenAI parameters are detected."""
    # provider type is set to "openai" and OpenAI-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "openai",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "deploment-name",
                "openai_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "tenant_id": "tenant-ID",
                    "client_id": "client-ID",
                    "client_secret_path": "tests/config/secret/apitoken",
                    "credentials_path": "tests/config/secret/apitoken",
                    "deployment_name": "deployment-name",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_watsonx_specific():
    """Test if Watsonx-specific config is loaded and validated."""
    # provider type is set to "watsonx" and Watsonx-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "watsonx",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "watsonx_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "*project id*",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # Watsonx-specific configuration must be present
    assert provider_config.watsonx_config is not None
    assert str(provider_config.watsonx_config.url) == "http://localhost/"
    assert provider_config.watsonx_config.project_id == "*project id*"
    assert provider_config.watsonx_config.api_key == "secret_key"

    # configuration for other providers must not be set
    assert provider_config.azure_config is None
    assert provider_config.openai_config is None
    assert provider_config.bam_config is None


def test_provider_config_watsonx_unknown_parameters():
    """Test if unknown Watsonx parameters are detected."""
    # provider type is set to "watsonx" and Watsonx-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "watsonx",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "watsonx_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "credentials_path": "tests/config/secret/apitoken",
                    "project_id": "*project id*",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_provider_config_bam_specific():
    """Test if BAM-specific config is loaded and validated."""
    # provider type is set to "bam" and BAM-specific configuration is there
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "bam_config": {
                "url": "http://localhost",
                "credentials_path": "tests/config/secret/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )
    # BAM-specific configuration must be present
    assert provider_config.bam_config is not None
    assert str(provider_config.bam_config.url) == "http://localhost/"
    assert provider_config.bam_config.api_key == "secret_key"

    # configuration for other providers must not be set
    assert provider_config.azure_config is None
    assert provider_config.openai_config is None
    assert provider_config.watsonx_config is None


def test_provider_config_bam_unknown_parameters():
    """Test if unknown BAM parameters are detected."""
    # provider type is set to "bam" and BAM-specific configuration is there
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "bam_config": {
                    "unknown_parameter": "unknown value",
                    "url": "http://localhost",
                    "credentials_path": "tests/config/secret/apitoken",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_improper_provider_specific_config():
    """Test if check for improper provider-specific config is performed."""
    with pytest.raises(
        InvalidConfigurationError,
        match="provider type bam selected, but configuration is set for different provider",
    ):
        # provider type is set to "bam" but OpenAI-specific configuration is there
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


def test_multiple_provider_specific_configs():
    """Test if check for multiple provider-specific configs is performed."""
    with pytest.raises(
        InvalidConfigurationError,
        match="multiple provider-specific configurations found, but just one is expected for provider bam",  # noqa E501
    ):
        # two provider-specific configurations is in the configuration
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "openai_config": {
                    "url": "http://localhost",
                },
                "watsonx_config": {
                    "url": "http://localhost",
                },
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            }
        )


providers = (
    constants.PROVIDER_BAM,
    constants.PROVIDER_OPENAI,
    constants.PROVIDER_AZURE_OPENAI,
    constants.PROVIDER_WATSONX,
)

models = (
    constants.GRANITE_13B_CHAT_V1,
    constants.GRANITE_13B_CHAT_V2,
    constants.GPT4_TURBO,
    constants.GPT35_TURBO,
    "test",
)


@pytest.mark.parametrize("provider_name", providers)
@pytest.mark.parametrize("model_name", models)
def test_provider_model_specific_tokens_limit(provider_name, model_name):
    """Test if the model specific token limits are set as default."""
    # provider config with attributes 'blended' for all providers
    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": provider_name,
            "url": "test_url",
            "deployment_name": "test",
            "project_id": 42,
            "models": [
                {
                    "name": model_name,
                }
            ],
        }
    )
    # expected token limit for given model
    expected_limit = constants.DEFAULT_CONTEXT_WINDOW_SIZE

    # some provider+model combinations are not specified; in this case
    # default value is used instead
    expected_limit = constants.CONTEXT_WINDOW_SIZES.get(provider_name).get(
        model_name, constants.DEFAULT_CONTEXT_WINDOW_SIZE
    )
    assert provider_config.models[model_name].context_window_size == expected_limit
    if model_name == "test":
        assert (
            provider_config.models[model_name].context_window_size
            == constants.DEFAULT_CONTEXT_WINDOW_SIZE
        )


@pytest.mark.parametrize("model_name", models)
def test_provider_config_explicit_tokens(model_name):
    """Test the ProviderConfig model when explicit tokens are specified."""
    context_window_size = 500
    response_token_limit = 100

    provider_config = ProviderConfig(
        {
            "name": "test_name",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "test_project_id",
            "models": [
                {
                    "name": model_name,
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "context_window_size": context_window_size,
                    "response_token_limit": response_token_limit,
                }
            ],
        }
    )
    assert provider_config.models[model_name].context_window_size == context_window_size
    assert (
        provider_config.models[model_name].response_token_limit == response_token_limit
    )


def test_provider_config_improper_context_window_size_value():
    """Test the ProviderConfig model when improper context window size is specified."""
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid context_window_size = -1, positive value expected",
    ):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "context_window_size": -1,
                    }
                ],
            }
        )


def test_provider_config_improper_context_window_size_type():
    """Test the ProviderConfig model when improper context window size is specified."""
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid context_window_size = not-a-number, positive value expected",
    ):
        ProviderConfig(
            {
                "name": "test_name",
                "type": "bam",
                "url": "test_url",
                "credentials_path": "tests/config/secret/apitoken",
                "project_id": "test_project_id",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "context_window_size": "not-a-number",
                    }
                ],
            }
        )


def test_provider_config_equality():
    """Test the ProviderConfig equality check."""
    provider_config_1 = ProviderConfig()
    provider_config_2 = ProviderConfig()

    # compare the same provider configs
    assert provider_config_1 == provider_config_2

    # compare different model configs
    provider_config_2.name = "some non-default name"
    assert provider_config_1 != provider_config_2

    # compare with value of different type
    other_value = "foo"
    assert provider_config_1 != other_value


def test_provider_config_validation_proper_config():
    """Test the ProviderConfig model validation."""
    provider_config = ProviderConfig(
        {
            "name": "bam",
            "url": "http://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )

    provider_config.validate_yaml()


def test_provider_config_validation_improper_url():
    """Test the ProviderConfig model validation for improper URL."""
    provider_config = ProviderConfig(
        {
            "name": "bam",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )

    with pytest.raises(InvalidConfigurationError, match="provider URL is invalid"):
        provider_config.validate_yaml()


def test_provider_config_validation_missing_name():
    """Test the ProviderConfig model validation for missing name."""
    provider_config = ProviderConfig(
        {
            "type": "bam",
            "url": "httpXXX://test.url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )

    with pytest.raises(InvalidConfigurationError, match="provider name is missing"):
        provider_config.validate_yaml()


def test_provider_config_validation_no_credentials_path():
    """Test the ProviderConfig model validation when path to credentials is not provided."""
    provider_config = ProviderConfig(
        {
            "name": "bam",
            "url": "http://test.url",
            "credentials_path": None,
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test.model.url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )

    provider_config.validate_yaml()
    assert provider_config.credentials is None


def test_llm_providers():
    """Test the LLMProviders model."""
    llm_providers = LLMProviders(
        [
            {
                "name": "test_provider_name",
                "type": "bam",
                "url": "test_provider_url",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [
                    {
                        "name": "test_model_name",
                        "url": "test_model_url",
                        "credentials_path": "tests/config/secret/apitoken",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider_name"].name == "test_provider_name"
    assert llm_providers.providers["test_provider_name"].type == "bam"
    assert llm_providers.providers["test_provider_name"].url == "test_provider_url"
    assert llm_providers.providers["test_provider_name"].credentials == "secret_key"
    assert len(llm_providers.providers["test_provider_name"].models) == 1
    assert (
        llm_providers.providers["test_provider_name"].models["test_model_name"].name
        == "test_model_name"
    )
    assert (
        llm_providers.providers["test_provider_name"].models["test_model_name"].url
        == "test_model_url"
    )
    assert (
        llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .credentials
        == "secret_key"
    )

    llm_providers = LLMProviders()
    assert len(llm_providers.providers) == 0

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [],
                },
            ]
        )
    assert "provider name is missing" in str(excinfo.value)


def test_llm_providers_type_defaulting():
    """Test that provider type is defaulted from provider name."""
    llm_providers = LLMProviders(
        [
            {
                "name": "bam",
                "models": [
                    {
                        "name": "m1",
                        "url": "https://test_model_url",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["bam"].name == "bam"
    assert llm_providers.providers["bam"].type == "bam"

    llm_providers = LLMProviders(
        [
            {
                "name": "test_provider",
                "type": "bam",
                "models": [
                    {
                        "name": "m1",
                        "url": "https://test_model_url",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider"].name == "test_provider"
    assert llm_providers.providers["test_provider"].type == "bam"


def test_llm_providers_type_validation():
    """Test that only known provider types are allowed."""
    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "name": "invalid_provider",
                },
            ]
        )
    assert "invalid provider type: invalid_provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {"name": "bam", "type": "invalid_type"},
            ]
        )
    assert "invalid provider type: invalid_type" in str(excinfo.value)


def test_llm_providers_watsonx_required_projectid():
    """Test that project_id is required for Watsonx provider."""
    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "name": "watsonx",
                },
            ]
        )
    assert "project_id is required for Watsonx provider" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        LLMProviders(
            [
                {
                    "name": "test_watsonx",
                    "type": "watsonx",
                },
            ]
        )
    assert "project_id is required for Watsonx provider" in str(excinfo.value)

    llm_providers = LLMProviders(
        [
            {
                "name": "watsonx",
                "project_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
                "models": [
                    {
                        "name": "m1",
                        "url": "https://test_model_url",
                    }
                ],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["watsonx"].name == "watsonx"
    assert llm_providers.providers["watsonx"].type == "watsonx"
    assert (
        llm_providers.providers["watsonx"].project_id
        == "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    )

    llm_providers = LLMProviders(
        [
            {
                "name": "test_provider",
                "type": "watsonx",
                "project_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
                "models": [{"name": "test_model_name", "url": "test_model_url"}],
            },
        ]
    )
    assert len(llm_providers.providers) == 1
    assert llm_providers.providers["test_provider"].name == "test_provider"
    assert llm_providers.providers["test_provider"].type == "watsonx"
    assert (
        llm_providers.providers["test_provider"].project_id
        == "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    )


def test_llm_providers_equality():
    """Test the LLMProviders equality check."""
    provider_config_1 = LLMProviders()
    provider_config_2 = LLMProviders()

    # compare same providers
    assert provider_config_1 == provider_config_2

    # compare different providers
    provider_config_2.providers = [ProviderConfig()]
    assert provider_config_1 != provider_config_2

    # compare with value of different type
    other_value = "foo"
    assert provider_config_1 != other_value


def test_valid_values():
    """Test valid values."""
    # test default values
    logging_config = LoggingConfig({})
    assert logging_config.app_log_level == logging.INFO
    assert logging_config.lib_log_level == logging.WARNING
    assert logging_config.uvicorn_log_level == logging.WARNING

    # test custom values
    logging_config = LoggingConfig(
        {
            "app_log_level": "debug",
            "lib_log_level": "debug",
            "uvicorn_log_level": "debug",
        }
    )
    assert logging_config.app_log_level == logging.DEBUG
    assert logging_config.lib_log_level == logging.DEBUG
    assert logging_config.uvicorn_log_level == logging.DEBUG

    logging_config = LoggingConfig()
    assert logging_config.app_log_level == logging.INFO


def test_invalid_values():
    """Test invalid values."""
    # value is not string
    with pytest.raises(InvalidConfigurationError, match="invalid log level for 5"):
        LoggingConfig({"app_log_level": 5})

    # value is not valid log level
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid log level for app_log_level: dingdong",
    ):
        LoggingConfig({"app_log_level": "dingdong"})

    # value is not valid log level
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid log level for uvicorn_log_level: foo",
    ):
        LoggingConfig({"uvicorn_log_level": "foo"})


def test_postgres_config_default_values():
    """Test the PostgresConfig model."""
    postgres_config = PostgresConfig()
    assert postgres_config.host == constants.POSTGRES_CACHE_HOST
    assert postgres_config.port == constants.POSTGRES_CACHE_PORT
    assert postgres_config.dbname == constants.POSTGRES_CACHE_DBNAME
    assert postgres_config.user == constants.POSTGRES_CACHE_USER
    assert postgres_config.max_entries == constants.POSTGRES_CACHE_MAX_ENTRIES


def test_postgres_config_correct_values():
    """Test the PostgresConfig model when correct values are used."""
    postgres_config = PostgresConfig(
        host="other_host",
        port=1234,
        dbname="my_database",
        user="admin",
        ssl_mode="allow",
        max_entries=42,
    )

    # explicitly set values
    assert postgres_config.host == "other_host"
    assert postgres_config.port == 1234
    assert postgres_config.dbname == "my_database"
    assert postgres_config.user == "admin"
    assert postgres_config.ssl_mode == "allow"
    assert postgres_config.max_entries == 42


def test_postgres_config_wrong_port():
    """Test the PostgresConfig model."""
    with pytest.raises(
        ValidationError, match="The port needs to be between 0 and 65536"
    ):
        PostgresConfig(
            host="other_host",
            port=9999999,
            dbname="my_database",
            user="admin",
            ssl_mode="allow",
        )


def test_postgres_config_equality():
    """Test the PostgresConfig equality check."""
    postgres_config_1 = PostgresConfig()
    postgres_config_2 = PostgresConfig()

    # compare the same Postgres configs
    assert postgres_config_1 == postgres_config_2

    # compare different Postgres configs
    postgres_config_2.host = "12.34.56.78"
    assert postgres_config_1 != postgres_config_2

    # compare with value of different type
    other_value = "foo"
    assert postgres_config_1 != other_value


def test_postgres_config_with_password():
    """Test the PostgresConfig model."""
    postgres_config = PostgresConfig(
        host="other_host",
        port=1234,
        dbname="my_database",
        user="admin",
        password_path="tests/config/postgres_password.txt",  # noqa: S106
        ssl_mode="allow",
        max_entries=42,
    )
    # check if password was read correctly from file
    assert postgres_config.password == "postgres_password"  # noqa: S105


def test_redis_config():
    """Test the RedisConfig model."""
    redis_config = RedisConfig({})
    # default values
    assert redis_config.retry_on_error == constants.REDIS_RETRY_ON_ERROR
    assert redis_config.retry_on_timeout == constants.REDIS_RETRY_ON_TIMEOUT
    assert redis_config.number_of_retries == constants.REDIS_NUMBER_OF_RETRIES

    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "retry_on_error": "false",
            "retry_on_timeout": "false",
            "number_of_retries": 42,
        }
    )

    # explicitly set values
    assert redis_config.host == "localhost"
    assert redis_config.port == 6379
    assert redis_config.max_memory == "200mb"
    assert redis_config.max_memory_policy == "allkeys-lru"
    assert redis_config.retry_on_error is False
    assert redis_config.retry_on_timeout is False
    assert redis_config.number_of_retries == 42

    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "retry_on_error": "true",
            "retry_on_timeout": "true",
            "number_of_retries": 100,
        }
    )
    assert redis_config.host == "localhost"
    assert redis_config.port == 6379
    assert redis_config.max_memory == "200mb"
    assert redis_config.max_memory_policy == "allkeys-lru"
    assert redis_config.retry_on_error is True
    assert redis_config.retry_on_timeout is True
    assert redis_config.number_of_retries == 100

    redis_config = RedisConfig()

    # initial values
    assert redis_config.host is None
    assert redis_config.port is None
    assert redis_config.max_memory is None
    assert redis_config.max_memory_policy is None
    assert redis_config.retry_on_error is None
    assert redis_config.retry_on_timeout is None
    assert redis_config.number_of_retries is None


def test_redis_config_with_ca_cert_path():
    """Test the RedisConfig model with CA certificate path."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "ca_cert_path": "tests/config/redis_ca_cert.crt",
        }
    )
    assert redis_config.ca_cert_path == "tests/config/redis_ca_cert.crt"


def test_redis_config_with_no_ca_cert_path():
    """Test the RedisConfig model with no CA certificate path."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
        }
    )
    assert redis_config.ca_cert_path is None


def test_redis_config_with_password():
    """Test the RedisConfig model."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
            "password_path": "tests/config/redis_password.txt",
        }
    )
    assert redis_config.password == "redis_password"  # noqa: S105


def test_redis_config_with_no_password():
    """Test the RedisConfig model with no password."""
    redis_config = RedisConfig(
        {
            "host": "localhost",
            "port": 6379,
            "max_memory": "200mb",
            "max_memory_policy": "allkeys-lru",
        }
    )
    assert redis_config.password is None


def test_redis_config_with_invalid_password_path():
    """Test the RedisConfig model with invalid password path."""
    with pytest.raises(Exception):
        RedisConfig(
            {
                "host": "localhost",
                "port": 6379,
                "max_memory": "200mb",
                "max_memory_policy": "allkeys-lru",
                "password_path": "/dev/null/foobar",
            }
        )


def test_redis_config_equality():
    """Test the RedisConfig equality check."""
    redis_config_1 = RedisConfig()
    redis_config_2 = RedisConfig()

    # compare the same Redis configs
    assert redis_config_1 == redis_config_2

    # compare different Redis configs
    redis_config_2.host = "12.34.56.78"
    assert redis_config_1 != redis_config_2

    # compare with value of different type
    other_value = "foo"
    assert redis_config_1 != other_value


def test_memory_cache_config():
    """Test the MemoryCacheConfig model."""
    memory_cache_config = InMemoryCacheConfig(
        {
            "max_entries": 100,
        }
    )
    assert memory_cache_config.max_entries == 100

    memory_cache_config = InMemoryCacheConfig()
    assert memory_cache_config.max_entries is None


def test_memory_cache_config_improper_entries():
    """Test the MemoryCacheConfig model if improper max_entries is used."""
    with pytest.raises(
        InvalidConfigurationError,
        match="invalid max_entries for memory conversation cache",
    ):
        InMemoryCacheConfig(
            {
                "max_entries": -100,
            }
        )


def test_memory_config_equality():
    """Test the MemoryConfig equality check."""
    memory_config_1 = InMemoryCacheConfig()
    memory_config_2 = InMemoryCacheConfig()

    # compare the same memory configs
    assert memory_config_1 == memory_config_2

    # compare different memory configs
    memory_config_2.max_entries = 123456
    assert memory_config_1 != memory_config_2

    # compare with value of different type
    other_value = "foo"
    assert memory_config_1 != other_value


def test_conversation_cache_config():
    """Test the ConversationCacheConfig model."""
    conversation_cache_config = ConversationCacheConfig(
        {
            "type": "memory",
            "memory": {
                "max_entries": 100,
            },
        }
    )
    assert conversation_cache_config.type == "memory"
    assert conversation_cache_config.memory.max_entries == 100

    conversation_cache_config = ConversationCacheConfig(
        {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
                "max_memory": "200mb",
                "max_memory_policy": "allkeys-lru",
            },
        }
    )
    assert conversation_cache_config.type == "redis"
    assert conversation_cache_config.redis.host == "localhost"
    assert conversation_cache_config.redis.port == 6379
    assert conversation_cache_config.redis.max_memory == "200mb"
    assert conversation_cache_config.redis.max_memory_policy == "allkeys-lru"

    conversation_cache_config = ConversationCacheConfig(
        {
            "type": "postgres",
            "postgres": {
                "host": "1.2.3.4",
                "port": 1234,
                "dbname": "testdb",
                "user": "user",
                "ssl_mode": "allow",
            },
        }
    )
    assert conversation_cache_config.type == "postgres"
    assert conversation_cache_config.postgres.host == "1.2.3.4"
    assert conversation_cache_config.postgres.port == 1234
    assert conversation_cache_config.postgres.dbname == "testdb"
    assert conversation_cache_config.postgres.user == "user"
    assert conversation_cache_config.postgres.ssl_mode == "allow"

    conversation_cache_config = ConversationCacheConfig()
    assert conversation_cache_config.type is None
    assert conversation_cache_config.redis is None
    assert conversation_cache_config.memory is None
    assert conversation_cache_config.postgres is None

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ConversationCacheConfig({"type": "redis"})
    assert "redis configuration is missing" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ConversationCacheConfig({"type": "memory"})
    assert "memory configuration is missing" in str(excinfo.value)

    with pytest.raises(InvalidConfigurationError) as excinfo:
        ConversationCacheConfig({"type": "postgres"})
    assert "Postgres configuration is missing" in str(excinfo.value)


def test_conversation_cache_config_validation():
    """Test the ConversationCacheConfig validation."""
    conversation_cache_config = ConversationCacheConfig()

    # not specified cache type case
    conversation_cache_config.type = None
    with pytest.raises(
        InvalidConfigurationError, match="missing conversation cache type"
    ):
        conversation_cache_config.validate_yaml()

    # unknown cache type case
    conversation_cache_config.type = "unknown"
    with pytest.raises(
        InvalidConfigurationError, match="unknown conversation cache type: unknown"
    ):
        conversation_cache_config.validate_yaml()


def test_conversation_cache_config_equality():
    """Test the ConversationCacheConfig equality check."""
    conversation_cache_config_1 = ConversationCacheConfig()
    conversation_cache_config_2 = ConversationCacheConfig()

    # compare the same conversation_cache configs
    assert conversation_cache_config_1 == conversation_cache_config_2

    # compare different conversation_cache configs
    conversation_cache_config_2.type = "some non-default type"
    assert conversation_cache_config_1 != conversation_cache_config_2

    # compare with value of different type
    other_value = "foo"
    assert conversation_cache_config_1 != other_value


def test_ols_config(tmpdir):
    """Test the OLSConfig model."""
    ols_config = OLSConfig(
        {
            "default_provider": "test_default_provider",
            "default_model": "test_default_model",
            "conversation_cache": {
                "type": "memory",
                "memory": {
                    "max_entries": 100,
                },
            },
            "logging_config": {
                "logging_level": "INFO",
            },
            "user_data_collection": {
                "feedback_disabled": False,
                "feedback_storage": tmpdir.strpath,
            },
        }
    )
    assert ols_config.default_provider == "test_default_provider"
    assert ols_config.default_model == "test_default_model"
    assert ols_config.conversation_cache.type == "memory"
    assert ols_config.conversation_cache.memory.max_entries == 100
    assert ols_config.logging_config.app_log_level == logging.INFO
    assert ols_config.query_validation_method == constants.QueryValidationMethod.LLM
    assert ols_config.user_data_collection.feedback_disabled is False
    assert ols_config.user_data_collection.feedback_storage == tmpdir.strpath
    assert ols_config.reference_content is None


def test_config():
    """Test the Config model of the Global service configuration."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "test_provider_name",
                    "type": "bam",
                    "url": "test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "test_model_url",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
            ],
            "ols_config": {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
                "query_validation_method": "disabled",
            },
            "dev_config": {"disable_tls": "true"},
        }
    )
    assert len(config.llm_providers.providers) == 1
    assert (
        config.llm_providers.providers["test_provider_name"].name
        == "test_provider_name"
    )
    assert (
        config.llm_providers.providers["test_provider_name"].url == "test_provider_url"
    )
    assert (
        config.llm_providers.providers["test_provider_name"].credentials == "secret_key"
    )
    assert len(config.llm_providers.providers["test_provider_name"].models) == 1
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .name
        == "test_model_name"
    )
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .url
        == "test_model_url"
    )
    assert (
        config.llm_providers.providers["test_provider_name"]
        .models["test_model_name"]
        .credentials
        == "secret_key"
    )
    assert config.ols_config.default_provider == "test_default_provider"
    assert config.ols_config.default_model == "test_default_model"
    assert config.ols_config.conversation_cache.type == "memory"
    assert config.ols_config.conversation_cache.memory.max_entries == 100
    assert config.ols_config.logging_config.app_log_level == logging.ERROR
    assert (
        config.ols_config.query_validation_method
        == constants.QueryValidationMethod.DISABLED
    )


def test_config_improper_missing_model():
    """Test the Config model of the Global service configuration when model is missing."""
    with pytest.raises(InvalidConfigurationError, match="default_model is missing"):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "type": "bam",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 1000,
                        },
                    },
                    "default_provider": "test_default_provider",
                },
                "dev_config": {"disable_tls": "true"},
            }
        ).validate_yaml()


def test_config_improper_missing_provider():
    """Test the Config model of the Global service configuration when provider is missing."""
    with pytest.raises(InvalidConfigurationError, match="default_provider is missing"):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "type": "bam",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 1000,
                        },
                    },
                    "default_model": "test_default_model",
                },
                "dev_config": {"disable_tls": "true"},
            }
        ).validate_yaml()


def test_config_improper_provider():
    """Test the Config model of the Global service configuration when improper provider is set."""
    with pytest.raises(
        InvalidConfigurationError,
        match="default_provider specifies an unknown provider test_default_provider",
    ):
        Config(
            {
                "llm_providers": [],
                "ols_config": {
                    "default_provider": "test_default_provider",
                    "default_model": "test_default_model",
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 100,
                        },
                    },
                    "logging_config": {
                        "app_log_level": "error",
                    },
                },
                "dev_config": {"disable_tls": "true"},
            }
        ).validate_yaml()


def test_config_with_fake_default_provider():
    """Test the config when fake provider is set as default."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "fake_provider",
                    "type": "fake_provider",
                    "models": [
                        {
                            "name": "fake_model",
                        }
                    ],
                },
            ],
            "ols_config": {
                "default_provider": "fake_provider",
                "default_model": "fake_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
                "query_validation_method": "disabled",
            },
        }
    )
    assert len(config.llm_providers.providers) == 1
    assert config.llm_providers.providers["fake_provider"].name == "fake_provider"
    assert len(config.llm_providers.providers["fake_provider"].models) == 1
    assert (
        config.llm_providers.providers["fake_provider"].models["fake_model"].name
        == "fake_model"
    )
    assert config.ols_config.default_provider == "fake_provider"
    assert config.ols_config.default_model == "fake_model"
    assert config.ols_config.logging_config.app_log_level == logging.ERROR
    assert (
        config.ols_config.query_validation_method
        == constants.QueryValidationMethod.DISABLED
    )


def test_config_improper_model():
    """Test the Config model of the Global service configuration when improper model is set."""
    with pytest.raises(
        InvalidConfigurationError,
        match="default_model specifies an unknown model test_default_model",
    ):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "test_provider_name",
                        "type": "bam",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "default_provider": "test_provider_name",
                    "default_model": "test_default_model",
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 100,
                        },
                    },
                    "logging_config": {
                        "app_log_level": "error",
                    },
                },
                "dev_config": {"disable_tls": "true"},
            }
        ).validate_yaml()


def test_ols_config_with_invalid_validation_method():
    """Test the Ols config with invalid validation method."""
    ols_config = {
        "conversation_cache": {
            "type": "memory",
            "memory": {
                "max_entries": 100,
            },
        },
        "query_validation_method": False,
    }

    with pytest.raises(
        InvalidConfigurationError,
        match="Invalid query validation method",
    ):
        OLSConfig(ols_config).validate_yaml(True)


def test_logging_config_equality():
    """Test the LoggingConfig equality check."""
    logging_config_1 = LoggingConfig()
    logging_config_2 = LoggingConfig()

    # compare the same logging configs
    assert logging_config_1 == logging_config_2

    # compare different logging configs
    logging_config_2.app_log_level = 42
    assert logging_config_1 != logging_config_2

    # compare with value of different type
    other_value = "foo"
    assert logging_config_1 != other_value


def test_reference_content_equality():
    """Test the ReferenceContent equality check."""
    reference_content_1 = ReferenceContent()
    reference_content_2 = ReferenceContent()

    # compare the same configs
    assert reference_content_1 == reference_content_2

    # compare different configs
    reference_content_2.product_docs_index_path = "foo"
    assert reference_content_1 != reference_content_2

    # compare with value of different type
    other_value = "foo"
    assert reference_content_1 != other_value


def test_config_no_query_filter_node():
    """Test the Config model when query filter is not set at all."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "openai",
                    "url": "http://test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                }
            ],
            "ols_config": {
                "default_provider": "openai",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
            },
            "dev_config": {
                "disable_tls": "true",
            },
        }
    )
    assert config.ols_config.query_filters is None


def test_config_no_query_filter():
    """Test the Config model when query filter list is empty."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "openai",
                    "url": "http://test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test_model_url",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                }
            ],
            "ols_config": {
                "default_provider": "openai",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "query_filters": [],
                "logging_config": {
                    "app_log_level": "error",
                },
            },
            "dev_config": {
                "disable_tls": "true",
            },
        }
    )
    assert len(config.ols_config.query_filters) == 0


def test_config_improper_query_filter():
    """Test the Config model with improper query filter (no name) is set."""
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "openai",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "default_provider": "openai",
                    "default_model": "test_default_model",
                    "query_filters": [
                        {
                            "pattern": "test_regular_expression",
                            "replace_with": "test_replace_with",
                        }
                    ],
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 100,
                        },
                    },
                    "logging_config": {
                        "app_log_level": "error",
                    },
                },
                "dev_config": {
                    "disable_tls": "true",
                },
            }
        ).validate_yaml()


def test_config_with_multiple_query_filter():
    """Test the Config model with multiple query filter is set."""
    config = Config(
        {
            "llm_providers": [
                {
                    "name": "openai",
                    "url": "http://test_provider_url",
                    "credentials_path": "tests/config/secret/apitoken",
                    "models": [
                        {
                            "name": "test_model_name",
                            "url": "http://test.io",
                            "credentials_path": "tests/config/secret/apitoken",
                        }
                    ],
                },
            ],
            "ols_config": {
                "default_provider": "openai",
                "default_model": "test_model_name",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
                "query_filters": [
                    {
                        "name": "filter1",
                        "pattern": r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
                        "replace_with": "redacted",
                    },
                    {
                        "name": "filter2",
                        "pattern": r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+",
                        "replace_with": "",
                    },
                ],
            },
            "dev_config": {
                "disable_tls": "true",
            },
        }
    )
    assert config.ols_config.query_filters[0].name == "filter1"
    assert (
        config.ols_config.query_filters[0].pattern
        == r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    )
    assert config.ols_config.query_filters[0].replace_with == "redacted"
    assert config.ols_config.query_filters[1].name == "filter2"
    assert (
        config.ols_config.query_filters[1].pattern
        == r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+"
    )
    assert config.ols_config.query_filters[1].replace_with == ""


def test_config_invalid_regex_query_filter():
    """Test the Config model with invalid query filter pattern."""
    with pytest.raises(
        InvalidConfigurationError,
        match="pattern is invalid",
    ):
        Config(
            {
                "llm_providers": [
                    {
                        "name": "openai",
                        "url": "http://test_provider_url",
                        "credentials_path": "tests/config/secret/apitoken",
                        "models": [
                            {
                                "name": "test_model_name",
                                "url": "http://test_model_url",
                                "credentials_path": "tests/config/secret/apitoken",
                            }
                        ],
                    }
                ],
                "ols_config": {
                    "default_provider": "openai",
                    "default_model": "test_default_model",
                    "query_filters": [
                        {
                            "name": "test_name",
                            "pattern": "[",
                            "replace_with": "test_replace_with",
                        }
                    ],
                    "conversation_cache": {
                        "type": "memory",
                        "memory": {
                            "max_entries": 100,
                        },
                    },
                    "logging_config": {
                        "app_log_level": "error",
                    },
                },
                "dev_config": {
                    "disable_tls": "true",
                },
            }
        ).validate_yaml()


def test_query_filter_constructor():
    """Test checks made by QueryFilter constructor."""
    # no input
    query_filter = QueryFilter(None)
    assert query_filter.name is None
    assert query_filter.pattern is None
    assert query_filter.replace_with is None

    # proper input
    query_filter = QueryFilter(
        {"name": "NAME", "pattern": "PATTERN", "replace_with": "REPLACE_WITH"}
    )
    assert query_filter.name == "NAME"
    assert query_filter.pattern == "PATTERN"
    assert query_filter.replace_with == "REPLACE_WITH"

    # missing inputs
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        QueryFilter({"pattern": "PATTERN", "replace_with": "REPLACE_WITH"})
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        QueryFilter({"name": "NAME", "replace_with": "REPLACE_WITH"})
    with pytest.raises(
        InvalidConfigurationError,
        match="name, pattern and replace_with need to be specified",
    ):
        QueryFilter({"name": "NAME", "pattern": "PATTERN"})


def test_query_filter_validation():
    """Test method to validate query filter settings."""

    def get_query_filter():
        """Construct new fully-configured filter from scratch."""
        return QueryFilter(
            {"name": "NAME", "pattern": "PATTERN", "replace_with": "REPLACE_WITH"}
        )

    query_filter = get_query_filter()
    query_filter.name = None
    with pytest.raises(InvalidConfigurationError, match="name is missing"):
        query_filter.validate_yaml()

    query_filter = get_query_filter()
    query_filter.pattern = None
    with pytest.raises(InvalidConfigurationError, match="pattern is missing"):
        query_filter.validate_yaml()

    query_filter = get_query_filter()
    query_filter.replace_with = None
    with pytest.raises(InvalidConfigurationError, match="replace_with is missing"):
        query_filter.validate_yaml()


def test_authentication_config_validation_proper_config():
    """Test method to validate authentication config."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "http://cluster.org/foo",
            "k8s_ca_cert_path": "tests/config/empty_cert.crt",
        }
    )
    auth_config.validate_yaml()


def test_authentication_config_validation_empty_cluster_api():
    """Test method to validate authentication config."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "",
            "k8s_ca_cert_path": "tests/config/empty_cert.crt",
        }
    )
    auth_config.validate_yaml()


def test_authentication_config_validation_missing_cluster_api():
    """Test method to validate authentication config when cluster API is missing."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_ca_cert_path": "tests/config/empty_cert.crt",
        }
    )
    # k8s_cluster_api is optional
    auth_config.validate_yaml()


def test_authentication_config_validation_invalid_cluster_api():
    """Test method to validate authentication config."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "this-is-not-valid-url",
            "k8s_ca_cert_path": "tests/config/empty_cert.crt",
        }
    )
    with pytest.raises(
        InvalidConfigurationError, match="k8s_cluster_api URL is invalid"
    ):
        auth_config.validate_yaml()
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "None",
            "k8s_ca_cert_path": "tests/config/empty_cert.crt",
        }
    )
    with pytest.raises(
        InvalidConfigurationError, match="k8s_cluster_api URL is invalid"
    ):
        auth_config.validate_yaml()


def test_authentication_config_validation_empty_cert_path():
    """Test method to validate authentication config when cert path is empty."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "http://cluster.org/foo",
            "k8s_ca_cert_path": "",
        }
    )
    # k8s_ca_cert_path is optional
    auth_config.validate_yaml()


def test_authentication_config_validation_missing_cert_path():
    """Test method to validate authentication config when cert path is missing."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "http://cluster.org/foo",
        }
    )
    # k8s_ca_cert_path is optional
    auth_config.validate_yaml()


def test_authentication_config_validation_invalid_cert_path():
    """Test method to validate authentication config."""
    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "http://cluster.org/foo",
            "k8s_ca_cert_path": "/dev/null/foo",  # that file can not exists
        }
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="k8s_ca_cert_path does not exist: /dev/null/foo",
    ):
        auth_config.validate_yaml()

    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "http://cluster.org/foo",
            "k8s_ca_cert_path": "/dev/null",
        }
    )
    with pytest.raises(
        InvalidConfigurationError, match="k8s_ca_cert_path is not a file: /dev/null"
    ):
        auth_config.validate_yaml()

    auth_config = AuthenticationConfig(
        {
            "skip_tls_verification": True,
            "k8s_cluster_api": "http://cluster.org/foo",
            "k8s_ca_cert_path": "None",
        }
    )
    with pytest.raises(
        InvalidConfigurationError, match="k8s_ca_cert_path does not exist: None"
    ):
        auth_config.validate_yaml()


def test_user_data_config__feedback(tmpdir):
    """Tests the UserDataCollection model, feedback part."""
    # valid configuration
    user_data = UserDataCollection(
        feedback_disabled=False, feedback_storage=tmpdir.strpath
    )
    assert user_data.feedback_disabled is False
    assert user_data.feedback_storage == tmpdir.strpath

    # enabled needs feedback_storage
    with pytest.raises(
        ValueError,
        match="feedback_storage is required when feedback is enabled",
    ):
        UserDataCollection(feedback_disabled=False)

    # disabled doesn't need feedback_storage
    user_data = UserDataCollection(feedback_disabled=True)
    assert user_data.feedback_disabled is True
    assert user_data.feedback_storage is None


def test_user_data_config__transcripts(tmpdir):
    """Tests the UserDataCollection model, transripts part."""
    # valid configuration
    user_data = UserDataCollection(
        transcripts_disabled=False, transcripts_storage=tmpdir.strpath
    )
    assert user_data.transcripts_disabled is False
    assert user_data.transcripts_storage == tmpdir.strpath

    # enabled needs transcripts_storage
    with pytest.raises(
        ValueError,
        match="transcripts_storage is required when transcripts capturing is enabled",
    ):
        UserDataCollection(transcripts_disabled=False)

    # disabled doesn't need transcripts_storage
    user_data = UserDataCollection(transcripts_disabled=True)
    assert user_data.transcripts_disabled is True
    assert user_data.transcripts_storage is None
