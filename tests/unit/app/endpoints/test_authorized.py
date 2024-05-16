"""Unit tests for authorized endpoints handlers."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request

from ols import constants
from ols.app.endpoints.authorized import (
    is_user_authorized,
)
from ols.app.models.models import AuthorizationResponse
from ols.utils import config
from tests.mock_classes.mock_k8s_api import (
    mock_subject_access_review_response,
    mock_token_review_response,
)


@pytest.fixture
def _setup():
    """Fixture for environment setup."""
    config.init_empty_config()


@pytest.fixture
def _disabled_auth():
    """Fixture for tests that expect disabled auth."""
    assert config.config.dev_config is not None
    config.config.dev_config.disable_auth = True


@pytest.fixture
def _enabled_auth():
    """Fixture for tests that expect enabled auth."""
    assert config.config.dev_config is not None
    config.config.dev_config.disable_auth = False


def test_is_user_authorized_auth_disabled(_setup, _disabled_auth):
    """Test the is_user_authorized function when the authentication is disabled."""
    # the tested function returns constant right now
    request = Request(scope={"type": "http"})
    response = is_user_authorized(request)
    assert response == AuthorizationResponse(
        user_id=constants.DEFAULT_USER_UID, username=constants.DEFAULT_USER_NAME
    )


def test_is_user_authorized_false_no_bearer_token(_setup, _enabled_auth):
    """Test the is_user_authorized function when its missing authorization header."""
    # the tested function returns constant right now
    request = Request(scope={"type": "http", "headers": []})

    # Expect an HTTPException for invalid tokens
    with pytest.raises(HTTPException) as exc_info:
        is_user_authorized(request)

    # Check if the correct status code is returned for unauthenticated access
    assert exc_info.value.status_code == 401


@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
def test_is_user_authorized_valid_token(
    mock_authz_api, mock_authn_api, _setup, _enabled_auth
):
    """Tests the is_user_authorized function with a mocked valid-token."""
    # Setup mock responses for valid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    # Simulate a request with a valid token
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer valid-token")]}
    )

    response = is_user_authorized(request)
    assert response == AuthorizationResponse(user_id="valid-uid", username="valid-user")


@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
def test_is_user_authorized_invalid_token(
    mock_authz_api, mock_authn_api, _setup, _enabled_auth
):
    """Test the is_user_authorized function with a mocked invalid-token."""
    # Setup mock responses for invalid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    # Simulate a request with an invalid token
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer invalid-token")]}
    )

    # Expect an HTTPException for invalid tokens
    with pytest.raises(HTTPException) as exc_info:
        is_user_authorized(request)

    # Check if the correct status code is returned for unauthenticated access
    assert exc_info.value.status_code == 403
