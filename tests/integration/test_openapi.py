"""Integration tests for REST API endpoint that provides OpenAPI specification."""

import json
import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols.utils.config import ConfigManager


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="function")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def _setup():
    """Setups the test client."""
    ConfigManager._instance = None
    config_manager = ConfigManager()
    config_manager.init_config("tests/config/valid_config.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    client = TestClient(app)

    return client, config_manager


def test_openapi_endpoint(_setup):
    """Check if REST API provides endpoint with OpenAPI specification."""
    client, _ = _setup
    response = client.get("/openapi.json")
    assert response.status_code == requests.codes.ok

    # this line ensures that response payload contains proper JSON
    payload = response.json()
    assert payload is not None, "Incorrect response"

    # check the metadata nodes
    for attribute in ("openapi", "info", "components", "paths"):
        assert (
            attribute in payload
        ), f"Required metadata attribute {attribute} not found"

    # check application description
    info = payload["info"]
    assert "description" in info, "Service description not provided"
    assert "OpenShift LightSpeed Service API specification" in info["description"]

    # elementary check that all mandatory endpoints are covered
    paths = payload["paths"]
    for endpoint in ("/readiness", "/liveness", "/v1/query", "/v1/feedback"):
        assert endpoint in paths, f"Endpoint {endpoint} is not described"


def test_openapi_endpoint_head_method(_setup):
    """Check if REST API allows HEAD HTTP method for endpoint with OpenAPI specification."""
    client, _ = _setup
    response = client.head("/openapi.json")
    assert response.status_code == requests.codes.ok
    assert response.text == ""


def test_openapi_content(_setup):
    """Check if the pre-generated OpenAPI schema is up-to date."""
    # retrieve pre-generated OpenAPI schema
    with open("docs/openapi.json") as fin:
        pre_generated_schema = json.load(fin)

    # retrieve current OpenAPI schema
    client, _ = _setup
    response = client.get("/openapi.json")
    assert response.status_code == requests.codes.ok
    current_schema = response.json()

    # remove node that is not included in pre-generated OpenAPI schema
    del current_schema["info"]["license"]

    # compare schemas (as dicts)
    assert (
        current_schema == pre_generated_schema
    ), "Pre-generated schema is not up to date. Fix it with `make schema`."
