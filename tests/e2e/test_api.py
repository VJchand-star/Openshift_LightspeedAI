"""Integration tests for basic OLS REST API endpoints."""

import os

import requests
from httpx import Client

url = os.getenv("OLS_URL", "http://localhost:8080")
token = os.getenv("OLS_TOKEN")
client = Client(base_url=url, verify=False)  # noqa: S501
if token:
    client.headers.update({"Authorization": f"Bearer {token}"})


conversation_id = "12345678-abcd-0000-0123-456789abcdef"


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": {"status": "healthy"}}


def test_raw_prompt():
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    r = client.post(
        "/v1/debug/query",
        json={
            "conversation_id": conversation_id,
            "query": "respond to this message with the word hello",
        },
        timeout=20,
    )
    print(vars(r))
    response = r.json()

    assert r.status_code == requests.codes.ok
    assert response["conversation_id"] == conversation_id
    assert "hello" in response["response"].lower()


def test_invalid_question():
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    response = client.post(
        "/v1/query",
        json={"conversation_id": conversation_id, "query": "test query"},
        timeout=20,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    expected_details = (
        "I can only answer questions about OpenShift and Kubernetes. "
        "Please rephrase your question"
    )
    expected_json = {
        "conversation_id": conversation_id,
        "response": expected_details,
        "referenced_documents": [],
        "truncated": False,
    }
    assert response.json() == expected_json


def test_query_call_without_payload():
    """Check the REST API /v1/query with POST HTTP method when no payload is provided."""
    response = client.post(
        "/v1/query",
        timeout=20,
    )
    print(vars(response))
    assert response.status_code == requests.codes.unprocessable_entity
    # the actual response might differ when new Pydantic version will be used
    # so let's do just primitive check
    assert "missing" in response.text


def test_query_call_with_improper_payload():
    """Check the REST API /v1/query with POST HTTP method when improper payload is provided."""
    response = client.post(
        "/v1/query",
        json={"parameter": "this-is-not-proper-question-my-friend"},
        timeout=20,
    )
    print(vars(response))
    assert response.status_code == requests.codes.unprocessable_entity
    # the actual response might differ when new Pydantic version will be used
    # so let's do just primitive check
    assert "missing" in response.text


def test_valid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for valid question and no yaml."""
    response = client.post(
        "/v1/query",
        json={"conversation_id": conversation_id, "query": "what is kubernetes?"},
        timeout=90,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    json_response = response.json()
    json_response["conversation_id"] == conversation_id
    # checking a few major information from response
    assert "Kubernetes is" in json_response["response"]
    assert (
        "orchestration tool" in json_response["response"]
        or "orchestration system" in json_response["response"]
        or "orchestration platform" in json_response["response"]
    )
    assert (
        "The following response was generated without access to reference content:"
        not in json_response["response"]
    )


def test_rag_question() -> None:
    """Ensure responses include rag references."""
    response = client.post(
        "/v1/query",
        json={"query": "what is the first step to install an openshift cluster?"},
        timeout=90,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    json_response = response.json()
    assert len(json_response["referenced_documents"]) > 0
    assert "install" in json_response["referenced_documents"][0]
    assert "https://" in json_response["referenced_documents"][0]

    assert (
        "The following response was generated without access to reference content:"
        not in json_response["response"]
    )


def test_query_filter() -> None:
    """Ensure responses does not include filtered words."""
    response = client.post(
        "/v1/query",
        json={"query": "what is foo in bar?"},
        timeout=90,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    json_response = response.json()
    assert len(json_response["referenced_documents"]) > 0
    assert "openshift" in json_response["referenced_documents"][0]
    assert "https://" in json_response["referenced_documents"][0]

    # values to be filtered and replaced are defined in:
    # tests/config/singleprovider.e2e.template.config.yaml
    assert "openshift" in json_response["response"].lower()
    assert "deployment" in json_response["response"].lower()
    assert "foo" not in json_response["response"]
    assert "bar" not in json_response["response"]


def test_metrics() -> None:
    """Check if service provides metrics endpoint with expected metrics."""
    response = client.get("/metrics/")
    assert response.status_code == requests.codes.ok
    assert response.text is not None

    # counters that are expected to be part of metrics
    expected_counters = (
        "rest_api_calls_total",
        "llm_calls_total",
        "llm_calls_failures_total",
        "llm_validation_errors_total",
        "llm_token_sent_total",
        "llm_token_received_total",
    )

    # check if all counters are present
    for expected_counter in expected_counters:
        assert f"{expected_counter} " in response.text

    # check the duration histogram presence
    assert 'response_duration_seconds_count{path="/metrics/"}' in response.text
    assert 'response_duration_seconds_sum{path="/metrics/"}' in response.text


def test_improper_token():
    """Test accessing /v1/query endpoint using improper auth. token."""
    # let's assume that auth. is enabled when token is specified
    if token:
        response = client.post(
            "/v1/query",
            json={"query": "what is foo in bar?"},
            timeout=90,
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == requests.codes.forbidden
