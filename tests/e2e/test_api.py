"""Integration tests for basic OLS REST API endpoints."""

import os

import pytest
import requests
from httpx import Client

from ols.constants import INVALID_QUERY_RESP, NO_RAG_CONTENT_RESP
from scripts.validate_response import ResponseValidation

url = os.getenv("OLS_URL", "http://localhost:8080")
token = os.getenv("OLS_TOKEN")
client = Client(base_url=url, verify=False)  # noqa: S501
if token:
    client.headers.update({"Authorization": f"Bearer {token}"})


conversation_id = "12345678-abcd-0000-0123-456789abcdef"


# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90

SCORE_THRESHOLD = 0.4  # low score is better

# Sample Question/Answer set
# TODO: Load this from QnA json file.
QUESTION1 = "what is kubernetes?"
ANSWER1 = (
    "Kubernetes is an open source container orchestration tool developed by Google. "
    "It allows you to run and manage container-based workloads, making it easier to "
    "deploy and scale applications in a cloud native way. With Kubernetes, "
    "you can create clusters that span hosts across on-premise, public, private, or "
    "hybrid clouds. It helps in sharing resources, orchestrating containers across "
    "multiple hosts, installing new hardware configurations, running health checks "
    "and self-healing applications, and scaling containerized applications."
)
QUESTION2 = "what is openshift virtualization?"
ANSWER2 = (
    "OpenShift Virtualization is an add-on to the Red Hat OpenShift Container Platform "
    "that enables the running and management of virtual machine workloads alongside "
    "container workloads. It introduces new objects into the OpenShift cluster using "
    "Kubernetes custom resources to facilitate virtualization tasks such as creating and "
    "managing Linux and Windows virtual machines, running pod and VM workloads together "
    "in a cluster, connecting to VMs through various consoles and CLI tools, importing "
    "and cloning existing VMs, managing network interface controllers and storage disks "
    "attached to VMs, as well as live migrating VMs between nodes."
)


def read_metrics(client):
    """Read all metrics using REST API call."""
    response = client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)

    # check that the /metrics endpoint is correct and we got
    # some response
    assert response.status_code == requests.codes.ok
    assert response.text is not None

    return response.text


def get_rest_api_counter_value(
    client, path, status_code=requests.codes.ok, default=None
):
    """Retrieve counter value from metrics."""
    response = read_metrics(client)
    counter_name = "rest_api_calls_total"

    # counters with labels have the following format:
    # rest_api_calls_total{path="/openapi.json",status_code="200"} 1.0
    prefix = f'{counter_name}{{path="{path}",status_code="{status_code}"}} '

    return get_counter_value(counter_name, prefix, response, default)


def get_response_duration_seconds_value(client, path, default=None):
    """Retrieve counter value from metrics."""
    response = read_metrics(client)
    counter_name = "response_duration_seconds_sum"

    # counters with response durations have the following format:
    # response_duration_seconds_sum{path="/v1/debug/query"} 0.123
    prefix = f'{counter_name}{{path="{path}"}} '

    return get_counter_value(counter_name, prefix, response, default, to_int=False)


def get_model_provider_counter_value(
    client, counter_name, model, provider, default=None
):
    """Retrieve counter value from metrics."""
    response = read_metrics(client)

    # counters with model and provider have the following format:
    # llm_token_sent_total{model="ibm/granite-13b-chat-v2",provider="bam"} 8.0
    # llm_token_received_total{model="ibm/granite-13b-chat-v2",provider="bam"} 2465.0
    prefix = f'{counter_name}{{model="{model}",provider="{provider}"}} '

    return get_counter_value(counter_name, prefix, response, default)


def read_info_attribute_value(lines, info_node_name):
    """Read value of attribute stored in some Info node."""
    prefix = f'{info_node_name}{{name="'

    for line in lines:
        if line.startswith(prefix):
            # strip prefix
            value = line[len(prefix) :]

            # strip suffix
            return value[: value.find('"')]

    # info node was not found
    return None


def get_model_and_provider(client):
    """Read configured model and provider from metrics."""
    response = read_metrics(client)
    lines = [line.strip() for line in response.split("\n")]

    model = read_info_attribute_value(lines, "selected_model_info")
    provider = read_info_attribute_value(lines, "selected_provider_info")

    return model, provider


def get_counter_value(counter_name, prefix, response, default=None, to_int=True):
    """Try to retrieve counter value from response with all metrics."""
    lines = [line.strip() for line in response.split("\n")]

    # try to find the given counter
    for line in lines:
        if line.startswith(prefix):
            without_prefix = line[len(prefix) :]
            # parse counter value as float
            value = float(without_prefix)
            # convert that float to integer if needed
            if to_int:
                return int(value)
            return value

    # counter was not found, which might be ok for first API call
    if default is not None:
        return default

    raise Exception(f"Counter {counter_name} was not found in metrics")


def check_counter_increases(endpoint, old_counter, new_counter, delta=1):
    """Check if the counter value increases as expected."""
    assert (
        new_counter >= old_counter + delta
    ), f"REST API counter for {endpoint} has not been updated properly"


def check_duration_sum_increases(endpoint, old_counter, new_counter):
    """Check if the counter value with total duration increases as expected."""
    assert (
        new_counter > old_counter
    ), f"Duration sum for {endpoint} has not been updated properly"


def check_token_counter_increases(counter, old_counter, new_counter, expect_change):
    """Check if the counter value increases as expected."""
    if expect_change:
        assert (
            new_counter > old_counter
        ), f"Counter for {counter} tokens has not been updated properly"
    else:
        assert (
            new_counter == old_counter
        ), f"Counter for {counter} tokens has changed, which is unexpected"


class RestAPICallCounterChecker:
    """Context manager to check if REST API counter is increased for given endpoint."""

    def __init__(self, client, endpoint, status_code=requests.codes.ok):
        """Register client and endpoint."""
        self.client = client
        self.endpoint = endpoint
        self.status_code = status_code

    def __enter__(self):
        """Retrieve old counter value before calling REST API."""
        self.old_counter = get_rest_api_counter_value(
            self.client, self.endpoint, status_code=self.status_code, default=0
        )
        self.old_duration = get_response_duration_seconds_value(
            self.client, self.endpoint, default=0
        )

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Retrieve new counter value after calling REST API, and check if it increased."""
        # test if REST API endpoint counter has been updated
        new_counter = get_rest_api_counter_value(
            self.client, self.endpoint, status_code=self.status_code
        )
        check_counter_increases(self.endpoint, self.old_counter, new_counter)

        # test if duration counter has been updated
        new_duration = get_response_duration_seconds_value(
            self.client,
            self.endpoint,
        )
        check_duration_sum_increases(self.endpoint, self.old_duration, new_duration)


class TokenCounterChecker:
    """Context manager to check if token counters are increased before and after LLL calls.

    Example:
    ```python
    with TokenCounterChecker(client, "ibm/granite-13b-chat-v2", "bam"):
        ...
        ...
        ...
    """

    def __init__(
        self,
        client,
        model,
        provider,
        expect_sent_change=True,
        expect_received_change=True,
    ):
        """Register model and provider which tokens will be checked."""
        self.model = model
        self.provider = provider
        self.client = client
        # when model nor provider are specified (OLS cluster), don't run checks
        self.skip_check = model is None or provider is None

        # expect change in number of sent tokens
        self.expect_sent_change = expect_sent_change

        # expect change in number of received tokens
        self.expect_received_change = expect_received_change

    def __enter__(self):
        """Retrieve old counter values before calling LLM."""
        if self.skip_check:
            return
        self.old_counter_token_sent_total = get_model_provider_counter_value(
            self.client, "llm_token_sent_total", self.model, self.provider, default=0
        )
        self.old_counter_token_received_total = get_model_provider_counter_value(
            self.client,
            "llm_token_received_total",
            self.model,
            self.provider,
            default=0,
        )

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Retrieve new counter value after calling REST API, and check if it increased."""
        if self.skip_check:
            return
        # check if counter for sent tokens has been updated
        new_counter_token_sent_total = get_model_provider_counter_value(
            self.client, "llm_token_sent_total", self.model, self.provider
        )
        check_token_counter_increases(
            "sent",
            self.old_counter_token_sent_total,
            new_counter_token_sent_total,
            self.expect_sent_change,
        )

        # check if counter for received tokens has been updated
        new_counter_token_received_total = get_model_provider_counter_value(
            self.client,
            "llm_token_received_total",
            self.model,
            self.provider,
            default=0,
        )
        check_token_counter_increases(
            "received",
            self.old_counter_token_received_total,
            new_counter_token_received_total,
            self.expect_received_change,
        )


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    endpoint = "/readiness"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    endpoint = "/liveness"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        assert response.json() == {"status": {"status": "healthy"}}


def test_raw_prompt():
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    endpoint = "/v1/debug/query"
    with RestAPICallCounterChecker(client, endpoint):
        r = client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "respond to this message with the word hello",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(r))
        response = r.json()

        assert r.status_code == requests.codes.ok
        assert response["conversation_id"] == conversation_id
        assert "hello" in response["response"].lower()


def test_invalid_question():
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "test query"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok

        expected_json = {
            "conversation_id": conversation_id,
            "response": INVALID_QUERY_RESP,
            "referenced_documents": [],
            "truncated": False,
        }
        assert response.json() == expected_json


def test_query_call_without_payload():
    """Check the REST API /v1/query with POST HTTP method when no payload is provided."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(
        client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        response = client.post(
            endpoint,
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.unprocessable_entity
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_query_call_with_improper_payload():
    """Check the REST API /v1/query with POST HTTP method when improper payload is provided."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(
        client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        response = client.post(
            endpoint,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=NON_LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.unprocessable_entity
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_valid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for valid question and no yaml."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": QUESTION1},
            timeout=LLM_REST_API_TIMEOUT,
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
        assert NO_RAG_CONTENT_RESP not in json_response["response"]

        score = ResponseValidation().get_similarity_score(
            json_response["response"], ANSWER1
        )
        assert score <= SCORE_THRESHOLD


@pytest.mark.standalone
def test_valid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = get_model_and_provider(client)

    endpoint = "/v1/query"
    with (
        RestAPICallCounterChecker(client, endpoint),
        TokenCounterChecker(client, model, provider),
    ):
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": QUESTION1},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok


@pytest.mark.standalone
def test_invalid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = get_model_and_provider(client)

    endpoint = "/v1/query"
    with (
        RestAPICallCounterChecker(client, endpoint),
        TokenCounterChecker(client, model, provider),
    ):
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "test query"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok


@pytest.mark.standalone
def test_token_counters_for_query_call_without_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = get_model_and_provider(client)

    endpoint = "/v1/query"
    with (
        RestAPICallCounterChecker(
            client, endpoint, status_code=requests.codes.unprocessable_entity
        ),
        TokenCounterChecker(
            client,
            model,
            provider,
            expect_sent_change=False,
            expect_received_change=False,
        ),
    ):
        response = client.post(
            endpoint,
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.standalone
def test_token_counters_for_query_call_with_improper_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = get_model_and_provider(client)

    endpoint = "/v1/query"
    with (
        RestAPICallCounterChecker(
            client, endpoint, status_code=requests.codes.unprocessable_entity
        ),
        TokenCounterChecker(
            client,
            model,
            provider,
            expect_sent_change=False,
            expect_received_change=False,
        ),
    ):
        response = client.post(
            endpoint,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.rag
def test_rag_question() -> None:
    """Ensure responses include rag references."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"query": QUESTION2},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert len(json_response["referenced_documents"]) > 0
        assert "about_virt" in json_response["referenced_documents"][0]
        assert "https://" in json_response["referenced_documents"][0]

        assert NO_RAG_CONTENT_RESP not in json_response["response"]

        score = ResponseValidation().get_similarity_score(
            json_response["response"], ANSWER2
        )
        assert score <= SCORE_THRESHOLD


def test_query_filter() -> None:
    """Ensure responses does not include filtered words."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"query": "what is foo in bar?"},
            timeout=LLM_REST_API_TIMEOUT,
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


def test_conversation_history() -> None:
    """Ensure conversations include previous query history."""
    endpoint = "/v1/query"
    conversation_id = "12345678-abcd-0000-0123-356789abcdef"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "what is ingress in kubernetes?",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert "ingress" in json_response["response"].lower()
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "what?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert "ingress" in json_response["response"].lower()


def test_metrics() -> None:
    """Check if service provides metrics endpoint with expected metrics."""
    response = client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)
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
        "selected_model_info",
        "selected_provider_info",
    )

    # check if all counters are present
    for expected_counter in expected_counters:
        assert f"{expected_counter} " in response.text

    # check the duration histogram presence
    assert 'response_duration_seconds_count{path="/metrics"}' in response.text
    assert 'response_duration_seconds_sum{path="/metrics"}' in response.text


@pytest.mark.cluster
def test_improper_token():
    """Test accessing /v1/query endpoint using improper auth. token."""
    # let's assume that auth. is enabled when token is specified
    if not token:
        pytest.skip("skipping authentication tests because OLS_TOKEN is not set")
    response = client.post(
        "/v1/query",
        json={"query": "what is foo in bar?"},
        timeout=NON_LLM_REST_API_TIMEOUT,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == requests.codes.forbidden


def test_feedback() -> None:
    """Check if feedback is properly stored.

    This is a full end-to-end scenario where the feedback is stored,
    retrieved and removed at the end (to avoid leftovers).
    """
    # check if feedback is enabled
    response = client.get("/v1/feedback/status", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.ok
    assert response.json()["status"]["enabled"] is True

    # check the feedback store is empty
    empty_feedback = client.get("/v1/feedback/list", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert empty_feedback.status_code == requests.codes.ok
    assert "feedbacks" in empty_feedback.json()
    assert len(empty_feedback.json()["feedbacks"]) == 0

    # store the feedback
    posted_feedback = client.post(
        "/v1/feedback",
        json={
            "conversation_id": conversation_id,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )
    assert posted_feedback.status_code == requests.codes.ok
    assert posted_feedback.json() == {"response": "feedback received"}

    # check the feedback store has one feedback
    stored_feedback = client.get("/v1/feedback/list", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert stored_feedback.status_code == requests.codes.ok
    assert "feedbacks" in stored_feedback.json()
    assert len(stored_feedback.json()["feedbacks"]) == 1

    # remove the feedback
    remove_feedback = client.delete(
        f'/v1/feedback/{stored_feedback.json()["feedbacks"][0]}',
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )
    assert remove_feedback.status_code == requests.codes.ok
    assert remove_feedback.json() == {"response": "feedback removed"}

    # check the feedback store is empty again
    removed_feedback = client.get("/v1/feedback/list", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert removed_feedback.status_code == requests.codes.ok
    assert "feedbacks" in removed_feedback.json()
    assert len(removed_feedback.json()["feedbacks"]) == 0


@pytest.mark.cluster
def test_feedback_can_post_with_wrong_token():
    """Test posting feedback with improper auth. token."""
    # let's assume that auth. is enabled when token is specified
    if not token:
        pytest.skip("skipping authentication tests because OLS_TOKEN is not set")
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": conversation_id,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == requests.codes.forbidden
