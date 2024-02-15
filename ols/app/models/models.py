"""Data models representing payloads for REST API calls."""

from typing import Optional

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class LLMRequest(BaseModel):
    """Model representing a request for the LLM (Language Model).

    Attributes:
        query: The query string.
        conversation_id: The optional conversation ID (UUID).
        provider: The optional provider.
        model: The optional model.

    Example:
        ```python
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        ```
    """

    query: str
    conversation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "write a deployment yaml for the mongodb image",
                    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                }
            ]
        }
    }

    @model_validator(mode="after")
    def validate_provider_and_model(self) -> Self:
        """Perform validation on the provider and model."""
        if self.model and not self.provider:
            raise ValueError(
                "LLM provider must be specified when the model is specified!"
            )
        if self.provider and not self.model:
            raise ValueError(
                "LLM model must be specified when the provider is specified!"
            )
        return self


class LLMResponse(BaseModel):
    """Model representing a response from the LLM (Language Model).

    Attributes:
        conversation_id: The optional conversation ID (UUID).
        response: The optional response.
        referenced_documents: The optional URLs for the documents used to generate the response.
        truncated: Set to True if conversation history was truncated to be within context window.
    """

    conversation_id: str
    response: str
    referenced_documents: list[str]
    truncated: bool

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "response": "Operator Lifecycle Manager (OLM) helps users install...",
                    "referenced_documents": [
                        "https://docs.openshift.com/container-platform/4.14/operators/"
                        "understanding/olm/olm-understanding-olm.html"
                    ],
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """Model representing a feedback request.

    Attributes:
        conversation_id: The required conversation ID (UUID).
        feedback_object: The JSON blob representing feedback.

    Example:
        ```python
        feedback_request = FeedbackRequest(
            conversation_id="12345678-abcd-0000-0123-456789abcdef",
            feedback_object={"rating": 5, "comment": "Great service!"}
        )
        ```
    """

    conversation_id: str
    feedback_object: dict  # TODO: proper object (OLS-77)

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "12345678-abcd-0000-0123-456789abcdef",
                    "feedback_object": {"rating": 5, "comment": "Great service!"},
                }
            ]
        }
    }


class FeedbackResponse(BaseModel):
    """Model representing a response to a feedback request.

    Attributes:
        response: The response of the feedback request.

    Example:
        ```python
        feedback_response = FeedbackResponse(response="feedback received")
        ```
    """

    response: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "feedback received",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Model representing a response to a health request.

    Attributes:
        status: The status of the app.

    Example:
        ```python
        health_response = HealthResponse(status={"status": "healthy"})
        ```
    """

    status: dict[str, str]

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": {"status": "healthy"},
                }
            ]
        }
    }
