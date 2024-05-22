"""Handlers for all OLS-related REST API endpoints."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import pytz
from fastapi import APIRouter, Depends, HTTPException, status

from ols import config, constants
from ols.app import metrics
from ols.app.models.models import (
    ErrorResponse,
    ForbiddenResponse,
    LLMRequest,
    LLMResponse,
    PromptTooLongResponse,
    ReferencedDocument,
    UnauthorizedResponse,
)
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.src.query_helpers.docs_summarizer import DocsSummarizer
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.utils import suid
from ols.utils.auth_dependency import AuthDependency
from ols.utils.keywords import KEYWORDS
from ols.utils.token_handler import PromptTooLongError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])
auth_dependency = AuthDependency(virtual_path="/ols-access")


query_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Query is valid and correct response from LLM is returned",
        "model": LLMResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    413: {
        "description": "Prompt is too long",
        "model": PromptTooLongResponse,
    },
    500: {
        "description": "Query can not be validated, LLM is not accessible or other internal error",
        "model": ErrorResponse,
    },
}


@router.post("/query", responses=query_responses)
def conversation_request(
    llm_request: LLMRequest, auth: Any = Depends(auth_dependency)
) -> LLMResponse:
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query and conversation ID.
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response containing the processed information.
    """
    # Initialize variables
    previous_input = []
    referenced_documents: list[ReferencedDocument] = []

    user_id = retrieve_user_id(auth)
    logger.info(f"User ID {user_id}")
    if not suid.check_suid(user_id):
        logger.error(f"Invalid user ID: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "response": "Invalid user ID",
                "cause": "User ID must be a valid UUID",
            },
        )

    conversation_id = retrieve_conversation_id(llm_request)
    if not suid.check_suid(conversation_id):
        logger.error(f"Invalid conversation ID: {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Invalid conversation ID",
                "cause": "Conversation ID must be a valid UUID",
            },
        )
    previous_input = retrieve_previous_input(user_id, llm_request)

    # Log incoming request
    logger.info(f"{conversation_id} Incoming request: {llm_request.query}")

    # Redact the query
    llm_request = redact_query(conversation_id, llm_request)

    # Validate the query
    if not previous_input:
        valid = validate_question(conversation_id, llm_request)
    else:
        logger.debug("follow-up conversation - skipping question validation")
        valid = True

    if not valid:
        response, referenced_documents, truncated = (
            constants.INVALID_QUERY_RESP,
            [],
            False,
        )
    else:
        response, referenced_documents, truncated = generate_response(
            conversation_id, llm_request, previous_input
        )

    store_conversation_history(user_id, conversation_id, llm_request, response)

    if config.ols_config.user_data_collection.transcripts_disabled:
        logger.debug("transcripts collections is disabled in configuration")
    else:
        store_transcript(
            user_id,
            conversation_id,
            valid,
            llm_request,
            response,
            referenced_documents,
            truncated,
        )

    return LLMResponse(
        conversation_id=conversation_id,
        response=response,
        referenced_documents=referenced_documents,
        truncated=truncated,
    )


def retrieve_user_id(auth: Any) -> str:
    """Retrieve user ID from the token processed by auth. mechanism."""
    # auth contains tuple with user ID (in UUID format) and user name
    return auth[0]


def retrieve_conversation_id(llm_request: LLMRequest) -> str:
    """Retrieve conversation ID based on existing ID or on newly generated one."""
    conversation_id = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if not conversation_id:
        conversation_id = suid.get_suid()
        logger.info(f"{conversation_id} New conversation")

    return conversation_id


def retrieve_previous_input(
    user_id: str, llm_request: LLMRequest
) -> list[dict[Literal["type", "content"], str]]:
    """Retrieve previous user input, if exists."""
    try:
        previous_input: list[dict[Literal["type", "content"], str]] = []
        if llm_request.conversation_id:
            cache_content = config.conversation_cache.get(
                user_id, llm_request.conversation_id
            )
            if cache_content is not None:
                previous_input = cache_content
            logger.info(
                f"{llm_request.conversation_id} Previous conversation input: {previous_input}"
            )
        return previous_input
    except Exception as e:
        logger.error(f"Error retrieving previous user input for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error retrieving conversation history",
                "cause": str(e),
            },
        )


def generate_response(
    conversation_id: str,
    llm_request: LLMRequest,
    previous_input: list[dict[Literal["type", "content"], str]],
) -> tuple[str, list[ReferencedDocument], bool]:
    """Generate response based on validation result, previous input, and model output."""
    # Summarize documentation
    try:
        docs_summarizer = DocsSummarizer(
            provider=llm_request.provider, model=llm_request.model
        )
        history = [
            conversation["type"] + ": " + conversation["content"].strip()
            for conversation in previous_input
            if conversation
        ]
        llm_response = docs_summarizer.summarize(
            conversation_id, llm_request.query, config.rag_index, history
        )
        return (
            llm_response["response"],
            llm_response["referenced_documents"],
            llm_response["history_truncated"],
        )
    except PromptTooLongError as summarizer_error:
        logger.error(f"Prompt is too long: {summarizer_error}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "response": "Prompt is too long",
                "cause": str(summarizer_error),
            },
        )
    except Exception as summarizer_error:
        logger.error("Error while obtaining answer for user question")
        logger.exception(summarizer_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error while obtaining answer for user question",
                "cause": str(summarizer_error),
            },
        )


def human_msg(content: str) -> dict[Literal["type", "content"], str]:
    """Create a human message dictionary."""
    return {"type": "human", "content": content}


def ai_msg(content: str) -> dict[Literal["type", "content"], str]:
    """Create an AI message dictionary."""
    return {"type": "ai", "content": content}


def store_conversation_history(
    user_id: str, conversation_id: str, llm_request: LLMRequest, response: Optional[str]
) -> None:
    """Store conversation history into selected cache.

    History is stored as simple dictionaries in the following format:
    ```python
        [
            {"type": "human/ai", "content": "..."},
            ...
        ]
    ```
    """
    try:
        if config.conversation_cache is not None:
            logger.info(f"{conversation_id} Storing conversation history.")
            chat_message_history = [
                human_msg(llm_request.query),
                ai_msg(response or ""),
            ]
            config.conversation_cache.insert_or_append(
                user_id,
                conversation_id,
                chat_message_history,
            )
    except Exception as e:
        logger.error(
            "Error storing conversation history for user "
            f"{user_id} and conversation {conversation_id}"
        )
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error storing conversation",
                "cause": str(e),
            },
        )


def redact_query(conversation_id: str, llm_request: LLMRequest) -> LLMRequest:
    """Redact query using query_redactor, raise HTTPException in case of any problem."""
    try:
        logger.debug(f"Redacting query for conversation {conversation_id}")
        if not config.query_redactor:
            logger.debug("query_redactor not found")
            return llm_request
        llm_request.query = config.query_redactor.redact_query(
            conversation_id, llm_request.query
        )
        return llm_request
    except Exception as redactor_error:
        logger.error(
            f"Error while redacting query {redactor_error} for conversation {conversation_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error while redacting query",
                "cause": str(redactor_error),
            },
        )


def _validate_question_llm(conversation_id: str, llm_request: LLMRequest) -> bool:
    """Validate user question using llm, raise HTTPException in case of any problem."""
    # Validate the query
    try:
        question_validator = QuestionValidator(
            provider=llm_request.provider, model=llm_request.model
        )
        return question_validator.validate_question(conversation_id, llm_request.query)
    except LLMConfigurationError as e:
        metrics.llm_calls_validation_errors_total.inc()
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"response": "Unable to process this request", "cause": str(e)},
        )
    except PromptTooLongError as e:
        logger.error(f"Prompt is too long: {e}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "response": "Prompt is too long",
                "cause": str(e),
            },
        )
    except Exception as validation_error:
        metrics.llm_calls_failures_total.inc()
        logger.error("Error while validating question")
        logger.exception(validation_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error while validating question",
                "cause": str(validation_error),
            },
        )


def _validate_question_keyword(query: str) -> bool:
    """Validate user question using keyword."""
    # Current implementation is without any tokenizer method, lemmatization/n-grams.
    # Add valid keywords to keywords.py file.
    query_temp = query.lower()
    for kw in KEYWORDS:
        if kw in query_temp:
            return True
    # query_temp = {q_word.lower().strip(".?,") for q_word in query.split()}
    # common_words = config.keywords.intersection(query_temp)
    # if len(common_words) > 0:
    #     return constants.SUBJECT_ALLOWED

    logger.debug(f"No matching keyword found for query: {query}")
    return False


def validate_question(conversation_id: str, llm_request: LLMRequest) -> bool:
    """Validate user question."""
    match config.ols_config.query_validation_method:

        case constants.QueryValidationMethod.DISABLED:
            logger.debug(
                f"{conversation_id} Question validation is disabled. "
                f"Treating question as valid."
            )
            return True

        case constants.QueryValidationMethod.KEYWORD:
            logger.debug("Keyword based query validation.")
            return _validate_question_keyword(llm_request.query)

        case _:
            logger.debug("LLM based query validation.")
            return _validate_question_llm(conversation_id, llm_request)


def store_transcript(
    user_id: str,
    conversation_id: str,
    query_is_valid: bool,
    llm_request: LLMRequest,
    response: str,
    referenced_documents: list[ReferencedDocument],
    truncated: bool,
) -> None:
    """Store transcript in the local filesystem.

    Args:
        user_id: The user ID (UUID).
        conversation_id: The conversation ID (UUID).
        query_is_valid: The result of the query validation.
        llm_request: The request containing a query.
        response: The response to store.
        referenced_documents: The list of referenced documents.
        truncated: The flag indicating if the history was truncated.
    """
    # ensures storage path exists
    transcripts_path = Path(
        config.ols_config.user_data_collection.transcripts_storage,
        user_id,
        conversation_id,
    )
    if not transcripts_path.exists():
        logger.debug(f"creating transcript storage directory '{transcripts_path}'")
        transcripts_path.mkdir(parents=True)

    data_to_store = {
        "metadata": {
            "provider": llm_request.provider or config.ols_config.default_provider,
            "model": llm_request.model or config.ols_config.default_model,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
        },
        "redacted_query": llm_request.query,
        "query_is_valid": query_is_valid,
        "llm_response": response,
        "referenced_documents": [doc.model_dump() for doc in referenced_documents],
        "truncated": truncated,
    }

    # stores feedback in a file under unique uuid
    transcript_file_path = transcripts_path / f"{suid.get_suid()}.json"
    with open(transcript_file_path, "w", encoding="utf-8") as transcript_file:
        json.dump(data_to_store, transcript_file)

    logger.debug(f"transcript stored in '{transcript_file_path}'")
