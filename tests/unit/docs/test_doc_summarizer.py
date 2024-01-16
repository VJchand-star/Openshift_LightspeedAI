"""Unit tests for DocsSummarizer class."""

import os
from unittest.mock import patch

from ols.src.docs.docs_summarizer import DocsSummarizer
from ols.utils import config
from tests.mock_classes.llm_loader import mock_llm_loader
from tests.mock_classes.mock_llama_index import MockLlamaIndex


@patch("ols.src.docs.docs_summarizer.LLMLoader", new=mock_llm_loader(None))
@patch("ols.src.docs.docs_summarizer.ServiceContext.from_defaults")
@patch("ols.src.docs.docs_summarizer.StorageContext.from_defaults")
@patch("ols.src.docs.docs_summarizer.load_index_from_storage", new=MockLlamaIndex)
@patch.dict(os.environ, {"TEI_SERVER_URL": "localhost"}, clear=True)
def test_summarize(storage_context, service_context):
    """Basic test for DocsSummarizer using mocked index and query engine."""
    config.load_empty_config()
    summarizer = DocsSummarizer()
    question = "What's the ultimate question with answer 42?"
    summary, documents = summarizer.summarize("1234", question)
    assert question in summary
    assert len(documents) == 0
