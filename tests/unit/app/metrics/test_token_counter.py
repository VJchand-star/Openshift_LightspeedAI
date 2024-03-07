"""Unit tests for GenericTokenCounter class."""

from langchain_core.outputs.llm_result import LLMResult

from ols.app.metrics import GenericTokenCounter


class MockLLM:
    """Mocked LLM to be used in unit tests."""

    def get_num_tokens(self, prompt):
        """Poor man's token counter."""
        return len(prompt.split(" "))


def test_on_llm_start():
    """Test the GenericTokenCounter.on_llm_start method."""
    llm = MockLLM()

    # initialize new token counter
    token_counter = GenericTokenCounter(llm)

    # a beginning the counters should be zeroed
    assert token_counter.llm_calls == 0
    assert token_counter.input_tokens_counted == 0

    # token count for empty input
    token_counter.on_llm_start({}, [])

    # token counter needs to be zero as mocked LLM does not process anything
    assert token_counter.llm_calls == 1
    assert token_counter.input_tokens_counted == 0

    # now the prompt will be tokenized into 5 tokens
    token_counter.on_llm_start({}, ["this is just a test"])
    assert token_counter.llm_calls == 2
    assert token_counter.input_tokens_counted == 5


class MockResult:
    """Mocked one result that can be made part of LLMResult."""

    def __init__(self, prompt_tokens, completion_tokens):
        """Perform setup of one result, initializing it's tokens counters."""
        self.llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }


class MockLLMResult:
    """Mocked LLMResult value."""

    def __init__(self, results):
        """Perform setup of LLMResult mocked object."""
        self.results = results

    def flatten(self) -> list[LLMResult]:
        """Return results that are already flattened at input."""
        return self.results


def test_on_llm_end():
    """Test the GenericTokenCounter.on_llm_end method."""
    llm = MockLLM()

    # initialize new token counter
    token_counter = GenericTokenCounter(llm)
    assert token_counter.input_tokens == 0
    assert token_counter.output_tokens == 0

    # empty response
    response = MockLLMResult([])
    token_counter.on_llm_end(response)

    # for empty response, counters should not change
    assert token_counter.input_tokens == 0
    assert token_counter.output_tokens == 0

    # non-empty response
    x = MockResult(10, 20)
    response = MockLLMResult([x])
    token_counter.on_llm_end(response)

    # for non-empty response, counters should change
    assert token_counter.input_tokens == 10
    assert token_counter.output_tokens == 20
