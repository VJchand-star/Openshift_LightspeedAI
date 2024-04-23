"""Mocked summary returned from query engine."""


class MockSummary:
    """Mocked summary returned from query engine."""

    # TODO: OLS-524 Mutable objects used as function argument defaults
    def __init__(self, query, nodes=[]):
        """Initialize all required object attributes."""
        self.query = query
        self.source_nodes = nodes

    def __str__(self):
        """Return string representation that is used by DocsSummarizer."""
        return f"Summary for query '{self.query}'"
