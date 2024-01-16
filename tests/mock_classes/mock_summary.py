"""Mocked summary returned from query engine."""


class MockSummary:
    """Mocked summary returned from query engine."""

    def __init__(self, query, nodes=[]):
        """Constructor that initializes all required object attributes."""
        self.query = query
        self.source_nodes = nodes

    def __str__(self):
        """String representation that is used by DocsSummarizer."""
        return f"Summary for query '{self.query}'"
