"""Mocked query engine."""

from .mock_summary import MockSummary


class Node:
    """Node containing source node metadata."""

    def __init__(self, docs_url, title):
        """Initialize docs_url metadata."""
        self.metadata = {"docs_url": docs_url, "title": title}


class SourceNode:
    """Node containing one reference to document."""

    def __init__(self, docs_url, title):
        """Initialize sub-node with metadata."""
        self.node = Node(docs_url, title)


class MockQueryEngine:
    """Mocked query engine."""

    def __init__(self, *args, **kwargs):
        """Store all provided arguments for later usage."""
        self.args = args
        self.kwargs = kwargs

    def query(self, query):
        """Return summary for given query."""
        # fill-in some documents for more realistic tests
        nodes = [
            SourceNode("/docs/test.", "Docs Test"),
            SourceNode("/known-bugs.", "Known Bugs"),
            SourceNode("/errata.", "Errata"),
        ]
        return MockSummary(query, nodes)
