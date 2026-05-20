"""
Placeholder for a future Brave Search API provider.

To implement:
1. Install: pip install requests
2. Obtain a Brave Search API key from https://brave.com/search/api/
3. Replace this class with a real implementation that:
   - Calls the Brave Search API with each generated query
   - Fetches and extracts text from result URLs
   - Passes extracted text to a local LLM or summarisation step
   - Returns a SearchResult matching the base interface

The dashboard and agents will work without any changes.
"""

from .base import SearchProvider, SearchResult


class BraveSearchProvider(SearchProvider):

    def __init__(self, api_key: str, summariser_api_key: str = ""):
        self.api_key = api_key
        self.summariser_api_key = summariser_api_key

    @property
    def name(self) -> str:
        return "Brave Search (not yet implemented)"

    def research(self, prompt: str, max_tokens: int = 6000) -> SearchResult:
        raise NotImplementedError(
            "BraveSearchProvider is not yet implemented. "
            "Use LLMWebProvider for Version 1."
        )
