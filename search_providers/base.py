from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SourceRef:
    url: str
    title: str = ""
    snippet: str = ""


@dataclass
class SearchResult:
    query: str
    content: str
    sources: List[SourceRef] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content)


class SearchProvider(ABC):
    """
    Abstract interface for all search/research providers.
    The dashboard and agents depend on this interface only,
    so swapping providers requires no changes upstream.
    """

    @abstractmethod
    def research(self, prompt: str, max_tokens: int = 6000) -> SearchResult:
        """
        Execute a research prompt and return structured results.
        The prompt should ask for a specific research task and specify
        that the response must be valid JSON.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'OpenAI gpt-4o'."""
