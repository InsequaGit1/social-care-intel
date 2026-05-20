"""
LLM-with-web-search provider for Version 1.

Supports:
  - OpenAI  : gpt-4o / gpt-4o-mini via the Responses API (web_search_preview)
  - Gemini  : gemini-2.0-flash / gemini-1.5-pro via google-genai + Google Search grounding
  - Claude  : claude-opus-4-7 / claude-sonnet-4-6 via Anthropic Messages API + web_search tool

All three paths return a SearchResult with the LLM's synthesised text and any
source references that can be extracted from the response metadata.
"""

import re
import time
from typing import List

from .base import SearchProvider, SearchResult, SourceRef


SUPPORTED_PROVIDERS = ("OpenAI", "Gemini", "Claude")


class LLMWebProvider(SearchProvider):

    def __init__(self, provider_name: str, model_name: str, api_key: str):
        if provider_name not in SUPPORTED_PROVIDERS:
            raise ValueError(f"provider_name must be one of {SUPPORTED_PROVIDERS}")

        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return f"{self.provider_name} {self.model_name}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def research(self, prompt: str, max_tokens: int = 6000) -> SearchResult:
        try:
            if self.provider_name == "OpenAI":
                return self._openai_research(prompt, max_tokens)
            elif self.provider_name == "Gemini":
                return self._gemini_research(prompt, max_tokens)
            elif self.provider_name == "Claude":
                return self._claude_research(prompt, max_tokens)
        except Exception as exc:
            return SearchResult(query=prompt[:120], content="", error=str(exc))

    # ------------------------------------------------------------------
    # OpenAI — Responses API with web_search_preview
    # ------------------------------------------------------------------

    def _openai_research(self, prompt: str, max_tokens: int) -> SearchResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        response = client.responses.create(
            model=self.model_name,
            tools=[{"type": "web_search_preview"}],
            input=prompt,
            max_output_tokens=max_tokens,
        )

        text_parts: List[str] = []
        sources: List[SourceRef] = []

        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for content_block in getattr(item, "content", []):
                    if getattr(content_block, "type", "") == "output_text":
                        text_parts.append(content_block.text)
            elif item_type == "web_search_call":
                pass  # logged internally; sources come from annotations
            elif item_type == "web_search_result":
                url = getattr(item, "url", "")
                title = getattr(item, "title", "")
                if url:
                    sources.append(SourceRef(url=url, title=title))

        # Also extract inline annotations if present
        for item in response.output:
            for content_block in getattr(item, "content", []):
                for annotation in getattr(content_block, "annotations", []):
                    url = getattr(annotation, "url", "")
                    title = getattr(annotation, "title", "")
                    if url and not any(s.url == url for s in sources):
                        sources.append(SourceRef(url=url, title=title))

        content = "\n\n".join(text_parts)
        return SearchResult(query=prompt[:120], content=content, sources=sources)

    # ------------------------------------------------------------------
    # Gemini — google-genai with Google Search grounding
    # ------------------------------------------------------------------

    def _gemini_research(self, prompt: str, max_tokens: int) -> SearchResult:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=max_tokens,
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        content = response.text or ""
        sources: List[SourceRef] = []

        # Extract grounding metadata if available
        try:
            metadata = response.candidates[0].grounding_metadata
            for chunk in getattr(metadata, "grounding_chunks", []):
                web = getattr(chunk, "web", None)
                if web:
                    url = getattr(web, "uri", "")
                    title = getattr(web, "title", "")
                    if url:
                        sources.append(SourceRef(url=url, title=title))
        except (IndexError, AttributeError):
            pass

        return SearchResult(query=prompt[:120], content=content, sources=sources)

    # ------------------------------------------------------------------
    # Claude — Messages API with web_search_20250305 built-in tool
    # ------------------------------------------------------------------

    def _claude_research(self, prompt: str, max_tokens: int) -> SearchResult:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        messages = [{"role": "user", "content": prompt}]
        text_parts: List[str] = []
        sources: List[SourceRef] = []
        max_iterations = 12

        for _ in range(max_iterations):
            response = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages,
            )

            tool_uses = []
            for block in response.content:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    text_parts.append(block.text)
                elif block_type == "tool_use":
                    tool_uses.append(block)
                elif block_type == "tool_result":
                    # Extract URLs from search result items
                    for item in getattr(block, "content", []):
                        url = getattr(item, "url", "")
                        title = getattr(item, "title", "")
                        if url and not any(s.url == url for s in sources):
                            sources.append(SourceRef(url=url, title=title))

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                # Append assistant turn and continue — the server handles search execution
                messages.append({"role": "assistant", "content": response.content})
                # Acknowledge each tool_use so the conversation can continue
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": [],
                    }
                    for tu in tool_uses
                ]
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
            else:
                break

        content = "\n\n".join(text_parts)
        return SearchResult(query=prompt[:120], content=content, sources=sources)
