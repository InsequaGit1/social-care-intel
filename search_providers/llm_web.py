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

        kwargs = dict(
            model=self.model_name,
            tools=[{"type": "web_search_preview"}],
            input=prompt,
            max_output_tokens=max_tokens,
        )
        # Some newer models reject the temperature parameter — adapt at runtime.
        if not getattr(self, "_temperature_unsupported", False):
            kwargs["temperature"] = 0
        try:
            response = client.responses.create(**kwargs)
        except Exception as exc:
            if "temperature" in str(exc).lower() and "temperature" in kwargs:
                self._temperature_unsupported = True
                kwargs.pop("temperature")
                response = client.responses.create(**kwargs)
            else:
                raise

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
            temperature=0,  # Determinism
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
        """
        Claude with the server-side web_search tool. Anthropic executes the
        searches internally and returns results in the response — we do NOT
        send tool_result blocks back. The only multi-turn case is stop_reason
        == "pause_turn" (long-running search), where we pass the assistant
        turn back unchanged to resume.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        messages = [{"role": "user", "content": prompt}]
        text_parts: List[str] = []
        sources: List[SourceRef] = []
        max_iterations = 8

        for _ in range(max_iterations):
            kwargs = dict(
                model=self.model_name,
                max_tokens=max_tokens,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
                messages=messages,
            )
            # Older Claude models accept temperature=0 (low-variance narrative);
            # newer generations (e.g. opus-4-8) reject the parameter with a 400
            # "`temperature` is deprecated for this model". Adapt at runtime and
            # remember for the rest of this provider's lifetime.
            if not getattr(self, "_temperature_unsupported", False):
                kwargs["temperature"] = 0
            try:
                response = client.messages.create(**kwargs)
            except anthropic.BadRequestError as exc:
                if "temperature" in str(exc).lower() and "temperature" in kwargs:
                    self._temperature_unsupported = True
                    kwargs.pop("temperature")
                    response = client.messages.create(**kwargs)
                else:
                    raise

            for block in response.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(block.text)
                    # Citations attached to text blocks carry source URLs
                    for cite in (getattr(block, "citations", None) or []):
                        url = getattr(cite, "url", "")
                        title = getattr(cite, "title", "")
                        if url and not any(s.url == url for s in sources):
                            sources.append(SourceRef(url=url, title=title))
                elif btype == "web_search_tool_result":
                    # Server-side search results: block.content is a list of
                    # web_search_result items with url/title.
                    for item in (getattr(block, "content", None) or []):
                        url = getattr(item, "url", "")
                        title = getattr(item, "title", "")
                        if url and not any(s.url == url for s in sources):
                            sources.append(SourceRef(url=url, title=title))

            if response.stop_reason == "pause_turn":
                # Resume a long-running server tool call — pass assistant turn back.
                messages.append({"role": "assistant", "content": response.content})
                continue
            break  # end_turn / max_tokens / stop_sequence → done

        content = "\n\n".join(text_parts)
        return SearchResult(query=prompt[:120], content=content, sources=sources)
