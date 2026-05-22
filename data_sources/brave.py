"""
Brave Search API client — authoritative web search, deterministic results.

Used in two ways:
1. Find specific source URLs (CQC profiles, Contracts Finder notices, etc.)
   then pass IDs to authoritative APIs.
2. As a general web search backbone that returns clean URLs + snippets
   rather than relying on an LLM's bundled search.

API docs: https://api.search.brave.com/app/documentation
"""

from typing import Any, Dict, List, Optional

import requests


class BraveClient:
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        })

    def search(self, query: str, count: int = 10, country: str = "GB") -> List[Dict[str, Any]]:
        """
        Run a Brave Web Search. Returns a list of result dicts each with
        keys: title, url, description, age (where available).
        """
        params = {
            "q": query,
            "count": min(count, 20),
            "country": country,
            "search_lang": "en",
            "safesearch": "moderate",
        }
        try:
            r = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            return [{"error": str(exc)}]

        data = r.json()
        web_results = (data.get("web") or {}).get("results") or []
        return [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "age": result.get("age", ""),
            }
            for result in web_results
        ]

    def find_cqc_profile(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Return the most likely cqc.org.uk profile result for a named provider."""
        results = self.search(f'site:cqc.org.uk "{provider_name}"', count=5)
        for r in results:
            url = r.get("url", "")
            if "cqc.org.uk" in url:
                return r
        return None

    def find_companies_house(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Return the Companies House page for a named company, if found."""
        results = self.search(
            f'site:find-and-update.company-information.service.gov.uk "{company_name}"',
            count=5,
        )
        for r in results:
            if "company-information.service.gov.uk" in r.get("url", ""):
                return r
        return None
