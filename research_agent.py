"""
ResearchAgent — orchestrates the market intelligence research phase.

Responsibilities:
  - Build targeted prompts from the research brief
  - Call the search provider (abstracted; no provider-specific code here)
  - Parse and validate the returned JSON
  - Accumulate results into the standard output schema
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from search_providers.base import SearchProvider


@dataclass
class ResearchConfig:
    commissioner: str
    service_area: str
    target_company: str
    target_website: str
    geographic_area: str
    time_period: str
    known_competitors: List[str]
    manual_urls: List[str]
    research_depth: str          # "quick" or "deep"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ---- Depth-dependent limits ----------------------------------------

    @property
    def is_quick(self) -> bool:
        return self.research_depth == "quick"

    @property
    def max_competitors(self) -> int:
        return 5 if self.is_quick else 10

    @property
    def max_sources(self) -> int:
        return 12 if self.is_quick else 30

    @property
    def max_procurement_notices(self) -> int:
        return 3 if self.is_quick else 8

    @property
    def max_commissioner_docs(self) -> int:
        return 3 if self.is_quick else 8

    @property
    def max_pages_per_website(self) -> int:
        return 2 if self.is_quick else 5


# ---------------------------------------------------------------------------

StatusCallback = Callable[[str], None]
_noop: StatusCallback = lambda msg: None


class ResearchAgent:

    PROMPT_PATH = Path(__file__).parent / "prompts" / "master_research_prompt.txt"

    def __init__(self, config: ResearchConfig, provider: SearchProvider):
        self.config = config
        self.provider = provider
        self._prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        cfg = self.config

        status_callback(f"  Searching procurement databases for **{cfg.service_area}** in **{cfg.geographic_area}**…")

        prompt = self._build_prompt()
        result = self.provider.research(prompt, max_tokens=7000)

        if not result.ok:
            status_callback(f"  ⚠️ Research call returned an error: {result.error}")
            raw_data: Dict[str, Any] = {}
        else:
            status_callback("  Parsing research results…")
            raw_data = _extract_json(result.content)

        status_callback(f"  Found {len(raw_data.get('procurement', []))} procurement notices, "
                        f"{len(raw_data.get('competitors', []))} competitors")

        # Merge provider-returned sources with any sources embedded in the response
        provider_sources = [
            {"url": s.url, "title": s.title, "used_for": "Research phase"}
            for s in result.sources
        ]
        combined_sources = _merge_sources(
            raw_data.get("sources", []), provider_sources
        )

        # Enforce depth limits
        procurement = raw_data.get("procurement", [])[: cfg.max_procurement_notices]
        competitors = raw_data.get("competitors", [])[: cfg.max_competitors]
        sources = combined_sources[: cfg.max_sources]

        output = {
            "metadata": {
                "run_id": cfg.run_id,
                "timestamp": datetime.now().isoformat(),
                "commissioner": cfg.commissioner,
                "service_area": cfg.service_area,
                "target_company": cfg.target_company,
                "target_website": cfg.target_website,
                "geographic_area": cfg.geographic_area,
                "time_period": cfg.time_period,
                "research_depth": cfg.research_depth,
                "model_provider": self.provider.name,
            },
            "procurement": procurement,
            "competitors": competitors,
            "commissioner_priorities": raw_data.get("commissioner_priorities", []),
            "sources": sources,
            "evidence_gaps": raw_data.get("evidence_gaps", []),
            "confidence_rating": raw_data.get("confidence_rating", "Low"),
            "confidence_rationale": raw_data.get("confidence_rationale", ""),
        }

        return output

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        cfg = self.config
        known = ", ".join(cfg.known_competitors) if cfg.known_competitors else "None specified"
        manual = "\n".join(cfg.manual_urls) if cfg.manual_urls else "None"
        depth_label = (
            f"Quick Scan — limit to {cfg.max_procurement_notices} procurement notices "
            f"and {cfg.max_competitors} competitors"
            if cfg.is_quick
            else f"Deeper Scan — find up to {cfg.max_procurement_notices} procurement notices "
                 f"and {cfg.max_competitors} competitors with fuller detail"
        )

        return self._prompt_template.format(
            commissioner=cfg.commissioner,
            service_area=cfg.service_area,
            target_company=cfg.target_company,
            target_website=cfg.target_website or "Not provided — please find it",
            geographic_area=cfg.geographic_area or "Not provided — infer from commissioner area",
            time_period=cfg.time_period,
            known_competitors=known,
            manual_urls=manual,
            research_depth=depth_label,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from LLM output.
    The LLM is instructed to return only JSON but may sometimes wrap it
    in markdown fences or add a short preamble.
    """
    if not text:
        return {}

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {}


def _merge_sources(existing: List[dict], new_sources: List[dict]) -> List[dict]:
    seen_urls = {s.get("url", "") for s in existing}
    merged = list(existing)
    for s in new_sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            merged.append(s)
            seen_urls.add(url)
    return merged
