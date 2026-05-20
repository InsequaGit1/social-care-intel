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
    time_period: str
    research_depth: str          # "quick" or "deep"
    target_website: str = ""
    geographic_area: str = ""
    known_competitors: List[str] = field(default_factory=list)
    manual_urls: List[str] = field(default_factory=list)
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
    ENRICH_PATH = Path(__file__).parent / "prompts" / "competitor_enrichment_prompt.txt"

    def __init__(self, config: ResearchConfig, provider: SearchProvider):
        self.config = config
        self.provider = provider
        self._prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")
        self._enrich_template = self.ENRICH_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        cfg = self.config

        geo_label = cfg.geographic_area or "(area to be inferred from commissioner)"
        status_callback(f"  Searching procurement databases for **{cfg.service_area}** in **{geo_label}**…")

        prompt = self._build_prompt()
        result = self.provider.research(prompt, max_tokens=7000)

        if not result.ok:
            status_callback(f"  ⚠️ Research call returned an error: {result.error}")
            raw_data: Dict[str, Any] = {}
        else:
            status_callback("  Parsing research results…")
            raw_data = _extract_json(result.content)
            if not raw_data and result.content:
                preview = result.content[:200].replace("\n", " ")
                status_callback(
                    f"  ⚠️ Model did not return valid JSON. First 200 chars: _{preview}…_"
                )

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

        # Deep Scan only — enrich each competitor with focused targeted searches
        if not cfg.is_quick and competitors:
            status_callback(f"  🔎 Deep enrichment: running targeted searches on {len(competitors)} competitors…")
            enriched = []
            for i, comp in enumerate(competitors, 1):
                name = comp.get("name", "Unknown")
                status_callback(f"    [{i}/{len(competitors)}] Enriching **{name}** — website, CQC, Companies House, contracts…")
                enriched_comp = self._enrich_competitor(comp)
                enriched.append(enriched_comp)
                # Roll up any new sources
                for url in enriched_comp.get("source_urls", []):
                    if url and not any(s.get("url") == url for s in sources):
                        sources.append({"url": url, "title": f"{name} enrichment", "used_for": "Competitor enrichment"})
            competitors = enriched
            sources = sources[: cfg.max_sources]

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
    # Per-competitor enrichment (Deep Scan only)
    # ------------------------------------------------------------------

    def _enrich_competitor(self, competitor: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        name = competitor.get("name", "")

        current_profile = json.dumps({
            k: competitor.get(k) for k in (
                "website", "cqc_rating", "companies_house_number",
                "services", "geographic_coverage", "selection_rationale",
            )
        }, indent=2)

        prompt = _fill_template(self._enrich_template,
            company_name=name,
            current_profile=current_profile,
            commissioner=cfg.commissioner,
            service_area=cfg.service_area,
            geographic_area=cfg.geographic_area or "Unknown",
        )

        result = self.provider.research(prompt, max_tokens=3500)
        if not result.ok:
            return competitor  # Keep original if enrichment fails

        enriched_data = _extract_json(result.content)
        if not enriched_data:
            return competitor

        # Merge — prefer enriched data where it's non-empty, but preserve rationale
        merged = dict(competitor)
        for key, value in enriched_data.items():
            if value and value not in ("Unknown", "", []):
                merged[key] = value
        # Always preserve original selection_rationale
        if competitor.get("selection_rationale"):
            merged["selection_rationale"] = competitor["selection_rationale"]
        return merged

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

        return _fill_template(self._prompt_template,
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

def _fill_template(template: str, **kwargs) -> str:
    """
    Safe template substitution that only replaces known {key} placeholders.
    Unlike str.format(), this leaves all other curly braces (e.g. JSON in
    the prompt) untouched, preventing KeyError on JSON schema examples.
    """
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template


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
