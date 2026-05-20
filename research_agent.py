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
    DISCOVERY_PATH = Path(__file__).parent / "prompts" / "competitor_discovery_prompt.txt"
    DRILLDOWN_PATH = Path(__file__).parent / "prompts" / "provider_drilldown_prompt.txt"

    def __init__(self, config: ResearchConfig, provider: SearchProvider):
        self.config = config
        self.provider = provider
        self._prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")
        self._enrich_template = self.ENRICH_PATH.read_text(encoding="utf-8")
        self._discovery_template = self.DISCOVERY_PATH.read_text(encoding="utf-8")
        self._drilldown_template = self.DRILLDOWN_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        cfg = self.config

        geo_label = cfg.geographic_area or "(area to be inferred from commissioner)"

        # ---- Phase 0 (Deep only): Dedicated competitor discovery ----
        discovered_competitors: List[Dict] = []
        if not cfg.is_quick:
            status_callback(f"  🎯 Discovering providers in **{geo_label}** for **{cfg.service_area}**…")
            discovered_competitors = self._discover_competitors(status_callback)
            status_callback(f"  Discovery returned {len(discovered_competitors)} providers")

        # ---- Phase 1: Master research (procurement + priorities + competitors) ----
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
        master_competitors = raw_data.get("competitors", [])
        sources = combined_sources[: cfg.max_sources]

        # Merge discovered competitors with master research competitors (dedupe by name)
        competitors = _merge_competitors(discovered_competitors, master_competitors)[: cfg.max_competitors]
        status_callback(f"  Combined competitor list: {len(competitors)} unique providers")

        # ---- Phase 1.5 (Deep only): Drill down on multi-provider contracts ----
        if not cfg.is_quick and procurement:
            multi = [p for p in procurement if _looks_multi_provider(p)]
            for p in multi[:3]:  # Cap at 3 to bound cost
                status_callback(f"  📋 Drilling down on contract **{p.get('title', '')[:60]}**…")
                drill = self._drilldown_contract(p)
                # Stash awarded provider list on the procurement record
                p["awarded_providers"] = drill.get("awarded_providers", [])
                p["shortlisted_providers"] = drill.get("shortlisted_providers", [])
                p["drilldown_notes"] = drill.get("notes", "")
                # Promote drilled-down provider names into competitor list (if not already there)
                for ap in drill.get("awarded_providers", []) + drill.get("shortlisted_providers", []):
                    name = ap.get("name", "").strip()
                    if name and not any(c.get("name", "").lower() == name.lower() for c in competitors):
                        competitors.append({
                            "name": name,
                            "selection_rationale": f"Drilled from contract '{p.get('title', '')}' — {ap.get('evidence_url', '')}",
                            "source_urls": [ap.get("evidence_url", "")],
                        })
            competitors = competitors[: cfg.max_competitors]

        # ---- Phase 2 (Deep only): Enrich each competitor with targeted searches ----
        if not cfg.is_quick and competitors:
            status_callback(f"  🔎 Deep enrichment: running targeted searches on {len(competitors)} competitors…")
            enriched = []
            for i, comp in enumerate(competitors, 1):
                name = comp.get("name", "Unknown")
                status_callback(f"    [{i}/{len(competitors)}] Enriching **{name}** — website, CQC, Companies House, contracts…")
                enriched_comp = self._enrich_competitor(comp)
                # Always ensure enrichment_status is present so dashboard can show it
                if "enrichment_status" not in enriched_comp:
                    enriched_comp["enrichment_status"] = {
                        "website_found": bool(enriched_comp.get("website") and enriched_comp.get("website") not in ("Unknown", "")),
                        "cqc_found": bool(enriched_comp.get("cqc_profile_url")),
                        "companies_house_found": bool(enriched_comp.get("companies_house_number") and enriched_comp.get("companies_house_number") != "Unknown"),
                        "contracts_found": bool(enriched_comp.get("known_contracts_with_commissioner")),
                        "searches_run": [],
                    }
                enriched.append(enriched_comp)
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
    # Phase 0: Dedicated competitor discovery (Deep Scan only)
    # ------------------------------------------------------------------

    def _discover_competitors(self, status_callback: StatusCallback = _noop) -> List[Dict[str, Any]]:
        cfg = self.config
        known = ", ".join(cfg.known_competitors) if cfg.known_competitors else "None"

        # Try to extract a commissioner domain hint for site: searches
        commissioner_domain = ""
        if cfg.commissioner:
            slug = cfg.commissioner.lower().replace(" ", "")
            commissioner_domain = f"{slug}.gov.uk"

        prompt = _fill_template(self._discovery_template,
            commissioner=cfg.commissioner,
            commissioner_domain=commissioner_domain,
            service_area=cfg.service_area,
            geographic_area=cfg.geographic_area or "the commissioner's area",
            time_period=cfg.time_period,
            known_competitors=known,
        )

        result = self.provider.research(prompt, max_tokens=5000)
        if not result.ok:
            status_callback(f"  ⚠️ Discovery call failed: {result.error}")
            return []

        data = _extract_json(result.content)
        if not data:
            return []

        discovered = data.get("competitors", [])
        # Flatten first_pass_data into top-level hints
        for c in discovered:
            hints = c.pop("first_pass_data", {}) or {}
            if hints.get("website_hint"):
                c["website"] = hints["website_hint"]
            if hints.get("cqc_hint"):
                c["cqc_rating"] = hints["cqc_hint"]
            if hints.get("headquarters_hint"):
                c["headquarters"] = hints["headquarters_hint"]
            # Map evidence_source_url -> source_urls list
            ev = c.pop("evidence_source_url", "")
            if ev:
                c.setdefault("source_urls", []).append(ev)

        return discovered

    # ------------------------------------------------------------------
    # Phase 1.5: Procurement provider drill-down (Deep Scan only)
    # ------------------------------------------------------------------

    def _drilldown_contract(self, procurement: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        prompt = _fill_template(self._drilldown_template,
            contract_title=procurement.get("title", ""),
            commissioner=cfg.commissioner,
            source_url=procurement.get("source_url", ""),
            contract_type=procurement.get("type", ""),
            contract_value=procurement.get("value", ""),
        )

        result = self.provider.research(prompt, max_tokens=3000)
        if not result.ok:
            return {"awarded_providers": [], "shortlisted_providers": [], "notes": f"Drilldown failed: {result.error}"}

        data = _extract_json(result.content)
        return data if data else {"awarded_providers": [], "shortlisted_providers": [], "notes": "No JSON returned"}

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


def _looks_multi_provider(procurement: Dict[str, Any]) -> bool:
    """Heuristic: should we drill down on this contract?"""
    awarded_to = (procurement.get("awarded_to") or "").lower()
    triggers = ("multiple", "various", "tbc", "not yet awarded", "framework",
                "to be announced", "see notice", "list", "panel")
    if any(t in awarded_to for t in triggers):
        return True
    if not awarded_to or awarded_to == "unknown":
        return True
    if procurement.get("type", "").lower() in ("framework", "dps"):
        return True
    return False


def _merge_competitors(discovered: List[Dict], master: List[Dict]) -> List[Dict]:
    """Dedupe by name (case insensitive). Discovered takes precedence for selection_rationale."""
    by_name: Dict[str, Dict] = {}
    for c in discovered:
        name = (c.get("name") or "").strip()
        if name:
            by_name[name.lower()] = dict(c)
    for c in master:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in by_name:
            # Merge: keep discovered's rationale, fill missing fields from master
            existing = by_name[key]
            for k, v in c.items():
                if k not in existing or not existing[k]:
                    existing[k] = v
        else:
            by_name[key] = dict(c)
    return list(by_name.values())


def _merge_sources(existing: List[dict], new_sources: List[dict]) -> List[dict]:
    seen_urls = {s.get("url", "") for s in existing}
    merged = list(existing)
    for s in new_sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            merged.append(s)
            seen_urls.add(url)
    return merged
