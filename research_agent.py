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
    # External data source API keys (read from Streamlit secrets / env vars)
    cqc_api_key: str = ""
    brave_api_key: str = ""
    companies_house_api_key: str = ""

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
    TARGET_PROFILE_PATH = Path(__file__).parent / "prompts" / "target_profile_prompt.txt"

    def __init__(self, config: ResearchConfig, provider: SearchProvider):
        self.config = config
        self.provider = provider
        self._prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")
        self._enrich_template = self.ENRICH_PATH.read_text(encoding="utf-8")
        self._discovery_template = self.DISCOVERY_PATH.read_text(encoding="utf-8")
        self._drilldown_template = self.DRILLDOWN_PATH.read_text(encoding="utf-8")
        self._target_profile_template = self.TARGET_PROFILE_PATH.read_text(encoding="utf-8")

        # Area context learned from the target's CQC record, used to
        # disambiguate same-named competitors in other towns.
        self._target_local_authority = ""

        # Initialise authoritative data source clients if keys present
        self.cqc = None
        self.brave = None
        if config.cqc_api_key:
            try:
                from data_sources.cqc import CQCClient
                self.cqc = CQCClient(config.cqc_api_key)
            except Exception:
                self.cqc = None
        if config.brave_api_key:
            try:
                from data_sources.brave import BraveClient
                self.brave = BraveClient(config.brave_api_key)
            except Exception:
                self.brave = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        cfg = self.config

        geo_label = cfg.geographic_area or "(area to be inferred from commissioner)"

        # ---- Phase 0a (Deep only): Dedicated TARGET company profile lookup ----
        target_profile: Dict[str, Any] = {}
        if not cfg.is_quick:
            status_callback(f"  🎯 Building verified profile for target: **{cfg.target_company}**…")
            target_profile = self._research_target_profile(status_callback)
            cqc_rating = target_profile.get("cqc", {}).get("rating", "Unknown")
            ch_number = target_profile.get("companies_house", {}).get("number", "Unknown")
            status_callback(f"    Target CQC rating: {cqc_rating} · Companies House: {ch_number}")

        # ---- Phase 0b (Deep only): Dedicated competitor discovery ----
        discovered_competitors: List[Dict] = []
        if not cfg.is_quick:
            status_callback(f"  🔍 Discovering providers in **{geo_label}** for **{cfg.service_area}**…")
            discovered_competitors = self._discover_competitors(status_callback)
            status_callback(f"    Discovery returned {len(discovered_competitors)} providers")

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
        procurement_raw = raw_data.get("procurement", [])[: cfg.max_procurement_notices]
        # Reject procurement entries with hallucinated-looking URLs
        procurement, rejected = _filter_hallucinated_procurement(procurement_raw)
        if rejected:
            status_callback(
                f"  🛑 Rejected {len(rejected)} procurement entries with suspicious URLs "
                f"(likely model fabrication). Kept {len(procurement)}."
            )
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
                # Stamp enrichment_status BEFORE running so it's always present
                comp["enrichment_status"] = {
                    "attempted": True,
                    "website_found": False,
                    "cqc_found": False,
                    "companies_house_found": False,
                    "contracts_found": False,
                    "searches_run": [],
                    "error": None,
                }
                status_callback(f"    [{i}/{len(competitors)}] Enriching **{name}** — website, CQC, Companies House, contracts…")
                try:
                    enriched_comp = self._enrich_competitor(comp)
                    # Recompute the status flags from actual data after enrichment
                    enriched_comp["enrichment_status"] = _compute_enrichment_status(
                        enriched_comp,
                        original_status=enriched_comp.get("enrichment_status", {}),
                    )
                except Exception as exc:
                    comp["enrichment_status"]["error"] = str(exc)
                    enriched_comp = comp
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
            "procurement_rejected": rejected,
            "target_profile": target_profile,
            "competitors": competitors,
            "commissioner_priorities": raw_data.get("commissioner_priorities", []),
            "sources": sources,
            "evidence_gaps": raw_data.get("evidence_gaps", []),
            "confidence_rating": raw_data.get("confidence_rating", "Low"),
            "confidence_rationale": raw_data.get("confidence_rationale", ""),
        }

        return output

    # ------------------------------------------------------------------
    # Authoritative CQC lookup (Brave → CQC API) with name confidence
    # ------------------------------------------------------------------

    CQC_MATCH_THRESHOLD = 0.6

    def _expected_area_tokens(self) -> set:
        """Tokens describing the target area, for CQC record disambiguation."""
        cfg = self.config
        text = " ".join([
            cfg.geographic_area or "",
            cfg.commissioner or "",
            self._target_local_authority or "",
        ]).lower()
        # Drop generic council words so "southend" survives but "council" doesn't
        text = re.sub(r"\b(city|council|borough|county|district|metropolitan|"
                      r"unitary|authority|icb|nhs|of|and|the)\b", " ", text)
        return {t for t in re.split(r"[^a-z]+", text) if len(t) >= 4}

    def _lookup_cqc(self, provider_name: str, status_callback: StatusCallback = _noop) -> Optional[Dict[str, Any]]:
        """
        Find a CQC profile for a named provider, using Brave to surface
        candidate URLs, a name-similarity check, and a geographic check to
        avoid attaching a same-named provider from a different town.
        Only returns data when confidence >= CQC_MATCH_THRESHOLD.
        """
        if not (self.brave and self.cqc):
            return None
        try:
            results = self.brave.search(f'site:cqc.org.uk "{provider_name}"', count=8)
            expected = self._expected_area_tokens()

            # Score every valid candidate by name confidence + area preference.
            # Area is used to PICK among similarly-named candidates (e.g. the
            # right "Victoria Court"), NOT to hard-reject — so a legitimate
            # provider registered in a neighbouring district is still kept.
            candidates = []
            for r in results:
                url = r.get("url", "") or ""
                title = r.get("title", "") or ""
                if "cqc.org.uk" not in url or not self.cqc.extract_id_from_url(url):
                    continue
                candidate_name = title.split(" - ")[0].split(" | ")[0].strip()
                name_score = _name_match_confidence(provider_name, candidate_name)
                if name_score < self.CQC_MATCH_THRESHOLD:
                    continue
                # Does the Brave title/snippet/url hint at the target area?
                hint_text = f"{title} {r.get('description', '')} {url}".lower()
                hint_tokens = {t for t in re.split(r"[^a-z]+", hint_text) if len(t) >= 4}
                area_bonus = 0.15 if (expected and not expected.isdisjoint(hint_tokens)) else 0.0
                candidates.append((name_score + area_bonus, name_score, r, candidate_name))

            if not candidates:
                status_callback(f"    ⚠️ CQC: no confident name match for **{provider_name}**")
                return None

            candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_score, best, best_candidate = candidates[0]

            raw = self.cqc.fetch_from_url(best.get("url", ""))
            if not raw:
                return None
            summary = self.cqc.summarise_provider_profile(raw)

            # Soft area verification: flag (don't drop) records outside the area.
            area_verified = True
            if expected:
                record_area = " ".join([
                    summary.get("local_authority", ""), summary.get("region", ""),
                    summary.get("town", ""), summary.get("postcode", ""),
                ]).lower()
                record_tokens = {t for t in re.split(r"[^a-z]+", record_area) if len(t) >= 4}
                if record_tokens and expected.isdisjoint(record_tokens):
                    area_verified = False

            summary["_brave_snippet"] = best.get("description", "")
            summary["_match_confidence"] = round(best_score, 2)
            summary["_matched_name"] = best_candidate
            summary["_area_verified"] = area_verified

            beds = summary.get("number_of_beds")
            beds_str = f", {beds} beds" if beds else ""
            area_note = "" if area_verified else f" ⚠️ in {summary.get('local_authority') or 'other area'}"
            status_callback(
                f"    ✅ CQC: **{provider_name}** → '{best_candidate}' → "
                f"{summary.get('overall_rating', 'Unknown')}{beds_str} (conf {best_score:.2f}){area_note}"
            )
            return summary
        except Exception as exc:
            status_callback(f"    ⚠️ CQC lookup error for {provider_name}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Phase 0a: Verified target-company profile (Deep Scan only)
    # ------------------------------------------------------------------

    def _research_target_profile(self, status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        cfg = self.config

        # First — try authoritative CQC lookup if API keys configured.
        # (Don't yet know the LA, so disambiguation uses commissioner/geo only.)
        cqc_data = self._lookup_cqc(cfg.target_company, status_callback)

        # Learn the target's local authority so we can disambiguate competitors
        if cqc_data and cqc_data.get("local_authority"):
            self._target_local_authority = cqc_data["local_authority"]

        # Build context to pass to the LLM about what we already verified
        verified_context = ""
        if cqc_data:
            verified_context = (
                "\n=== AUTHORITATIVE CQC DATA (use these values verbatim) ===\n"
                f"CQC Rating: {cqc_data.get('overall_rating', 'Unknown')}\n"
                f"Last Inspection: {cqc_data.get('last_inspection_date', 'Unknown')}\n"
                f"CQC Profile URL: {cqc_data.get('cqc_url', '')}\n"
                f"Registration Status: {cqc_data.get('registration_status', '')}\n"
                f"Registration Date: {cqc_data.get('registration_date', '')}\n"
                f"Number of Beds: {cqc_data.get('number_of_beds', 'Unknown')}\n"
                f"Address: {cqc_data.get('address', '')}\n"
                f"Local Authority: {cqc_data.get('local_authority', '')}\n"
                f"Service Types: {cqc_data.get('service_types', [])}\n"
                f"Specialisms: {cqc_data.get('specialisms', [])}\n"
                f"Sub-ratings: {cqc_data.get('sub_ratings', {})}\n"
            )

        prompt = _fill_template(self._target_profile_template,
            target_company=cfg.target_company,
            target_website=cfg.target_website or "Not provided — search for it",
            commissioner=cfg.commissioner,
            service_area=cfg.service_area,
            geographic_area=cfg.geographic_area or "Not specified",
        ) + verified_context

        result = self.provider.research(prompt, max_tokens=4000)
        if not result.ok:
            status_callback(f"  ⚠️ Target profile lookup failed: {result.error}")
            return {"cqc": {"rating": cqc_data.get("overall_rating", "Unknown")} if cqc_data else {}}

        data = _extract_json(result.content) or {}

        # Overwrite CQC section with authoritative API data if we have it
        if cqc_data:
            data["cqc"] = {
                "rating": cqc_data.get("overall_rating", "Unknown"),
                "last_inspection_date": cqc_data.get("last_inspection_date", "Unknown"),
                "registration_date": cqc_data.get("registration_date", ""),
                "number_of_beds": cqc_data.get("number_of_beds"),
                "provider_id": cqc_data.get("provider_id", ""),
                "profile_url": cqc_data.get("cqc_url", ""),
                "sub_ratings": cqc_data.get("sub_ratings", {}),
                "service_types": cqc_data.get("service_types", []),
                "specialisms": cqc_data.get("specialisms", []),
                "local_authority": cqc_data.get("local_authority", ""),
                "registered_locations": [{
                    "name": cqc_data.get("name", ""),
                    "rating": cqc_data.get("overall_rating", "Unknown"),
                    "url": cqc_data.get("cqc_url", ""),
                }],
                "verified_source": "CQC Syndication API",
            }
            # If CQC includes a website and we don't have a good one yet, use it
            cqc_site = cqc_data.get("website")
            current_site = data.get("official_website")
            if cqc_site and (not current_site or current_site in ("Unknown", "")):
                data["official_website"] = cqc_site
            # Ensure lookup_status reflects authoritative data
            ls = data.get("lookup_status", {})
            ls["cqc_found"] = True
            if cqc_data.get("website"):
                ls["website_found"] = True
            data["lookup_status"] = ls

        # Strip any hallucinated procurement URLs from the contracts list
        contracts = data.get("contracts_with_commissioner", []) or []
        cleaned = []
        for c in contracts:
            if isinstance(c, dict):
                src = c.get("source_url", "")
                if src and _is_url_suspicious(src):
                    status_callback(f"    🛑 Dropped fabricated contract URL: {src}")
                    continue
            cleaned.append(c)
        data["contracts_with_commissioner"] = cleaned

        return data

    # ------------------------------------------------------------------
    # Phase 0b: Dedicated competitor discovery (Deep Scan only)
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

        # First — try authoritative CQC lookup
        cqc_data = self._lookup_cqc(name)

        verified_context = ""
        if cqc_data:
            verified_context = (
                f"\n\nAUTHORITATIVE CQC DATA (verbatim — do not change):\n"
                f"- CQC Rating: {cqc_data.get('overall_rating', 'Unknown')}\n"
                f"- CQC Profile URL: {cqc_data.get('cqc_url', '')}\n"
                f"- Last Inspection: {cqc_data.get('last_inspection_date', 'Unknown')}\n"
                f"- Address: {cqc_data.get('address', '')}\n"
                f"- Local Authority: {cqc_data.get('local_authority', '')}\n"
            )

        current_profile = json.dumps({
            k: competitor.get(k) for k in (
                "website", "cqc_rating", "companies_house_number",
                "services", "geographic_coverage", "selection_rationale",
            )
        }, indent=2) + verified_context

        prompt = _fill_template(self._enrich_template,
            company_name=name,
            current_profile=current_profile,
            commissioner=cfg.commissioner,
            service_area=cfg.service_area,
            geographic_area=cfg.geographic_area or "Unknown",
        )

        result = self.provider.research(prompt, max_tokens=3500)
        if not result.ok:
            # Even if LLM call failed, stamp CQC data so it's not lost
            base = dict(competitor)
            if cqc_data:
                base["cqc_rating"] = cqc_data.get("overall_rating", "Unknown")
                base["cqc_profile_url"] = cqc_data.get("cqc_url", "")
            return base

        enriched_data = _extract_json(result.content)
        if not enriched_data:
            base = dict(competitor)
            if cqc_data:
                base["cqc_rating"] = cqc_data.get("overall_rating", "Unknown")
                base["cqc_profile_url"] = cqc_data.get("cqc_url", "")
            return base

        # Merge — prefer enriched data where it's non-empty, but preserve rationale
        merged = dict(competitor)
        for key, value in enriched_data.items():
            if value and value not in ("Unknown", "", []):
                merged[key] = value
        # Always preserve original selection_rationale
        if competitor.get("selection_rationale"):
            merged["selection_rationale"] = competitor["selection_rationale"]
        # ALWAYS overwrite CQC fields with authoritative API data if present
        if cqc_data:
            merged["cqc_rating"] = cqc_data.get("overall_rating", "Unknown")
            merged["cqc_profile_url"] = cqc_data.get("cqc_url", "")
            merged["cqc_verified"] = True
            # Store the rich CQC structured data for benchmarking + dashboard
            merged["cqc_data"] = {
                "sub_ratings": cqc_data.get("sub_ratings", {}),
                "number_of_beds": cqc_data.get("number_of_beds"),
                "registration_date": cqc_data.get("registration_date", ""),
                "last_inspection_date": cqc_data.get("last_inspection_date", ""),
                "service_types": cqc_data.get("service_types", []),
                "specialisms": cqc_data.get("specialisms", []),
                "local_authority": cqc_data.get("local_authority", ""),
                "town": cqc_data.get("town", ""),
            }
            # If CQC has a website and we don't already have one, use it
            existing_site = merged.get("website") or ""
            if cqc_data.get("website") and existing_site in ("", None, "Unknown"):
                merged["website"] = cqc_data["website"]
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


def _normalise_company_name(name: str) -> str:
    """Lowercase, strip parenthetical content, punctuation, and company-form suffixes."""
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", " ", s)              # remove (...) blocks
    s = re.sub(r"[^\w\s]", " ", s)                # strip punctuation
    s = re.sub(r"\b(ltd|limited|plc|llp|llc|inc|the|co)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_match_confidence(query: str, candidate: str) -> float:
    """
    Return a 0-1 confidence that `candidate` refers to the same company as `query`.

    Examples (threshold 0.6):
      "Ashley Care Ltd" vs "Ashley Care Limited"                       → 1.0    accept
      "Bluebird Care" vs "Bluebird Care (Southend & Rochford)"         → 1.0    accept
      "Ashley Care Ltd" vs "Ashley Community Care Services Limited"    → 0.35   reject
      "Mears Care" vs "Mears Group Ltd"                                → ~0.1   reject
    """
    nq = _normalise_company_name(query)
    nc = _normalise_company_name(candidate)
    if not nq or not nc:
        return 0.0

    if nq == nc:
        return 1.0

    # Query phrase appears as contiguous substring of candidate (token boundaries)
    if f" {nq} " in f" {nc} ":
        len_ratio = len(nq) / len(nc)
        return min(1.0, 0.7 + 0.3 * len_ratio)

    # Candidate is a shorter version of the query
    if f" {nc} " in f" {nq} ":
        len_ratio = len(nc) / len(nq)
        return min(1.0, 0.6 + 0.3 * len_ratio)

    # Token-level fallback — all query tokens present but not contiguous = weak
    qt = set(nq.split())
    ct = set(nc.split())
    if not qt or not ct:
        return 0.0
    overlap = len(qt & ct)
    if overlap == len(qt) and overlap >= 2:
        return 0.35
    return (overlap / max(len(qt), len(ct))) * 0.3


def _compute_enrichment_status(comp: Dict, original_status: Dict) -> Dict:
    """Recompute enrichment flags from the actual data fields after enrichment ran."""
    website = comp.get("website", "")
    cqc_url = comp.get("cqc_profile_url", "")
    ch_num = comp.get("companies_house_number", "")
    contracts = comp.get("known_contracts_with_commissioner", [])

    return {
        "attempted": True,
        "website_found": bool(website and website not in ("Unknown", "") and website.startswith("http")),
        "cqc_found": bool(cqc_url and "cqc.org.uk" in cqc_url),
        "companies_house_found": bool(ch_num and ch_num != "Unknown" and len(str(ch_num)) >= 6),
        "contracts_found": bool(contracts),
        "searches_run": original_status.get("searches_run", []),
        "error": original_status.get("error"),
    }


def _is_url_suspicious(url: str) -> bool:
    """
    Detect URLs that look like LLM hallucinations.
    Common patterns: sequential digits, repeated digit pairs, placeholder IDs.
    """
    if not url or not isinstance(url, str):
        return True
    # Strip protocol and find any long digit sequence in the path
    import re as _re
    m = _re.search(r"/Notice/(\d{6,})", url)
    if m:
        digits = m.group(1)
        # Sequential ascending: 1234567890, 12345678
        if digits in "0123456789" or digits in "9876543210":
            return True
        # Repeated digit pairs like 11223344, 33445566, 55667788, 66778899
        pairs = [digits[i:i+2] for i in range(0, len(digits) - 1, 2)]
        if len(pairs) >= 3 and all(p[0] == p[1] for p in pairs):
            return True
        # Each pair is +11 from the previous (1122, 2233, 3344, 4455 pattern)
        if len(pairs) >= 3:
            ints = []
            try:
                ints = [int(p[0]) for p in pairs]
            except ValueError:
                pass
            if ints and all(ints[i+1] - ints[i] == 1 for i in range(len(ints) - 1)):
                return True
    # Reverse-sequential markers like 0987654321
    if "/Notice/0987654321" in url or "/Notice/1234567890" in url:
        return True
    return False


def _filter_hallucinated_procurement(procurement: List[Dict]) -> tuple:
    """Return (kept, rejected) splitting procurement by suspicious-URL test."""
    kept = []
    rejected = []
    for p in procurement:
        url = p.get("source_url", "")
        if _is_url_suspicious(url):
            rejected.append(p)
        else:
            kept.append(p)
    return kept, rejected


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
