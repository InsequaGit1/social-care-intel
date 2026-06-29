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
    def service_area_label(self) -> str:
        """Service area for prompts; sensible default when the user leaves it blank
        (the target's CQC service types drive competitor filtering regardless)."""
        return self.service_area or "social care services (infer from the target company's CQC registration)"

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

        # Area + service context learned from the target's CQC record, used to
        # disambiguate competitors and filter the area list to like-for-like
        # service types (CQC's own classification, not the user's free text).
        self._target_local_authority = ""
        self._target_service_types: List[str] = []

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

        # ---- Phase 0a (BOTH modes): Verified TARGET company profile ----
        # Cheap and authoritative (CQC + Companies House + website). Always worth it.
        status_callback(f"  🎯 Building verified profile for target: **{cfg.target_company}**…")
        target_profile = self._research_target_profile(status_callback)
        cqc_rating = (target_profile.get("cqc", {}) or {}).get("rating", "Unknown")
        ch_number = (target_profile.get("companies_house", {}) or {}).get("number", "Unknown")
        status_callback(f"    Target CQC rating: {cqc_rating} · Companies House: {ch_number}")

        # If the target lookup didn't yield a local authority (e.g. target name
        # misspelled or not CQC-registered), derive it independently from the
        # commissioner/area so authoritative CQC area discovery can still run.
        if not self._target_local_authority and self.cqc and self.brave:
            self._target_local_authority = self._derive_local_authority(status_callback)

        # ---- Phase 0b: Competitor discovery ----
        # CQC area discovery (API-only, authoritative) runs in BOTH modes.
        # LLM discovery (an extra web-search call) is Deep-only.
        discovered_competitors: List[Dict] = []
        discovery_method = "none"
        cqc_found: List[Dict] = []
        if self.cqc and self._target_local_authority:
            cqc_found = self._discover_competitors_cqc(status_callback)

        llm_found: List[Dict] = []
        if not cfg.is_quick:
            status_callback(f"  🔍 LLM discovery for **{cfg.service_area}** in **{geo_label}**…")
            llm_found = self._discover_competitors(status_callback)

        # CQC-found take priority (authoritative + already carry CQC data)
        discovered_competitors = _merge_competitors(cqc_found, llm_found)
        if cqc_found and llm_found:
            discovery_method = "cqc+llm"
        elif cqc_found:
            discovery_method = "cqc"
        elif llm_found:
            discovery_method = "llm"
        status_callback(
            f"    Discovery: {len(cqc_found)} from CQC area list + {len(llm_found)} from LLM "
            f"→ {len(discovered_competitors)} unique"
        )

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
                # Retry once, instructing the model to emit ONLY JSON.
                status_callback("  ↻ First response wasn't valid JSON — retrying with strict JSON instruction…")
                retry_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous response could not be parsed. "
                    "Return ONLY the JSON object — no explanation, no markdown code fences, "
                    "no text before or after. Start your response with { and end with }."
                )
                retry = self.provider.research(retry_prompt, max_tokens=7000)
                if retry.ok:
                    raw_data = _extract_json(retry.content)
                    if retry.sources:
                        result.sources.extend(retry.sources)
                if not raw_data:
                    preview = (result.content or "")[:200].replace("\n", " ")
                    status_callback(
                        f"  ⚠️ Still no valid JSON after retry. First 200 chars: _{preview}…_"
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

        # ---- Phase 2a (BOTH modes): attach authoritative CQC data ----
        # Cheap CQC lookup (API only, no LLM) for any competitor that doesn't
        # already carry CQC data from the area-discovery step. This is what
        # makes Quick Scan genuinely useful: real ratings, beds, specialisms.
        if competitors and self.cqc and self.brave:
            need = [c for c in competitors if not c.get("cqc_data")]
            if need:
                status_callback(f"  🏷 Attaching CQC data to {len(need)} competitor(s)…")
                for comp in need:
                    self._attach_cqc_only(comp, status_callback)

        # ---- Phase 2b (Deep only): LLM enrichment (Companies House, contracts) ----
        if not cfg.is_quick and competitors:
            status_callback(f"  🔎 Deep enrichment: Companies House + contracts on {len(competitors)} competitors…")
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

        # Ensure every competitor carries an enrichment_status (Quick mode
        # skips Phase 2b, so stamp from whatever data we have).
        for comp in competitors:
            if "enrichment_status" not in comp:
                comp["enrichment_status"] = _compute_enrichment_status(comp, {})

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
                "discovery_method": discovery_method,
                "target_local_authority": self._target_local_authority,
                "cqc_enabled": bool(self.cqc),
                "brave_enabled": bool(self.brave),
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

    def _area_query_hint(self) -> str:
        """A short area string to bias CQC searches to the right town."""
        cfg = self.config
        hint = cfg.geographic_area or self._target_local_authority or cfg.commissioner or ""
        hint = re.sub(r"(?i)\b(city|borough|county|council|district|metropolitan|"
                      r"unitary|authority|icb|nhs)\b", " ", hint)
        return re.sub(r"\s+", " ", hint).strip()

    def _lookup_cqc(self, provider_name: str, status_callback: StatusCallback = _noop,
                    allow_fuzzy: bool = False, strict_area: bool = False) -> Optional[Dict[str, Any]]:
        """
        Find a CQC profile for a named provider, using Brave to surface
        candidate URLs, a name-similarity check, and a geographic check to
        avoid attaching a same-named provider from a different town.

        - The target area is included in the search query so Brave returns the
          right-town record (e.g. "Aspen Court" in Tower Hamlets, not Derby).
        - strict_area=True (used for the TARGET) HARD-REJECTS an out-of-area
          match, so a wrong-town record never poisons the local-authority used
          for area discovery.
        - allow_fuzzy=True (TARGET only) tolerates a typo in the user-typed name.
        """
        if not (self.brave and self.cqc):
            return None
        try:
            expected = self._expected_area_tokens()
            area_hint = self._area_query_hint()
            query = f'site:cqc.org.uk "{provider_name}"'
            if area_hint:
                query += f" {area_hint}"
            results = self.brave.search(query, count=8)

            # Score every valid candidate by name confidence + area preference.
            candidates = []          # strict matches
            fuzzy_candidates = []    # typo-tolerant matches (target only)
            for r in results:
                url = r.get("url", "") or ""
                title = r.get("title", "") or ""
                if "cqc.org.uk" not in url or not self.cqc.extract_id_from_url(url):
                    continue
                candidate_name = title.split(" - ")[0].split(" | ")[0].strip()
                name_score = _name_match_confidence(provider_name, candidate_name)
                hint_text = f"{title} {r.get('description', '')} {url}".lower()
                hint_tokens = {t for t in re.split(r"[^a-z]+", hint_text) if len(t) >= 4}
                area_bonus = 0.15 if (expected and not expected.isdisjoint(hint_tokens)) else 0.0
                if name_score >= self.CQC_MATCH_THRESHOLD:
                    candidates.append((name_score + area_bonus, name_score, r, candidate_name))
                elif allow_fuzzy:
                    fz = _fuzzy_name_ratio(provider_name, candidate_name)
                    if fz >= 0.84:
                        fuzzy_candidates.append((fz + area_bonus, fz, r, candidate_name))

            is_fuzzy = False
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, best_score, best, best_candidate = candidates[0]
            elif fuzzy_candidates:
                fuzzy_candidates.sort(key=lambda x: x[0], reverse=True)
                _, best_score, best, best_candidate = fuzzy_candidates[0]
                is_fuzzy = True
                status_callback(
                    f"    🔁 CQC: no exact match for '{provider_name}', "
                    f"fuzzy-matched '{best_candidate}' (similarity {best_score:.2f}) — please verify."
                )
            else:
                status_callback(f"    ⚠️ CQC: no confident name match for **{provider_name}**")
                return None

            raw = self.cqc.fetch_from_url(best.get("url", ""))
            if not raw:
                return None
            summary = self.cqc.summarise_provider_profile(raw)
            summary = self._augment_rating(summary)  # provider-level rating fallback

            # Area verification.
            area_verified = True
            if expected:
                record_area = " ".join([
                    summary.get("local_authority", ""), summary.get("region", ""),
                    summary.get("town", ""), summary.get("postcode", ""),
                ]).lower()
                record_tokens = {t for t in re.split(r"[^a-z]+", record_area) if len(t) >= 4}
                if record_tokens and expected.isdisjoint(record_tokens):
                    area_verified = False

            # For the TARGET (strict_area), a wrong-town match is almost
            # certainly the wrong company — reject it so it can't poison the
            # local authority used for area discovery.
            if strict_area and not area_verified:
                status_callback(
                    f"    🛑 CQC: rejected '{best_candidate}' for **{provider_name}** — "
                    f"record is in {summary.get('local_authority') or summary.get('town') or 'another area'}, "
                    f"not the target area ({area_hint or 'specified'})."
                )
                return None

            summary["_brave_snippet"] = best.get("description", "")
            summary["_match_confidence"] = round(best_score, 2)
            summary["_matched_name"] = best_candidate
            summary["_area_verified"] = area_verified
            summary["_fuzzy_match"] = is_fuzzy

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
    # Derive local authority independently of the target (typo-resilience)
    # ------------------------------------------------------------------

    def _derive_local_authority(self, status_callback: StatusCallback = _noop) -> str:
        """
        Find the CQC local authority for the target area without relying on the
        target company's record. Searches CQC for any local provider, fetches
        one, and reads its localAuthority — verifying it matches the expected
        area tokens (from commissioner/geographic_area).
        """
        cfg = self.config
        area_hint = cfg.geographic_area or cfg.commissioner or ""
        area_hint = re.sub(r"(?i)\b(city|borough|county|council|district|"
                           r"metropolitan|unitary|authority|icb|nhs)\b", " ", area_hint).strip()
        if not area_hint:
            return ""
        expected = self._expected_area_tokens()
        status_callback(f"  🧭 Deriving CQC local authority for **{area_hint}**…")
        try:
            results = self.brave.search(
                f'site:cqc.org.uk {cfg.service_area} {area_hint}', count=6,
            )
            for r in results:
                url = r.get("url", "") or ""
                if not self.cqc.extract_id_from_url(url):
                    continue
                raw = self.cqc.fetch_from_url(url)
                if not raw:
                    continue
                s = self.cqc.summarise_provider_profile(raw)
                la = s.get("local_authority", "")
                if not la:
                    continue
                la_tokens = {t for t in re.split(r"[^a-z]+", la.lower()) if len(t) >= 4}
                if not expected or not expected.isdisjoint(la_tokens):
                    status_callback(f"    Derived local authority: **{la}**")
                    return la
        except Exception as exc:
            status_callback(f"    ⚠️ LA derivation failed: {exc}")
        return ""

    # ------------------------------------------------------------------
    # Phase 0a: Verified target-company profile
    # ------------------------------------------------------------------

    def _research_target_profile(self, status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        cfg = self.config

        # First — try authoritative CQC lookup. allow_fuzzy=True tolerates a
        # typo in the user-entered target name; strict_area=True rejects a
        # wrong-town match (e.g. an "Aspen Court" in Derby when the target is
        # in Tower Hamlets) so it can't poison area discovery.
        cqc_data = self._lookup_cqc(cfg.target_company, status_callback,
                                    allow_fuzzy=True, strict_area=True)

        # Learn the target's local authority + service types from its CQC record
        if cqc_data and cqc_data.get("local_authority"):
            self._target_local_authority = cqc_data["local_authority"]
        if cqc_data and cqc_data.get("service_types"):
            self._target_service_types = list(cqc_data["service_types"])
            status_callback(f"    Target CQC service types: {', '.join(self._target_service_types)}")

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
            service_area=cfg.service_area_label,
            geographic_area=cfg.geographic_area or "Not specified",
        ) + verified_context

        result = self.provider.research(prompt, max_tokens=4000)
        if not result.ok:
            status_callback(f"  ⚠️ Target profile lookup failed: {result.error}")
            return {"cqc": {"rating": cqc_data.get("overall_rating", "Unknown")} if cqc_data else {}}

        data = _extract_json(result.content) or {}

        # Overwrite CQC section with authoritative API data if we have it
        if cqc_data:
            api_rating = cqc_data.get("overall_rating") or "Unknown"
            # If the CQC API has no current rating (e.g. during methodology transition),
            # fall back to whatever the LLM extracted from the CQC profile web page.
            if api_rating == "Unknown":
                api_rating = (data.get("cqc") or {}).get("rating") or "Unknown"
            data["cqc"] = {
                "rating": api_rating,
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
                    "rating": api_rating,
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
            # Surface fuzzy-match status so the dashboard can ask for verification
            data["cqc"]["matched_name"] = cqc_data.get("_matched_name", "")
            data["cqc"]["fuzzy_match"] = cqc_data.get("_fuzzy_match", False)

        # Flag whether the target was confidently identified at all
        data["target_identified"] = bool(cqc_data)
        if not cqc_data:
            status_callback(
                f"  ⚠️ Target '{cfg.target_company}' could not be confidently matched on CQC. "
                f"Check the spelling — results for the target will be limited."
            )

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
    # Phase 0b (primary): Authoritative CQC area-based discovery
    # ------------------------------------------------------------------

    _RATING_RANK = {"Outstanding": 4, "Good": 3, "Requires improvement": 2, "Inadequate": 1}
    CQC_DETAIL_CAP = 60  # max location detail fetches (higher because we filter by service type)

    # CQC service types that count as in-scope social-care competitors.
    _SOCIAL_CARE_HINTS = (
        "homecare", "domiciliary", "supported living", "care home", "nursing",
        "extra care", "shared lives", "personal care", "hospice", "rehabilitation",
    )
    # Clinical/medical CQC service types to exclude (GPs, dentists, hospitals…).
    _CLINICAL_EXCLUDE_HINTS = (
        "doctor", "gp", "dentist", "dental", "hospital", "clinic", "pharmacy",
        "ambulance", "diagnostic", "surgery", "urgent care", "slimming",
        "fertility", "dialysis", "prison",
    )

    def _service_is_care_home(self) -> Optional[bool]:
        """
        Decide CQC's careHome Y/N filter. Prefer the TARGET's actual CQC
        service types (authoritative); fall back to the user's free-text
        service area only when the target's types are unknown. None = ambiguous.
        """
        # 1. Authoritative: the target's own CQC classification
        if getattr(self, "_target_service_types", None):
            joined = " ".join(self._target_service_types).lower()
            if "care home" in joined or "nursing" in joined:
                return True
            if any(t in joined for t in ("homecare", "domiciliary", "supported living",
                                         "extra care", "shared lives")):
                return False
            # Some social-care types (e.g. hospice) — don't force a filter
            return None

        # 2. Fallback: the user's typed service area
        s = (self.config.service_area or "").lower()
        care_home_terms = ("residential", "care home", "nursing home", "nursing care",
                           "rest home", "care homes")
        community_terms = ("domiciliary", "home care", "homecare", "supported living",
                          "live-in", "live in", "extra care", "reablement", "at home",
                          "outreach", "community")
        if any(t in s for t in care_home_terms):
            return True
        if any(t in s for t in community_terms):
            return False
        return None

    def _wanted_service_types(self) -> List[str]:
        """
        The CQC service types to filter competitors to. Prefer the TARGET's
        actual CQC service types (e.g. 'Homecare agencies') so we use CQC's
        own classification rather than the user's free-text label. Falls back
        to the social-care allowlist when the target's types are unknown.
        """
        if self._target_service_types:
            return [t.lower() for t in self._target_service_types]
        return list(self._SOCIAL_CARE_HINTS)

    def _service_type_in_scope(self, service_types: List[str], wanted: List[str]) -> bool:
        """Keep a location if its service types match the wanted set and aren't clinical."""
        st_lower = [str(s).lower() for s in (service_types or [])]
        if not st_lower:
            return False
        # Exclude clearly clinical/medical providers
        if any(any(ex in s for ex in self._CLINICAL_EXCLUDE_HINTS) for s in st_lower):
            return False
        # Keep if any service type matches the wanted set (target's types or allowlist)
        for s in st_lower:
            if any(w in s or s in w for w in wanted):
                return True
        return False

    def _augment_rating(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        If a location has no current overall rating, fall back to the provider's
        rating (ratings often sit at provider level for multi-site or
        domiciliary providers). Mutates and returns the summary.
        """
        if summary.get("overall_rating") not in ("Unknown", "", None):
            return summary
        pid = summary.get("provider_id")
        if not pid:
            return summary
        try:
            prov = self.cqc.get_provider(pid)
        except Exception:
            prov = None
        if prov:
            ps = self.cqc.summarise_provider_profile(prov)
            if ps.get("overall_rating") not in ("Unknown", "", None):
                summary["overall_rating"] = ps["overall_rating"]
                if ps.get("sub_ratings") and not summary.get("sub_ratings"):
                    summary["sub_ratings"] = ps["sub_ratings"]
                summary["_rating_from_provider"] = True
        return summary

    def _discover_competitors_cqc(self, status_callback: StatusCallback = _noop) -> List[Dict[str, Any]]:
        """
        Query the CQC API for all registered providers in the target's local
        authority, rank by significance (beds + rating), and return the top
        competitors with CQC data already attached.
        """
        la = self._target_local_authority
        care_home = self._service_is_care_home()
        # If the careHome flag is ambiguous but we know the target's service
        # types, still run (list without the flag and filter by service type).
        if care_home is None and not self._target_service_types:
            status_callback(
                f"    ℹ️ Service area '{self.config.service_area}' is ambiguous for CQC "
                f"care-home filter and target type unknown — using LLM discovery only."
            )
            return []

        wanted = self._wanted_service_types()
        wanted_label = ", ".join(self._target_service_types) if self._target_service_types else "social care (general)"
        status_callback(
            f"  🏛 Querying CQC for registered providers in **{la}** "
            f"matching service type(s): _{wanted_label}_…"
        )
        try:
            locations = self.cqc.list_locations(
                local_authority=la, care_home=care_home, per_page=100, max_pages=3,
            )
        except Exception as exc:
            status_callback(f"    ⚠️ CQC area query failed: {exc}")
            return []

        if not locations:
            status_callback(f"    CQC returned no locations for {la}")
            return []
        status_callback(f"    CQC returned {len(locations)} locations in {la}; filtering by service type…")

        out: List[Dict[str, Any]] = []
        fetched = 0
        skipped_clinical = 0
        target_limit = max(self.config.max_competitors * 3, 15)  # gather enough to rank
        for loc in locations:
            if fetched >= self.CQC_DETAIL_CAP or len(out) >= target_limit:
                break
            name = loc.get("locationName", "").strip()
            if not name:
                continue
            if _name_match_confidence(self.config.target_company, name) >= 0.6:
                continue  # skip the target itself
            try:
                raw = self.cqc.get_location(loc["locationId"])
            except Exception:
                raw = None
            fetched += 1
            if not raw:
                continue
            raw["_cqc_url"] = f"https://www.cqc.org.uk/location/{loc['locationId']}"
            raw["_lookup_type"] = "location"
            s = self.cqc.summarise_provider_profile(raw)

            # Filter by service type — exclude GPs/dentists/clinics, keep only
            # providers whose CQC service types match the target's.
            if not self._service_type_in_scope(s.get("service_types", []), wanted):
                skipped_clinical += 1
                continue

            s = self._augment_rating(s)  # provider-level rating fallback
            out.append({
                "name": name,
                "selection_rationale": (
                    f"CQC-registered {', '.join(s.get('service_types', [])) or 'provider'} "
                    f"in {la} (authoritative CQC area list). Rating: {s.get('overall_rating', 'Unknown')}."
                ),
                "website": s.get("website", ""),
                "cqc_rating": s.get("overall_rating", "Unknown"),
                "cqc_profile_url": s.get("cqc_url", ""),
                "cqc_verified": True,
                "cqc_data": {
                    "sub_ratings": s.get("sub_ratings", {}),
                    "number_of_beds": s.get("number_of_beds"),
                    "registration_date": s.get("registration_date", ""),
                    "last_inspection_date": s.get("last_inspection_date", ""),
                    "service_types": s.get("service_types", []),
                    "specialisms": s.get("specialisms", []),
                    "local_authority": s.get("local_authority", ""),
                    "town": s.get("town", ""),
                },
                "source_urls": [s.get("cqc_url", "")],
            })

        status_callback(
            f"    Matched {len(out)} in-scope providers "
            f"(fetched {fetched}, excluded {skipped_clinical} out-of-scope/clinical)"
        )

        # Rank by significance: rating quality first, then beds (beds=0 for domiciliary)
        out.sort(
            key=lambda c: (
                self._RATING_RANK.get(c.get("cqc_rating"), 0),
                c["cqc_data"].get("number_of_beds") or 0,
            ),
            reverse=True,
        )
        return out

    # ------------------------------------------------------------------
    # Phase 0b (supplement): LLM-based competitor discovery (Deep Scan only)
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
            service_area=cfg.service_area_label,
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

    def _attach_cqc_only(self, comp: Dict[str, Any], status_callback: StatusCallback = _noop) -> Dict[str, Any]:
        """
        Cheap CQC-only enrichment (no LLM call): look up the competitor's CQC
        record and attach rating + structured data. Used in both Quick and Deep.
        """
        if comp.get("cqc_data"):
            return comp
        cqc_data = self._lookup_cqc(comp.get("name", ""), status_callback)
        if cqc_data:
            comp["cqc_rating"] = cqc_data.get("overall_rating", "Unknown")
            comp["cqc_profile_url"] = cqc_data.get("cqc_url", "")
            comp["cqc_verified"] = True
            comp["cqc_data"] = {
                "sub_ratings": cqc_data.get("sub_ratings", {}),
                "number_of_beds": cqc_data.get("number_of_beds"),
                "registration_date": cqc_data.get("registration_date", ""),
                "last_inspection_date": cqc_data.get("last_inspection_date", ""),
                "service_types": cqc_data.get("service_types", []),
                "specialisms": cqc_data.get("specialisms", []),
                "local_authority": cqc_data.get("local_authority", ""),
                "town": cqc_data.get("town", ""),
            }
            site = comp.get("website") or ""
            if cqc_data.get("website") and site in ("", "Unknown", None):
                comp["website"] = cqc_data["website"]
        return comp

    def _enrich_competitor(self, competitor: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        name = competitor.get("name", "")

        # If CQC data is already attached (from CQC area discovery or Phase 2a),
        # skip the redundant Brave→CQC lookup — saves an API round-trip.
        if competitor.get("cqc_data"):
            cqc_data = None
        else:
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
            service_area=cfg.service_area_label,
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
            service_area=cfg.service_area_label,
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


def _clean_json_text(s: str) -> str:
    """Remove trailing commas before } or ] which are invalid in strict JSON."""
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _find_balanced_json_blocks(text: str) -> List[str]:
    """
    Return all top-level balanced {...} blocks, respecting strings/escapes.
    Robust to prose containing stray braces before/after the real JSON object.
    """
    blocks: List[str] = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    blocks.append(text[start : i + 1])
                    start = -1
    return blocks


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from LLM output. Handles markdown fences,
    leading/trailing prose (common with Claude + web search), stray braces,
    and trailing commas.
    """
    if not text:
        return {}

    candidates: List[str] = []

    # 1. Markdown-fenced block(s)
    for m in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        candidates.append(m)
    # 2. The raw text
    candidates.append(text)
    # 3. All balanced {...} blocks, longest first (real JSON is usually biggest)
    balanced = _find_balanced_json_blocks(text)
    balanced.sort(key=len, reverse=True)
    candidates.extend(balanced)
    # 4. Outermost braces (last resort)
    bs, be = text.find("{"), text.rfind("}")
    if bs != -1 and be > bs:
        candidates.append(text[bs : be + 1])

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        for attempt in (cand, _clean_json_text(cand)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

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


def _fuzzy_name_ratio(query: str, candidate: str) -> float:
    """
    Edit-distance similarity between normalised names, for typo tolerance.
    Uses difflib SequenceMatcher. "ashley car" vs "ashley care" ≈ 0.95.
    Only used as a fallback for the TARGET (user-typed) name, never for
    competitor auto-matching, to avoid false positives.
    """
    from difflib import SequenceMatcher
    nq = _normalise_company_name(query)
    nc = _normalise_company_name(candidate)
    if not nq or not nc:
        return 0.0
    return SequenceMatcher(None, nq, nc).ratio()


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
