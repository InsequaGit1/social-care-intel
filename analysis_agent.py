"""
AnalysisAgent — website analysis and benchmarking phase.

Responsibilities:
  - Analyse the target company's website and each competitor's website
  - Score all companies against the benchmarking criteria
  - Produce bid positioning recommendations
  - Return additions to the main results dict
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from research_agent import ResearchConfig, _extract_json, _fill_template
from search_providers.base import SearchProvider


StatusCallback = Callable[[str], None]
_noop: StatusCallback = lambda msg: None


def _safe_join(items, sep: str = ", ", limit=None) -> str:
    """
    Join a list of items to a string, tolerating dicts and other non-string
    members. Dicts get reduced to their 'title' or 'name' field if possible.
    """
    if not items:
        return ""
    if limit is not None:
        items = items[:limit]
    parts = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            label = item.get("title") or item.get("name") or item.get("point") or str(item)
            parts.append(str(label))
        elif item is None:
            continue
        else:
            parts.append(str(item))
    return sep.join(parts)


class AnalysisAgent:

    WEB_PROMPT_PATH = Path(__file__).parent / "prompts" / "website_analysis_prompt.txt"
    BENCH_PROMPT_PATH = Path(__file__).parent / "prompts" / "benchmarking_prompt.txt"
    BENCH_SINGLE_PATH = Path(__file__).parent / "prompts" / "benchmarking_single_prompt.txt"
    SYNTH_PATH = Path(__file__).parent / "prompts" / "synthesis_prompt.txt"

    # 7 criteria. cqc_rating is a WORD value pulled from API/research;
    # the other 6 are 1-5 scored by the LLM.
    CRITERIA = [
        "cqc_rating",                 # word: Outstanding|Good|Requires Improvement|Inadequate|Unknown
        "service_location_fit",       # 1-5 — service offer match + geographic proximity to commissioner
        "quality_compliance",         # 1-5
        "local_track_record",         # 1-5 — actual delivered contracts in area / with commissioner
        "delivery_strength",          # 1-5 — workforce_capacity + mobilisation_capability
        "strategic_differentiators",  # 1-5 — digital + social value + partnership + website credibility
        "overall_bid_threat",         # 1-5
    ]
    SCORED_CRITERIA = [c for c in CRITERIA if c != "cqc_rating"]

    def __init__(self, config: ResearchConfig, provider: SearchProvider):
        self.config = config
        self.provider = provider
        self._web_template = self.WEB_PROMPT_PATH.read_text(encoding="utf-8")
        self._bench_template = self.BENCH_PROMPT_PATH.read_text(encoding="utf-8")
        self._bench_single_template = self.BENCH_SINGLE_PATH.read_text(encoding="utf-8")
        self._synth_template = self.SYNTH_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        research_results: Dict[str, Any],
        status_callback: StatusCallback = _noop,
    ) -> Dict[str, Any]:
        cfg = self.config

        # Build list of companies to analyse
        competitors = research_results.get("competitors", [])
        companies_to_analyse = self._build_company_list(competitors)

        # ---- Website analysis ----------------------------------------
        website_analyses: Dict[str, Any] = {}

        status_callback(f"  Analysing website: **{cfg.target_company}** ({cfg.target_website})…")
        target_analysis = self._analyse_website(
            company_name=cfg.target_company,
            website_url=cfg.target_website,
        )
        website_analyses[cfg.target_company] = target_analysis

        for company in companies_to_analyse:
            name = company.get("name", "Unknown")
            url = company.get("website", "")
            if not url or url == "Unknown":
                status_callback(f"  Skipping website analysis for **{name}** — no URL found")
                website_analyses[name] = _empty_analysis(name, url)
                continue

            status_callback(f"  Analysing website: **{name}** ({url})…")
            analysis = self._analyse_website(company_name=name, website_url=url)
            website_analyses[name] = analysis

        # ---- Benchmarking (per-company) ------------------------------
        # Hydrate the target company row with verified target_profile data if present
        target_profile = research_results.get("target_profile", {}) or {}
        target_row = {
            "name": cfg.target_company,
            "is_target": True,
            "website": target_profile.get("official_website", cfg.target_website or "Unknown"),
            "cqc_rating": (target_profile.get("cqc", {}) or {}).get("rating", "Unknown"),
            "cqc_profile_url": (target_profile.get("cqc", {}) or {}).get("profile_url", ""),
            "companies_house_number": (target_profile.get("companies_house", {}) or {}).get("number", "Unknown"),
            "known_contracts_with_commissioner": target_profile.get("contracts_with_commissioner", []),
            "selection_rationale": "Target bidding company for this opportunity.",
        }
        all_companies = [target_row] + [
            {**c, "is_target": False} for c in companies_to_analyse
        ]
        status_callback(f"  📊 Scoring {len(all_companies)} companies across 14 criteria (one call per company)…")
        bench_data = self._benchmark(
            all_companies=all_companies,
            research_results=research_results,
            website_analyses=website_analyses,
            status_callback=status_callback,
        )

        # ---- New sources from web analysis ---------------------------
        new_sources = _collect_sources_from_analyses(website_analyses)
        existing_sources = research_results.get("sources", [])
        from research_agent import _merge_sources
        all_sources = _merge_sources(existing_sources, new_sources)[: cfg.max_sources]

        return {
            "website_analyses": website_analyses,
            "benchmarking": bench_data.get("scores", {}),
            "benchmarking_criteria": bench_data.get("criteria", []),
            "bid_positioning": bench_data.get("bid_positioning_summary", []),
            "executive_summary": bench_data.get("executive_summary", ""),
            "evidence_gaps": _merge_evidence_gaps(
                research_results.get("evidence_gaps", []),
                bench_data.get("evidence_gaps", []),
            ),
            "sources": all_sources,
            "confidence_rating": research_results.get("confidence_rating", "Low"),
            "confidence_rationale": research_results.get("confidence_rationale", ""),
        }

    # ------------------------------------------------------------------
    # Website analysis
    # ------------------------------------------------------------------

    def _analyse_website(self, company_name: str, website_url: str) -> Dict[str, Any]:
        cfg = self.config

        prompt = _fill_template(self._web_template,
            company_name=company_name,
            website_url=website_url,
            service_area=cfg.service_area,
            commissioner=cfg.commissioner,
            geographic_area=cfg.geographic_area or "Not specified",
            max_pages=cfg.max_pages_per_website,
        )

        result = self.provider.research(prompt, max_tokens=5000)

        if not result.ok:
            return _empty_analysis(company_name, website_url, error=result.error)

        data = _extract_json(result.content)
        if not data:
            return _empty_analysis(company_name, website_url, error="Could not parse response")

        # Ensure company name and URL are set correctly
        data.setdefault("company_name", company_name)
        data.setdefault("website_url", website_url)

        # Append any sources the provider extracted from HTTP metadata
        existing_pages = set(data.get("pages_accessed", []))
        for s in result.sources:
            if s.url not in existing_pages:
                data.setdefault("pages_accessed", []).append(s.url)

        return data

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def _benchmark(
        self,
        all_companies: List[Dict],
        research_results: Dict[str, Any],
        website_analyses: Dict[str, Any],
        status_callback: StatusCallback = _noop,
    ) -> Dict[str, Any]:
        """
        Score companies one at a time, then run a synthesis call for
        the executive summary, bid positioning, and evidence gaps.
        """
        cfg = self.config
        research_summary = _build_research_summary(
            research_results=research_results,
            website_analyses=website_analyses,
        )

        # ---- Per-company scoring -------------------------------------
        scores: Dict[str, Any] = {}
        for i, comp in enumerate(all_companies, 1):
            name = comp.get("name", "Unknown")
            is_target = comp.get("is_target", False)
            label = f"{name}{' (TARGET)' if is_target else ''}"
            status_callback(f"    [{i}/{len(all_companies)}] Scoring **{label}**…")

            company_scores = self._score_single_company(
                company=comp,
                website_analysis=website_analyses.get(name, {}),
                research_summary=research_summary,
            )
            if not company_scores:
                company_scores = {
                    c: {"score": 0, "justification": "Scoring call failed", "source": "", "analyst_inference": False}
                    for c in self.SCORED_CRITERIA
                }

            # Stamp authoritative CQC rating as a word value, with verification flag
            company_scores["cqc_rating"] = {
                "value": comp.get("cqc_rating", "Unknown") or "Unknown",
                "url": comp.get("cqc_profile_url", "") or "",
                "verified": bool(comp.get("cqc_verified", False)),
            }

            scores[name] = company_scores

        # ---- Synthesis pass ------------------------------------------
        status_callback("    Synthesising executive summary and bid positioning…")
        synthesis = self._synthesise(
            scores=scores,
            research_summary=research_summary,
        )

        return {
            "scores": scores,
            "criteria": self.CRITERIA,
            "executive_summary": synthesis.get("executive_summary", ""),
            "bid_positioning_summary": synthesis.get("bid_positioning", []),
            "evidence_gaps": synthesis.get("evidence_gaps", []),
        }

    def _score_single_company(
        self,
        company: Dict,
        website_analysis: Dict,
        research_summary: str,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.config
        name = company.get("name", "Unknown")

        # Build a focused company profile from research + website analysis
        profile_parts = [
            f"Name: {name}",
            f"Website: {company.get('website', 'Unknown')}",
            f"CQC rating: {company.get('cqc_rating', 'Unknown')}",
            f"CQC profile: {company.get('cqc_profile_url', '')}",
            f"Companies House: {company.get('companies_house_number', 'Unknown')}",
            f"Size: {company.get('size_description', 'Unknown')}",
            f"Services: {_safe_join(company.get('services', []))}",
            f"Geographic coverage: {_safe_join(company.get('geographic_coverage', []))}",
            f"Known contracts with commissioner: {company.get('known_contracts_with_commissioner', [])}",
            f"Selection rationale: {company.get('selection_rationale', 'N/A')}",
        ]
        if website_analysis and website_analysis.get("accessible", True):
            ev = website_analysis.get("evidence_quality", {})
            profile_parts.append(f"Evidence quality from website: {ev.get('overall_evidence_quality', 'Unknown')}")
            profile_parts.append(f"Strengths: {_safe_join(website_analysis.get('strengths', []), sep='; ', limit=5)}")
            profile_parts.append(f"Gaps: {_safe_join(website_analysis.get('weaknesses_and_gaps', []), sep='; ', limit=5)}")

        company_profile = "\n".join(profile_parts)

        prompt = _fill_template(self._bench_single_template,
            target_company=cfg.target_company,
            commissioner=cfg.commissioner,
            service_area=cfg.service_area,
            geographic_area=cfg.geographic_area or "Not specified",
            company_name=name,
            company_profile=company_profile,
            research_summary=research_summary[:4000],  # Cap context size
        )

        result = self.provider.research(prompt, max_tokens=3500)
        if not result.ok:
            return None

        data = _extract_json(result.content)
        if not data:
            return None
        return data.get("scores", {})

    def _synthesise(self, scores: Dict[str, Any], research_summary: str) -> Dict[str, Any]:
        cfg = self.config

        # Build a compact, name-explicit summary of scores for the synthesizer.
        # Surfaces the verified CQC rating (word) plus each 1-5 criterion.
        scored_lines = []
        for company, cs in scores.items():
            cqc = cs.get("cqc_rating", {})
            cqc_label = cqc.get("value", "Unknown") if isinstance(cqc, dict) else "Unknown"
            cqc_flag = "✓verified" if (isinstance(cqc, dict) and cqc.get("verified")) else "unverified"

            scored_parts = [f"CQC={cqc_label} ({cqc_flag})"]
            for k in self.SCORED_CRITERIA:
                v = cs.get(k, {})
                s = v.get("score", 0) if isinstance(v, dict) else 0
                scored_parts.append(f"{k}={s}")
            overall = cs.get("overall_bid_threat", {})
            overall_score = overall.get("score", 0) if isinstance(overall, dict) else 0
            scored_lines.append(
                f"  - {company} [overall threat {overall_score}/5]: " + " · ".join(scored_parts)
            )
        scored_summary = "\n".join(scored_lines)

        prompt = _fill_template(self._synth_template,
            target_company=cfg.target_company,
            commissioner=cfg.commissioner,
            service_area=cfg.service_area,
            geographic_area=cfg.geographic_area or "Not specified",
            scored_summary=scored_summary,
            research_summary=research_summary[:3000],
        )

        result = self.provider.research(prompt, max_tokens=3500)
        if not result.ok:
            return {}

        data = _extract_json(result.content)
        return data if data else {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_company_list(self, competitors: List[Dict]) -> List[Dict]:
        seen = set()
        result = []
        for c in competitors:
            name = c.get("name", "").strip()
            if name and name not in seen:
                seen.add(name)
                result.append(c)
        return result[: self.config.max_competitors]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _empty_analysis(name: str, url: str, error: str = "") -> Dict[str, Any]:
    return {
        "company_name": name,
        "website_url": url,
        "accessible": False,
        "pages_reviewed": [],
        "positioning": {
            "headline_proposition": "No reliable public source found",
            "target_commissioner_message": "",
            "differentiators": [],
            "category": "Unknown",
        },
        "service_offer": {"services_listed": [], "service_area_match": "", "geographic_coverage_claimed": []},
        "evidence_quality": {
            "cqc_rating_cited": "Not cited",
            "outcomes_data_present": False,
            "outcomes_description": "",
            "case_studies_count": 0,
            "awards_mentioned": [],
            "accreditations": [],
            "overall_evidence_quality": "Very weak",
        },
        "commissioner_relevance": {
            "local_authority_mentioned": False,
            "commissioner_named": "",
            "local_contract_references": [],
            "local_relevance_score": "Low",
        },
        "claims_by_type": {"explicit": [], "evidenced": [], "unsupported": [], "analyst_inference": []},
        "strengths": [],
        "weaknesses_and_gaps": ["Website could not be accessed or analysed"],
        "bid_positioning_implications": [],
        "pages_accessed": [],
        "access_notes": error or "Website not accessible",
    }


def _collect_sources_from_analyses(website_analyses: Dict[str, Any]) -> List[Dict]:
    sources = []
    for company_name, analysis in website_analyses.items():
        for url in analysis.get("pages_accessed", []):
            if url:
                sources.append({
                    "url": url,
                    "title": f"{company_name} — website page",
                    "used_for": "Website analysis",
                })
    return sources


def _merge_evidence_gaps(existing: List, new: List) -> List:
    seen = {g.get("area", "") for g in existing}
    merged = list(existing)
    for g in new:
        area = g.get("area", "")
        if area not in seen:
            merged.append(g)
            seen.add(area)
    return merged


def _build_companies_list_text(target_name: str, competitors: List[Dict]) -> str:
    lines = [f"- {target_name} (TARGET COMPANY)"]
    for c in competitors:
        lines.append(f"- {c.get('name', 'Unknown')}")
    return "\n".join(lines)


def _build_research_summary(
    research_results: Dict[str, Any],
    website_analyses: Dict[str, Any],
) -> str:
    """
    Build a condensed text summary of research findings to pass into the
    benchmarking prompt. Keeps token usage manageable.
    """
    parts = []

    procurement = research_results.get("procurement", [])
    if procurement:
        parts.append("PROCUREMENT HISTORY:")
        for p in procurement[:6]:
            parts.append(
                f"  - {p.get('title', 'Unknown')} | Awarded to: {p.get('awarded_to', 'Unknown')} | "
                f"Value: {p.get('value', 'Unknown')} | Source: {p.get('source_url', '')}"
            )

    competitors = research_results.get("competitors", [])
    if competitors:
        parts.append("\nCOMPETITOR INTELLIGENCE:")
        for c in competitors:
            parts.append(
                f"  - {c.get('name')} | CQC: {c.get('cqc_rating', 'Unknown')} | "
                f"Size: {c.get('size_description', 'Unknown')} | "
                f"Local contracts: {_safe_join(c.get('known_contracts_with_commissioner', []) or ['None found'])}"
            )

    priorities = research_results.get("commissioner_priorities", [])
    if priorities:
        parts.append("\nCOMMISSIONER PRIORITIES:")
        for p in priorities[:5]:
            parts.append(f"  - {p.get('priority')}: {p.get('detail', '')}")

    if website_analyses:
        parts.append("\nWEBSITE ANALYSIS SUMMARY:")
        for company, analysis in website_analyses.items():
            eq = analysis.get("evidence_quality", {})
            parts.append(
                f"  - {company}: CQC cited={eq.get('cqc_rating_cited', 'N/A')} | "
                f"Evidence quality={eq.get('overall_evidence_quality', 'Unknown')} | "
                f"Strengths={_safe_join(analysis.get('strengths', []), sep='; ', limit=2)} | "
                f"Gaps={_safe_join(analysis.get('weaknesses_and_gaps', []), sep='; ', limit=2)}"
            )

    return "\n".join(parts) if parts else "No research data available."
