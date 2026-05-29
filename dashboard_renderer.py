"""
DashboardRenderer — renders Streamlit tabs and produces export files.

Designed so the dashboard has zero knowledge of which search provider
or LLM was used. It reads only from the standard results dict.
"""

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Scoring colour helpers
# ---------------------------------------------------------------------------

SCORE_COLOURS = {
    1: "#d32f2f",   # red
    2: "#f57c00",   # orange
    3: "#fbc02d",   # amber
    4: "#388e3c",   # green
    5: "#1565c0",   # blue
}
CONFIDENCE_COLOURS = {
    "High": "#388e3c",
    "Medium": "#f57c00",
    "Low": "#d32f2f",
}
EVIDENCE_COLOURS = {
    "Strong": "#388e3c",
    "Moderate": "#f57c00",
    "Weak": "#f57c00",
    "Very weak": "#d32f2f",
}


def _score_badge(score: int) -> str:
    colour = SCORE_COLOURS.get(score, "#999")
    return f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">{score}/5</span>'


def _conf_badge(level: str) -> str:
    colour = CONFIDENCE_COLOURS.get(level, "#999")
    return f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">{level}</span>'


def _ev_badge(level: str) -> str:
    colour = EVIDENCE_COLOURS.get(level, "#999")
    return f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;">{level}</span>'


CRITERION_LABELS = {
    # Current 7-criteria schema
    "cqc_rating": "CQC Rating",
    "service_location_fit": "Service & Location Fit",
    "quality_compliance": "Quality & Compliance",
    "local_track_record": "Local Track Record",
    "delivery_strength": "Delivery Strength",
    "strategic_differentiators": "Strategic Differentiators",
    "overall_bid_threat": "Overall Bid Threat",
    # Legacy 14-criteria schema (for older saved runs)
    "service_match": "Service Match",
    "local_presence": "Local Presence",
    "commissioner_relationship": "Commissioner Relationship",
    "contract_history": "Contract History",
    "cqc_position": "CQC Position",
    "workforce_capacity": "Workforce Capacity",
    "mobilisation_capability": "Mobilisation Capability",
    "digital_innovation": "Digital & Innovation",
    "specialist_capability": "Specialist Capability",
    "social_value": "Social Value",
    "partnership_working": "Partnership Working",
    "website_credibility": "Website Credibility",
    "overall_competitor_strength": "Overall Strength",
}

# Colour mapping for CQC word ratings
CQC_RATING_COLOURS = {
    "Outstanding": "#1565c0",            # blue
    "Good": "#388e3c",                   # green
    "Requires improvement": "#f57c00",   # orange
    "Inadequate": "#d32f2f",             # red
    "No published rating": "#9e9e9e",    # grey
    "Unknown": "#9e9e9e",                # grey
}


def _cqc_badge_html(value: str, verified: bool = False) -> str:
    # Normalise to title-case for lookup
    norm = (value or "Unknown").strip()
    colour = CQC_RATING_COLOURS.get(norm, "#9e9e9e")
    # Try title-case variants if not directly found
    if colour == "#9e9e9e" and norm.lower() != "unknown":
        for k, v in CQC_RATING_COLOURS.items():
            if k.lower() == norm.lower():
                colour = v
                break
    suffix = " ✓" if verified else ""
    return (
        f'<span style="background:{colour};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em;">'
        f'{norm}{suffix}</span>'
    )


# ---------------------------------------------------------------------------

class DashboardRenderer:

    def __init__(self, results: Dict[str, Any]):
        self.r = results
        self.meta = results.get("metadata", {})
        self.target = self.meta.get("target_company", "Target Company")

    # ------------------------------------------------------------------
    # Main render — creates all tabs
    # ------------------------------------------------------------------

    def render(self):
        tabs = st.tabs([
            "📋 Executive Summary",
            "📄 Contract History",
            "🏢 Competitor Landscape",
            "🌐 Website Analysis",
            "📊 Benchmarking Matrix",
            "🎯 Commissioner Priorities",
            "💡 Bid Positioning",
            "⚠️ Evidence Gaps",
            "🔗 Source Audit",
        ])

        with tabs[0]:
            self._tab_executive_summary()
        with tabs[1]:
            self._tab_contract_history()
        with tabs[2]:
            self._tab_competitor_landscape()
        with tabs[3]:
            self._tab_website_analysis()
        with tabs[4]:
            self._tab_benchmarking()
        with tabs[5]:
            self._tab_commissioner_priorities()
        with tabs[6]:
            self._tab_bid_positioning()
        with tabs[7]:
            self._tab_evidence_gaps()
        with tabs[8]:
            self._tab_source_audit()

        st.divider()
        self._render_exports()

    # ------------------------------------------------------------------
    # Tab: Executive Summary
    # ------------------------------------------------------------------

    def _tab_executive_summary(self):
        st.subheader("Executive Summary")

        # Meta cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Target Company", self.target)
        with col2:
            st.metric("Commissioner", self.meta.get("commissioner", "—"))
        with col3:
            st.metric("Service Area", self.meta.get("service_area", "—"))
        with col4:
            confidence = self.r.get("confidence_rating", "Low")
            st.metric("Confidence Rating", confidence)

        # Target company verified profile (from dedicated lookup phase)
        target_profile = self.r.get("target_profile") or {}
        if target_profile:
            st.divider()
            st.markdown("### Target Company — Verified Profile")
            tcol1, tcol2, tcol3, tcol4 = st.columns(4)
            cqc = target_profile.get("cqc", {}) or {}
            ch = target_profile.get("companies_house", {}) or {}
            with tcol1:
                rating = cqc.get("rating", "Unknown")
                st.metric("CQC Rating", rating)
                if cqc.get("profile_url"):
                    st.caption(f"[CQC profile]({cqc['profile_url']})")
            with tcol2:
                st.metric("Last Inspection", cqc.get("last_inspection_date", "Unknown"))
            with tcol3:
                st.metric("Companies House #", ch.get("number", "Unknown"))
            with tcol4:
                st.metric("Contracts with Commissioner", len(target_profile.get("contracts_with_commissioner", [])))
            status = target_profile.get("lookup_status", {})
            if status:
                flags = []
                flags.append("✅ Website" if status.get("website_found") else "❌ Website")
                flags.append("✅ CQC" if status.get("cqc_found") else "❌ CQC")
                flags.append("✅ Companies House" if status.get("companies_house_found") else "❌ Companies House")
                flags.append("✅ Contracts" if status.get("contracts_found") else "❌ Contracts")
                st.caption("Verified data sources: " + " · ".join(flags))

        # Flag rejected procurement entries (hallucinated URLs)
        rejected = self.r.get("procurement_rejected", [])
        if rejected:
            st.warning(
                f"⚠️ {len(rejected)} procurement notice(s) were rejected by validation "
                f"because their URLs appeared to be fabricated. These have been removed from the report. "
                f"Re-run the research if this seems wrong."
            )

        st.divider()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            summary = self.r.get("executive_summary", "")
            if summary:
                st.markdown("### Research Summary")
                st.markdown(summary)
            else:
                st.info("No executive summary generated. Review individual tabs for findings.")

        with col_b:
            st.markdown("### Run Details")
            st.markdown(f"**Run ID:** `{self.meta.get('run_id', '')[:8]}`")
            st.markdown(f"**Timestamp:** {self.meta.get('timestamp', '')[:19]}")
            st.markdown(f"**Depth:** {self.meta.get('research_depth', '').title()}")
            st.markdown(f"**Provider:** {self.meta.get('model_provider', '')}")
            st.markdown(f"**Time Period:** {self.meta.get('time_period', '')}")
            geo = self.meta.get('geographic_area', '') or self.meta.get('target_local_authority', '') or '—'
            st.markdown(f"**Geography:** {geo}")
            # Data provenance — how were competitors found, which APIs were live
            dm = self.meta.get("discovery_method", "")
            dm_label = {
                "cqc+llm": "CQC area list + LLM",
                "llm": "LLM only",
                "none": "—",
            }.get(dm, dm or "—")
            st.markdown(f"**Discovery:** {dm_label}")
            la = self.meta.get("target_local_authority", "")
            if la:
                st.markdown(f"**CQC Local Authority:** {la}")
            badges = []
            if self.meta.get("cqc_enabled"):
                badges.append("🟢 CQC API")
            if self.meta.get("brave_enabled"):
                badges.append("🟢 Brave")
            if badges:
                st.caption("Live data sources: " + " · ".join(badges))

        st.divider()

        col_c, col_d, col_e = st.columns(3)
        procurement = self.r.get("procurement", [])
        competitors = self.r.get("competitors", [])
        sources = self.r.get("sources", [])
        with col_c:
            st.metric("Procurement Notices Found", len(procurement))
        with col_d:
            st.metric("Competitors Identified", len(competitors))
        with col_e:
            st.metric("Sources Reviewed", len(sources))

        rationale = self.r.get("confidence_rationale", "")
        if rationale:
            st.info(f"**Confidence note:** {rationale}")

    # ------------------------------------------------------------------
    # Tab: Contract History
    # ------------------------------------------------------------------

    def _tab_contract_history(self):
        st.subheader("Contract History & Procurement Activity")
        procurement = self.r.get("procurement", [])

        if not procurement:
            st.warning("No procurement notices or contracts were found for this commissioner and service area.")
            return

        for p in procurement:
            conf_colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                p.get("confidence", "low"), "⚪"
            )
            notice_type = p.get("type", "contract").replace("_", " ").title()
            title = p.get("title", "Untitled")

            with st.expander(f"{conf_colour} {notice_type} — {title}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Awarded to:** {p.get('awarded_to', 'Not yet awarded')}")
                    st.markdown(f"**Commissioner:** {p.get('commissioner', '—')}")
                    st.markdown(f"**Value:** {p.get('value', 'Not published')}")
                with col2:
                    st.markdown(f"**Award date:** {p.get('award_date', 'Unknown')}")
                    st.markdown(f"**Start date:** {p.get('start_date', 'Unknown')}")
                    st.markdown(f"**End date:** {p.get('end_date', 'Unknown')}")

                if p.get("notes"):
                    st.markdown(f"**Notes:** {p['notes']}")

                awarded_providers = p.get("awarded_providers", [])
                shortlisted_providers = p.get("shortlisted_providers", [])
                if awarded_providers or shortlisted_providers:
                    st.markdown("**Provider drill-down:**")
                    if awarded_providers:
                        st.markdown("*Awarded providers:*")
                        for ap in awarded_providers:
                            lot = f" — Lot: {ap.get('lot')}" if ap.get('lot') else ""
                            url = ap.get('evidence_url', '')
                            if url and url.startswith('http'):
                                st.markdown(f"  - {ap.get('name', 'Unknown')}{lot} [↗]({url})")
                            else:
                                st.markdown(f"  - {ap.get('name', 'Unknown')}{lot}")
                    if shortlisted_providers:
                        st.markdown("*Shortlisted:*")
                        for sp in shortlisted_providers:
                            st.markdown(f"  - {sp.get('name', 'Unknown')}")
                    if p.get("drilldown_notes"):
                        st.caption(f"Drilldown notes: {p['drilldown_notes']}")

                src_url = p.get("source_url", "")
                if src_url and src_url.startswith("http"):
                    st.markdown(f"**Source:** [{src_url}]({src_url})")
                elif src_url:
                    st.markdown(f"**Source:** {src_url}")
                else:
                    st.markdown("**Source:** No reliable public source found")

    # ------------------------------------------------------------------
    # Tab: Competitor Landscape
    # ------------------------------------------------------------------

    def _tab_competitor_landscape(self):
        st.subheader("Competitor Landscape")
        competitors = self.r.get("competitors", [])

        if not competitors:
            st.warning("No competitors were identified. Broaden your geographic area or add known competitors manually.")
            return

        # Summary table — now includes authoritative CQC structured data
        rows = []
        for c in competitors:
            cqc = c.get("cqc_data", {}) or {}
            beds = cqc.get("number_of_beds")
            rows.append({
                "Company": c.get("name", ""),
                "CQC Rating": c.get("cqc_rating", "Unknown"),
                "Beds": beds if beds else "—",
                "Registered": (cqc.get("registration_date", "") or "")[:4] or "—",
                "Local Contracts": len(c.get("known_contracts_with_commissioner", [])),
                "Specialisms": ", ".join(cqc.get("specialisms", [])[:2]) if cqc.get("specialisms") else "—",
                "Website": c.get("website", "") or "—",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Competitor Profiles")

        for c in competitors:
            with st.expander(f"🏢 {c.get('name', 'Unknown')}", expanded=False):
                rationale = c.get("selection_rationale", "")
                if rationale:
                    st.markdown(f"**Why included:** _{rationale}_")
                else:
                    st.caption("⚠️ No selection rationale provided by the model.")

                enrichment = c.get("enrichment_status", {})
                if enrichment:
                    flags = []
                    flags.append("✅ Website" if enrichment.get("website_found") else "❌ Website")
                    flags.append("✅ CQC" if enrichment.get("cqc_found") else "❌ CQC")
                    flags.append("✅ Companies House" if enrichment.get("companies_house_found") else "❌ Companies House")
                    flags.append("✅ Contracts" if enrichment.get("contracts_found") else "❌ Contracts")
                    st.caption("Enrichment: " + " · ".join(flags))

                col1, col2 = st.columns(2)
                with col1:
                    website = c.get("website", "Unknown")
                    if website and website.startswith("http"):
                        st.markdown(f"**Website:** [{website}]({website})")
                    else:
                        st.markdown(f"**Website:** {website}")
                    st.markdown(f"**Headquarters:** {c.get('headquarters', 'Unknown')}")
                    rating = c.get("cqc_rating", "Unknown")
                    verified = " ✓" if c.get("cqc_verified") else ""
                    st.markdown(f"**CQC Rating:** {rating}{verified}", unsafe_allow_html=True)
                    if c.get("cqc_profile_url"):
                        st.markdown(f"**CQC Profile:** [View on cqc.org.uk]({c['cqc_profile_url']})")
                    ch_num = c.get("companies_house_number", "Unknown")
                    ch_url = c.get("companies_house_url", "")
                    if ch_url and ch_url.startswith("http"):
                        st.markdown(f"**Companies House:** [{ch_num}]({ch_url})")
                    else:
                        st.markdown(f"**Companies House:** {ch_num}")

                with col2:
                    # Authoritative CQC structured data
                    cqc = c.get("cqc_data", {}) or {}
                    if cqc:
                        if cqc.get("number_of_beds"):
                            st.markdown(f"**Registered beds:** {cqc['number_of_beds']}")
                        if cqc.get("registration_date"):
                            st.markdown(f"**CQC registered since:** {cqc['registration_date'][:10]}")
                        if cqc.get("last_inspection_date"):
                            st.markdown(f"**Last inspection:** {str(cqc['last_inspection_date'])[:10]}")
                        subs = cqc.get("sub_ratings", {})
                        if subs:
                            st.markdown("**CQC sub-ratings:**")
                            for k, v in subs.items():
                                st.markdown(f"  - {k}: {v}")
                        if cqc.get("specialisms"):
                            st.markdown(f"**Specialisms:** {', '.join(cqc['specialisms'][:6])}")
                        if cqc.get("service_types"):
                            st.markdown(f"**Service types:** {', '.join(cqc['service_types'][:4])}")

                    services = c.get("services", [])
                    if services and not cqc.get("service_types"):
                        st.markdown("**Services:**")
                        for s in services[:6]:
                            st.markdown(f"  - {s}")

                local_contracts = c.get("known_contracts_with_commissioner", [])
                if local_contracts:
                    st.markdown("**Known local contracts:**")
                    for lc in local_contracts:
                        if isinstance(lc, dict):
                            title = lc.get("title", "Untitled")
                            date = lc.get("date", "")
                            value = lc.get("value", "")
                            url = lc.get("source_url", "")
                            line = f"  - **{title}**"
                            if date:
                                line += f" ({date})"
                            if value:
                                line += f" — {value}"
                            if url and url.startswith("http"):
                                line += f" [↗]({url})"
                            st.markdown(line)
                        else:
                            st.markdown(f"  - {lc}")
                else:
                    st.markdown("**Known local contracts:** No reliable public source found")

                if c.get("notes"):
                    st.markdown(f"**Notes:** {c['notes']}")

                sources = c.get("source_urls", [])
                if sources:
                    st.markdown("**Sources:**")
                    for url in sources:
                        if url.startswith("http"):
                            st.markdown(f"  - [{url}]({url})")

    # ------------------------------------------------------------------
    # Tab: Website Analysis
    # ------------------------------------------------------------------

    def _tab_website_analysis(self):
        st.subheader("Website Analysis")
        analyses = self.r.get("website_analyses", {})

        if not analyses:
            st.warning("No website analyses were completed.")
            return

        # Show target first, then competitors
        target_analysis = analyses.get(self.target)
        if target_analysis:
            st.markdown(f"### 🎯 {self.target} (Target Company)")
            self._render_website_card(target_analysis)
            st.divider()

        competitor_names = [k for k in analyses if k != self.target]
        if competitor_names:
            st.markdown("### Competitor Websites")
            for name in competitor_names:
                with st.expander(f"🌐 {name}", expanded=False):
                    self._render_website_card(analyses[name])

    def _render_website_card(self, analysis: Dict):
        if not analysis.get("accessible", True) or analysis.get("access_notes", "").startswith("Website not"):
            st.warning(f"Website could not be analysed: {analysis.get('access_notes', '')}")
            return

        positioning = analysis.get("positioning", {})
        evidence = analysis.get("evidence_quality", {})
        relevance = analysis.get("commissioner_relevance", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            eq = evidence.get("overall_evidence_quality", "Unknown")
            st.markdown(f"**Evidence Quality:** {_ev_badge(eq)}", unsafe_allow_html=True)
        with col2:
            lr = relevance.get("local_relevance_score", "Unknown")
            colour = {"High": "#388e3c", "Medium": "#f57c00", "Low": "#d32f2f"}.get(lr, "#999")
            st.markdown(
                f'**Local Relevance:** <span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;">{lr}</span>',
                unsafe_allow_html=True,
            )
        with col3:
            cqc_cited = evidence.get("cqc_rating_cited", "Not cited")
            st.markdown(f"**CQC Cited:** {cqc_cited}")

        if positioning.get("headline_proposition"):
            st.markdown(f"**Headline proposition:** _{positioning['headline_proposition']}_")

        if positioning.get("target_commissioner_message"):
            st.markdown(f"**Commissioner-facing message:** {positioning['target_commissioner_message']}")

        claims = analysis.get("claims_by_type", {})

        col_a, col_b = st.columns(2)
        with col_a:
            evidenced = claims.get("evidenced", [])
            if evidenced:
                st.markdown("**✅ Evidenced claims:**")
                for claim in evidenced[:5]:
                    st.markdown(f"  - {claim}")

            strengths = analysis.get("strengths", [])
            if strengths:
                st.markdown("**💪 Strengths:**")
                for s in strengths[:5]:
                    st.markdown(f"  - {s}")

        with col_b:
            unsupported = claims.get("unsupported", [])
            if unsupported:
                st.markdown("**⚠️ Unsupported claims:**")
                for claim in unsupported[:5]:
                    st.markdown(f"  - {claim}")

            gaps = analysis.get("weaknesses_and_gaps", [])
            if gaps:
                st.markdown("**🔍 Gaps / weaknesses:**")
                for g in gaps[:5]:
                    st.markdown(f"  - {g}")

        implications = analysis.get("bid_positioning_implications", [])
        if implications:
            st.markdown("**💡 Bid positioning implications:**")
            for imp in implications[:4]:
                if isinstance(imp, dict):
                    st.markdown(f"  - **{imp.get('implication', '')}** — {imp.get('action', '')}")
                else:
                    st.markdown(f"  - {imp}")

        pages = analysis.get("pages_accessed", [])
        if pages:
            with st.expander("Pages reviewed", expanded=False):
                for url in pages:
                    st.markdown(f"  - [{url}]({url})" if url.startswith("http") else f"  - {url}")

    # ------------------------------------------------------------------
    # Tab: Benchmarking Matrix
    # ------------------------------------------------------------------

    def _tab_benchmarking(self):
        st.subheader("Benchmarking Matrix")

        benchmarking = self.r.get("benchmarking", {})

        if not benchmarking:
            st.warning("Benchmarking data was not generated.")
            return

        companies = list(benchmarking.keys())

        # Derive the criteria list from the actual data, NOT from the saved
        # benchmarking_criteria field. Self-heals against version mismatches
        # where the criteria list and scored keys drift apart.
        first_scores = next(iter(benchmarking.values()), {})
        criteria_list = list(first_scores.keys())

        # Put cqc_rating first if present, then everything else in saved order
        if "cqc_rating" in criteria_list:
            criteria_list.remove("cqc_rating")
            criteria_list.insert(0, "cqc_rating")
        # Push overall to the end
        for tail_key in ("overall_bid_threat", "overall_competitor_strength"):
            if tail_key in criteria_list:
                criteria_list.remove(tail_key)
                criteria_list.append(tail_key)

        # Render the matrix as HTML so we can colour-code CQC word ratings
        # and 1-5 scores in the same table cell-by-cell.
        col_labels = [CRITERION_LABELS.get(c, c) for c in criteria_list]
        html_parts = [
            '<div style="overflow-x:auto;">',
            '<table style="border-collapse:collapse;width:100%;font-size:0.9em;">',
            '<thead><tr>',
            '<th style="text-align:left;padding:8px 10px;background:#1a237e;color:white;">Company</th>',
        ]
        for label in col_labels:
            html_parts.append(
                f'<th style="text-align:center;padding:8px 6px;background:#1a237e;color:white;">{label}</th>'
            )
        html_parts.append('</tr></thead><tbody>')

        for company in companies:
            scores = benchmarking[company]
            cells = [f'<td style="padding:6px 10px;border-bottom:1px solid #e0e0e0;font-weight:bold;">{company}</td>']
            for crit in criteria_list:
                val = scores.get(crit, {})
                if crit == "cqc_rating":
                    if isinstance(val, dict):
                        word = val.get("value", "Unknown")
                        verified = val.get("verified", False)
                        cell = _cqc_badge_html(word, verified)
                    else:
                        cell = _cqc_badge_html(str(val) if val else "Unknown")
                    cells.append(
                        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:center;">{cell}</td>'
                    )
                else:
                    s = val.get("score", 0) if isinstance(val, dict) else (val if isinstance(val, (int, float)) else 0)
                    try:
                        s_int = int(s)
                    except (ValueError, TypeError):
                        s_int = 0
                    colour = SCORE_COLOURS.get(s_int, "#eee")
                    text_colour = "white" if s_int >= 4 else ("black" if s_int == 3 else "white")
                    display = f"{s_int}/5" if s_int else "—"
                    cells.append(
                        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:center;'
                        f'background:{colour};color:{text_colour};font-weight:bold;">{display}</td>'
                    )
            html_parts.append('<tr>' + ''.join(cells) + '</tr>')
        html_parts.append('</tbody></table></div>')

        st.markdown('\n'.join(html_parts), unsafe_allow_html=True)

        st.divider()
        st.markdown("### Score Justifications")

        for company in companies:
            with st.expander(f"📊 {company}", expanded=False):
                scores = benchmarking[company]
                for crit in criteria_list:
                    val = scores.get(crit, {})
                    label = CRITERION_LABELS.get(crit, crit)

                    if crit == "cqc_rating":
                        if isinstance(val, dict):
                            word = val.get("value", "Unknown")
                            verified = val.get("verified", False)
                            url = val.get("url", "")
                            st.markdown(
                                f"**{label}:** {_cqc_badge_html(word, verified)}",
                                unsafe_allow_html=True,
                            )
                            if url and url.startswith("http"):
                                st.markdown(f"  → [CQC profile]({url})")
                            elif not verified:
                                st.markdown("  → *Not verified against CQC Syndication API*")
                        continue

                    if not isinstance(val, dict):
                        continue
                    score = val.get("score", 0)
                    justification = val.get("justification", "No justification provided")
                    source = val.get("source", "")
                    is_inference = val.get("analyst_inference", False)

                    badge = _score_badge(score)
                    inf_note = " *(analyst inference)*" if is_inference else ""

                    st.markdown(
                        f"**{label}** {badge}{inf_note}: {justification}",
                        unsafe_allow_html=True,
                    )
                    if source and source != "No source found" and source.startswith("http"):
                        st.markdown(f"  → [{source}]({source})")
                    elif source and source != "No source found":
                        st.markdown(f"  → {source}")
                    else:
                        st.markdown("  → *No reliable public source found*")

    # ------------------------------------------------------------------
    # Tab: Commissioner Priorities
    # ------------------------------------------------------------------

    def _tab_commissioner_priorities(self):
        st.subheader("Commissioner Priorities")
        priorities = self.r.get("commissioner_priorities", [])

        if not priorities:
            st.warning("No commissioner priorities were identified from public sources.")
            return

        for p in priorities:
            conf = p.get("confidence", "low")
            icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
            with st.expander(f"{icon} {p.get('priority', 'Unknown')}", expanded=True):
                st.markdown(p.get("detail", ""))
                src = p.get("source_url", "")
                if src and src.startswith("http"):
                    st.markdown(f"**Source:** [{src}]({src})")
                elif src:
                    st.markdown(f"**Source:** {src}")
                else:
                    st.markdown("**Source:** No reliable public source found")

    # ------------------------------------------------------------------
    # Tab: Bid Positioning
    # ------------------------------------------------------------------

    def _tab_bid_positioning(self):
        st.subheader("Bid Positioning Recommendations")
        positioning = self.r.get("bid_positioning", [])

        if not positioning:
            st.warning("No bid positioning recommendations were generated.")
            return

        high = [p for p in positioning if p.get("priority") == "high"]
        medium = [p for p in positioning if p.get("priority") == "medium"]
        low = [p for p in positioning if p.get("priority") == "low"]
        other = [p for p in positioning if p.get("priority") not in ("high", "medium", "low")]

        for label, group, colour in [
            ("🔴 High Priority", high, "#d32f2f"),
            ("🟡 Medium Priority", medium, "#f57c00"),
            ("🟢 Lower Priority", low + other, "#388e3c"),
        ]:
            if group:
                st.markdown(f"### {label}")
                for p in group:
                    point = p.get("point", "")
                    detail = p.get("detail", "")
                    evidence = p.get("evidence", "")
                    st.markdown(f"**{point}**")
                    if detail:
                        st.markdown(f"{detail}")
                    if evidence:
                        st.caption(f"Evidence basis: {evidence}")
                    st.markdown("---")

    # ------------------------------------------------------------------
    # Tab: Evidence Gaps
    # ------------------------------------------------------------------

    def _tab_evidence_gaps(self):
        st.subheader("Evidence Gaps")
        gaps = self.r.get("evidence_gaps", [])

        if not gaps:
            st.success("No significant evidence gaps were flagged.")
            return

        st.info(
            "The following information could not be confirmed from reliable public sources. "
            "These gaps should be addressed before bid submission."
        )

        for g in gaps:
            with st.expander(f"⚠️ {g.get('area', 'Unknown')}", expanded=False):
                st.markdown(g.get("detail", ""))
                action = g.get("suggested_action", "")
                if action:
                    st.markdown(f"**Suggested action:** {action}")

    # ------------------------------------------------------------------
    # Tab: Source Audit
    # ------------------------------------------------------------------

    def _tab_source_audit(self):
        st.subheader("Source Audit")
        sources = self.r.get("sources", [])

        if not sources:
            st.warning("No sources were logged during research.")
            return

        st.caption(f"{len(sources)} sources reviewed during this research run.")

        rows = []
        for s in sources:
            url = s.get("url", "")
            rows.append({
                "URL": url,
                "Title": s.get("title", ""),
                "Used for": s.get("used_for", ""),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def _render_exports(self):
        st.subheader("Export Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                label="📄 Download Markdown Report",
                data=self.to_markdown(),
                file_name=f"competitor-intel-{self.meta.get('run_id', 'report')[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="🌐 Download HTML Report",
                data=self.to_html(),
                file_name=f"competitor-intel-{self.meta.get('run_id', 'report')[:8]}.html",
                mime="text/html",
                use_container_width=True,
            )
        with col3:
            st.download_button(
                label="📊 Download JSON Results",
                data=json.dumps(self.r, indent=2, default=str),
                file_name=f"competitor-intel-{self.meta.get('run_id', 'results')[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col4:
            st.download_button(
                label="📋 Download Source CSV",
                data=self.to_source_csv(),
                file_name=f"sources-{self.meta.get('run_id', 'audit')[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Export renderers
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        lines = []
        meta = self.meta
        r = self.r

        lines.append(f"# Social Care Competitor Intelligence Report")
        lines.append(f"")
        lines.append(f"**Target Company:** {self.target}  ")
        lines.append(f"**Commissioner:** {meta.get('commissioner', '')}  ")
        lines.append(f"**Service Area:** {meta.get('service_area', '')}  ")
        lines.append(f"**Geographic Area:** {meta.get('geographic_area', '')}  ")
        lines.append(f"**Time Period:** {meta.get('time_period', '')}  ")
        lines.append(f"**Research Depth:** {meta.get('research_depth', '').title()}  ")
        lines.append(f"**Run ID:** {meta.get('run_id', '')[:8]}  ")
        lines.append(f"**Timestamp:** {meta.get('timestamp', '')[:19]}  ")
        lines.append(f"**Confidence Rating:** {r.get('confidence_rating', 'Low')}  ")
        lines.append(f"")

        summary = r.get("executive_summary", "")
        if summary:
            lines.append("## Executive Summary")
            lines.append(summary)
            lines.append("")

        procurement = r.get("procurement", [])
        if procurement:
            lines.append("## Contract History")
            for p in procurement:
                lines.append(f"### {p.get('title', 'Untitled')}")
                lines.append(f"- **Type:** {p.get('type', '')}")
                lines.append(f"- **Awarded to:** {p.get('awarded_to', 'Not yet awarded')}")
                lines.append(f"- **Value:** {p.get('value', 'Not published')}")
                lines.append(f"- **Award date:** {p.get('award_date', 'Unknown')}")
                lines.append(f"- **Source:** {p.get('source_url', 'No reliable public source found')}")
                if p.get("notes"):
                    lines.append(f"- **Notes:** {p['notes']}")
                lines.append("")

        competitors = r.get("competitors", [])
        if competitors:
            lines.append("## Competitor Landscape")
            for c in competitors:
                lines.append(f"### {c.get('name', 'Unknown')}")
                lines.append(f"- **Website:** {c.get('website', 'Unknown')}")
                lines.append(f"- **CQC Rating:** {c.get('cqc_rating', 'Unknown')}")
                lines.append(f"- **Size:** {c.get('size_description', 'Unknown')}")
                services = c.get("services", [])
                if services:
                    lines.append(f"- **Services:** {', '.join(services)}")
                lines.append("")

        benchmarking = r.get("benchmarking", {})
        if benchmarking:
            lines.append("## Benchmarking Scores")
            criteria = r.get("benchmarking_criteria", list(CRITERION_LABELS.keys()))
            header = "| Company | " + " | ".join(CRITERION_LABELS.get(c, c) for c in criteria) + " |"
            separator = "| --- | " + " | ".join("---" for _ in criteria) + " |"
            lines.append(header)
            lines.append(separator)
            for company, scores in benchmarking.items():
                row_vals = []
                for crit in criteria:
                    val = scores.get(crit, {})
                    if crit == "cqc_rating":
                        if isinstance(val, dict):
                            word = val.get("value", "Unknown")
                            tick = " ✓" if val.get("verified") else ""
                            row_vals.append(f"{word}{tick}")
                        else:
                            row_vals.append(str(val) if val else "Unknown")
                    else:
                        score = val.get("score", "—") if isinstance(val, dict) else val
                        row_vals.append(str(score))
                lines.append(f"| {company} | " + " | ".join(row_vals) + " |")
            lines.append("")

        gaps = r.get("evidence_gaps", [])
        if gaps:
            lines.append("## Evidence Gaps")
            for g in gaps:
                lines.append(f"- **{g.get('area', '')}:** {g.get('detail', '')} — *{g.get('suggested_action', '')}*")
            lines.append("")

        sources = r.get("sources", [])
        if sources:
            lines.append("## Sources")
            for s in sources:
                url = s.get("url", "")
                title = s.get("title", url)
                used_for = s.get("used_for", "")
                if url.startswith("http"):
                    lines.append(f"- [{title}]({url}) — {used_for}")
                else:
                    lines.append(f"- {title} — {used_for}")
            lines.append("")

        lines.append(f"\n---\n*Generated by Social Care Competitor Intelligence Dashboard — {datetime.now().strftime('%d %B %Y')}*")

        return "\n".join(lines)

    def to_html(self) -> str:
        md_content = self.to_markdown()
        try:
            import markdown as md_lib
            body = md_lib.markdown(md_content, extensions=["tables", "fenced_code"])
        except ImportError:
            body = f"<pre>{md_content}</pre>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Competitor Intelligence — {self.target}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px;
          color: #222; line-height: 1.6; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; border-bottom: 1px solid #c5cae9; padding-bottom: 4px; margin-top: 2em; }}
  h3 {{ color: #3949ab; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }}
  th {{ background: #283593; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e8eaf6; }}
  tr:nth-child(even) {{ background: #f5f5ff; }}
  a {{ color: #1a237e; }}
  code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
  .footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #ccc;
             color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
{body}
<div class="footer">Generated by Social Care Competitor Intelligence Dashboard &mdash; {datetime.now().strftime('%d %B %Y')}</div>
</body>
</html>"""

    def to_source_csv(self) -> str:
        sources = self.r.get("sources", [])
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["url", "title", "used_for"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for s in sources:
            writer.writerow(s)
        return output.getvalue()
