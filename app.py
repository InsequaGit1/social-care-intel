"""
Social Care Competitor Intelligence Dashboard
---------------------------------------------
Local one-shot research tool. No database, no login, no hosted backend.
Results are saved to the local outputs/ folder.

Run with:  streamlit run app.py
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from research_agent import ResearchAgent, ResearchConfig
from analysis_agent import AnalysisAgent
from dashboard_renderer import DashboardRenderer
from search_providers.llm_web import LLMWebProvider

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Social Care Competitor Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .main-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
    padding: 1.5rem 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1.5rem;
  }
  .main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
  .main-header p  { color: #c5cae9; margin: 0.3rem 0 0; font-size: 0.9rem; }
  div[data-testid="stMetricValue"] { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

# Claude is the default provider (listed first). Gemini removed.
PROVIDERS = {
    "Claude": {
        # Opus 4.6 is the default model. Sonnet for cheaper/faster runs,
        # newer Opus for top quality, Haiku for cheapest. IDs per Anthropic's
        # dateless 4.6-generation scheme.
        "models": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        ],
        "env_var": "ANTHROPIC_API_KEY",
        "note": "Uses the server-side web_search tool. Opus = top quality (default); "
                "Sonnet/Haiku = cheaper for high-volume Deep Scans.",
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini"],
        "env_var": "OPENAI_API_KEY",
        "note": "Uses the Responses API with web_search_preview grounding.",
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown("""
    <div class="main-header">
      <h1>🔍 Social Care Competitor Intelligence Dashboard</h1>
      <p>UK social care market research — one-shot tool, runs locally, outputs saved to outputs/</p>
    </div>
    """, unsafe_allow_html=True)

    if "results" in st.session_state and st.session_state.results:
        _show_results()
    else:
        _show_input_form()


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

def _show_input_form():
    st.markdown("### Research Brief")

    with st.form("research_form", border=True):

        # ---- Required fields ------------------------------------------
        st.markdown("#### Required")
        c1, c2 = st.columns(2)

        with c1:
            commissioner = st.text_input(
                "Commissioner / Local Authority / ICB *",
                placeholder="e.g. Birmingham City Council",
            )
            target_company = st.text_input(
                "Target Company Name *",
                placeholder="e.g. Acorn Care Services Ltd",
            )

        with c2:
            time_period = st.text_input(
                "Time Period to Review *",
                placeholder="e.g. Last 3 years (2022–2025)",
            )

        # ---- Optional inputs ------------------------------------------
        st.markdown("#### Optional — leave blank and the model will find these")
        c3, c4 = st.columns(2)

        with c3:
            service_area = st.text_input(
                "Service Area",
                placeholder="e.g. domiciliary care — inferred from the target's CQC registration if blank",
            )
            target_website = st.text_input(
                "Target Company Website",
                placeholder="e.g. https://www.acorncare.co.uk — found automatically if blank",
            )
            geographic_area = st.text_input(
                "Geographic Area",
                placeholder="e.g. Birmingham and West Midlands — inferred if blank",
            )

        with c4:
            known_competitors_raw = st.text_area(
                "Known Competitors (one per line)",
                placeholder="Mencap\nScopeUK\nRehabcare Ltd",
                height=80,
            )
            manual_urls_raw = st.text_area(
                "Manual URLs to check (one per line)",
                placeholder="https://www.birmingham.gov.uk/procurement",
                height=80,
            )

        # ---- Provider & depth -----------------------------------------
        st.markdown("#### Research Settings")
        c5, c6, c7 = st.columns([1, 1, 1])

        with c5:
            research_depth = st.radio(
                "Research Depth",
                options=["Quick Scan", "Deeper Scan"],
                help=(
                    "**Quick Scan** — typically 3–6 minutes. "
                    "Authoritative CQC market map: target profile, real local providers "
                    "with verified CQC ratings/beds/specialisms, procurement notices, and "
                    "benchmarking on CQC data. Up to 5 competitors.\n\n"
                    "**Deeper Scan** — typically 10–25 minutes. "
                    "Everything in Quick PLUS competitor website analysis, Companies House + "
                    "contract enrichment per competitor, procurement provider drill-down, and "
                    "LLM-based discovery of framework/contract players. Up to 10 competitors."
                ),
            )

        with c6:
            provider_name = st.selectbox(
                "Model Provider",
                options=list(PROVIDERS.keys()),
                help="All providers use built-in web search / grounding.",
            )
            provider_cfg = PROVIDERS[provider_name]
            st.caption(provider_cfg["note"])

        with c7:
            model_name = st.selectbox("Model", options=provider_cfg["models"])

        # ---- API key --------------------------------------------------
        env_var = provider_cfg["env_var"]
        env_value = os.environ.get(env_var, "")
        if not env_value:
            try:
                env_value = st.secrets.get(env_var, "")
            except Exception:
                env_value = ""

        if env_value:
            st.success(f"API key detected (from `{env_var}` env var or Streamlit secrets).")
            api_key_input = st.text_input(
                f"Override API key (leave blank to use stored key)",
                type="password",
                value="",
            )
        else:
            api_key_input = st.text_input(
                f"API Key (`{env_var}`) *",
                type="password",
                placeholder=f"Paste your API key, or set {env_var} in env / Streamlit secrets",
            )

        # ---- Submit ---------------------------------------------------
        submitted = st.form_submit_button(
            "🚀 Run Research",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        resolved_key = api_key_input.strip() or env_value
        errors = _validate_form(
            target_company=target_company,
            commissioner=commissioner,
            service_area=service_area,
            time_period=time_period,
            api_key=resolved_key,
            env_var=env_var,
        )

        if errors:
            for e in errors:
                st.error(e)
        else:
            _run_research(
                commissioner=commissioner,
                service_area=service_area,
                target_company=target_company,
                target_website=target_website,
                geographic_area=geographic_area,
                time_period=time_period,
                known_competitors=[
                    line.strip()
                    for line in known_competitors_raw.splitlines()
                    if line.strip()
                ],
                manual_urls=[
                    line.strip()
                    for line in manual_urls_raw.splitlines()
                    if line.strip()
                ],
                research_depth="quick" if research_depth == "Quick Scan" else "deep",
                provider_name=provider_name,
                model_name=model_name,
                api_key=resolved_key,
            )


# ---------------------------------------------------------------------------
# Research execution
# ---------------------------------------------------------------------------

def _run_research(
    commissioner, service_area, target_company, target_website,
    geographic_area, time_period, known_competitors, manual_urls,
    research_depth, provider_name, model_name, api_key,
):
    run_id = str(uuid.uuid4())

    # Read optional external data source keys from env vars or Streamlit secrets
    def _read_secret(name: str) -> str:
        val = os.environ.get(name, "")
        if not val:
            try:
                val = st.secrets.get(name, "")
            except Exception:
                val = ""
        return val

    cqc_key = _read_secret("CQC_API_KEY")
    brave_key = _read_secret("BRAVE_API_KEY")
    ch_key = _read_secret("COMPANIES_HOUSE_API_KEY")

    # Surface which authoritative sources are active
    enabled = []
    if cqc_key:
        enabled.append("CQC API")
    if brave_key:
        enabled.append("Brave Search")
    if ch_key:
        enabled.append("Companies House")
    if enabled:
        st.info("🔐 Authoritative data sources active: " + ", ".join(enabled))
    else:
        st.info("ℹ️ Running on LLM web search only. Add CQC_API_KEY / BRAVE_API_KEY to Streamlit secrets for higher accuracy.")

    config = ResearchConfig(
        commissioner=commissioner,
        service_area=service_area,
        target_company=target_company,
        target_website=target_website,
        geographic_area=geographic_area,
        time_period=time_period,
        known_competitors=known_competitors,
        manual_urls=manual_urls,
        research_depth=research_depth,
        run_id=run_id,
        cqc_api_key=cqc_key,
        brave_api_key=brave_key,
        companies_house_api_key=ch_key,
    )

    provider = LLMWebProvider(
        provider_name=provider_name,
        model_name=model_name,
        api_key=api_key,
    )

    with st.status("Running research…", expanded=True) as status:
        try:
            st.write("🔍 **Phase 1:** Market intelligence — procurement, competitors, commissioner priorities…")
            research_agent = ResearchAgent(config, provider)
            phase1 = research_agent.run(status_callback=st.write)

            st.write("🌐 **Phase 2:** Website analysis — target company and competitors…")
            analysis_agent = AnalysisAgent(config, provider)
            phase2 = analysis_agent.run(
                research_results=phase1,
                status_callback=st.write,
            )

            st.write("💾 Saving outputs…")
            results = _merge_results(phase1, phase2)
            output_dir = _save_outputs(results, config)

            status.update(
                label=f"✅ Research complete — saved to `{output_dir}`",
                state="complete",
            )

            st.session_state.results = results
            st.rerun()

        except Exception as exc:
            status.update(label=f"❌ Research failed: {exc}", state="error")
            st.error(f"An error occurred: {exc}")
            st.exception(exc)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _show_results():
    results = st.session_state.results
    meta = results.get("metadata", {})

    info_col, btn_col = st.columns([4, 1])
    with info_col:
        st.info(
            f"Showing results for **{meta.get('target_company', '')}** | "
            f"Commissioner: **{meta.get('commissioner', '')}** | "
            f"Run `{meta.get('run_id', '')[:8]}` | "
            f"Confidence: **{results.get('confidence_rating', 'Low')}**"
        )
    with btn_col:
        if st.button("🔄 New Research Run", type="primary", use_container_width=True):
            del st.session_state.results
            st.rerun()

    renderer = DashboardRenderer(results)
    renderer.render()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_form(**kwargs) -> list:
    errors = []
    required = {
        "target_company": "Target company name",
        "commissioner": "Commissioner / local authority",
        "time_period": "Time period",
    }
    for field, label in required.items():
        if not kwargs.get(field, "").strip():
            errors.append(f"{label} is required.")

    if not kwargs.get("api_key", "").strip():
        env_var = kwargs.get("env_var", "API_KEY")
        errors.append(
            f"API key is required — paste it above or set the `{env_var}` environment variable."
        )
    return errors


def _merge_results(phase1: dict, phase2: dict) -> dict:
    merged = {**phase1}
    merged["website_analyses"] = phase2.get("website_analyses", {})
    merged["benchmarking"] = phase2.get("benchmarking", {})
    merged["benchmarking_criteria"] = phase2.get("benchmarking_criteria", [])
    merged["bid_positioning"] = phase2.get("bid_positioning", [])
    merged["executive_summary"] = phase2.get("executive_summary", "")
    merged["evidence_gaps"] = phase2.get("evidence_gaps", phase1.get("evidence_gaps", []))
    merged["sources"] = phase2.get("sources", phase1.get("sources", []))
    merged["confidence_rating"] = phase2.get("confidence_rating", phase1.get("confidence_rating", "Low"))
    merged["confidence_rationale"] = phase2.get("confidence_rationale", phase1.get("confidence_rationale", ""))
    return merged


def _save_outputs(results: dict, config: ResearchConfig) -> str:
    output_dir = Path("outputs") / config.run_id[:8]
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    renderer = DashboardRenderer(results)

    # Markdown
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(renderer.to_markdown())

    # HTML
    with open(output_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(renderer.to_html())

    # CSV
    with open(output_dir / "sources.csv", "w", encoding="utf-8", newline="") as f:
        f.write(renderer.to_source_csv())

    return str(output_dir)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
