# Social Care Competitor Intelligence Dashboard

A local one-shot research tool for UK social care bid teams. Enter a research brief, choose your AI provider, and get a structured competitor intelligence report in minutes — saved locally, no database, no login.

---

## Quick start

### 1. Install dependencies

```bash
cd social-care-intel
pip install -r requirements.txt
```

### 2. Set your API key

Set an environment variable for whichever provider you're using:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Google Gemini
export GOOGLE_API_KEY=AIza...

# Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...
```

Or paste it directly into the API key field in the app — it is never stored or transmitted anywhere other than the API call.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## What it does

You fill in a research brief (commissioner, service area, target company, geography, time period). The tool then:

1. **Market Intelligence** — searches for procurement notices on Contracts Finder and Find a Tender, council contract registers, decision papers, CQC profiles, and Companies House records.
2. **Competitor Identification** — finds providers operating in the same space with the same commissioner.
3. **Website Analysis** — analyses the target company and each competitor's website, classifying claims as explicit, evidenced, unsupported, or analyst inference.
4. **Benchmarking** — scores all companies 1–5 across 14 criteria with justifications and source references.
5. **Bid Positioning** — generates prioritised recommendations for differentiation.

---

## Research depth

| Setting | Competitors | Sources | Procurement notices | Time (approx) |
|---|---|---|---|---|
| Quick Scan | Up to 5 | Up to 12 | Up to 3 | 2–5 min |
| Deeper Scan | Up to 10 | Up to 30 | Up to 8 | 8–20 min |

---

## Supported providers

| Provider | Model options | Web search method |
|---|---|---|
| OpenAI | gpt-4o, gpt-4o-mini | Responses API + `web_search_preview` |
| Gemini | gemini-2.0-flash, gemini-1.5-pro | `google_search` grounding |
| Claude | claude-opus-4-7, claude-sonnet-4-6 | `web_search_20250305` built-in tool |

All three use the model's native web search capability. No separate search API is needed for Version 1.

---

## Outputs

Each run saves four files to `outputs/<run-id>/`:

| File | Contents |
|---|---|
| `results.json` | Full structured results (matches `sample_output_schema.json`) |
| `report.md` | Full Markdown report |
| `report.html` | Standalone HTML report — open in any browser |
| `sources.csv` | Source audit — all URLs reviewed |

All four files are also available as download buttons inside the dashboard.

---

## Deterministic benchmarking (reliability by design)

The benchmarking matrix is **computed, not guessed**. Every 1–5 score comes from a
pure function (`scoring.py`) applied to authoritative CQC structured data:

| Criterion | Derived from |
|---|---|
| CQC Rating | CQC overall rating (word value, verified against the API) |
| Service & Location Fit | service-type + local-authority + specialism overlap with the target |
| Quality & Compliance | the five CQC sub-ratings (Safe/Effective/Caring/Responsive/Well-led) |
| Local Track Record | longevity in the target's local authority + named contracts |
| Delivery Strength | registered beds + registration longevity |
| Strategic Differentiators | CQC specialism/service breadth (+ website evidence in Deep Scan) |
| Overall Bid Threat | weighted composite (shown to one decimal for fine ranking) |

Consequences:
- **Reproducible** — the same CQC data always yields the same scores (proven by `test_scoring.py`).
- **Verifiable** — each score links to the exact CQC profile it used.
- **Consistent** — the identical rubric is applied to every competitor.

The LLM is never used to assign scores; it only writes the executive-summary and
bid-positioning narrative *on top of* the fixed scores.

Run the test suites any time:
```bash
python3 test_scoring.py   # scoring: reproducibility, monotonicity, edge cases
python3 test_logic.py     # CQC parsing, name-matching, JSON extraction
```

## Guardrails

The tool is designed to avoid hallucinated intelligence:

- Claims without a public source are labelled **"No reliable public source found"**
- Procurement notices with fabricated-looking URLs are automatically rejected
- A wrong-town CQC match for the target is hard-rejected (no cross-area contamination)
- A final **Confidence Rating** (High / Medium / Low) reflects overall data quality
- The tool will never invent contract award decisions, CQC ratings, or provider relationships

---

## Adding a Brave Search provider (Version 2)

The search provider is abstracted behind `search_providers/base.py`. To add Brave Search:

1. Implement `BraveSearchProvider` in `search_providers/brave_placeholder.py`
2. It must inherit `SearchProvider` and implement `research()` and `name`
3. Add it to the provider selector in `app.py`

The dashboard, research agent, and analysis agent need no changes.

---

## File structure

```
app.py                          Streamlit UI and orchestration
research_agent.py               Phase 1 — market intelligence
analysis_agent.py               Phase 2 — website analysis and benchmarking
dashboard_renderer.py           Renders all tabs and produces exports
search_providers/
    base.py                     Abstract SearchProvider interface
    llm_web.py                  OpenAI / Gemini / Claude implementations
    brave_placeholder.py        Stub for future Brave Search integration
prompts/
    master_research_prompt.txt  Market intelligence prompt
    website_analysis_prompt.txt Website analysis prompt
    benchmarking_prompt.txt     Benchmarking and scoring prompt
outputs/                        Research run outputs (git-ignored)
sample_output_schema.json       JSON schema for the results format
requirements.txt
README.md
```

---

## Notes for non-technical users

- The app runs entirely on your machine. Nothing is stored externally.
- Research quality depends on what public information exists. If a commissioner does not publish contracts openly, the tool will say so rather than guessing.
- Quick Scan is good for a first pass before a bid decision. Deeper Scan is better for active bid support.
- Always verify critical facts (CQC ratings, contract values, award decisions) against the original source before using them in a bid.
