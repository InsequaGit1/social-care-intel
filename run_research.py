"""
Command-line runner for a real research pass — used for live testing outside
Streamlit. Reads API keys from a local, git-ignored .env file (or the
environment) so keys never appear on the command line or in chat.

Setup (do this yourself in your editor/terminal, not in chat):
  Create a file called `.env` in this folder containing:

    OPENAI_API_KEY=sk-...            # or ANTHROPIC_API_KEY for Claude
    CQC_API_KEY=...
    BRAVE_API_KEY=...

  `.env` is git-ignored, so it is never committed.

Run:
  python3 run_research.py \
      --commissioner "Southend-on-Sea City Council" \
      --target "Ashley Care Ltd" \
      --service "domiciliary care" \
      --geography "Southend-on-Sea" \
      --period "last 3 years" \
      --depth quick \
      --provider Claude --model claude-opus-4-6

Output: prints a summary and writes the full JSON to outputs/<run-id>/results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env if python-dotenv is available (keys stay off the command line)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

from research_agent import ResearchAgent, ResearchConfig
from analysis_agent import AnalysisAgent
from search_providers.llm_web import LLMWebProvider

PROVIDER_ENV = {
    "OpenAI": "OPENAI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
    "Gemini": "GOOGLE_API_KEY",
}


def _status(msg: str) -> None:
    # Strip simple markdown bold for clean console output
    print("   " + msg.replace("**", ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live research run (CLI)")
    ap.add_argument("--commissioner", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--service", default="")
    ap.add_argument("--geography", default="")
    ap.add_argument("--website", default="")
    ap.add_argument("--period", default="last 3 years")
    ap.add_argument("--depth", choices=["quick", "deep"], default="quick")
    ap.add_argument("--provider", choices=list(PROVIDER_ENV), default="OpenAI")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    llm_key = os.environ.get(PROVIDER_ENV[args.provider], "")
    if not llm_key:
        print(f"ERROR: {PROVIDER_ENV[args.provider]} not found in environment/.env", file=sys.stderr)
        return 2

    cqc_key = os.environ.get("CQC_API_KEY", "")
    brave_key = os.environ.get("BRAVE_API_KEY", "")
    ch_key = os.environ.get("COMPANIES_HOUSE_API_KEY", "")

    print(f"Provider: {args.provider} / {args.model}")
    print(f"Live data sources: "
          f"{'CQC ' if cqc_key else ''}{'Brave ' if brave_key else ''}"
          f"{'CompaniesHouse' if ch_key else ''}".strip() or "LLM web search only")
    print("-" * 60)

    config = ResearchConfig(
        commissioner=args.commissioner,
        service_area=args.service,
        target_company=args.target,
        target_website=args.website,
        geographic_area=args.geography,
        time_period=args.period,
        research_depth=args.depth,
        cqc_api_key=cqc_key,
        brave_api_key=brave_key,
        companies_house_api_key=ch_key,
    )
    provider = LLMWebProvider(args.provider, args.model, llm_key)

    print("PHASE 1 — research")
    phase1 = ResearchAgent(config, provider).run(status_callback=_status)
    print("PHASE 2 — analysis + benchmarking")
    phase2 = AnalysisAgent(config, provider).run(phase1, status_callback=_status)

    results = {**phase1, **{k: v for k, v in phase2.items()}}

    out_dir = Path("outputs") / config.run_id[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Console summary
    print("\n" + "=" * 60)
    print(f"Target LA: {phase1.get('metadata', {}).get('target_local_authority', '?')}")
    print(f"Discovery: {phase1.get('metadata', {}).get('discovery_method', '?')}")
    print(f"Competitors: {len(phase1.get('competitors', []))}")
    print(f"Procurement notices: {len(phase1.get('procurement', []))} "
          f"(rejected {len(phase1.get('procurement_rejected', []))})")
    bench = phase2.get("benchmarking", {})
    if bench:
        print("\nBenchmarking (overall bid threat):")
        rows = []
        for name, cs in bench.items():
            ov = cs.get("overall_bid_threat", {})
            raw = ov.get("raw_score", ov.get("score", 0)) if isinstance(ov, dict) else 0
            cqc = cs.get("cqc_rating", {})
            rating = cqc.get("value", "?") if isinstance(cqc, dict) else "?"
            rows.append((raw, name, rating))
        for raw, name, rating in sorted(rows, reverse=True):
            print(f"  {raw:>4}/5  {name}  (CQC: {rating})")
    print(f"\nSaved: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
