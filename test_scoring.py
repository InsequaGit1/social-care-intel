"""
Tests for the deterministic benchmarking scorer (scoring.py).

Proves the three properties the tool needs:
  1. Reproducibility — same input gives identical output every time.
  2. Monotonicity — better facts give higher (never lower) scores.
  3. Correctness — realistic fixtures produce sensible, defensible scores.

Run:  python3 test_scoring.py
"""

import copy
import sys

import scoring

FAILURES = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILURES.append(name)


# --- Fixtures (realistic CQC shapes) ---------------------------------------

TARGET = {
    "name": "Ashley Care Ltd",
    "is_target": True,
    "cqc_rating": "Good",
    "cqc_profile_url": "https://www.cqc.org.uk/location/1-2430518179",
    "cqc_verified": True,
    "cqc_data": {
        "sub_ratings": {"Safe": "Good", "Effective": "Good", "Caring": "Good",
                        "Responsive": "Good", "Well-led": "Good"},
        "number_of_beds": 0,
        "registration_date": "2016-01-22",
        "service_types": ["Homecare agencies"],
        "specialisms": ["Caring for adults over 65 yrs", "Dementia", "Physical disabilities"],
        "local_authority": "Southend-on-Sea",
    },
}

STRONG_COMP = {
    "name": "Big Outstanding Homecare",
    "cqc_rating": "Outstanding",
    "cqc_profile_url": "https://www.cqc.org.uk/location/1-999",
    "cqc_verified": True,
    "cqc_data": {
        "sub_ratings": {"Safe": "Good", "Effective": "Outstanding", "Caring": "Outstanding",
                        "Responsive": "Good", "Well-led": "Outstanding"},
        "number_of_beds": 0,
        "registration_date": "2009-05-01",
        "service_types": ["Homecare agencies"],
        "specialisms": ["Caring for adults over 65 yrs", "Dementia", "Physical disabilities",
                        "Sensory impairment", "Mental health conditions", "Learning disabilities"],
        "local_authority": "Southend-on-Sea",
    },
}

WEAK_COMP = {
    "name": "New Small Agency",
    "cqc_rating": "Requires improvement",
    "cqc_profile_url": "https://www.cqc.org.uk/location/1-111",
    "cqc_verified": True,
    "cqc_data": {
        "sub_ratings": {"Safe": "Requires improvement", "Effective": "Good", "Caring": "Good",
                        "Responsive": "Requires improvement", "Well-led": "Requires improvement"},
        "number_of_beds": 0,
        "registration_date": "2024-02-01",
        "service_types": ["Homecare agencies"],
        "specialisms": ["Caring for adults over 65 yrs"],
        "local_authority": "Southend-on-Sea",
    },
}

OUT_OF_AREA = {
    "name": "Faraway Care",
    "cqc_rating": "Good",
    "cqc_profile_url": "https://www.cqc.org.uk/location/1-222",
    "cqc_verified": True,
    "cqc_data": {
        "sub_ratings": {"Safe": "Good", "Effective": "Good", "Caring": "Good",
                        "Responsive": "Good", "Well-led": "Good"},
        "number_of_beds": 0,
        "registration_date": "2010-01-01",
        "service_types": ["Homecare agencies"],
        "specialisms": ["Caring for adults over 65 yrs"],
        "local_authority": "Manchester",
    },
}


def test_reproducibility():
    print("\n== Reproducibility (same input → identical output) ==")
    a = scoring.score_company(copy.deepcopy(STRONG_COMP), copy.deepcopy(TARGET))
    b = scoring.score_company(copy.deepcopy(STRONG_COMP), copy.deepcopy(TARGET))
    # Compare just the score integers across all criteria
    sa = {k: v["score"] for k, v in a.items() if k != "cqc_rating"}
    sb = {k: v["score"] for k, v in b.items() if k != "cqc_rating"}
    check("identical scores on repeat", sa == sb)


def test_monotonicity():
    print("\n== Monotonicity (better facts → higher scores) ==")
    strong = scoring.score_company(STRONG_COMP, TARGET)
    weak = scoring.score_company(WEAK_COMP, TARGET)
    check("strong quality > weak quality",
          strong["quality_compliance"]["score"] > weak["quality_compliance"]["score"])
    check("strong delivery > weak delivery (longevity)",
          strong["delivery_strength"]["score"] > weak["delivery_strength"]["score"])
    check("strong overall > weak overall",
          strong["overall_bid_threat"]["score"] > weak["overall_bid_threat"]["score"])


def test_area_effect():
    print("\n== Area effect (same LA scores higher on fit + track record) ==")
    local = scoring.score_company(STRONG_COMP, TARGET)          # Southend, same as target
    distant = scoring.score_company(OUT_OF_AREA, TARGET)         # Manchester
    check("same-LA fit > out-of-area fit",
          local["service_location_fit"]["score"] > distant["service_location_fit"]["score"])
    check("same-LA track record > out-of-area track record",
          local["local_track_record"]["score"] > distant["local_track_record"]["score"])


def test_quality_caps():
    print("\n== Quality caps (Inadequate/RI domains cap the score) ==")
    inadequate = {"cqc_data": {"sub_ratings": {
        "Safe": "Inadequate", "Effective": "Good", "Caring": "Good",
        "Responsive": "Good", "Well-led": "Good"}}}
    q = scoring.score_quality_compliance(inadequate["cqc_data"])
    check("one Inadequate domain caps quality at <=2", q["score"] <= 2)

    ri = {"cqc_data": {"sub_ratings": {
        "Safe": "Requires improvement", "Effective": "Good", "Caring": "Good",
        "Responsive": "Good", "Well-led": "Good"}}}
    q2 = scoring.score_quality_compliance(ri["cqc_data"])
    check("one RI domain caps quality at <=3", q2["score"] <= 3)


def test_unrated_handling():
    print("\n== Unrated provider handling ==")
    unrated = {
        "name": "Brand New Co", "cqc_rating": "Unknown", "cqc_verified": True,
        "cqc_data": {"sub_ratings": {}, "number_of_beds": 0,
                     "registration_date": "2025-06-01", "service_types": ["Homecare agencies"],
                     "specialisms": [], "local_authority": "Southend-on-Sea"},
    }
    s = scoring.score_company(unrated, TARGET)
    check("unrated quality is conservative (<=2)", s["quality_compliance"]["score"] <= 2)
    check("unrated still scored (no crash, 1-5)", 1 <= s["overall_bid_threat"]["score"] <= 5)


def test_bed_scaling():
    print("\n== Bed scaling for care homes ==")
    def home(beds, reg="2010-01-01"):
        return scoring.score_delivery_strength({"number_of_beds": beds, "registration_date": reg})
    big = home(80)["score"]
    mid = home(30)["score"]
    small = home(8)["score"]
    check("80 beds >= 30 beds >= 8 beds (delivery)", big >= mid >= small)
    check("80-bed established home scores high (>=4)", big >= 4)


def test_all_scores_in_range():
    print("\n== All scores within 1-5 ==")
    for comp in (TARGET, STRONG_COMP, WEAK_COMP, OUT_OF_AREA):
        s = scoring.score_company(comp, TARGET)
        for k, v in s.items():
            if k == "cqc_rating":
                continue
            if not (1 <= v["score"] <= 5):
                check(f"{comp['name']}/{k} in range", False)
                break
        else:
            check(f"{comp['name']}: all criteria in 1-5", True)


def test_justifications_present():
    print("\n== Every score carries a justification ==")
    s = scoring.score_company(STRONG_COMP, TARGET)
    ok = all(v.get("justification") for k, v in s.items() if k != "cqc_rating")
    check("all criteria have justifications", ok)


def test_target_without_cqc():
    print("\n== Graceful handling when TARGET has no CQC data ==")
    blank_target = {"name": "Mystery Co", "is_target": True, "cqc_rating": "Unknown",
                    "cqc_data": {}}
    try:
        s = scoring.score_company(STRONG_COMP, blank_target)
        ok_range = all(1 <= s[k]["score"] <= 5 for k in s if k != "cqc_rating")
        check("competitor still scored 1-5 with blank target", ok_range)
        check("no crash on blank target", True)
    except Exception as exc:
        check(f"no crash on blank target (got {exc})", False)


def test_end_to_end_reproducibility():
    print("\n== End-to-end benchmark reproducibility (identical matrix twice) ==")
    from analysis_agent import AnalysisAgent
    from research_agent import ResearchConfig
    from search_providers.base import SearchProvider, SearchResult

    class Stub(SearchProvider):
        name = "stub"
        def research(self, prompt, max_tokens=6000):
            return SearchResult(query="x", content='{"executive_summary":"","bid_positioning":[],"evidence_gaps":[]}')

    cfg = ResearchConfig(commissioner="Southend-on-Sea", service_area="",
                         target_company="Ashley Care Ltd", time_period="3y", research_depth="quick")
    agent = AnalysisAgent(cfg, Stub())
    companies = [TARGET, STRONG_COMP, WEAK_COMP, OUT_OF_AREA]

    def matrix():
        b = agent._benchmark(all_companies=[copy.deepcopy(c) for c in companies],
                             research_results={"competitors": companies}, website_analyses={})
        return {name: {k: (v.get("score") if isinstance(v, dict) and "score" in v else v.get("value"))
                       for k, v in cs.items()}
                for name, cs in b["scores"].items()}

    m1, m2 = matrix(), matrix()
    check("two full benchmark runs produce identical matrices", m1 == m2)


if __name__ == "__main__":
    test_reproducibility()
    test_monotonicity()
    test_area_effect()
    test_quality_caps()
    test_unrated_handling()
    test_bed_scaling()
    test_all_scores_in_range()
    test_justifications_present()
    test_target_without_cqc()
    test_end_to_end_reproducibility()
    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} failed: {FAILURES}")
        sys.exit(1)
    print("✅ All scoring tests passed.")
