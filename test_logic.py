"""
Offline logic tests — no network, no API keys required.

Run:  python3 test_logic.py

Validates the deterministic helper functions that underpin reliability:
  - CQC name disambiguation (right company, right town)
  - URL hallucination detection
  - CQC API response parsing (sub-ratings, beds, specialisms)
  - safe_join tolerance of mixed types

These are the parts that have repeatedly broken in production, so they
get explicit regression coverage.
"""

import sys

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"        got:  {got!r}")
        print(f"        want: {want!r}")
        FAILURES.append(name)


def check_true(name, cond):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILURES.append(name)


def test_name_matching():
    print("\n== CQC name-match confidence ==")
    from research_agent import _name_match_confidence
    cases = [
        ("Ashley Care Ltd", "Ashley Care Limited", True),
        ("Ashley Care Ltd", "Ashley Community Care Services Limited", False),
        ("Bluebird Care", "Bluebird Care (Southend & Rochford)", True),
        ("Caremark", "Caremark Care Ltd", True),
        ("Mears Care", "Mears Group Ltd", False),
        ("Caremark", "Allied Healthcare", False),
    ]
    for query, candidate, want_accept in cases:
        score = _name_match_confidence(query, candidate)
        check(f"{query!r} ~ {candidate!r} -> accept={want_accept}", score >= 0.6, want_accept)


def test_url_filter():
    print("\n== Hallucinated URL detection ==")
    from research_agent import _is_url_suspicious
    fakes = [
        "https://www.contractsfinder.service.gov.uk/Notice/1234567890",
        "https://www.contractsfinder.service.gov.uk/Notice/1122334455",
        "https://www.contractsfinder.service.gov.uk/Notice/6677889900",
    ]
    reals = [
        "https://www.find-tender.service.gov.uk/Notice/005986-2026",
        "https://www.contractsfinder.service.gov.uk/Notice/2c4e7f2b-1a2b-4c3d-9e8f-abc123456789",
    ]
    for u in fakes:
        check(f"fake flagged: ...{u[-12:]}", _is_url_suspicious(u), True)
    for u in reals:
        check(f"real allowed: ...{u[-12:]}", _is_url_suspicious(u), False)


def test_cqc_parsing():
    print("\n== CQC response parsing (sub-ratings, beds, specialisms) ==")
    from data_sources.cqc import CQCClient
    client = CQCClient("dummy")
    # Realistic CQC location payload shape
    raw = {
        "locationId": "1-123456789",
        "providerId": "1-987654321",
        "name": "Sunnyfields Care Home",
        "registrationStatus": "Registered",
        "registrationDate": "2011-04-01",
        "numberOfBeds": 42,
        "website": "https://sunnyfields.example.co.uk",
        "postalAddressTownCity": "Southend-on-Sea",
        "postalCode": "SS1 1AA",
        "region": "East of England",
        "localAuthority": "Southend-on-Sea",
        "currentRatings": {
            "overall": {
                "rating": "Good",
                "reportDate": "2023-05-01",
                "keyQuestionRatings": [
                    {"name": "Safe", "rating": "Requires improvement"},
                    {"name": "Effective", "rating": "Good"},
                    {"name": "Caring", "rating": "Good"},
                    {"name": "Responsive", "rating": "Good"},
                    {"name": "Well-led", "rating": "Good"},
                ],
            }
        },
        "gacServiceTypes": [{"name": "Care home service with nursing"}],
        "specialisms": [{"name": "Dementia"}, {"name": "Caring for adults over 65 yrs"}],
        "regulatedActivities": [{"name": "Accommodation for persons who require nursing or personal care"}],
        "_cqc_url": "https://www.cqc.org.uk/location/1-123456789",
        "_lookup_type": "location",
    }
    s = client.summarise_provider_profile(raw)
    check("overall_rating", s["overall_rating"], "Good")
    check("number_of_beds", s["number_of_beds"], 42)
    check("registration_date", s["registration_date"], "2011-04-01")
    check("local_authority", s["local_authority"], "Southend-on-Sea")
    check("sub_ratings Safe", s.get("sub_ratings", {}).get("Safe"), "Requires improvement")
    check("sub_ratings count", len(s.get("sub_ratings", {})), 5)
    check("specialisms", s["specialisms"], ["Dementia", "Caring for adults over 65 yrs"])
    check("service_types", s["service_types"], ["Care home service with nursing"])
    check("website extracted", s["website"], "https://sunnyfields.example.co.uk")
    check("url id extraction", CQCClient.extract_id_from_url(raw["_cqc_url"]),
          {"type": "location", "id": "1-123456789"})


def test_safe_join():
    print("\n== safe_join tolerance ==")
    from analysis_agent import _safe_join
    check("mixed types", _safe_join([{"title": "A"}, "B", None, {"name": "C"}]), "A, B, C")
    check("empty", _safe_join([]), "")
    check("limit", _safe_join(["a", "b", "c"], limit=2), "a, b")


def test_fuzzy_target_match():
    print("\n== Fuzzy target name matching (typo tolerance) ==")
    from research_agent import _fuzzy_name_ratio
    # Typos that SHOULD fuzzy-match (>= 0.84)
    check_true("'Ashley Car Ltd' ~ 'Ashley Care'", _fuzzy_name_ratio("Ashley Car Ltd", "Ashley Care") >= 0.84)
    check_true("'Bluebird Cair' ~ 'Bluebird Care'", _fuzzy_name_ratio("Bluebird Cair", "Bluebird Care") >= 0.84)
    # Different companies that should NOT match
    check_true("'Ashley Care' !~ 'Allied Healthcare'", _fuzzy_name_ratio("Ashley Care", "Allied Healthcare") < 0.84)
    check_true("'Mencap' !~ 'Mears Care'", _fuzzy_name_ratio("Mencap", "Mears Care") < 0.84)


def test_service_type_filter():
    print("\n== CQC service-type scope filter (exclude GPs/dentists) ==")
    from research_agent import ResearchConfig, ResearchAgent
    cfg = ResearchConfig(commissioner="X", service_area="domiciliary care",
                         target_company="Y", time_period="z", research_depth="deep")
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.config = cfg
    agent._target_service_types = ["Homecare agencies"]
    wanted = agent._wanted_service_types()
    check_true("homecare agency in scope", agent._service_type_in_scope(["Homecare agencies"], wanted))
    check_true("supported living in scope (fallback hints)",
               agent._service_type_in_scope(["Supported living"], list(ResearchAgent._SOCIAL_CARE_HINTS)))
    check_true("GP surgery EXCLUDED", not agent._service_type_in_scope(["Doctors/Gps"], wanted))
    check_true("dentist EXCLUDED", not agent._service_type_in_scope(["Dentists"], wanted))
    check_true("empty EXCLUDED", not agent._service_type_in_scope([], wanted))


def test_care_home_mapping():
    print("\n== Service area → CQC careHome filter ==")
    from research_agent import ResearchConfig, ResearchAgent
    from search_providers.llm_web import LLMWebProvider

    def make(service):
        cfg = ResearchConfig(
            commissioner="X", service_area=service, target_company="Y",
            time_period="z", research_depth="deep",
        )
        agent = ResearchAgent.__new__(ResearchAgent)  # skip __init__ (no prompts/keys)
        agent.config = cfg
        agent._target_service_types = []  # no target lookup in this unit test
        return agent._service_is_care_home()

    check("residential care -> True", make("residential care"), True)
    check("nursing home -> True", make("nursing home"), True)
    check("domiciliary care -> False", make("domiciliary care"), False)
    check("supported living -> False", make("supported living"), False)
    check("home care -> False", make("home care"), False)
    check("unknown thing -> None", make("widget manufacturing"), None)


def test_cqc_list_parsing():
    print("\n== CQC list_locations response parsing ==")
    # Validate the list endpoint shape handling without network
    sample = {"locations": [
        {"locationId": "1-111", "locationName": "Alpha Home"},
        {"locationId": "1-222", "locationName": "Beta Lodge"},
    ]}
    locs = []
    for loc in sample.get("locations", []):
        if loc.get("locationId"):
            locs.append({"locationId": loc["locationId"],
                         "locationName": loc.get("locationName") or ""})
    check("parsed 2 locations", len(locs), 2)
    check("first name", locs[0]["locationName"], "Alpha Home")


def test_json_extraction():
    print("\n== Robust JSON extraction ==")
    from research_agent import _extract_json
    # Prose before and after (common with Claude + web search)
    t1 = 'Here is the analysis you requested:\n{"a": 1, "b": [2, 3]}\nLet me know if you need more.'
    check("prose-wrapped", _extract_json(t1), {"a": 1, "b": [2, 3]})
    # Markdown fences
    t2 = '```json\n{"x": "y"}\n```'
    check("fenced", _extract_json(t2), {"x": "y"})
    # Trailing comma (invalid strict JSON)
    t3 = '{"a": 1, "b": 2,}'
    check("trailing comma", _extract_json(t3), {"a": 1, "b": 2})
    # Stray brace in prose before the object
    t4 = 'Note: use {curly} carefully. Result: {"ok": true}'
    check("stray brace then object", _extract_json(t4), {"ok": True})
    # Nested braces
    t5 = '{"outer": {"inner": {"deep": 1}}}'
    check("nested", _extract_json(t5), {"outer": {"inner": {"deep": 1}}})
    # Garbage
    check("no json", _extract_json("totally not json"), {})


def test_enrichment_status():
    print("\n== enrichment status computation ==")
    from research_agent import _compute_enrichment_status
    comp = {
        "website": "https://x.co.uk",
        "cqc_profile_url": "https://www.cqc.org.uk/location/1-1",
        "companies_house_number": "09782291",
        "known_contracts_with_commissioner": [{"title": "X"}],
    }
    s = _compute_enrichment_status(comp, {})
    check("website_found", s["website_found"], True)
    check("cqc_found", s["cqc_found"], True)
    check("companies_house_found", s["companies_house_found"], True)
    check("contracts_found", s["contracts_found"], True)


if __name__ == "__main__":
    test_name_matching()
    test_url_filter()
    test_cqc_parsing()
    test_safe_join()
    test_fuzzy_target_match()
    test_service_type_filter()
    test_care_home_mapping()
    test_cqc_list_parsing()
    test_json_extraction()
    test_enrichment_status()
    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    print("✅ All logic tests passed.")
